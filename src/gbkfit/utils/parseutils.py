"""
Parsing of configurations.

A configuration describes an object with a dict of options: the
parameters of a constructor of its class, __init__ or a named one (e.g.
from_file; see Serializable.config_constructors). A configuration with
the option 'type' describes an object of one of several classes (e.g. a
component), which is chosen by its type. The errors in a configuration
have the path to the part with the error (see ConfigError).
"""

import abc
import contextlib
import contextvars
import copy
import difflib
import importlib
import importlib.util
import inspect
import logging
import os
import re
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, ClassVar, Literal, NamedTuple, Self

import astropy.units

from . import funcutils, iterutils, typeutils


_log = logging.getLogger(__name__)


def make_typed_desc(
        cls: type['TypedSerializable'],
        label: str | None = None
) -> str:
    """
    Describe a typed class, for messages.

    Parameters
    ----------
    cls : type
        The class.
    label : str, optional
        What the class is (e.g. 'observable').

    Returns
    -------
    str
        The description (e.g. 'observable pixel_moments
        (class=PixelMoments)').
    """
    desc = f'{cls.type()} (class={cls.__qualname__})'
    return f'{label} {desc}' if label else desc


# Whether unknown options are errors (strict, the default) or warnings
# (lenient)
_strict = contextvars.ContextVar('strict', default=True)


@contextlib.contextmanager
def strict_mode(strict: bool = True) -> Iterator[None]:
    """
    Make unknown options errors (strict, the default) or warnings
    (lenient), in the block.

    This also applies to unknown sections and parameters. Lenient mode
    leaves them out with a warning, e.g. to load a configuration with an
    option of a newer version of GBKFIT.

    Parameters
    ----------
    strict : bool, optional
        Whether unknown options are errors.
    """
    token = _strict.set(strict)
    try:
        yield
    finally:
        _strict.reset(token)


def is_strict() -> bool:
    """
    Check whether unknown options are errors (see strict_mode).

    Returns
    -------
    bool
        Whether unknown options are errors.
    """
    return _strict.get()


def describe_unknown(unknown: Iterable, known: Iterable[str]) -> str:
    """
    Describe unknown names, each with the most similar known name.

    Parameters
    ----------
    unknown : Iterable
        The unknown names.
    known : Iterable of str
        The known names.

    Returns
    -------
    str
        The unknown names, each with the most similar known name, if any
        (e.g. "'rnmx' (did you mean 'rnmax'?), 'zzz'").
    """
    known = list(known)
    descs = []
    for name in sorted(map(str, unknown)):
        similar = difflib.get_close_matches(name, known, n=1)
        descs.append(
            f"'{name}' (did you mean '{similar[0]}'?)" if similar
            else f"'{name}'")
    return ', '.join(descs)


def report_unknown(message: str, unknown: Iterable, known: Iterable[str]):
    """
    Report unknown names: an error, or in lenient mode a warning that they
    are ignored (see strict_mode).

    Parameters
    ----------
    message : str
        What the names are (e.g. 'unknown options').
    unknown : Iterable
        The unknown names.
    known : Iterable of str
        The known names, to suggest similar ones.

    Raises
    ------
    UnknownNamesError
        Unless in lenient mode.
    """
    error = UnknownNamesError(
        f"{message}: {describe_unknown(unknown, known)}")
    if is_strict():
        raise error
    _log_problem(error, "they are ignored (lenient mode)")


def warn(message: str) -> None:
    """
    Log a warning about the part of the configuration being loaded.

    The warning starts with the path to that part, as errors do (see
    config_path). Its record has it as a ConfigError too (its attribute
    config_problem), for programs that collect the warnings (e.g. to
    show them in place).

    Parameters
    ----------
    message : str
        The warning.
    """
    _log_problem(ConfigError(message))


def _log_problem(error: 'ConfigError', note: str | None = None) -> None:
    """
    Log a problem of the part of the configuration being loaded as a
    warning (see warn), with a note after it in the log only.
    """
    # (an error raised here would get the path of each config_path block
    # around it, from the innermost out)
    for segments, context in reversed(_location.get()):
        _add_location(error, segments, context)
    text = str(error) if note is None else f"{error}; {note}"
    _log.warning(text, extra=dict(config_problem=error))


class Problem(NamedTuple):
    """
    A problem of a configuration: an error or a warning, its path in the
    configuration (option names and list indices; e.g. ['galaxies', 0]),
    and its message (with the type of the part, if known).
    """
    severity: Literal['error', 'warning']
    path: list[str | int]
    message: str


@contextlib.contextmanager
def collect_problems() -> Iterator[list[Problem]]:
    """
    Collect the problems of the configuration loaded in the block: the
    warnings logged about it (see warn), its unknown names, as errors (see
    report_unknown, in lenient mode), and the ConfigError that ends the
    block, if any, which it stops; each with its path, in order.

    Yields
    ------
    list of Problem
        The problems, as they come.
    """
    problems: list[Problem] = []
    handler = _ProblemHandler(problems)
    logger = logging.getLogger('gbkfit')
    logger.addHandler(handler)
    try:
        yield problems
    except ConfigError as e:
        problems.append(_problem('error', e))
    finally:
        logger.removeHandler(handler)


class _ProblemHandler(logging.Handler):
    """
    The handler of the warnings about a configuration logged (see warn),
    and its unknown names, as errors (see report_unknown), which adds them
    to a list of problems.
    """

    def __init__(self, problems: list[Problem]):
        super().__init__(logging.WARNING)
        self._problems = problems

    def emit(self, record: logging.LogRecord) -> None:
        if (error := getattr(record, 'config_problem', None)) is not None:
            severity = 'error' if isinstance(error, UnknownNamesError) \
                else 'warning'
            self._problems.append(_problem(severity, error))


def _problem(
        severity: Literal['error', 'warning'], error: 'ConfigError'
) -> Problem:
    """Return the problem of an error (or of a warning)."""
    message = error.message if error.context is None \
        else f"[{error.context}] {error.message}"
    return Problem(severity, list(error.path), message)


class ConfigError(RuntimeError):
    """
    An error in a configuration.

    Parameters
    ----------
    message : str
        The description of the error.
    path : Sequence of str or int, optional
        The path to the part of the configuration with the error: option
        names and list indices (e.g. models[0].components[1]).
    context : str, optional
        The type of that part (e.g. 'disk'), if any.
    """

    def __init__(
            self,
            message: str,
            path: Sequence[str | int] = (),
            context: str | None = None
    ):
        super().__init__(message)
        self.message = message
        self.path = list(path)
        self.context = context

    def __str__(self):
        path = ''.join(
            f'[{segment}]' if isinstance(segment, int) else f'.{segment}'
            for segment in self.path).lstrip('.')
        if self.context:
            path = f'{path} [{self.context}]' if path else self.context
        return f'{path}: {self.message}' if path else self.message


class UnknownNamesError(ConfigError):
    """
    An error of unknown names in a configuration: options, sections or
    parameters (see report_unknown).
    """


# The parts of the configuration being loaded (see config_path): the
# segments and the context of each
_location = contextvars.ContextVar('location', default=())


def _add_location(error: ConfigError, segments, context) -> None:
    """Add the segments and the context of a part to an error."""
    if error.context is None and not error.path:
        error.context = context
    error.path[:0] = segments


@contextlib.contextmanager
def config_path(
        *segments: str | int,
        context: str | None = None
) -> Iterator[None]:
    """
    Add to the path of the configuration errors raised in the block.

    Other errors become configuration errors. The warnings of warn get
    the path too.

    Parameters
    ----------
    *segments : str or int
        The option names and list indices to add.
    context : str, optional
        The type of the part of the configuration loaded in the block
        (e.g. 'disk'), for the errors in that part itself (not in its
        options).
    """
    token = _location.set(_location.get() + ((segments, context),))
    try:
        yield
    except ConfigError as e:
        _add_location(e, segments, context)
        raise
    except Exception as e:
        raise ConfigError(str(e), segments, context) from e
    finally:
        _location.reset(token)


# The names of items of lists (e.g. components): letters, digits and
# underscores
_NAME = re.compile(r'[A-Za-z0-9_]+')


def check_name(name: str | None) -> None:
    """
    Check the name of an item of a list (e.g. a component).

    Parameters
    ----------
    name : str or None
        The name: letters, digits and underscores; or None.

    Raises
    ------
    ConfigError
        If the name is invalid.
    """
    if name is not None and not (
            isinstance(name, str) and _NAME.fullmatch(name)):
        raise ConfigError(
            f"invalid name {name!r}: use only letters, digits and "
            f"underscores")


def item_prefixes(
        names: Sequence[str | None],
        label: str,
        prefix: str,
        prefix_first: bool
) -> list[str]:
    """
    Return the prefixes of the parameter names of the items of a list.

    The prefixes also apply to the names of the constants of the items.
    They come from the names of the items, if every item has one, or else
    from their positions.

    Parameters
    ----------
    names : Sequence of str or None
        The names of the items.
    label : str
        What the items are, for messages (e.g. 'components').
    prefix : str
        The prefix of the items without a name (e.g. 'cmp').
    prefix_first : bool
        Whether the first item without a name has a prefix.

    Returns
    -------
    list of str
        The prefix of each item: its name and '_' (e.g. 'disk_'), or the
        prefix, its index (none for the first item) and '_' (e.g. 'cmp1_');
        an empty string for the first item without prefix_first.

    Raises
    ------
    ConfigError
        If only some of the items have a name, or the names are repeated.
    """
    named = [name is not None for name in names]
    if not any(named):
        return [
            f'{prefix}{i or ""}_' if i or prefix_first else ''
            for i in range(len(names))]
    if not all(named):
        raise ConfigError(
            f"either all or none of the {label} must have a name; only "
            f"{sum(named)} of {len(names)} have one")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigError(
            f"the names of the {label} must be unique; repeated: {repeated}")
    return [f'{name}_' for name in names]


def _quoted(names: Iterable) -> str:
    """Join the names in quotes (e.g. "'a', 'b'")."""
    return ', '.join(f"'{name}'" for name in sorted(names))


def parse_options(
        info: dict[str, Any],
        required: Iterable[str] | None = None,
        optional: Iterable[str] | None = None
) -> dict[str, Any]:
    """
    Check the options of a configuration, and return the known ones.

    Unknown options are reported (see report_unknown) and left out.

    Parameters
    ----------
    info : dict
        The options.
    required : Iterable of str, optional
        The names of the required options.
    optional : Iterable of str, optional
        The names of the optional options.

    Returns
    -------
    dict
        The known options.

    Raises
    ------
    ConfigError
        If a required option is missing, or an option is unknown (unless
        in lenient mode, see strict_mode).
    """
    required = iterutils.setify(required)
    optional = iterutils.setify(optional)
    if both := required & optional:
        raise RuntimeError(
            f"options cannot be both required and optional: {_quoted(both)}")
    known = required | optional
    if unknown := set(info) - known:
        report_unknown("unknown options", unknown, known)
    if missing := required - set(info):
        raise ConfigError(f"missing required options: {_quoted(missing)}")
    return {k: v for k, v in info.items() if k in known}


# The directory that relative paths in configurations are relative to,
# while they are loaded (see paths_relative_to)
_base_dir: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    'base_dir', default=None)


@contextlib.contextmanager
def paths_relative_to(directory: str) -> Iterator[None]:
    """
    Resolve the relative paths of configurations (see resolve_path)
    against a directory, e.g. that of the configuration file, while
    loading them.

    Parameters
    ----------
    directory : str
        The directory.
    """
    token = _base_dir.set(os.path.abspath(directory))
    try:
        yield
    finally:
        _base_dir.reset(token)


def resolve_path(path: str) -> str:
    """
    Return a path of a configuration: a relative path is relative to the
    directory of paths_relative_to, if any (and to the current directory
    otherwise).
    """
    base = _base_dir.get()
    if base is None or os.path.isabs(path):
        return path
    return os.path.join(base, path)


def parse_file(x: str | dict[str, Any]) -> tuple[str, int | str]:
    """
    Parse a file option (a relative filename is resolved by resolve_path).

    Parameters
    ----------
    x : str or dict
        A filename, or a dict with the filename ('file') and, optionally,
        the HDU ('hdu', e.g. 'SCI').

    Returns
    -------
    tuple of str and (int or str)
        The filename and the HDU (by default the first, 0).

    Raises
    ------
    ConfigError
        If the dict has no filename.
    """
    if isinstance(x, str):
        x = dict(file=x)
    options = parse_options(x, required={'file'}, optional={'hdu'})
    return resolve_path(options['file']), options.get('hdu', 0)


def parse_options_for_callable(
        info: dict[str, Any],
        func: Callable,
        ignore_params: Iterable[str] | None = None
) -> dict[str, Any]:
    """
    Check the options of a configuration against the parameters of a
    callable, and return them as its arguments.

    The parameters without a default value are required options, and the
    others optional ones; a ``**kwargs`` parameter takes the other
    options. An option must match the type annotation of its parameter,
    if any, and its range (see typeutils.problem). Unknown options are
    reported (see report_unknown) and left out.

    Options of some types are given in a form of their own (see
    from_config): a FilePath is relative to the directory of the
    configuration (see resolve_path), a FitsFile is a filename or a dict
    with the filename (file) and the HDU (hdu), a record (a NamedTuple,
    e.g. a Column) is a dict of its fields, a Quantity is a dict with its
    value and unit (e.g. {value: [1, 2], unit: kpc}), and the values of a
    Mapping are in the forms of their type.

    Parameters
    ----------
    info : dict
        The options.
    func : Callable
        The callable (e.g. the __init__ of a class).
    ignore_params : Iterable of str, optional
        The parameters that are not options (e.g. those given by the
        caller).

    Returns
    -------
    dict
        The options, by parameter.

    Raises
    ------
    ConfigError
        If a required option is missing, an option does not match its
        type, or an option is unknown (unless in lenient mode).
    RuntimeError
        If func has no parameters of the names of ignore_params.
    """
    ignore_params = iterutils.setify(ignore_params)
    required, optional = funcutils.parameter_names(func)
    if unknown := ignore_params - set(required) - set(optional):
        raise RuntimeError(
            f"{func.__qualname__} has no parameters {_quoted(unknown)}")
    # (object.__init__, of a class without options, refuses the **kwargs
    # of its signature)
    others = None if func is object.__init__ else next((
        p.name for p in inspect.signature(func).parameters.values()
        if p.kind is inspect.Parameter.VAR_KEYWORD), None)
    known = set(required) | set(optional) | ignore_params
    options = parse_options(
        info,
        [p for p in required if p not in ignore_params],
        [p for p in optional if p not in ignore_params]
        + ([] if others is None else [k for k in info if k not in known]))
    types = typeutils.option_types(func)
    args = {}
    for option, value in options.items():
        type_ = types.get(option if option in known else others)
        if type_ is not None:
            with config_path(option):
                value = from_config(value, type_)
            if problem := typeutils.problem(value, type_):
                raise ConfigError(f"option '{option}' {problem}")
        args[option] = value
    return args


def from_config(value: Any, type_: Any) -> Any:
    """
    Return the value of an option of a type annotation from the form a
    configuration gives it in (see parse_options_for_callable); values in
    no form of their own are returned as they are.

    Parameters
    ----------
    value : Any
        The value in the configuration.
    type_ : Any
        The annotation.

    Returns
    -------
    Any
        The value.

    Raises
    ------
    ConfigError
        If the value is in a form of its own and invalid.
    """
    for member in typeutils.union_members(type_):
        if member == typeutils.FilePath and isinstance(value, str):
            return resolve_path(value)
        if member == typeutils.FitsFile and isinstance(value, str | Mapping):
            file, hdu = parse_file(value)
            return (file, hdu) if isinstance(value, Mapping) else file
        if is_record(member) and isinstance(value, Mapping):
            return member(**parse_options_for_callable(dict(value), member))
        if typing.get_origin(member) is Mapping \
                and isinstance(value, Mapping):
            item_type = typing.get_args(member)[1]
            items = {}
            for key, item in value.items():
                with config_path(key):
                    items[key] = from_config(item, item_type)
            return items
        if member is astropy.units.Quantity and isinstance(value, Mapping):
            return _quantity(value)
    return value


def is_record(type_: Any) -> bool:
    """
    Return whether a type annotation is a record (a NamedTuple, e.g.
    typeutils.Column), which configurations give as a dict of its fields.
    """
    return isinstance(type_, type) and issubclass(type_, tuple) \
        and hasattr(type_, '_fields')


def _quantity(info: Mapping[str, Any]) -> astropy.units.Quantity:
    """Return the Quantity of a dict with its value and unit."""
    # (unitutils imports this module)
    from gbkfit.utils import unitutils
    options = parse_options(dict(info), required={'value', 'unit'})
    with config_path('unit'):
        unit = unitutils.parse_unit(options['unit'])
    try:
        return astropy.units.Quantity(options['value'], unit, dtype=float)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"option 'value' must be a number or a list of numbers; it is "
            f"{options['value']!r}") from e


def check_options(func: Callable, values: Mapping[str, Any]) -> None:
    """
    Check the arguments of a call of a constructor against the type
    annotations of its parameters: their types, ranges (e.g.
    ``Annotated[float, Gt(0)]``) and choices (``Literal``; see
    typeutils.problem).

    Each constructor that configurations call (see
    Serializable.config_constructors) calls it first, with itself, named
    by the class that defines it, and its arguments, e.g.
    ``check_options(PSFGauss.__init__, locals())``: options are checked
    alike in Python and in configurations. (``self.__init__`` would be
    that of a subclass, in a constructor that subclasses call.)

    Parameters
    ----------
    func : Callable
        The constructor.
    values : Mapping
        The arguments, by name (others are ignored, e.g. self, and so are
        those of ``*args`` and ``**kwargs``).

    Raises
    ------
    ConfigError
        If an argument does not match its annotation.
    """
    types = typeutils.option_types(func)
    for name in funcutils.parameter_names(func).all:
        if name in types and name in values \
                and (problem := typeutils.problem(values[name], types[name])):
            raise ConfigError(f"option '{name}' {problem}")


def sanitize_dimensional_options(
        info: dict[str, Any],
        options: dict[str, Any],
        length: int
) -> None:
    """
    Make the options with one value per dimension lists, in place.

    A single value is repeated, and the extra values of a longer list are
    dropped with a warning (so that, e.g., the configuration of a spectral
    cube can be used for an image by changing its type). Options that are
    missing or null are left as they are.

    Parameters
    ----------
    info : dict
        The options.
    options : dict
        The type of the values of each option (e.g. dict(size=int,
        step=float)).
    length : int
        The number of dimensions.

    Raises
    ------
    ConfigError
        If an option has values of another type, or too few values.
    """
    for option, type_ in options.items():
        value = info.get(option)
        if value is None:
            continue
        if typeutils.matches_type(value, type_):
            info[option] = [value] * length
        elif (iterutils.is_sequence(value) and len(value) >= length
              and all(typeutils.matches_type(x, type_) for x in value)):
            if len(value) > length:
                warn(
                    f"option '{option}' has {len(value)} values, but only "
                    f"{length} are needed; the values will be trimmed from "
                    f"{list(value)} to {list(value[:length])}")
            info[option] = list(value[:length])
        else:
            raise ConfigError(
                f"option '{option}' must be a value of type "
                f"{typeutils.describe_type(type_)}, or a sequence of {length} "
                f"such values; it is {value!r}")


def load_option(
        loader: Callable,
        info: dict[str, Any],
        key: str,
        *,
        required: bool = False,
        **kwargs
) -> Any:
    """
    Load an option.

    Use it when the value goes elsewhere (e.g. under another name, or
    unpacked); load_option_and_update_info replaces the option in place.

    Parameters
    ----------
    loader : Callable
        The loader of the value of the option (e.g. a parser's load).
    info : dict
        The options.
    key : str
        The name of the option.
    required : bool, optional
        Whether the option must be given and not be null.
    **kwargs
        Passed to the loader.

    Returns
    -------
    Any
        The loaded value, or None if the option is missing or null.

    Raises
    ------
    ConfigError
        If a required option is missing or null, or the loader fails.
    """
    if info.get(key) is None:
        if required:
            raise ConfigError(
                f"option '{key}' cannot be null" if key in info else
                f"option '{key}' is required but not provided")
        return None
    with config_path(key):
        return loader(info[key], **kwargs)


def load_option_and_update_info(
        parser: 'Parser',
        info: dict[str, Any],
        key: str,
        *,
        required: bool = False,
        **kwargs
) -> None:
    """
    Load an option with a parser, and replace its value with the result.

    Use it when the loaded value replaces the option; load_option returns
    it instead. A missing option stays missing.

    Parameters
    ----------
    parser : Parser
        The parser of the option.
    info : dict
        The options, updated in place.
    key : str
        The name of the option.
    required : bool, optional
        Whether the option must be given and not be null.
    **kwargs
        Passed to the parser.

    Raises
    ------
    ConfigError
        If a required option is missing or null, or the parser fails.
    """
    value = load_option(parser.load, info, key, required=required, **kwargs)
    if key in info:
        info[key] = value


class Serializable(abc.ABC):
    """
    An object that can be loaded from, and dumped to, a configuration.
    """

    # The constructors that configurations call, by name: '__init__' or
    # named ones (e.g. 'from_file'). A configuration has the options of
    # one of them, which it is the only one to have (see
    # config_constructor)
    config_constructors: tuple[str, ...] = ('__init__',)

    # The options of its configurations that are not those of its
    # config_constructors, or are in other forms, and their types, which
    # its load reads (e.g. the named geometries of a galaxy, which its
    # components name: a configuration cannot hold one object twice)
    config_options: ClassVar[Mapping[str, Any]] = {}

    # The parameters of its config_constructors that the caller of its
    # load gives, not its configurations (e.g. the data of an observable,
    # which its observation gives)
    given_options: tuple[str, ...] = ()

    # The modules it needs beyond those GBKFIT always has (e.g. optional
    # packages), each with how to install it: without them it can be
    # described, but not made (see missing_modules)
    requires: ClassVar[Mapping[str, str]] = {}

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        """
        Load an object from its configuration.

        By default, the options are the parameters of one of its
        config_constructors (see config_constructor and
        parse_options_for_callable).

        Parameters
        ----------
        info : dict
            The options.

        Returns
        -------
        Self
            The object.
        """
        name = config_constructor(cls, info)
        func = cls.__init__ if name == '__init__' else getattr(cls, name)
        args = parse_options_for_callable(info, func)
        return cls(**args) if name == '__init__' else func(**args)

    @abc.abstractmethod
    def dump(self, *args, **kwargs) -> dict[str, Any]:
        """
        Dump the object to its configuration.

        Returns
        -------
        dict
            The options.
        """


def missing_modules(cls: type[Serializable]) -> dict[str, str]:
    """
    Return the modules that a class needs (see Serializable.requires) and
    that are not installed, each with how to install it.

    Parameters
    ----------
    cls : type
        The class.

    Returns
    -------
    dict
        How to install each missing module, by its name.
    """
    return {
        module: how for module, how in cls.requires.items()
        if not _is_installed(module)}


def _is_installed(module: str) -> bool:
    """Return whether a module can be found, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        # (a missing package of the module)
        return False


def config_constructor(
        cls: type[Serializable], info: Mapping[str, Any]
) -> str:
    """
    Return the name of the constructor of a class (one of its
    config_constructors) whose options a configuration has: the one
    whose own options (those the others do not have) it has, or, if it
    has none of those, the one without own options (whose options the
    others all have, e.g. DatasetRegionMoments.from_files).

    Parameters
    ----------
    cls : type
        The class.
    info : Mapping
        The options.

    Returns
    -------
    str
        The name of the constructor.

    Raises
    ------
    ConfigError
        If the configuration has the own options of none or of several of
        the constructors.
    """
    names = cls.config_constructors
    if len(names) == 1:
        return names[0]
    params = {}
    for name in names:
        func = cls.__init__ if name == '__init__' else getattr(cls, name)
        params[name] = set(funcutils.parameter_names(func).all) \
            - set(cls.given_options)
    own = {
        name: params[name].difference(
            *(params[other] for other in names if other != name))
        for name in names}
    given = [name for name in names if own[name] & set(info)]
    default = [name for name in names if not own[name]]
    if len(given) == 1 or (not given and len(default) == 1):
        return (given or default)[0]
    forms = '; or '.join(_quoted(own[name]) for name in names)
    raise ConfigError(
        f"give the options of one of its forms, which have {forms}; it has "
        + (f"those of several: "
           f"{_quoted(set(info) & set().union(*own.values()))}"
           if given else "none of them"))


class TypedSerializable(Serializable, abc.ABC):
    """
    A serializable object of one of several classes, chosen by the type
    in its configuration (see TypedParser).
    """

    @staticmethod
    @abc.abstractmethod
    def type() -> str:
        """
        Return the type of the class in configurations.

        Returns
        -------
        str
            The type (e.g. 'disk').
        """


def _prepare_args_and_kwargs(
        length: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any]
) -> tuple[list[list[Any]], list[dict[str, Any]]]:
    """
    Split the arguments for a list of items into those of each item.

    Each argument is a sequence with a value for each item, or None for
    None for each item.
    """
    def validate_sequence(value, name):
        if value is None:
            return [None] * length
        if not iterutils.is_sequence(value) or len(value) != length:
            raise RuntimeError(
                f"expected '{name}' to be a sequence of length {length}, "
                f"but got: {value}")
        return value
    args_list = [[] for _ in range(length)]
    for i, arg in enumerate(args):
        for j, value_ in enumerate(validate_sequence(arg, f"args[{i}]")):
            args_list[j].append(value_)
    kwargs_list = [{} for _ in range(length)]
    for key, val in kwargs.items():
        for j, value_ in enumerate(validate_sequence(val, f"kwargs['{key}']")):
            kwargs_list[j][key] = value_

    return args_list, kwargs_list


class Parser(abc.ABC):
    """
    The loader and dumper of the configurations of the objects of a class.

    A configuration is a dict, which describes one object, or a list of
    dicts, which describes a list of objects. A null configuration
    describes no object (None).

    Parameters
    ----------
    cls : type
        The class.
    """

    def __init__(self, cls: type[Serializable]):
        self._cls = cls

    def cls(self) -> type[Serializable]:
        return self._cls

    def cls_name(self) -> str:
        return self.cls().__qualname__

    def load(self, x, *args, **kwargs) -> Any:
        """
        Load an object, or a list of objects, from its configuration.

        The configuration is not changed.

        Parameters
        ----------
        x : dict or list of dict or None
            The configuration.
        *args, **kwargs
            Passed to the load of the class; for a list, sequences with a
            value for each item.

        Returns
        -------
        Any
            The object, or the list of objects.

        Raises
        ------
        ConfigError
            If the configuration has an error.
        """
        if iterutils.is_sequence(x):
            result = self.load_many(x, *args, **kwargs)
        else:
            result = self.load_one(x, *args, **kwargs)
        return result

    def load_one(self, x, *args, **kwargs) -> Any:
        return self._load_one_impl_wrapper(x, None, *args, **kwargs)

    def _load_one_impl_wrapper(self, x, index, *args, **kwargs):
        # Create a copy of the configuration for safety
        x = copy.deepcopy(x)
        with config_path(*([] if index is None else [index])):
            if not isinstance(x, (dict, type(None))):
                raise ConfigError(
                    f"expected configuration in the form of a dictionary; "
                    f"instead it found the following value: {x}")
            return self._load_one_impl(x, *args, **kwargs) \
                if x is not None else None

    @abc.abstractmethod
    def _load_one_impl(self, x, *args, **kwargs):
        pass

    def load_many(self, x, *args, **kwargs) -> Any:
        args_list, kwargs_list = _prepare_args_and_kwargs(
            len(x), args, kwargs)
        results = []
        for i, (item, item_args, item_kwargs) in enumerate(
                zip(x, args_list, kwargs_list)):
            results.append(self._load_one_impl_wrapper(
                item, i, *item_args, **item_kwargs))
        return results

    def dump(self, x, *args, **kwargs):
        """
        Dump an object, or a list of objects, to its configuration.

        Parameters
        ----------
        x : Serializable or list of Serializable or None
            The object, or the list of objects.
        *args, **kwargs
            Passed to the dump of the class; for a list, sequences with a
            value for each item.

        Returns
        -------
        dict or list of dict or None
            The configuration.
        """
        if iterutils.is_sequence(x):
            result = self.dump_many(x, *args, **kwargs)
        else:
            result = self.dump_one(x, *args, **kwargs)
        return result

    def dump_one(self, x: Serializable, *args, **kwargs) -> dict[str, Any]:
        return None if x is None else self._dump_one_impl(x, *args, **kwargs)

    @abc.abstractmethod
    def _dump_one_impl(self, x: Serializable, *args, **kwargs) -> dict[str, Any]:
        pass

    def dump_many(
            self,
            x: list[Serializable],
            *args,
            **kwargs
    ) -> list[dict[str, Any]]:
        args_list, kwargs_list = _prepare_args_and_kwargs(len(x), args, kwargs)
        results = []
        for item, item_args, item_kwargs in zip(x, args_list, kwargs_list):
            results.append(self.dump_one(item, *item_args, **item_kwargs))
        return results


class BasicParser(Parser):
    """
    The parser of a class with one form of configuration.
    """

    def _load_one_impl(self, x, *args, **kwargs):
        return self.cls().load(x, *args, **kwargs)

    def _dump_one_impl(self, x, *args, **kwargs):
        return x.dump(*args, **kwargs)


class TypedParser(Parser):
    """
    The parser of a class whose configurations choose one of its
    subclasses by their type (the option 'type').

    Subclasses that need modules GBKFIT does not always have (see
    Serializable.requires) are registered as the others, and their
    options can be read without them; asking for one whose modules are
    missing says what to install.

    Parameters
    ----------
    cls : type
        The class.
    classes : type or list of type, optional
        The subclasses to register (see register).
    """

    def __init__(
            self,
            cls: type[TypedSerializable],
            classes: (type[TypedSerializable] | list[type[TypedSerializable]]
                      | None) = None
    ):
        super().__init__(cls)
        self._classes: dict[str, type[TypedSerializable]] = {}
        self.register(classes)

    def register(
            self,
            classes: (type[TypedSerializable] | list[type[TypedSerializable]]
                      | None)
    ) -> None:
        """
        Register subclasses, each for its type.

        Parameters
        ----------
        classes : type or list of type or None
            The subclasses.

        Raises
        ------
        RuntimeError
            If a class is not a subclass, or its type is registered.
        """
        for cls in iterutils.listify(classes):
            desc = f"{cls.__qualname__} (type '{cls.type()}')"
            if not issubclass(cls, self.cls()):
                raise RuntimeError(
                    f"{self.cls_name()} parser could not register {desc}; "
                    f"it is not a subclass of {self.cls_name()}")
            if cls.type() in self._classes:
                raise RuntimeError(
                    f"{self.cls_name()} parser could not register {desc}; "
                    f"a class of the same type is already registered")
            self._classes[cls.type()] = cls

    def types(self) -> list[str]:
        """
        Return all the registered types, those of classes whose modules
        are missing included (see Serializable.requires).
        """
        return sorted(self._classes)

    def registered_class(self, type_: str) -> type[TypedSerializable]:
        """
        Return the class registered for a type.

        Parameters
        ----------
        type_ : str
            The type.

        Returns
        -------
        type
            The class.

        Raises
        ------
        ConfigError
            If no class is registered for the type, or the modules it
            needs are missing (see Serializable.requires).
        """
        if not isinstance(type_, str) or type_ not in self._classes:
            raise ConfigError(
                f"unknown type {describe_unknown([type_], self.types())}; "
                f"the types are: {_quoted(self.types())}")
        cls = self._classes[type_]
        if missing := missing_modules(cls):
            how = '; '.join(dict.fromkeys(missing.values()))
            raise ConfigError(f"type '{type_}' needs {_quoted(missing)}: {how}")
        return cls

    def registered_classes(self) -> dict[str, type[TypedSerializable]]:
        """
        Return the registered classes, those whose modules are missing
        included (see Serializable.requires).

        Returns
        -------
        dict
            The class registered for each type.
        """
        return dict(self._classes)

    def _load_one_impl(
            self,
            x: dict[str, Any],
            *args, **kwargs
    ) -> TypedSerializable:
        if 'type' not in x:
            raise ConfigError(
                f"option 'type' is required but not provided; the types "
                f"are: {_quoted(self.types())}")
        type_ = x.pop('type')
        cls = self.registered_class(type_)
        with config_path(context=type_):
            return cls.load(x, *args, **kwargs)

    def _dump_one_impl(
            self,
            x: Serializable,
            *args, **kwargs
    ) -> dict[str, Any]:
        """Dump the object, with its type."""
        if not isinstance(x, TypedSerializable):
            raise RuntimeError(
                f"unsupported object; value: {x}, type: {type(x)}")
        return dict(type=x.type()) | x.dump(*args, **kwargs)
