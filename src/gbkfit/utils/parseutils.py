"""
Parsing of configurations.

A configuration describes an object with a dict of options: the
parameters of the __init__ of its class. A configuration with the option
'type' describes an object of one of several classes (e.g. a component),
which is chosen by its type. The errors in a configuration have the path
to the part with the error (see ConfigError).
"""

import abc
import contextlib
import contextvars
import copy
import difflib
import importlib
import inspect
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Self

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


# Whether unknown options are errors (strict) or warnings
_strict = contextvars.ContextVar('strict', default=False)


@contextlib.contextmanager
def strict_mode(strict: bool = True):
    """
    Make unknown options errors instead of warnings, in the block.

    This also applies to unknown sections and parameters.

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
    Report unknown names: a warning that they are ignored, or an error in
    strict mode.

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
    ConfigError
        In strict mode.
    """
    message = f"{message}: {describe_unknown(unknown, known)}"
    if is_strict():
        raise ConfigError(message)
    warn(f"{message}; they will be ignored")


def warn(message: str) -> None:
    """
    Log a warning about the part of the configuration being loaded.

    The warning starts with the path to that part, as errors do (see
    config_path).

    Parameters
    ----------
    message : str
        The warning.
    """
    # (an error raised here would get the path of each config_path block
    # around it, from the innermost out)
    located = ConfigError(message)
    for segments, context in reversed(_location.get()):
        _add_location(located, segments, context)
    _log.warning(str(located))


class ConfigError(RuntimeError):
    """
    An error in a configuration.

    Parameters
    ----------
    message : str
        The description of the error.
    path : Sequence of str or int, optional
        The path to the part of the configuration with the error: option
        names and list indices (e.g. models[0].gmodel.components[1]).
    context : str, optional
        The type of that part (e.g. 'smdisk'), if any.
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


# The parts of the configuration being loaded (see config_path): the
# segments and the context of each
_location = contextvars.ContextVar('location', default=())


def _add_location(error: ConfigError, segments, context) -> None:
    """Add the segments and the context of a part to an error."""
    if error.context is None and not error.path:
        error.context = context
    error.path[:0] = segments


@contextlib.contextmanager
def config_path(*segments: str | int, context: str | None = None):
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
        (e.g. 'smdisk'), for the errors in that part itself (not in its
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

    The prefixes also apply to the names of the constants and the extra
    outputs of the items. They come from the names of the items, if every
    item has one, or else from their positions.

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
        If a required option is missing, or an option is unknown in strict
        mode.
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


def parse_file(x: str | dict[str, Any]) -> tuple[str, int | str]:
    """
    Parse a file option.

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
    return options['file'], options.get('hdu', 0)


def parse_options_for_callable(
        info: dict[str, Any],
        func: Callable,
        ignore_params: Iterable[str] | None = None,
        rename_params: dict[str, str] | None = None,
        add_required: dict[str, Any] | None = None,
        add_optional: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Check the options of a configuration against the parameters of a
    callable, and return them as its arguments.

    The parameters without a default value are required options, and the
    others optional ones. An option must match the type annotation of its
    parameter, if any (see typeutils.matches_type). Unknown options are
    reported (see report_unknown) and left out.

    Parameters
    ----------
    info : dict
        The options.
    func : Callable
        The callable (e.g. the __init__ of a class).
    ignore_params : Iterable of str, optional
        The parameters that are not options (e.g. those given by the
        caller).
    rename_params : dict, optional
        The option name of each renamed parameter (e.g. dict(minimum='min')).
    add_required, add_optional : dict, optional
        The required and the optional options that are not parameters,
        with their types (None for any type).

    Returns
    -------
    dict
        The options: those of the parameters keyed by parameter name, and
        the added ones by option name.

    Raises
    ------
    ConfigError
        If a required option is missing, an option does not match its
        type, or an option is unknown in strict mode.
    """
    ignore_params = iterutils.setify(ignore_params)
    rename_params = rename_params or {}
    add_required = add_required or {}
    add_optional = add_optional or {}
    required, optional = funcutils.parameter_names(func)
    _check_param_options(
        func, set(required) | set(optional), ignore_params, rename_params,
        add_required, add_optional)
    options = parse_options(
        info,
        {rename_params.get(p, p) for p in required if p not in ignore_params}
        | set(add_required),
        {rename_params.get(p, p) for p in optional if p not in ignore_params}
        | set(add_optional))
    # The types of the options, and their parameters
    types = {
        rename_params.get(p, p): type_
        for p, type_ in inspect.get_annotations(func).items()}
    types |= add_required | add_optional
    option_params = {option: p for p, option in rename_params.items()}
    args = {}
    for option, value in options.items():
        type_ = types.get(option)
        if type_ is not None and not typeutils.matches_type(value, type_):
            raise ConfigError(
                f"option '{option}' must be of type "
                f"{typeutils.describe_type(type_)}; it is {value!r}")
        args[option_params.get(option, option)] = value
    return args


def _check_param_options(
        func, params, ignore_params, rename_params, add_required,
        add_optional):
    """
    Check the options of parse_options_for_callable against the
    parameters (params) of func.
    """
    name = func.__qualname__
    renamed = set(rename_params)
    new_names = list(rename_params.values())
    added = set(add_required) | set(add_optional)
    if unknown := (ignore_params | renamed) - params:
        raise RuntimeError(f"{name} has no parameters {_quoted(unknown)}")
    if both := ignore_params & renamed:
        raise RuntimeError(
            f"the parameters of {name} cannot be both ignored and renamed: "
            f"{_quoted(both)}")
    if repeated := {x for x in new_names if new_names.count(x) > 1}:
        raise RuntimeError(
            f"several parameters of {name} cannot be renamed to the same "
            f"name: {_quoted(repeated)}")
    if clashing := set(new_names) & params:
        raise RuntimeError(
            f"{name} has parameters with the new names {_quoted(clashing)}")
    if both := set(add_required) & set(add_optional):
        raise RuntimeError(
            f"added options cannot be both required and optional: "
            f"{_quoted(both)}")
    if clashing := added & (params | set(new_names)):
        raise RuntimeError(
            f"added options cannot have the names of the parameters of "
            f"{name}, or their new names: {_quoted(clashing)}")


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

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        """
        Load an object from its configuration.

        By default, the options are the parameters of __init__ (see
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
        return cls(**parse_options_for_callable(info, cls.__init__))

    @abc.abstractmethod
    def dump(self, *args, **kwargs) -> dict[str, Any]:
        """
        Dump the object to its configuration.

        Returns
        -------
        dict
            The options.
        """


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
            The type (e.g. 'smdisk').
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
        self._classes = {}
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
            If no class is registered for the type.
        """
        if not isinstance(type_, str) or type_ not in self._classes:
            raise ConfigError(
                f"unknown type {describe_unknown([type_], self._classes)}; "
                f"the available types are: {_quoted(self._classes)}")
        return self._classes[type_]

    def registered_classes(self) -> dict[str, type[TypedSerializable]]:
        """
        Return the registered classes.

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
                f"option 'type' is required but not provided; the available "
                f"types are: {_quoted(self._classes)}")
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


def register_optional_parsers(
        parser: TypedParser,
        classes: list[str],
        desc: str
) -> None:
    """
    Register the classes that can be imported.

    A class that cannot be imported (e.g. because it needs an optional
    package) is skipped with a warning.

    Parameters
    ----------
    parser : TypedParser
        The parser.
    classes : list of str
        The classes, by module and name (e.g.
        'gbkfit.driver.drivers.cuda.DriverCuda').
    desc : str
        What the classes are, for messages (e.g. 'driver').
    """
    for path in classes:
        try:
            module_name, class_name = path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            parser.register(getattr(module, class_name))
        except Exception as e:
            _log.warning(
                f"could not register {desc} {path}; "
                f"{e.__class__.__name__}: {e}")
