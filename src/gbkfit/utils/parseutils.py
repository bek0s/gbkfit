import abc
import contextlib
import contextvars
import copy
import difflib
import importlib
import inspect
import logging
import re
import typing
from collections.abc import Callable
from typing import Any, TypeAlias

from . import funcutils, iterutils, typeutils


_log = logging.getLogger(__name__)


TypeInfoOne: TypeAlias = dict[str, Any]
TypeInfoMany: TypeAlias = list[TypeInfoOne]
TypeInfo: TypeInfoOne | TypeInfoMany


def make_basic_desc(
        cls: type,
        label: str
) -> str:
    """
    Generates a basic description string for a class.
    """
    return f'{label} (class={cls.__qualname__})'


def make_typed_desc(
        cls: type['TypedSerializable'],
        label: str | None = None
) -> str:
    """
    Generates a description string for a class, incorporating its type
    and name.
    """
    desc = f'{cls.type()} (class={cls.__qualname__})'
    return f'{label} {desc}' if label else desc


# Whether unknown options are errors (strict) or warnings
_strict = contextvars.ContextVar('strict', default=False)


@contextlib.contextmanager
def strict_mode(strict: bool = True):
    """
    In the block, unknown configuration options (and sections, and
    parameters) are errors instead of warnings.
    """
    token = _strict.set(strict)
    try:
        yield
    finally:
        _strict.reset(token)


def is_strict() -> bool:
    return _strict.get()


def describe_unknown(unknown, known) -> str:
    """
    The unknown names, each with the most similar known name, if any
    (e.g. "'rnmx' (did you mean 'rnmax'?)").
    """
    descs = []
    for name in sorted(unknown):
        similar = difflib.get_close_matches(name, list(known), n=1)
        descs.append(
            f"'{name}' (did you mean '{similar[0]}'?)" if similar
            else f"'{name}'")
    return ', '.join(descs)


def report_unknown(message: str, unknown, known) -> None:
    """
    Report unknown names: a warning that they will be ignored, or an
    error in strict mode.
    """
    message = f"{message}: {describe_unknown(unknown, known)}"
    if is_strict():
        raise ConfigError(message)
    _log.warning(f"{message}; they will be ignored")


class ConfigError(RuntimeError):
    """
    An error in a configuration. It has the path to the part of the
    configuration with the error (e.g. models[0].gmodel.components[1]),
    and the description of the type of that part, if any.
    """

    def __init__(self, message: str, path=(), context: str | None = None):
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


@contextlib.contextmanager
def config_path(*segments: str | int, context: str | None = None):
    """
    Add the given segments (option names or list indices) to the path of
    the configuration errors raised in the block, and turn any other
    errors into configuration errors. context describes the type of the
    part of the configuration loaded in the block, for the errors in that
    part itself (not in its options).
    """
    try:
        yield
    except ConfigError as e:
        if e.context is None and not e.path:
            e.context = context
        e.path[:0] = segments
        raise
    except Exception as e:
        raise ConfigError(str(e), segments, context) from e


# The names of items of lists (e.g. components): letters, digits and
# underscores
_NAME = re.compile(r'[A-Za-z0-9_]+')


def check_name(name: str | None) -> None:
    """Raise ConfigError unless the name is None or a valid name."""
    if name is not None and not (
            isinstance(name, str) and _NAME.fullmatch(name)):
        raise ConfigError(
            f"invalid name {name!r}: use only letters, digits and "
            f"underscores")


def item_prefixes(
        names: typing.Sequence[str | None],
        label: str,
        prefix: str,
        prefix_first: bool
) -> list[str]:
    """
    The prefix of the names of the parameters (and the constants and the
    extra outputs) of each item of a list, from the names of the items:
    their name, if every item has one, or else their position (prefix and
    index, e.g. 'cmp1_'; the first item has no index, and has the prefix
    only if prefix_first). Raise ConfigError if only some of the items have
    a name, or if the names are not unique; label names the items in the
    message (e.g. 'components').
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


def parse_options(
        info: dict[str, Any],
        desc: str,
        required: set[str] | None = None,
        optional: set[str] | None = None
) -> dict[str, Any]:
    """
    Validates and filters a dictionary of options.
    Returns a filtered dictionary containing only recognized options.
    """
    required = iterutils.setify(required, False)
    optional = iterutils.setify(optional, False)
    # Required options must not clash with optional options
    if conflicting := required & optional:
        raise RuntimeError(
            f"the following required and optional options are conflicting: "
            f"{conflicting}")
    # Check for unknown options
    if unknown := set(info) - (required | optional):
        report_unknown(
            f"unknown options for {desc}", unknown, required | optional)
    # Check for missing options
    if missing := required - set(info):
        raise RuntimeError(
            f"the following options for {desc} are "
            f"required but missing: {str(missing)}")
    # Return all recognised options
    return {k: v for k, v in info.items() if k in required | optional}


def parse_options_for_callable(
        info: dict[str, Any],
        desc: str,
        fun: Callable,
        fun_ignore_args: list[str] | None = None,
        fun_rename_args: dict[str, str] | None = None,
        add_required: dict[str, type] = None,
        add_optional: dict[str, type] = None
) -> dict[str, Any]:
    """
    Parses options for a callable function, handling required/optional
    arguments, renaming, ignoring arguments, and type validation.
    """
    fun_ignore_args = iterutils.setify(fun_ignore_args, False)
    fun_rename_args = fun_rename_args or {}
    add_required = add_required or {}
    add_optional = add_optional or {}
    add_required_keys = set(add_required)
    add_optional_keys = set(add_optional)
    # Extract required and optional options/arguments from callable
    fun_required, fun_optional = funcutils.extract_args(fun)[1:]
    fun_required = set(fun_required)
    fun_optional = set(fun_optional)
    fun_all = fun_required | fun_optional
    # Ignore callable options if requested
    if fun_ignore_args:
        # Ensure the ignored options are known
        if unknown := fun_ignore_args - fun_all:
            raise RuntimeError(
                f"the following ignored options do not exist "
                f"in the callable's argument list: {unknown}")
        fun_required -= fun_ignore_args
        fun_optional -= fun_ignore_args
    # Rename callable options if requested
    if fun_rename_args:
        # Ensure the old names of the rename options are known
        if unknown := set(fun_rename_args.keys()) - fun_all:
            raise RuntimeError(
                f"the following rename options do not exist "
                f"in the callable's argument list: {unknown}")
        # Ensure the new names of the rename options are not
        # in conflict with the arguments of the callable
        if conflicting := set(fun_rename_args.values()) & fun_all:
            raise RuntimeError(
                f"the following rename options are in conflict "
                f"with callable's argument list: {conflicting}")
        for old_name, new_name in fun_rename_args.items():
            if old_name in fun_required:
                fun_required.discard(old_name)
                fun_required.add(new_name)
            if old_name in fun_optional:
                fun_optional.discard(old_name)
                fun_optional.add(new_name)
    # These will hold the total required and optional options
    required = set(fun_required)
    optional = set(fun_optional)
    # Added required options must not clash with added optional options
    if intersection := add_required_keys & add_optional_keys:
        raise RuntimeError(
            f"the following options should not exist "
            f"in both add_required and add_optional: "
            f"{intersection}")
    add_all_keys = add_required_keys | add_optional_keys
    # Added options should not conflict with callable's options
    if conflicting := add_all_keys & fun_all:
        raise RuntimeError(
            f"the following added options are in conflict "
            f"with callable's argument list: {conflicting}")
    # Renamed options must not clash with added options
    rename_old_names = set(fun_rename_args.keys())
    rename_new_names = set(fun_rename_args.values())
    if intersection := add_all_keys & (rename_old_names | rename_new_names):
        raise RuntimeError(
            f"the following options should not exist "
            f"in both fun_rename_args and add_[required|optional]: "
            f"{intersection}")
    # Update total options with added options
    required.update(add_required_keys)
    optional.update(add_optional_keys)
    # Parse required and optional options
    options = parse_options(info, desc, required, optional)
    # Gather, prepare, and validate option types
    option_types = inspect.get_annotations(fun)
    option_types = option_types | add_required | add_optional
    option_types = iterutils.remove_from_mapping_by_value(option_types, None)
    for old_name, new_name in fun_rename_args.items():
        if old_name in option_types:
            option_types[new_name] = option_types.pop(old_name)
    for option_name, option_value in options.items():
        if option_name in option_types:
            option_type = option_types[option_name]
            _log.debug(
                f"parsing option name: '{option_name}', "
                f"value: '{option_value}', "
                f"type: '{option_type}'")
            if not typeutils.validate_type(option_value, option_type):
                option_type_has_name = not bool(typing.get_args(option_type))
                option_type_label = option_type.__name__ \
                    if option_type_has_name else str(option_type)
                option_type_label = (
                    option_type_label
                    .replace('collections.abc.', '')
                    .replace('types.', '')
                    .replace('typing.', ''))
                raise RuntimeError(
                    f"option '{option_name}' is set to '{option_value}', "
                    f"however, the expected type is: {option_type_label}")
    # Rename options back to their argument name if needed
    if fun_rename_args:
        for arg_name, opt_name in fun_rename_args.items():
            if opt_name in options:
                options[arg_name] = options.pop(opt_name)
    return options


class Serializable(abc.ABC):
    """
    Abstract base class for serializable objects.

    This class defines the interface for objects that can be serialized
    to and deserialized from dictionaries.
    """
    @classmethod
    @abc.abstractmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'Serializable':
        """Load an instance from a dictionary."""
        pass

    @abc.abstractmethod
    def dump(self, *args, **kwargs) -> dict[str, Any]:
        """Serialize the instance into a dictionary."""
        pass


class BasicSerializable(Serializable, abc.ABC):
    pass


class TypedSerializable(Serializable, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type() -> str:
        pass


def _prepare_args_and_kwargs(
        length: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any]
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Prepares `args` and `kwargs` by ensuring they contain sequences of
    the correct length.

    This function ensures that all elements in `args` and `kwargs` are
    either sequences of the specified `length` or `None`. If an element
    is `None`, it is replaced with a list of `None` values of the
    appropriate length. If any element is not a sequence of the
    required length, an exception is raised.
    """
    def validate_sequence(value, name):
        if value is None:
            return [None] * length
        if not iterutils.is_sequence(value) or len(value) != length:
            raise RuntimeError(
                f"expected '{name}' to be a sequence of length {length},"
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

    def __init__(self, cls: type[Serializable]):
        self._cls = cls

    def cls(self) -> type[Serializable]:
        return self._cls

    def cls_name(self) -> str:
        return self.cls().__qualname__

    def load(self, x, *args, **kwargs) -> Any:
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

    def __init__(self, cls, name='no name'):
        super().__init__(cls)
        self._name = name

    def _load_one_impl(self, x, *args, **kwargs):
        return self.cls().load(x, *args, **kwargs)

    def _dump_one_impl(self, x, *args, **kwargs):
        return x.dump(*args, **kwargs)


class TypedParser(Parser):

    def __init__(
            self,
            cls: type[TypedSerializable],
            parsers: type[TypedSerializable] | list[type[TypedSerializable]] | None = None
    ):
        super().__init__(cls)
        self._parsers = {}
        self.register(parsers)

    def register(
            self,
            parsers: type[TypedSerializable] | list[type[TypedSerializable]] | None
    ) -> None:
        parsers = iterutils.listify(parsers, False)
        for parser in parsers:
            parser_desc = make_typed_desc(parser)
            if not issubclass(parser, self.cls()):
                raise RuntimeError(
                    f"{self.cls_name()} parser "
                    f"could not register parser of type {parser_desc}; "
                    f"the two parsers are incompatible")
            if parser.type() in self._parsers:  # type: ignore
                raise RuntimeError(
                    f"{self.cls_name()} parser "
                    f"could not register parser of type {parser_desc}; "
                    f"a parser of the same type is already registered")
            self._parsers[parser.type()] = parser  # type: ignore

    def registered_class(self, type_: str) -> type[TypedSerializable]:
        """The class registered for the given type."""
        if type_ not in self._parsers:
            raise ConfigError(
                f"unknown {self.cls_name()} type '{type_}'; "
                f"the available types are: {list(self._parsers.keys())}")
        return self._parsers[type_]

    def _load_one_impl(
            self,
            x: dict[str, Any],
            *args, **kwargs
    ) -> TypedSerializable:
        if 'type' not in x:
            raise ConfigError(
                f"a {self.cls_name()} configuration must define a 'type'")
        type_ = x.pop('type')
        if type_ not in self._parsers:
            raise ConfigError(
                f"unknown {self.cls_name()} type '{type_}'; "
                f"the available types are: {list(self._parsers.keys())}")
        parser = self._parsers[type_]
        with config_path(context=make_typed_desc(parser)):
            return parser.load(x, *args, **kwargs)

    def _dump_one_impl(
            self,
            x: Serializable,
            *args, **kwargs
    ) -> dict[str, Any]:
        """
        Serializes an object of type `TypedSerializable` into a
        dictionary. It also serializes the type automatically.
        """
        if not isinstance(x, TypedSerializable):
            raise RuntimeError(
                f"unsupported object; value: {x}, type: {type(x)}")
        return dict(type=x.type()) | x.dump(*args, **kwargs)


def _sanitize_dimensional_option(option, value, lengths, type_):
    args = typing.get_args(type_)
    types_ = args if args else [type_]
    type_name = " | ".join([t.__name__ for t in types_])
    lengths = iterutils.listify(lengths)
    max_length = max(lengths)

    def is_value(x):
        # (a bool is an int in Python, but not a number in a configuration)
        return isinstance(x, type_) and not isinstance(x, bool)
    if is_value(value):
        return iterutils.make_list(max_length, value)
    if iterutils.is_sequence(value) and all(is_value(x) for x in value):
        if len(value) < max_length:
            raise RuntimeError(
                f"option '{option}' has a value "
                f"with a length shorter than expected; "
                f"expected length: {' or '.join(map(str, lengths))}, "
                f"current length: {len(value)}")
        if len(value) > max_length:
            new_value = value[:max_length]
            _log.warning(
                f"option '{option}' has a value "
                f"with a length longer than expected; "
                f"current length: {len(value)}; "
                f"expected length: {' or '.join(map(str, lengths))}, "
                f"the value will be trimmed from {value} to {new_value}")
            value = new_value
        return value
    raise RuntimeError(
        f"option '{option}' should be a scalar of type {type_name}, "
        f"or a sequence of type {type_name} and "
        f"length of {' or '.join(map(str, lengths))}")


def sanitize_dimensional_options(
        info: dict[str, Any],
        options: dict[str, type],
        lengths: int | list[int]
) -> None:
    """
    Make the given options of a configuration (e.g. 'size', 'step'),
    which have one value for each dimension, lists of the given type and
    length, in place. A single value is repeated, and the extra values of
    a longer list are dropped with a warning, so that, e.g., a spectral
    cube configuration can be used for an image by changing its type.
    Options that are missing or null are left as they are.
    """
    for option, type_ in options.items():
        if info.get(option) is not None:
            info[option] = _sanitize_dimensional_option(
                option, info[option], lengths, type_)


def load_option(
        loader: Callable,
        info: dict[str, Any],
        key: str,
        required: bool = False,
        allow_none: bool = True,
        *args, **kwargs
) -> dict[str, Any] | None:
    exists = key in info
    if not exists and required:
        raise ConfigError(f"option '{key}' is required but not provided")
    if exists and info[key] is None and not allow_none:
        raise ConfigError(f"option '{key}' cannot be null")
    if not exists:
        return None
    with config_path(key):
        return loader(info[key], *args, **kwargs)


def load_option_and_update_info(
        parser: Parser,
        info: dict[str, Any],
        key: str,
        required: bool = False,
        allow_none: bool = True,
        *args, **kwargs
) -> dict[str, Any] | None:
    """
    Loads and updates a dictionary key using a parser.

    Notes
    -----
    - Note regarding the `required` parameter: This function is usually
      called from a `load()` factory method. These factory methods
      typically validate the presence of required arguments
      automatically using the `parse_options_for_callable()` function.
      Therefore, it is strongly recommended to use `required=True` only
      if you fully understand its implications.
    """
    exists = key in info
    if not exists and required:
        raise ConfigError(f"option '{key}' is required but not provided")
    if exists and info[key] is None and not allow_none:
        raise ConfigError(f"option '{key}' cannot be null")
    if exists:
        with config_path(key):
            info[key] = parser.load(info[key], *args, **kwargs)
    return info


def register_optional_parsers(
        abstract_parser: TypedParser,
        parsers: list[str],
        desc: str
) -> None:
    """
    Dynamically registers optional parsers.
    Notes
    -----
    - The function expects each parser string to contain at least one
      dot, separating the module path from the class name.
    - If registration fails for a parser, a warning is logged, but
      execution continues.
    """
    for parser in parsers:
        try:
            mod_name, cls_name = parser.rsplit('.', 1)
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            abstract_parser.register(cls)
        except Exception as e:
            _log.warning(
                f"could not register {desc} parser {parser}; "
                f"{e.__class__.__name__}: {e}")
