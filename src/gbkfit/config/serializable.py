"""
The objects that configurations describe, and how their options are
read: each is a parameter of a constructor of the class of the object,
__init__ or a named one (e.g. from_file; see
Serializable.config_constructors), in the form of its annotation (see
parse_options_for_callable).
"""

import abc
import importlib.util
import inspect
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, ClassVar, Self

import astropy.units
import numpy as np

import gbkfit.units
from gbkfit.utils import funcutils, iterutils
from .errors import ConfigError, config_path, quoted_names
from .options import parse_options
from .paths import parse_data_file, resolve_path
from .types import config_form, option_types, type_problem, union_members

if typing.TYPE_CHECKING:
    from .parsers import TypedParser


__all__ = [
    'Serializable',
    'TypedSerializable',
    'config_constructor',
    'constructor_of',
    'dump_quantity',
    'from_config',
    'is_record',
    'make_typed_desc',
    'missing_modules',
    'parse_options_for_callable'
]


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
    # its load reads (e.g. the drivers of the placements of workers, which
    # configurations name instead of giving the Drivers)
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
        config_constructors (see config_constructor), each in the form
        of its annotation, objects of other classes included (e.g. the
        PSF of an Instrument; see parse_options_for_callable). A class
        overrides it only for options in other forms (e.g. the tables of
        a PSF that varies) or loaded with what its caller gives (e.g. the
        catalogue of lines of an emission), and then calls it. The
        configuration is not changed.

        Parameters
        ----------
        info : dict
            The options.

        Returns
        -------
        Self
            The object.
        """
        func, make = constructor_of(cls, info)
        return make(**parse_options_for_callable(info, func))

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """
        Return the JSON Schema (2020-12) of the configurations of the class.

        Those of the base of a family (e.g. PSF) are those of any of its
        members, with their type, as the parser of the family reads them;
        those of another class (e.g. PSFGauss, or Galaxy) are its own
        options, as its load reads them. See gbkfit.schema, of whole
        configurations.

        Returns
        -------
        dict
            The schema.

        Raises
        ------
        TypeError
            If the class has no configurations (an abstract class that is
            the base of no family).

        Examples
        --------
        >>> import jsonschema
        >>> from gbkfit.instrument import PSF, PSFGauss
        >>> jsonschema.validate(dict(type='gauss', sigma=0.6), PSF.schema())
        >>> jsonschema.validate(dict(sigma=0.6), PSFGauss.schema())
        """
        # (the schema imports the catalogue, which imports every family)
        from .schema import class_schema
        return class_schema(cls)

    @abc.abstractmethod
    def dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
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
        func = _constructor_function(cls, name)
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
    forms = '; or '.join(quoted_names(own[name]) for name in names)
    raise ConfigError(
        f"give the options of one of its forms, which have {forms}; it has "
        + (f"those of several: "
           f"{quoted_names(set(info) & set().union(*own.values()))}"
           if given else "none of them"))


def constructor_of(
        cls: type[Serializable], info: Mapping[str, Any]
) -> tuple[Callable, Callable]:
    """
    Return the constructor of a class whose options a configuration has
    (see config_constructor): the function of its parameters (__init__,
    or a named constructor), and the callable that makes the object (the
    class, or the named constructor).

    Parameters
    ----------
    cls : type
        The class.
    info : Mapping
        The options of the configuration.

    Returns
    -------
    tuple of Callable
        The function of the parameters, and the callable.

    Raises
    ------
    ConfigError
        If the configuration has the own options of none or of several of
        the constructors.
    """
    name = config_constructor(cls, info)
    func = _constructor_function(cls, name)
    return func, cls if name == '__init__' else func


def _constructor_function(cls: type, name: str) -> Callable:
    """The function of the parameters of a constructor of a class."""
    return cls.__init__ if name == '__init__' else getattr(cls, name)


class TypedSerializable(Serializable, abc.ABC):
    """
    A serializable object of one of several classes, chosen by the type
    in its configuration (see TypedParser), and named for people by the
    title of its type.
    """

    @staticmethod
    @abc.abstractmethod
    def type() -> str:
        """
        Return the type of the class in configurations.

        Returns
        -------
        str
            The type (e.g. 'thick_disk').
        """

    @staticmethod
    @abc.abstractmethod
    def type_title() -> str:
        """
        Return the name of the type for people: a few words in sentence
        case, unique in its family (e.g. 'Thick disk', 'Generalised
        Gaussian'). Unlike the type, it is never parsed, and can change.

        Returns
        -------
        str
            The title.
        """


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
    if any, and its range (see type_problem). Unknown options are
    reported (see report_unknown) and left out.

    Options of some types are given in a form of their own (see
    from_config): a FilePath is relative to the directory of the
    configuration (see resolve_path), a ScalarQuantity is its text (e.g.
    '2.5 Angstrom'; see gbkfit.units.parse_quantity), a DataFile is a
    filename or a dict with the filename (file) and the HDU (hdu) or the
    path of its array (path), a record (a NamedTuple, e.g. a Column) is a
    dict of its fields, another Quantity is a dict with its value and
    unit (e.g. {value: [1, 2], unit: kpc}), an object of a family (e.g. a
    PSF) is a dict of its options with its type, one of another
    serializable class (e.g. an Instrument) is a dict of its options,
    and the items of a sequence and the values of a Mapping are in the
    forms of their type.

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
            f"{func.__qualname__} has no parameters {quoted_names(unknown)}")
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
    types = option_types(func)
    args = {}
    for option, value in options.items():
        type_ = types.get(option if option in known else others)
        if type_ is not None:
            with config_path(option):
                value = from_config(value, type_)
            if problem := type_problem(value, type_):
                raise ConfigError(f"option '{option}' {problem}")
        args[option] = value
    return args


def from_config(value: Any, type_: Any) -> Any:
    """
    Return the value of an option of a type annotation from the form a
    configuration gives it in (see parse_options_for_callable); values in
    no form of their own are returned as they are (e.g. numbers, and
    objects given in Python).

    A dict is in the form of a member of a union annotation that takes
    dicts: a member of a family, if the dict has a type (or no other
    member takes dicts), or else the first other one. A list is in the
    form of the first sequence member: each item in that of its items.

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
    # (an object given in Python, which may be a mapping, e.g. a Dataset)
    if isinstance(value, Serializable):
        return value
    members = tuple(
        member for x in union_members(type_)
        for member in union_members(_bare(x)))
    if isinstance(value, str):
        return _from_string(value, members)
    if isinstance(value, Mapping):
        return _from_dict(value, members)
    if isinstance(value, list | tuple):
        return _from_list(value, members)
    return value


def _bare(type_: Any) -> Any:
    """
    Return an annotation without its metadata (e.g. its Unit), whose form
    in configurations is that of its type; but for the types of a form of
    their own (e.g. FilePath), which are annotations.
    """
    if typing.get_origin(type_) is Annotated and config_form(type_) is None:
        return typing.get_args(type_)[0]
    return type_


def _from_string(value: str, members: tuple[Any, ...]) -> Any:
    """Return the value of a string, of the members of a union annotation."""
    for member in members:
        form = config_form(member)
        if form == 'path':
            return resolve_path(value)
        if form == 'data_file':
            return parse_data_file(value)
        if form == 'quantity':
            try:
                return gbkfit.units.parse_quantity(value)
            except ValueError as e:
                raise ConfigError(str(e)) from None
    return value


def _from_dict(value: Mapping[str, Any], members: tuple[Any, ...]) -> Any:
    """Return the value of a dict, of the members of a union annotation."""
    forms = [x for x in members if _takes_dicts(x)]
    parsers = [p for p in map(_family_parser, forms) if p is not None]
    others = [x for x in forms if _family_parser(x) is None]
    if parsers and ('type' in value or not others):
        return parsers[0].load(dict(value))
    if not others:
        return value
    member = others[0]
    if config_form(member) == 'data_file':
        return parse_data_file(value)
    if member is astropy.units.Quantity:
        return _quantity(value)
    if typing.get_origin(member) in (dict, Mapping):
        item_type = typing.get_args(member)[1]
        items = {}
        for key, item in value.items():
            with config_path(key):
                items[key] = _from_item(item, item_type)
        return items
    if is_record(member):
        return member(**parse_options_for_callable(dict(value), member))
    return member.load(dict(value))


def _from_list(value: list | tuple, members: tuple[Any, ...]) -> Any:
    """Return the value of a list, of the members of a union annotation."""
    item_types = [x for x in map(_item_type, members) if x is not None]
    if not item_types:
        return value
    items = []
    for i, item in enumerate(value):
        with config_path(i):
            items.append(_from_item(item, item_types[0]))
    return tuple(items) if isinstance(value, tuple) else items


def _from_item(value: Any, type_: Any) -> Any:
    """
    Return the value of an item of a sequence or a Mapping of an
    annotation. An item of objects given as dicts (e.g. of PSFs) is
    checked against it, so that an error names the item.

    Raises
    ------
    ConfigError
        If the item is invalid.
    """
    value = from_config(value, type_)
    if any(map(_takes_dicts, union_members(type_))) \
            and (problem := type_problem(value, type_)):
        raise ConfigError(problem)
    return value


def _takes_dicts(type_: Any) -> bool:
    """
    Return whether configurations give the values of a member of a union
    annotation as dicts (see from_config).
    """
    return (
        config_form(type_) == 'data_file'
        or type_ is astropy.units.Quantity
        or typing.get_origin(type_) in (dict, Mapping)
        or is_record(type_)
        or _family_parser(type_) is not None
        or (isinstance(type_, type) and issubclass(type_, Serializable)
            and not inspect.isabstract(type_)))


def _item_type(type_: Any) -> Any:
    """
    Return the annotation of the items of a sequence annotation (e.g.
    ``Sequence[PSF]``, or ``tuple[PSF, ...]``), or None if it is not one.
    """
    origin, args = typing.get_origin(type_), typing.get_args(type_)
    if origin in (list, Sequence) and args:
        return args[0]
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        return args[0]
    return None


def _family_parser(type_: Any) -> 'TypedParser | None':
    """
    Return the parser of the family whose base is an annotation (e.g.
    that of the PSFs, for PSF; see the catalogue), or None if it is
    the base of none.
    """
    if not (isinstance(type_, type) and issubclass(type_, TypedSerializable)):
        return None
    # (the catalogue imports every family, and so this module)
    from .catalogue import family_of
    family = family_of(type_)
    return None if family is None else family.parser


def is_record(type_: Any) -> bool:
    """
    Return whether a type annotation is a record (a NamedTuple, e.g.
    Column), which configurations give as a dict of its fields.
    """
    return isinstance(type_, type) and issubclass(type_, tuple) \
        and hasattr(type_, '_fields')


def dump_quantity(quantity: astropy.units.Quantity) -> dict[str, Any]:
    """
    Return the configuration of a Quantity: a dict with its value and its
    unit (as gbkfit.units.format_unit writes it), which configurations
    read back (see from_config).

    Parameters
    ----------
    quantity : astropy.units.Quantity
        The quantity: a number, or an array.

    Returns
    -------
    dict
        The configuration.
    """
    value = quantity.value
    return dict(
        value=value.tolist() if np.ndim(value) else float(value),
        unit=gbkfit.units.format_unit(quantity.unit))


def _quantity(info: Mapping[str, Any]) -> astropy.units.Quantity:
    """Return the Quantity of a dict with its value and unit."""
    options = parse_options(dict(info), required={'value', 'unit'})
    # The unit is parsed here, so that what is wrong with it is a mistake
    # of the configuration
    try:
        unit = gbkfit.units.parse_unit(options['unit'])
    except ValueError as e:
        raise ConfigError(str(e), ['unit']) from e
    try:
        return astropy.units.Quantity(options['value'], unit, dtype=float)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"option 'value' must be a number or a list of numbers; it is "
            f"{options['value']!r}") from e
