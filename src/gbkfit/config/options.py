"""
Checks of the options of configurations, and of the arguments of the
constructors that configurations call.
"""

import re
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from gbkfit.utils import funcutils, iterutils
from .errors import ConfigError, config_path, quoted_names, report_unknown
from .types import option_types, type_error

if typing.TYPE_CHECKING:
    from .parsers import Parser


__all__ = [
    'check_name',
    'check_options',
    'item_prefixes',
    'load_option',
    'load_option_and_update_info',
    'name_problem',
    'parse_options'
]


# The names of items of lists (e.g. components): letters, digits and
# underscores
_NAME = re.compile(r'[A-Za-z0-9_]+')


def name_problem(name: Any) -> str | None:
    """
    Return what is wrong with the name of an item of a list (e.g. a
    component), or None if nothing (see check_name).

    Parameters
    ----------
    name : Any
        The name: letters, digits and underscores; or None.

    Returns
    -------
    str or None
        The problem, or None.
    """
    if name is None or (isinstance(name, str) and _NAME.fullmatch(name)):
        return None
    return (
        f"invalid name {name!r}: use only letters, digits and "
        f"underscores")


def check_name(name: str | None) -> None:
    """
    Check the name of an item of a list (e.g. a component), an argument
    of its constructor.

    Parameters
    ----------
    name : str or None
        The name: letters, digits and underscores; or None.

    Raises
    ------
    TypeError
        If the name is not a string (or None).
    ValueError
        If the name has other characters.
    """
    if (problem := name_problem(name)) is not None:
        raise (ValueError if isinstance(name, str) else TypeError)(problem)


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
    ValueError
        If only some of the items have a name, or the names are repeated.
    """
    named = [name is not None for name in names]
    if not any(named):
        return [
            f'{prefix}{i or ""}_' if i or prefix_first else ''
            for i in range(len(names))]
    if not all(named):
        raise ValueError(
            f"either all or none of the {label} must have a name; only "
            f"{sum(named)} of {len(names)} have one")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"the names of the {label} must be unique; repeated: {repeated}")
    return [f'{name}_' for name in names]


def parse_options(
        info: Mapping[str, Any],
        required: Iterable[str] | None = None,
        optional: Iterable[str] | None = None
) -> dict[str, Any]:
    """
    Check the options of a configuration, and return the known ones.

    Unknown options are reported (see report_unknown) and left out.

    Parameters
    ----------
    info : Mapping
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
        If the options are not a dictionary, a required option is
        missing, or an option is unknown (unless in lenient mode, see
        strict_mode).
    """
    required = iterutils.setify(required)
    optional = iterutils.setify(optional)
    if both := required & optional:
        raise RuntimeError(
            f"options cannot be both required and optional: "
            f"{quoted_names(both)}")
    if not isinstance(info, Mapping):
        raise ConfigError(
            f"the options must be a dictionary; they are {info!r}")
    known = required | optional
    if unknown := set(info) - known:
        report_unknown("unknown options", unknown, known)
    if missing := required - set(info):
        raise ConfigError(
            f"missing required options: {quoted_names(missing)}")
    return {k: v for k, v in info.items() if k in known}


def check_options(func: Callable, values: Mapping[str, Any]) -> None:
    """
    Check the arguments of a call of a constructor against the type
    annotations of its parameters: their types, ranges (e.g.
    ``Annotated[float, Gt(0)]``) and choices (``Literal``; see
    type_problem).

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
    TypeError
        If an argument is not of the type of its annotation.
    ValueError
        If an argument is outside the range or the choices of its
        annotation.
    """
    # Not ConfigError: a constructor does not know whether a configuration
    # calls it, and the parsing checks the options of configurations
    # itself first (see parse_options_for_callable)
    types = option_types(func)
    for name in funcutils.parameter_names(func).all:
        if name in types and name in values \
                and (error := type_error(values[name], types[name])):
            raise type(error)(f"option '{name}' {error}")


def load_option(
        loader: Callable,
        info: dict[str, Any],
        key: str,
        *,
        required: bool = False,
        **kwargs: Any
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
                f"missing required options: '{key}'")
        return None
    with config_path(key):
        return loader(info[key], **kwargs)


def load_option_and_update_info(
        parser: 'Parser',
        info: dict[str, Any],
        key: str,
        *,
        required: bool = False,
        **kwargs: Any
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
