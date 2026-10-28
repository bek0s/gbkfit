"""
Helpers for the type annotations of options.

The options of the constructors of GBKFIT are described by the type
annotations of their parameters: their types, their ranges and choices
(e.g. ``Annotated[float, Gt(0)]``, ``Literal['a', 'b']``) and their
units (``Unit``). They are checked against them in Python calls (see
`gbkfit.utils.parseutils.check_options`) and in configurations (see
`gbkfit.utils.parseutils.parse_options_for_callable`).
"""

import dataclasses
import functools
import math
import types
import typing
import warnings
from collections.abc import Callable
from typing import Annotated, Any, Literal, NamedTuple, TypeAlias

import annotated_types
import numpy as np
import pydantic
import pydantic.warnings


__all__ = [
    'Column',
    'FilePath',
    'FitsFile',
    'Unit',
    'describe_type',
    'matches_type',
    'option_types',
    'problem',
    'split_fits_file',
    'union_members',
    'unit_of'
]


@dataclasses.dataclass(frozen=True)
class Unit:
    """
    The unit of an option, as a marker of its annotation, e.g.
    ``Annotated[float, Gt(0), Unit('arcsec')]``.

    It labels the values (in descriptions and schemas); it does not
    convert them.

    Parameters
    ----------
    unit : str
        The unit (e.g. 'arcsec', 'km/s').
    """

    unit: str


@dataclasses.dataclass(frozen=True)
class _ConfigForm:
    """
    The marker of a type whose values configurations give in a form of
    their own (see parseutils.parse_options_for_callable).
    """

    name: str


# The path of a file; in a configuration, relative to the directory of
# the configuration (see parseutils.resolve_path)
FilePath: TypeAlias = Annotated[str, _ConfigForm('path')]

# A FITS file: its filename, or its filename and the HDU to read (by
# default the first); in a configuration, a filename, or a dict with the
# filename (file) and the HDU (hdu)
FitsFile: TypeAlias = Annotated[
    str | tuple[str, int | str], _ConfigForm('fits_file')]


class Column(NamedTuple):
    """
    A column of a table file; in a configuration, a dict with its name
    (column) and its unit (unit).

    Parameters
    ----------
    column : str
        Its name.
    unit : str, optional
        The unit of its values, if the file does not give it (e.g. a CSV
        file) or to read them in another; by default, that of the file.
    """

    column: str
    unit: str | None = None


def split_fits_file(file: FitsFile) -> tuple[str, int | str]:
    """Return the filename and the HDU of a FITS file (by default 0)."""
    return (file, 0) if isinstance(file, str) else file


# The bounds of the ranges of options, and finiteness (e.g. of
# pydantic.FiniteFloat; see problem)
_BOUNDS = (
    annotated_types.Gt, annotated_types.Ge, annotated_types.Lt,
    annotated_types.Le, pydantic.AllowInfNan(False))


@functools.cache
def _adapter(type_: typing.Any) -> pydantic.TypeAdapter:
    """
    Return the strict pydantic validator of a type annotation.

    Raise TypeError if pydantic cannot validate values of the annotation.
    For an annotation that is not a type (e.g. 5), pydantic would instead
    warn and accept any value.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                'error', pydantic.warnings.ArbitraryTypeWarning)
            return pydantic.TypeAdapter(type_, config=pydantic.ConfigDict(
                strict=True, arbitrary_types_allowed=True))
    except (pydantic.PydanticUserError,
            pydantic.warnings.ArbitraryTypeWarning) as e:
        raise TypeError(
            f"the following type is not recognized: "
            f"{describe_type(type_)}") from e


def matches_type(value: typing.Any, type_: typing.Any) -> bool:
    """
    Check whether a configuration value matches a type annotation.

    A bool is not a number, an int is a float, and a str is not a
    sequence of strings.

    Parameters
    ----------
    value : Any
        The value of a configuration option.
    type_ : Any
        A type annotation (e.g. ``Sequence[float] | None``).

    Returns
    -------
    bool
        Whether the value matches the annotation.

    Raises
    ------
    TypeError
        If pydantic cannot validate values of the annotation.
    """
    try:
        _adapter(type_).validate_python(value)
    except pydantic.ValidationError:
        return False
    return True


def describe_type(type_: typing.Any) -> str:
    """
    Return the label of a type annotation, for messages.

    Parameters
    ----------
    type_ : Any
        A type annotation.

    Returns
    -------
    str
        The label: the name of a type, or the annotation without the
        names of its modules.
    """
    if type_ is type(None):
        return 'None'
    if typing.get_origin(type_) is Annotated:
        return describe_type(typing.get_args(type_)[0])
    if len(members := union_members(type_)) > 1:
        return ' | '.join(describe_type(x) for x in members)
    name = getattr(type_, '__name__', None)
    label = str(type_) if typing.get_args(type_) or name is None else name
    return (
        label
        .replace('collections.abc.', '')
        .replace('types.', '')
        .replace('typing.', ''))


def option_types(func: Callable) -> dict[str, Any]:
    """
    Return the type annotations of the parameters of a callable, with
    their markers (the metadata of Annotated).

    Parameters
    ----------
    func : Callable
        A function or a method (e.g. ``cls.__init__``).

    Returns
    -------
    dict
        The annotation of each annotated parameter.
    """
    return dict(_option_types(getattr(func, '__func__', func)))


@functools.cache
def _option_types(func: Callable) -> dict[str, Any]:
    """The annotations of a function, resolved once (see option_types)."""
    hints = typing.get_type_hints(func, include_extras=True)
    hints.pop('return', None)
    return hints


def union_members(type_: Any) -> tuple[Any, ...]:
    """
    Return the members of a union annotation (e.g. ``int | None``), or
    the annotation itself.
    """
    if typing.get_origin(type_) in (typing.Union, types.UnionType):
        return typing.get_args(type_)
    return (type_,)


def _without_none(type_: Any) -> Any:
    """Return an annotation without None, if it is a union with it."""
    members = [x for x in union_members(type_) if x is not type(None)]
    return functools.reduce(lambda a, b: a | b, members) \
        if len(members) > 1 else members[0]


def _is_bound(x: Any) -> bool:
    """Return whether the metadata of an Annotated is a bound (see _BOUNDS)."""
    return any(
        x == bound if isinstance(bound, pydantic.AllowInfNan)
        else isinstance(x, bound) for bound in _BOUNDS)


def _split_bounds(type_: Any) -> tuple[Any, list[Any]]:
    """
    Return an annotation without the bounds of its range (those of an
    Annotated, see _BOUNDS), and the bounds.
    """
    if typing.get_origin(type_) is not Annotated:
        return type_, []
    inner, *metadata = typing.get_args(type_)
    bounds = [x for x in metadata if _is_bound(x)]
    others = [x for x in metadata if not _is_bound(x)]
    return (Annotated[inner, *others] if others else inner), bounds


def problem(value: Any, type_: Any) -> str | None:
    """
    Return what is wrong with the value of an option of a type
    annotation, or None if nothing.

    The value must be of the type, and within the range of the bounds of
    the annotation (Gt, Ge, Lt and Le of annotated_types, and finite, of
    pydantic.FiniteFloat). A numpy value is checked as its Python value
    (e.g. an array as a list).

    Parameters
    ----------
    value : Any
        The value.
    type_ : Any
        The annotation (e.g. ``Annotated[float, Gt(0)] | None``).

    Returns
    -------
    str or None
        The problem (e.g. "must be greater than 0; it is -1"), or None.

    Raises
    ------
    TypeError
        If pydantic cannot validate values of the annotation.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None and type(None) in union_members(type_):
        return None
    type_, bounds = _split_bounds(_without_none(type_))
    # (a plain array, not a subclass, e.g. a Quantity)
    if not matches_type(value, type_) and not (
            type(value) is np.ndarray
            and matches_type(value.tolist(), type_)):
        if typing.get_origin(type_) is Literal:
            choices = ', '.join(repr(x) for x in typing.get_args(type_))
            return f"must be one of {choices}; it is {value!r}"
        return f"must be of type {describe_type(type_)}; it is {value!r}"
    bounds = [_bound(x) for x in bounds]
    if not all(check(value) for _, check in bounds):
        within = ' and '.join(words for words, _ in bounds)
        return f"must be {within}; it is {value!r}"
    return None


def _bound(bound: Any) -> tuple[str, Callable[[Any], bool]]:
    """
    Return the wording of a bound of a range (e.g. 'greater than 0'), and
    its check.
    """
    match bound:
        case annotated_types.Gt(gt=b):
            return f"greater than {b}", lambda x: x > b
        case annotated_types.Ge(ge=b):
            return f"at least {b}", lambda x: x >= b
        case annotated_types.Lt(lt=b):
            return f"less than {b}", lambda x: x < b
        case annotated_types.Le(le=b):
            return f"at most {b}", lambda x: x <= b
        case pydantic.AllowInfNan(allow_inf_nan=False):
            return "finite", math.isfinite
    raise TypeError(f"unknown bound {bound!r}")


def unit_of(type_: Any) -> str | None:
    """
    Return the unit of an option of a type annotation (its Unit marker;
    that of the annotation without None), or None if it has none.
    """
    type_ = _without_none(type_)
    if typing.get_origin(type_) is not Annotated:
        return None
    units = [x.unit for x in type_.__metadata__ if isinstance(x, Unit)]
    return units[0] if units else None
