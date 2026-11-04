"""
The properties of the parameters of a configuration (see ParamSpace):
their parsing into the elements that a ParamSpace takes (see elements),
and back.
"""

import copy
import numbers
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

from gbkfit.config import ConfigError, config_path, report_unknown
from ._keys import InvalidKeyError, Key, parse_key
from .elements import (
    DerivedValue, Element, Elements, FreeElement, Solved, Tied, Transformed)
from .pdescs import ParamDesc
from .solves import solve_parser


__all__ = [
    'Property',
    'dump_elements',
    'is_list',
    'parse_derived',
    'parse_properties',
    'problems_text'
]


# The property of a parameter, or of some of its elements, in a
# configuration (see ParamSpace)
Property: TypeAlias = (
    float | str | Mapping[str, Any] | None
    | Sequence[float | str | Mapping[str, Any] | None])


def is_list(value: Any) -> bool:
    """Check whether a property or an element is a list, one per element."""
    return isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.ndim == 1)


def problems_text(problems: Sequence[str]) -> str:
    """Return the problems of the parameters as the message of an error."""
    return "invalid parameter properties:\n- " + "\n- ".join(problems)


def parse_properties(
        properties: Mapping[str, Property],
        pdescs: Mapping[str, ParamDesc]
) -> tuple[dict[str, Element | list[Element]], list[str]]:
    """
    Parse the properties of the parameters of a configuration into what
    sets their elements, by key (see ParamSpace): a number fixes them, an
    expression ties them (Tied), a dict frees them (FreeElement; its
    attributes whose names start with '*' are lists spread over the
    elements), a dict of 'solve' alone solves them (Solved), None leaves
    them to the transforms function (Transformed), and a list has one of
    these for each element. Properties of unknown parameters are reported
    (see report_unknown) and left out.

    Parameters
    ----------
    properties : Mapping
        The properties, by key.
    pdescs : Mapping
        The descriptions of the parameters, by name.

    Returns
    -------
    dict
        What sets the elements, by key: one element for all those of the
        key, or a list of one for each; without those of invalid keys or
        properties.
    list of str
        The problems: invalid keys or properties, and unknown parameters
        (unless in lenient mode), which the caller reports together with
        its own (see ParamSpace.from_properties).
    """
    problems = []
    elements: dict[str, Element | list[Element]] = {}
    unknown = []
    for key, value in properties.items():
        try:
            parsed = parse_key(key, pdescs)
        except InvalidKeyError as e:
            problems.append(f"{key!r}: {e}")
            continue
        if parsed is None:
            unknown.append(key)
            continue
        # The errors of the free elements and the solves are those of
        # their own parsing and constructors, at the path of the key
        try:
            with config_path('properties', key):
                elements[key] = _parse_property(parsed, value)
        except _Problem as e:
            problems.append(f"'{key}': {e}")
    if unknown:
        # Unless in lenient mode, an error among the others
        try:
            report_unknown("unknown parameters", unknown, pdescs)
        except ConfigError as e:
            problems.append(e.message)
    return elements, problems


class _Problem(Exception):
    """A problem of a property, collected with the others."""


def _parse_property(key: Key, value: Any) -> Element | list[Element]:
    """Return what the property of a key sets its elements to."""
    if is_list(value):
        if key.indices is None or key.element:
            raise _Problem(
                "a list of values needs a key of more than one element")
        if len(value) != len(key.indices):
            raise _Problem(
                f"a list of {len(value)} values for {len(key.indices)} "
                f"elements")
        return [_parse_one(item) for item in value]
    if isinstance(value, Mapping) and 'solve' not in value \
            and any(k.startswith('*') for k in value):
        if key.indices is None or key.element:
            raise _Problem(
                "'*' attributes need a key of more than one element")
        return [
            _parse_one(item) for item in _spread(value, len(key.indices))]
    return _parse_one(value)


def _parse_one(value: Any) -> Element:
    """Return what a property sets an element (or all those of a key) to."""
    if isinstance(value, bool):
        raise _Problem(f"{value} is not a number")
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        return Tied(value)
    if value is None:
        return Transformed()
    if isinstance(value, Mapping) and 'solve' in value:
        if others := sorted(set(value) - {'solve'}):
            raise _Problem(
                f"a solved element takes its 'solve' alone (its prior is "
                f"in it); not {others}")
        with config_path('solve'):
            return Solved(solve_parser.load(value['solve']))
    if isinstance(value, Mapping):
        return FreeElement.load(value)
    raise _Problem(f"invalid value: {value!r}")


def _spread(properties: Mapping[str, Any], size: int) -> list[dict[str, Any]]:
    """
    Return the properties of each of the given number of elements: the
    same for each element, except those of attributes starting with '*',
    whose values (lists) are spread over the elements.
    """
    spread = {k[1:]: v for k, v in properties.items() if k.startswith('*')}
    common = {k: v for k, v in properties.items() if not k.startswith('*')}
    for attr, values in spread.items():
        if not is_list(values) or len(values) != size:
            raise _Problem(
                f"the value of '*{attr}' must be a list of {size} values, "
                f"one for each element")
    return [
        copy.deepcopy(common) | {k: v[i] for k, v in spread.items()}
        for i in range(size)]


def dump_elements(elements: Elements) -> dict[str, Property]:
    """
    Return the properties of a configuration that set the elements of
    the parameters as given (see parse_properties): a list of free
    elements of the same options as one dict, of the options that they
    share and of those that they do not spread over them (e.g. '*value').
    """
    return {key: _dump_property(value) for key, value in elements.items()}


def _dump_property(value: Element | Sequence[Element]) -> Property:
    """Return the property of a key."""
    if not is_list(value):
        return _dump_one(value)
    items = list(value)
    if not (items and all(isinstance(x, FreeElement) for x in items)):
        return [_dump_one(x) for x in items]
    dumped = [x.dump() for x in items]
    if any(info.keys() != dumped[0].keys() for info in dumped):
        return dumped
    info = {}
    for option in dumped[0]:
        values = [d[option] for d in dumped]
        if all(v == values[0] for v in values):
            info[option] = values[0]
        else:
            info[f'*{option}'] = values
    return info


def _dump_one(value: Element) -> Property:
    """Return the property of an element (or of all those of a key)."""
    if isinstance(value, Tied):
        return value.expr
    if isinstance(value, FreeElement):
        return value.dump()
    if isinstance(value, Solved):
        return dict(solve=solve_parser.dump(value.solve))
    if isinstance(value, Transformed):
        return None
    return float(value)


def parse_derived(
        derived: Mapping[str, Any]
) -> dict[str, str | DerivedValue]:
    """
    Parse the derived values of a configuration (see ParamSpace): each an
    expression, or {expr, unit}.

    Raises
    ------
    ConfigError
        If the derived values are not a dict, or one is a dict of other
        options.
    """
    if not isinstance(derived, Mapping):
        raise ConfigError(
            "option 'derived' must be a dict of the expressions of the "
            "derived values, by name")
    parsed: dict[str, str | DerivedValue] = {}
    for name, value in derived.items():
        if isinstance(value, Mapping):
            if set(value) - {'expr', 'unit'} or 'expr' not in value:
                raise ConfigError(
                    f"a derived value is its expression, or {{expr: ..., "
                    f"unit: ...}}; it is {dict(value)!r}", ['derived', name])
            value = DerivedValue(value['expr'], value.get('unit'))
        parsed[name] = value
    return parsed


