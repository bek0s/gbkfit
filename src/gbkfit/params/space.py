"""
ParamSpace: the parameters of a model, given their descriptions and the
properties of a configuration. Each element of each parameter is:

- fixed: its property is a number
- tied: its property is an expression (a string) of other parameters,
  or None, in which case a user function (transforms) sets it
- free: its property is a dict (e.g. the initial value and bounds of a
  fit), which is kept for the fitter

ParamSpace evaluates all parameters, given the values of the free ones.
"""

import collections
import copy
import dataclasses
import graphlib
import logging
import numbers
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from gbkfit.params.expressions import Expression, InvalidExpressionError
from gbkfit.params.keys import (
    InvalidKeyError, element_name, parse_key)
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc


__all__ = [
    'ParamSpace',
    'InvalidParamsError'
]


_log = logging.getLogger(__name__)


class InvalidParamsError(ValueError):
    """Invalid parameter properties; the message lists all problems."""


@dataclasses.dataclass
class _Assignment:
    """An expression, and the elements of a parameter it sets."""
    key: str
    name: str
    indices: np.ndarray | None
    expression: Expression


def _is_list(value):
    return isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.ndim == 1)


def _spread(properties, size):
    """
    The properties of each of the given number of elements: the same
    properties for each element, except those of attributes starting
    with '*', whose values (lists) are spread over the elements.
    """
    spread = {k[1:]: v for k, v in properties.items() if k.startswith('*')}
    common = {k: v for k, v in properties.items() if not k.startswith('*')}
    for attr, values in spread.items():
        if not _is_list(values) or len(values) != size:
            raise ValueError(
                f"the value of '*{attr}' must be a list of {size} values, "
                f"one for each element")
    return [
        copy.deepcopy(common) | {k: v[i] for k, v in spread.items()}
        for i in range(size)]


class ParamSpace:

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            properties: dict[str, Any],
            transforms: Callable | None = None,
            constants: dict[str, Any] | None = None,
            unknown: str = 'warn'):
        """
        constants are read-only values that expressions can use (e.g. the
        radial nodes of a disk). unknown is what to do with properties of
        unknown parameters: 'warn' (and ignore them) or 'error'.
        """
        if unknown not in ('warn', 'error'):
            raise ValueError("unknown must be 'warn' or 'error'")
        self._pdescs = dict(pdescs)
        self._properties = copy.deepcopy(properties)
        self._transforms = transforms
        self._constants = {}
        for name, value in (constants or {}).items():
            value = np.array(value, dtype=float)
            value.flags.writeable = False
            self._constants[name] = value
        # All elements of all parameters, in the order of the pdescs
        self._elements = [
            (name, index)
            for name, pdesc in pdescs.items()
            for index in (
                [None] if isinstance(pdesc, ParamScalarDesc)
                else range(pdesc.size()))]
        # What sets each element: 'fixed', 'free', 'tied' (an expression)
        # or 'none' (transforms)
        self._kinds = {}
        self._fixed = {}
        self._free = {}
        self._assignments = []
        errors = []
        if conflicts := sorted(set(self._constants) & set(pdescs)):
            errors.append(
                f"constants cannot have the names of parameters: "
                f"{conflicts}")
        keys_of = collections.defaultdict(list)
        unknown_keys = []
        for key, value in properties.items():
            try:
                parsed = parse_key(key, pdescs)
            except InvalidKeyError as e:
                errors.append(f"'{key}': {e}")
                continue
            if parsed is None:
                unknown_keys.append(key)
                continue
            # A key with an invalid value still sets its elements, so
            # that they are not also reported as missing
            for element in self._elements_of(parsed.name, parsed.indices):
                keys_of[element].append(key)
            try:
                for indices, item in self._items(parsed, value):
                    self._set(key, parsed.name, indices, item)
            except (InvalidExpressionError, ValueError) as e:
                errors.append(f"'{key}': {e}")
        for element, keys in keys_of.items():
            if len(keys) > 1:
                errors.append(
                    f"'{element_name(*element)}' is set more than once, "
                    f"by: {keys}")
        if missing := [
                element_name(*e) for e in self._elements if e not in keys_of]:
            errors.append(f"these parameters have no value: {missing}")
        if unknown_keys:
            message = f"these parameters are unknown: {unknown_keys}"
            if unknown == 'error':
                errors.append(message)
            else:
                _log.warning(f"{message}; they will be ignored")
        errors += self._check_transforms()
        if not errors:
            errors += self._order_assignments()
        if errors:
            raise InvalidParamsError(
                "invalid parameter properties:\n- " + "\n- ".join(errors))
        # The values of all elements are kept in one buffer, in the order
        # of the elements: the position of each element and parameter in
        # it, and its values before each evaluation (those of the fixed
        # elements, and nan for the others)
        self._flat = {e: i for i, e in enumerate(self._elements)}
        self._layout = {}
        start = 0
        for name, pdesc in pdescs.items():
            scalar = isinstance(pdesc, ParamScalarDesc)
            self._layout[name] = (start, start + pdesc.size(), scalar)
            start += pdesc.size()
        self._template = np.array(
            [self._fixed.get(e, np.nan) for e in self._elements])
        self._free_elements = [e for e in self._elements if e in self._free]
        self._free_names = [element_name(*e) for e in self._free_elements]
        self._free_flat = np.array(
            [self._flat[e] for e in self._free_elements], dtype=int)

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    def properties(self) -> dict[str, Any]:
        return self._properties

    def transforms(self) -> Callable | None:
        return self._transforms

    def names(
            self, free: bool = True, tied: bool = True, fixed: bool = True
    ) -> list[str]:
        """
        The names of the elements of the given kinds (e.g. 'a', 'v[3]'),
        in the order of the pdescs. The elements set by the transforms
        are tied.
        """
        kinds = set()
        kinds.update(['free'] if free else [])
        kinds.update(['tied', 'none'] if tied else [])
        kinds.update(['fixed'] if fixed else [])
        return [
            element_name(*e) for e in self._elements
            if self._kinds[e] in kinds]

    def free_properties(self) -> dict[str, dict[str, Any]]:
        """The properties of each free element, by name."""
        return {
            element_name(*e): copy.deepcopy(self._free[e])
            for e in self._free_elements}

    def evaluate(
            self, free: Mapping[str, Any] | Any = None
    ) -> dict[str, float | np.ndarray]:
        """
        The values of all parameters (new arrays every time), given the
        values of the free elements: a dict by name, or a sequence in the
        order of names(free=True, tied=False, fixed=False).
        """
        buffer = self._template.copy()
        buffer[self._free_flat] = self._free_values(free)
        namespace = dict(self._constants)
        for name, (start, stop, scalar) in self._layout.items():
            namespace[name] = float(buffer[start]) if scalar \
                else buffer[start:stop]
        for assignment in self._assignments:
            self._assign(buffer, namespace, assignment)
        values = {name: namespace[name] for name in self._pdescs}
        if self._transforms:
            self._apply_transforms(buffer, values)
        if not np.all(np.isfinite(buffer)):
            self._raise_not_finite(values)
        return values

    def exploded_values(self, values) -> dict[str, float]:
        """The value of each element, by name (e.g. 'v[3]')."""
        return {
            element_name(name, index): float(
                values[name] if index is None else values[name][index])
            for name, index in self._elements}

    def _items(self, key, value):
        """
        Split the value of a key into the values of its elements: pairs of
        the indices of elements (None for a scalar) and a single value.
        """
        if _is_list(value):
            if key.indices is None or key.element:
                raise ValueError(
                    "a list of values needs a key of more than one element")
            if len(value) != len(key.indices):
                raise ValueError(
                    f"a list of {len(value)} values for "
                    f"{len(key.indices)} elements")
            return [(key.indices[i:i + 1], v) for i, v in enumerate(value)]
        if isinstance(value, Mapping) and key.indices is not None:
            return [
                (key.indices[i:i + 1], v)
                for i, v in enumerate(_spread(value, len(key.indices)))]
        if isinstance(value, Mapping) and any(
                k.startswith('*') for k in value):
            raise ValueError(
                "'*' attributes need a key of more than one element")
        return [(key.indices, value)]

    @staticmethod
    def _elements_of(name, indices):
        if indices is None:
            return [(name, None)]
        return [(name, int(i)) for i in indices]

    def _set(self, key, name, indices, value):
        """
        Record what sets the given elements. Raise ValueError if the value
        is invalid.
        """
        elements = self._elements_of(name, indices)
        if isinstance(value, bool):
            raise ValueError(f"{value} is not a number")
        if isinstance(value, numbers.Real):
            kind = 'fixed'
            self._fixed.update(dict.fromkeys(elements, float(value)))
        elif isinstance(value, str):
            kind = 'tied'
            expression = Expression(value, self._pdescs, self._constants)
            self._assignments.append(
                _Assignment(key, name, indices, expression))
        elif value is None:
            kind = 'none'
        elif isinstance(value, Mapping):
            kind = 'free'
            self._free.update({e: copy.deepcopy(value) for e in elements})
        else:
            raise ValueError(f"invalid value: {value!r}")
        self._kinds.update(dict.fromkeys(elements, kind))

    def _check_transforms(self):
        nones = [element_name(*e) for e, k in self._kinds.items()
                 if k == 'none']
        if self._transforms and self._assignments:
            return [
                "expressions and a transforms function are mutually "
                "exclusive"]
        if nones and not self._transforms:
            return [
                f"these parameters are set to None, so they must be set by "
                f"a transforms function, but there is none: {nones}"]
        if self._transforms and not nones:
            _log.warning(
                "a transforms function was given, but no parameters are "
                "set to None (tied to it)")
        return []

    def _order_assignments(self):
        """
        Order the assignments so that each one comes after those that set
        the elements it reads.
        """
        writer = {}
        for i, a in enumerate(self._assignments):
            for index in [None] if a.indices is None else a.indices:
                writer[(a.name, None if index is None else int(index))] = i
        graph = graphlib.TopologicalSorter()
        for i, a in enumerate(self._assignments):
            graph.add(i)
            for name, indices in a.expression.reads.items():
                for index in [None] if indices is None else sorted(indices):
                    if (name, index) in writer:
                        graph.add(i, writer[(name, index)])
        try:
            order = list(graph.static_order())
        except graphlib.CycleError as e:
            cycle = ' -> '.join(
                f"'{self._assignments[i].key}'" for i in e.args[1])
            return [f"the expressions depend on each other in a cycle: "
                    f"{cycle}"]
        self._assignments = [self._assignments[i] for i in order]
        return []

    def _free_values(self, free):
        if free is None:
            free = {}
        if not isinstance(free, Mapping):
            free = list(free)
            if len(free) != len(self._free_names):
                raise RuntimeError(
                    f"expected {len(self._free_names)} free values, "
                    f"got {len(free)}")
            return free
        problems = []
        if missing := [n for n in self._free_names if n not in free]:
            problems.append(f"these free parameters are missing: {missing}")
        if extra := [n for n in free if n not in self._free_names]:
            problems.append(f"these parameters are not free: {extra}")
        if problems:
            raise RuntimeError('; '.join(problems))
        return [free[n] for n in self._free_names]

    def _assign(self, buffer, namespace, assignment):
        a = assignment
        try:
            result = np.asarray(a.expression.evaluate(namespace), dtype=float)
        except Exception as e:
            raise RuntimeError(
                f"the expression of '{a.key}' ('{a.expression.source}') "
                f"failed: {e}") from e
        if a.indices is None:
            if result.ndim != 0:
                raise RuntimeError(
                    f"the expression of '{a.key}' gives a vector of size "
                    f"{result.size}, but '{a.name}' is a scalar")
            namespace[a.name] = buffer[self._flat[(a.name, None)]] = float(
                result)
            return
        if result.ndim > 1 or (
                result.ndim == 1 and result.size != len(a.indices)):
            raise RuntimeError(
                f"the expression of '{a.key}' gives a vector of size "
                f"{result.size}, for a key of size {len(a.indices)}")
        namespace[a.name][a.indices] = result

    def _apply_transforms(self, buffer, values):
        # The function works on a copy, and only sets the elements set to
        # None
        result = copy.deepcopy(values)
        self._transforms(result)
        for element, kind in self._kinds.items():
            if kind != 'none':
                continue
            name, index = element
            try:
                if index is None:
                    value = float(result[name])
                    values[name] = buffer[self._flat[element]] = value
                else:
                    values[name][index] = result[name][index]
            except Exception as e:
                raise RuntimeError(
                    f"the transforms function did not set "
                    f"'{element_name(name, index)}' to a number: {e}") from e

    def _raise_not_finite(self, values):
        bad = [
            f"'{element_name(name, index)}' is {value}"
            for name, index in self._elements
            if not np.isfinite(value := (
                values[name] if index is None else values[name][index]))]
        raise RuntimeError(
            f"the values of parameters must be finite: {', '.join(bad)}")
