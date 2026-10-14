"""
ParamSpace: the parameters of a model evaluation or a fit, given their
descriptions and the properties of a configuration.
"""

import collections
import copy
import dataclasses
import enum
import graphlib
import logging
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from gbkfit.params._expressions import (
    Expression, InvalidExpressionError, check_name)
from gbkfit.params._keys import (
    InvalidKeyError, Key, element_name, element_names, parse_key)
from gbkfit.params.modes import ParamMode, param_mode_parser
from gbkfit.params.pdescs import ParamDesc
from gbkfit.utils import funcutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'ParamEvaluationError',
    'ParamKind',
    'ParamSpace',
    'ParamValues',
    'Property',
    'Transforms',
    'param_space_parser'
]


_log = logging.getLogger(__name__)


# The values of parameters, by name: a float for a scalar, an array for a
# vector
ParamValues: TypeAlias = dict[str, float | np.ndarray]

# The property of a parameter, or of some of its elements (see ParamSpace)
Property: TypeAlias = (
    float | str | Mapping[str, Any] | None
    | Sequence[float | str | Mapping[str, Any] | None])

# A function that sets the parameters whose property is None, in the
# values of all parameters (see ParamSpace)
Transforms: TypeAlias = Callable[[ParamValues], None]

# An element of a parameter: its name, and its index (None for a scalar)
_Element: TypeAlias = tuple[str, int | None]


class ParamEvaluationError(RuntimeError):
    """
    An evaluation of the parameters failed: an expression or the
    transforms function failed, or a value is not finite or not valid.
    """


class ParamKind(enum.Enum):
    """What sets the value of an element of a parameter."""
    FIXED = 'fixed'              # a number
    TIED = 'tied'                # an expression
    FREE = 'free'                # the caller (e.g. a fitter)
    TRANSFORMED = 'transformed'  # the transforms function (set to None)


@dataclasses.dataclass
class _Assignment:
    """An expression, and the elements of a parameter it sets."""
    key: str
    name: str
    indices: np.ndarray | None
    expression: Expression


def _is_list(value: Any) -> bool:
    """Check whether a property is a list of properties."""
    return isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.ndim == 1)


def _spread(properties: Mapping[str, Any], size: int) -> list[dict[str, Any]]:
    """
    Return the properties of each of the given number of elements: the
    same for each element, except those of attributes starting with '*',
    whose values (lists) are spread over the elements.
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


def _load_transforms(info: dict[str, Any]) -> Transforms:
    """Load a transforms function from its file: info is {file, func}."""
    opts = parseutils.parse_options(info, {'file', 'func'})
    return funcutils.load_function_from_file(
        parseutils.resolve_path(opts['file']), opts['func'])


def _dump_transforms(transforms: Transforms) -> dict[str, str]:
    """
    Return the file and the name of a transforms function. It must be
    defined at the top level of a file, from which it loads again.
    """
    code = getattr(transforms, '__code__', None)
    name = getattr(transforms, '__name__', None)
    if (code is None or code.co_filename.startswith('<')
            or getattr(transforms, '__qualname__', None) != name):
        raise ConfigError(
            f"the transforms function {transforms!r} cannot be dumped: it "
            f"must be a function defined at the top level of a file")
    return dict(file=code.co_filename, func=name)


class ParamSpace(parseutils.Serializable):
    """
    The parameters of a model evaluation or a fit: what sets each element
    of each parameter, and their evaluation.

    Each parameter, or some of its elements, has a property:

    - a number: the elements are fixed to it
    - a string: an expression of other parameters and constants, which
      ties the elements to it (e.g. 'hi_vpt_vt': 'vmax * 0.9')
    - a dict: the elements are free, and the dict holds the options of
      the fitter (e.g. {'value': 1, 'min': 0, 'max': 2}); the values of
      the attributes whose names start with '*' are lists, spread over
      the elements (e.g. {'*value': [1, 2, 3], 'min': 0})
    - None: the transforms function sets the elements
    - a list of these: one for each element

    The keys of the properties are the names of the parameters, with a
    subscript of integer constants to select elements of a vector (e.g.
    'v[0]', 'v[-1]', 'v[1:4]', 'v[::2]', 'v[[0, 2, 5]]'). Properties of
    unknown parameters are reported like unknown options (see
    parseutils.report_unknown).

    Expressions can use numbers, the values of parameters and their
    elements, constants, arithmetic, comparisons, conditional expressions
    (a if c else b), a few builtins (abs, min, max, round, sum, len), and
    of numpy (np, np.linalg) its ufuncs, numbers, scalar types and its
    functions on arrays (e.g. np.sin, np.pi, np.sum). An expression that
    reads a parameter with a mode reads its decoded values. Expressions
    and a transforms function are mutually exclusive.

    Parameters
    ----------
    pdescs : Mapping
        The descriptions of the parameters, by name.
    properties : Mapping
        The properties, by key.
    transforms : Transforms, optional
        A function that sets the elements whose property is None. It gets
        the values of all parameters (ParamValues), and sets these.
    constants : Mapping, optional
        Values that expressions can use (e.g. the radial nodes of the
        disks, from the constants() of the models), by name.
    modes : Mapping, optional
        The modes of vector parameters, by name: their properties then
        give coded values (e.g. offsets), which the mode decodes into
        their values (see ParamMode).

    Raises
    ------
    ConfigError
        If the properties, constants, modes or transforms are invalid
        (one error lists all the problems).
    """

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            pdescs: Mapping[str, ParamDesc],
            constants: Mapping[str, npt.ArrayLike] | None = None
    ) -> 'ParamSpace':
        """
        Load the parameters from their configuration: their properties,
        modes and transforms function ({file, func}).

        Parameters
        ----------
        info : dict
            The configuration.
        pdescs : Mapping
            The descriptions of the parameters, by name.
        constants : Mapping, optional
            The values that expressions can use, by name.

        Returns
        -------
        ParamSpace
            The parameters.

        Raises
        ------
        ConfigError
            If the configuration is invalid.
        """
        info = dict(info)
        # The properties are not checked against their annotation, whose
        # message would be unreadable: ParamSpace explains each problem
        properties = info.pop('properties', None)
        if not isinstance(properties, Mapping):
            raise ConfigError(
                "option 'properties' must be a dict of the properties of "
                "the parameters, by key")
        if 'transforms' in info:
            info['transforms'] = parseutils.load_option(
                _load_transforms, info, 'transforms')
        if info.get('modes') is not None:
            with parseutils.config_path('modes'):
                info['modes'] = {
                    name: parseutils.load_option(
                        param_mode_parser.load, info['modes'], name)
                    for name in dict(info['modes'])}
        opts = parseutils.parse_options_for_callable(
            info, cls.__init__,
            ignore_params=['pdescs', 'properties', 'constants'])
        return cls(pdescs, properties, **opts, constants=constants)

    def dump(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(properties=copy.deepcopy(self._properties))
        if self._modes:
            info['modes'] = {
                name: param_mode_parser.dump(mode)
                for name, mode in self._modes.items()}
        if self._transforms is not None:
            info['transforms'] = _dump_transforms(self._transforms)
        return info

    def __init__(
            self,
            pdescs: Mapping[str, ParamDesc],
            properties: Mapping[str, Property],
            transforms: Transforms | None = None,
            constants: Mapping[str, npt.ArrayLike] | None = None,
            modes: Mapping[str, ParamMode] | None = None
    ):
        if bad := [n for n, p in pdescs.items()
                   if not isinstance(p, ParamDesc)]:
            raise TypeError(
                f"the descriptions of these parameters are not ParamDesc: "
                f"{bad}")
        self._pdescs = dict(pdescs)
        self._properties = copy.deepcopy(dict(properties))
        self._transforms = transforms
        self._modes = dict(modes or {})
        self._constants = {}
        for name, value in (constants or {}).items():
            value = np.array(value, dtype=float)
            value.flags.writeable = False
            self._constants[name] = value
        # All elements of all parameters, in the order of the pdescs
        self._elements: list[_Element] = [
            (name, index)
            for name, pdesc in self._pdescs.items()
            for index in (
                [None] if pdesc.is_scalar() else range(pdesc.size()))]
        # What sets each element, and the fixed values, the properties of
        # the free elements, and the expressions
        self._kinds: dict[_Element, ParamKind] = {}
        self._fixed: dict[_Element, float] = {}
        self._free: dict[_Element, dict[str, Any]] = {}
        self._assignments: list[_Assignment] = []
        errors = self._check_names()
        errors += self._read_properties()
        errors += self._check_transforms()
        errors += self._check_modes()
        errors += self._check_fixed_ranges()
        if not errors:
            errors += self._order_steps()
        if errors:
            raise ConfigError(
                "invalid parameter properties:\n- " + "\n- ".join(errors))
        self._lay_out_buffer()

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the descriptions of the parameters, by name."""
        return self._pdescs

    def properties(self) -> dict[str, Property]:
        """Return the properties, by key."""
        return self._properties

    def transforms(self) -> Transforms | None:
        """Return the transforms function, if any."""
        return self._transforms

    def modes(self) -> dict[str, ParamMode]:
        """Return the modes of the vector parameters that have one."""
        return self._modes

    def kinds(self) -> dict[str, ParamKind]:
        """
        Return what sets each element of each parameter, by element name
        (e.g. 'a', 'v[3]'), in the order of the pdescs.
        """
        return {element_name(*e): self._kinds[e] for e in self._elements}

    def free_properties(self) -> dict[str, dict[str, Any]]:
        """
        Return the properties of the free elements, by element name, in
        the order in which evaluate takes their values.
        """
        return {
            element_name(*e): copy.deepcopy(self._free[e])
            for e in self._free_elements}

    def free_valid_ranges(self) -> dict[str, tuple[float, float]]:
        """
        Return the valid range of each free element whose value is that of
        its parameter (see ParamDesc), by element name: of all but those
        of the parameters with a mode, whose values are coded.
        """
        return {
            element_name(name, index): (
                self._pdescs[name].minimum(), self._pdescs[name].maximum())
            for name, index in self._free_elements
            if name not in self._modes}

    def evaluate(
            self,
            free: Mapping[str, float] | Sequence[float] | None = None
    ) -> ParamValues:
        """
        Evaluate the parameters.

        Parameters
        ----------
        free : Mapping or Sequence, optional
            The values of the free elements: by element name, or in the
            order of free_properties. By default, their 'value' property
            (e.g. the initial value of a fit).

        Returns
        -------
        ParamValues
            The values of all parameters, by name (new arrays each time).
            The values of a parameter with a mode are decoded.

        Raises
        ------
        ValueError
            If the values of the free elements are missing or unknown.
        ParamEvaluationError
            If an expression or the transforms function fails, or a value
            is not finite or outside its valid range.
        """
        buffer = self._template.copy()
        buffer[self._free_flat] = self._free_values(free)
        namespace: dict[str, Any] = dict(self._constants)
        for name, (start, stop, scalar) in self._layout.items():
            namespace[name] = (
                float(buffer[start]) if scalar else buffer[start:stop])
        for step in self._steps:
            if isinstance(step, _Assignment):
                self._assign(buffer, namespace, step)
            else:
                namespace[step] = self._modes[step].decode(namespace[step])
        values = {name: namespace[name] for name in self._pdescs}
        if self._transforms:
            self._apply_transforms(buffer, values)
        self._check_values(buffer, values)
        return values

    def with_values(self, values: Mapping[str, float]) -> 'ParamSpace':
        """
        Return a copy of the parameters whose free elements start at the
        given values (their 'value' properties): e.g. the elements of the
        solution of a fit (FitSolution.elements), to start another fit
        there, with another fitter or with other parameters free or tied.

        Parameters
        ----------
        values : Mapping
            The values of elements, by element name (e.g. 'a', 'v[3]').
            Those of a parameter with a mode are its decoded values, all of
            them. Values of elements that are not free here are ignored,
            and free elements without a value keep theirs.

        Returns
        -------
        ParamSpace
            The parameters, with their other properties unchanged.

        Raises
        ------
        ValueError
            If some but not all the values of a parameter with a mode are
            given, or a key of several free elements gets values for some
            of them only, and the others have none.
        """
        starts = self._start_values(values)
        properties = {}
        for key, prop in self._properties.items():
            parsed = parse_key(key, self._pdescs)
            properties[key] = copy.deepcopy(prop) if parsed is None \
                else self._start_property(key, parsed, prop, starts)
        return ParamSpace(
            self._pdescs, properties, self._transforms, self._constants,
            self._modes)

    def element_values(self, values: ParamValues) -> dict[str, float]:
        """
        Return the value of each element of each parameter, by element
        name (e.g. 'a', 'v[3]').

        Parameters
        ----------
        values : ParamValues
            The values of the parameters (see evaluate).

        Returns
        -------
        dict
            The value of each element.
        """
        return {
            element_name(name, index): float(
                values[name] if index is None else values[name][index])
            for name, index in self._elements}

    def _start_values(
            self, values: Mapping[str, float]
    ) -> dict[_Element, float]:
        """
        Return the start values of the free elements that the given values
        (by element name) have, coded for the parameters with a mode.
        """
        starts = {}
        for element in self._free:
            name, index = element
            if name not in self._modes \
                    and (ename := element_name(name, index)) in values:
                starts[element] = float(values[ename])
        for name, mode in self._modes.items():
            free = [e for e in self._free if e[0] == name]
            names = element_names(name, range(self._pdescs[name].size()))
            given = [n for n in names if n in values]
            if not free or not given:
                continue
            if len(given) < len(names):
                raise ValueError(
                    f"'{name}' has a mode: give all its values, or none; "
                    f"missing: {[n for n in names if n not in values]}")
            coded = mode.encode(np.array([values[n] for n in names]))
            starts.update({e: float(coded[e[1]]) for e in free})
        return starts

    def _start_property(
            self,
            key: str,
            parsed: Key,
            prop: Property,
            starts: dict[_Element, float]
    ) -> Property:
        """
        Return the property of a key with the 'value' of its free elements
        set from the start values (one list entry, or one '*value' item,
        per element for a key of several elements).
        """
        if _is_list(prop):
            return [
                self._start_property(
                    key, Key(parsed.name, parsed.indices[i:i + 1], True),
                    item, starts)
                for i, item in enumerate(prop)]
        elements = self._elements_of(parsed.name, parsed.indices)
        if not isinstance(prop, Mapping) \
                or not any(e in starts for e in elements):
            return copy.deepcopy(prop)
        if parsed.indices is None or parsed.element or len(elements) == 1:
            return copy.deepcopy(dict(prop)) | {'value': starts[elements[0]]}
        if missing := [
                element_name(*e) for e in elements
                if e not in starts and 'value' not in self._free[e]]:
            raise ValueError(
                f"'{key}': no start values for {missing}, but for the "
                f"other elements of the key")
        result = {
            k: copy.deepcopy(v) for k, v in prop.items()
            if k not in ('value', '*value')}
        result['*value'] = [
            starts.get(e, self._free[e].get('value')) for e in elements]
        return result

    def _check_names(self) -> list[str]:
        """Return the problems of the names of parameters and constants."""
        errors = []
        for what, names in [
                ('parameter', self._pdescs), ('constant', self._constants)]:
            for name in names:
                if reason := check_name(name):
                    errors.append(f"the {what} {name!r} has an invalid "
                                  f"name: {reason}")
        if conflicts := sorted(set(self._constants) & set(self._pdescs)):
            errors.append(
                f"constants cannot have the names of parameters: "
                f"{conflicts}")
        return errors

    def _read_properties(self) -> list[str]:
        """
        Record what sets each element from the properties, and return the
        problems: invalid keys or values, elements set more than once or
        never, and unknown parameters.
        """
        errors = []
        keys_of = collections.defaultdict(list)
        unknown_keys = []
        for key, value in self._properties.items():
            try:
                parsed = parse_key(key, self._pdescs)
            except InvalidKeyError as e:
                errors.append(f"{key!r}: {e}")
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
            # In strict mode, an error among the others
            try:
                parseutils.report_unknown(
                    "unknown parameters", unknown_keys, self._pdescs)
            except ConfigError as e:
                errors.append(e.message)
        return errors

    @staticmethod
    def _items(
            key: Key, value: Any
    ) -> list[tuple[np.ndarray | None, Any]]:
        """
        Split the property of a key into those of its elements: pairs of
        the indices of elements (None for a scalar) and one property.
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
    def _elements_of(
            name: str, indices: np.ndarray | None
    ) -> list[_Element]:
        """Return the given elements of a parameter (all if a scalar)."""
        if indices is None:
            return [(name, None)]
        return [(name, int(i)) for i in indices]

    def _set(
            self,
            key: str,
            name: str,
            indices: np.ndarray | None,
            value: Any
    ) -> None:
        """
        Record what sets the given elements. Raise ValueError (or
        InvalidExpressionError) if the property is invalid.
        """
        elements = self._elements_of(name, indices)
        if isinstance(value, bool):
            raise ValueError(f"{value} is not a number")
        if isinstance(value, numbers.Real):
            kind = ParamKind.FIXED
            self._fixed.update(dict.fromkeys(elements, float(value)))
        elif isinstance(value, str):
            kind = ParamKind.TIED
            expression = Expression(value, self._pdescs, self._constants)
            self._assignments.append(
                _Assignment(key, name, indices, expression))
        elif value is None:
            kind = ParamKind.TRANSFORMED
        elif isinstance(value, Mapping):
            kind = ParamKind.FREE
            self._free.update(
                {e: copy.deepcopy(dict(value)) for e in elements})
        else:
            raise ValueError(f"invalid value: {value!r}")
        self._kinds.update(dict.fromkeys(elements, kind))

    def _check_transforms(self) -> list[str]:
        """
        Return the problems of the transforms function: expressions too,
        or elements set to None without one.
        """
        nones = [element_name(*e) for e, k in self._kinds.items()
                 if k is ParamKind.TRANSFORMED]
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

    def _check_modes(self) -> list[str]:
        """
        Return the problems of the modes: of unknown or scalar parameters,
        of parameters they do not suit, or of elements that the transforms
        function sets.
        """
        errors = []
        for name, mode in self._modes.items():
            pdesc = self._pdescs.get(name)
            if pdesc is None:
                errors.append(f"the mode of an unknown parameter: '{name}'")
            elif pdesc.is_scalar():
                errors.append(
                    f"only vector parameters can have a mode; '{name}' is "
                    f"a scalar")
            else:
                try:
                    mode.check(pdesc.size())
                except ConfigError as e:
                    errors.append(f"the mode of '{name}': {e.message}")
                if any(self._kinds.get((name, i)) is ParamKind.TRANSFORMED
                       for i in range(pdesc.size())):
                    errors.append(
                        f"the transforms function cannot set elements of "
                        f"'{name}', which has a mode")
        return errors

    def _check_fixed_ranges(self) -> list[str]:
        """
        Return the fixed values outside the valid ranges of their
        parameters. Those of parameters with a mode are coded, so they are
        checked when evaluated.
        """
        return [
            f"'{element_name(*e)}' is {value}, outside its valid range "
            f"{self._range_text(e[0])}"
            for e, value in self._fixed.items()
            if e[0] not in self._modes and not (
                self._pdescs[e[0]].minimum() <= value
                <= self._pdescs[e[0]].maximum())]

    def _range_text(self, name: str) -> str:
        """Return the valid range of a parameter, as text."""
        pdesc = self._pdescs[name]
        return f"[{pdesc.minimum()}, {pdesc.maximum()}]"

    def _order_steps(self) -> list[str]:
        """
        Order the assignments and the decoding of the parameters with a
        mode (their names), so that each comes after those that set the
        elements it reads, and return the problems (a cycle). An
        expression that reads a parameter with a mode reads its decoded
        values, so it comes after its decoding, which comes after the
        assignments of its elements.
        """
        writer = {}
        for i, a in enumerate(self._assignments):
            for index in [None] if a.indices is None else a.indices:
                writer[(a.name, None if index is None else int(index))] = i
        graph: graphlib.TopologicalSorter = graphlib.TopologicalSorter()
        for name in self._modes:
            graph.add(name)
            for (written, _), i in writer.items():
                if written == name:
                    graph.add(name, i)
        for i, a in enumerate(self._assignments):
            graph.add(i)
            for name, indices in a.expression.reads.items():
                if name in self._modes:
                    graph.add(i, name)
                    continue
                for index in [None] if indices is None else sorted(indices):
                    if (name, index) in writer:
                        graph.add(i, writer[(name, index)])
        try:
            order = list(graph.static_order())
        except graphlib.CycleError as e:
            cycle = ' -> '.join(
                f"the mode of '{step}'" if isinstance(step, str)
                else f"'{self._assignments[step].key}'"
                for step in e.args[1])
            return [f"the expressions depend on each other in a cycle: "
                    f"{cycle}"]
        self._steps: list[_Assignment | str] = [
            step if isinstance(step, str) else self._assignments[step]
            for step in order]
        return []

    def _lay_out_buffer(self) -> None:
        """
        Lay out the values of all elements in one buffer, in the order of
        the elements: the position of each element and parameter in it,
        and its values before each evaluation (those of the fixed
        elements, and nan for the others).
        """
        self._flat = {e: i for i, e in enumerate(self._elements)}
        self._layout: dict[str, tuple[int, int, bool]] = {}
        start = 0
        for name, pdesc in self._pdescs.items():
            self._layout[name] = (
                start, start + pdesc.size(), pdesc.is_scalar())
            start += pdesc.size()
        self._template = np.array(
            [self._fixed.get(e, np.nan) for e in self._elements])
        self._free_elements = [e for e in self._elements if e in self._free]
        self._free_names = [element_name(*e) for e in self._free_elements]
        self._free_flat = np.array(
            [self._flat[e] for e in self._free_elements], dtype=int)
        self._ranged = [
            name for name, pdesc in self._pdescs.items() if pdesc.has_range()]

    def _free_values(
            self, free: Mapping[str, float] | Sequence[float] | None
    ) -> list[float]:
        """
        Return the values of the free elements, in order (see evaluate).
        """
        if free is None:
            if missing := [
                    n for n, e in zip(self._free_names, self._free_elements)
                    if 'value' not in self._free[e]]:
                raise ValueError(
                    f"these free parameters have no 'value' to evaluate "
                    f"them at: {missing}")
            return [self._free[e]['value'] for e in self._free_elements]
        if not isinstance(free, Mapping):
            free = list(free)
            if len(free) != len(self._free_names):
                raise ValueError(
                    f"expected {len(self._free_names)} free values, "
                    f"got {len(free)}")
            return free
        problems = []
        if missing := [n for n in self._free_names if n not in free]:
            problems.append(f"these free parameters are missing: {missing}")
        if extra := [n for n in free if n not in self._free_names]:
            problems.append(f"these parameters are not free: {extra}")
        if problems:
            raise ValueError('; '.join(problems))
        return [free[n] for n in self._free_names]

    def _assign(
            self,
            buffer: np.ndarray,
            namespace: dict[str, Any],
            assignment: _Assignment
    ) -> None:
        """Evaluate an expression, and set the elements it sets."""
        a = assignment
        try:
            result = np.asarray(a.expression.evaluate(namespace), dtype=float)
        except Exception as e:
            raise ParamEvaluationError(
                f"the expression of '{a.key}' ('{a.expression.source}') "
                f"failed: {e}") from e
        if a.indices is None:
            if result.ndim != 0:
                raise ParamEvaluationError(
                    f"the expression of '{a.key}' gives a vector of size "
                    f"{result.size}, but '{a.name}' is a scalar")
            namespace[a.name] = buffer[self._flat[(a.name, None)]] = float(
                result)
            return
        if result.ndim > 1 or (
                result.ndim == 1 and result.size != len(a.indices)):
            raise ParamEvaluationError(
                f"the expression of '{a.key}' gives a vector of size "
                f"{result.size}, for a key of size {len(a.indices)}")
        # The namespace has a view of the buffer
        namespace[a.name][a.indices] = result

    def _apply_transforms(
            self, buffer: np.ndarray, values: ParamValues
    ) -> None:
        """
        Apply the transforms function. It works on a copy of the values,
        and only the elements set to None are taken from it.
        """
        result = copy.deepcopy(values)
        try:
            self._transforms(result)
        except Exception as e:
            raise ParamEvaluationError(
                f"the transforms function failed: {e}") from e
        for element, kind in self._kinds.items():
            if kind is not ParamKind.TRANSFORMED:
                continue
            name, index = element
            try:
                if index is None:
                    value = float(result[name])
                    values[name] = buffer[self._flat[element]] = value
                else:
                    values[name][index] = result[name][index]
            except Exception as e:
                raise ParamEvaluationError(
                    f"the transforms function did not set "
                    f"'{element_name(name, index)}' to a number: {e}") from e

    def _check_values(self, buffer: np.ndarray, values: ParamValues) -> None:
        """
        Raise ParamEvaluationError if a value is not finite, or outside
        its valid range. The buffer has the coded values of the
        parameters with a mode, so those are checked in values.
        """
        if not (np.all(np.isfinite(buffer)) and all(
                np.all(np.isfinite(values[name])) for name in self._modes)):
            bad = [
                f"'{element_name(name, index)}' is {value}"
                for name, index in self._elements
                if not np.isfinite(value := (
                    values[name] if index is None else values[name][index]))]
            raise ParamEvaluationError(
                f"the values of parameters must be finite: {', '.join(bad)}")
        bad = []
        for name in self._ranged:
            pdesc = self._pdescs[name]
            value = np.atleast_1d(values[name])
            outside = (value < pdesc.minimum()) | (value > pdesc.maximum())
            if np.any(outside):
                names = element_names(
                    name, None if pdesc.is_scalar()
                    else np.flatnonzero(outside))
                bad += [
                    f"'{n}' is {v} (valid: {self._range_text(name)})"
                    for n, v in zip(names, value[outside])]
        if bad:
            raise ParamEvaluationError(
                f"the values of parameters must be in their valid ranges: "
                f"{', '.join(bad)}")


param_space_parser = parseutils.BasicParser(ParamSpace)
