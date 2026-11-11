"""
ParamSpace: the parameters of a model evaluation or a fit, given their
descriptions and the properties of a configuration.
"""

import abc
import collections
import copy
import dataclasses
import enum
import graphlib
import inspect
import logging
import numbers
import typing
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

import gbkfit.units
from gbkfit.config import (
    BasicParser, ConfigError, Serializable, config_path, describe_unknown,
    load_option, parse_options, parse_options_for_callable, resolve_path)
from gbkfit.params._errors import ParamEvaluationError
from gbkfit.params._expressions import (
    Expression, InvalidExpressionError, check_name)
from gbkfit.params._keys import (
    InvalidKeyError, Key, element_name, element_names, parse_key)
from gbkfit.params.elements import (
    DerivedValue, Element, Elements, FreeElement, Solved, Tied, Transformed)
from gbkfit.params.modes import ParamMode, param_mode_parser
from gbkfit.params.pdescs import (
    ParamDesc, ParamUnit, ValidRange, make_param_unit)
from gbkfit.params.properties import (
    Property, dump_elements, is_list, parse_derived, parse_properties,
    problems_text)
from gbkfit.params.solves import Solve
from gbkfit.utils import funcutils

if typing.TYPE_CHECKING:
    from gbkfit.params.joint_priors import JointPrior


__all__ = [
    'ParamFunctions',
    'ParamKind',
    'ParamSpace',
    'ParamValues',
    'Transforms',
    'param_space_parser'
]


_log = logging.getLogger(__name__)


# The values of parameters, by name: a float for a scalar, an array for a
# vector
ParamValues: TypeAlias = dict[str, float | np.ndarray]

# A function that sets the parameters whose property is None, in the
# values of all parameters (see ParamSpace)
Transforms: TypeAlias = Callable[[ParamValues], None]

# An element of a parameter: its name, and its index (None for a scalar)
_Element: TypeAlias = tuple[str, int | None]


class ParamFunctions(abc.ABC):
    """
    Names that the derived values of parameters can use besides the
    parameters and the constants (see ParamSpace): functions of the
    models (e.g. the circular velocity of a galaxy at any radius) and
    numbers (e.g. the gravitational constant), whose values can depend
    on those of the parameters.
    """

    @abc.abstractmethod
    def names(self) -> tuple[str, ...]:
        """Return the names."""

    @abc.abstractmethod
    def bind(self, values: ParamValues) -> dict[str, Any]:
        """
        Return the value of each name (a function or a number) at the
        values of the parameters, by name.
        """


class ParamKind(enum.Enum):
    """What sets the value of an element of a parameter."""
    FIXED = 'fixed'              # a number
    TIED = 'tied'                # an expression
    FREE = 'free'                # the caller (e.g. a fitter)
    SOLVED = 'solved'            # its solve, of the data (see Solve)
    TRANSFORMED = 'transformed'  # the transforms function (set to None)


@dataclasses.dataclass
class _Assignment:
    """An expression, and the elements of a parameter it sets."""
    key: str
    name: str
    indices: np.ndarray | None
    expression: Expression


def _load_transforms(info: dict[str, Any]) -> Transforms:
    """Load a transforms function from its file: info is {file, func}."""
    opts = parse_options(info, {'file', 'func'})
    return funcutils.load_function_from_file(
        resolve_path(opts['file']), opts['func'])


def _dump_transforms(transforms: Transforms) -> dict[str, str]:
    """
    Return the file and the name of a transforms function. It must be
    defined at the top level of a file, from which it loads again. Raise
    TypeError for another function, which has no configuration.
    """
    # The function itself, of the wrapper of a function of a file (see
    # load_function_from_file)
    transforms = inspect.unwrap(transforms)
    code = getattr(transforms, '__code__', None)
    name = getattr(transforms, '__name__', None)
    if (code is None or code.co_filename.startswith('<')
            or getattr(transforms, '__qualname__', None) != name):
        raise TypeError(
            f"the transforms function {transforms!r} cannot be dumped: it "
            f"must be a function defined at the top level of a file")
    return dict(file=code.co_filename, func=name)


class ParamSpace(Serializable):
    """
    The parameters of a model evaluation or a fit: what sets each element
    of each parameter, and their evaluation.

    Each parameter, or some of its elements, is set by an element (see
    elements):

    - a number: the elements are fixed to it
    - Tied: an expression of other parameters and constants, which ties
      the elements to it (e.g. Tied('vmax * 0.9'))
    - FreeElement: the elements are free, with their properties, the same
      for every fitter (e.g. FreeElement(value=1, min=0, max=2))
    - Solved: the elements are solved at each evaluation of an objective,
      from the data, by a solve (see Solve), in the valid range of the
      parameter; only parameters that the model is linear in can be (the
      objective says which)
    - Transformed: the transforms function sets the elements
    - a list of these: one for each element

    Configurations give them as properties instead (see from_properties
    and parse_properties): a number, an expression ('vmax * 0.9'), a dict
    of the options of a free element ({'value': 1, 'min': 0, 'max': 2},
    whose attributes whose names start with '*' are lists spread over the
    elements, e.g. {'*value': [1, 2, 3], 'min': 0}), a dict of 'solve'
    alone ({'solve': {'type': 'gaussian', 'mean': 1, 'sigma': 0.5}}),
    None, or a list of these.

    The keys are the names of the parameters, with a subscript of integer
    constants to select elements of a vector (e.g. 'v[0]', 'v[-1]',
    'v[1:4]', 'v[::2]', 'v[[0, 2, 5]]').

    Expressions can use numbers, the values of parameters and their
    elements, constants, arithmetic, comparisons, conditional expressions
    (a if c else b), a few builtins (abs, min, max, round, sum, len), of
    numpy (np, np.linalg) its ufuncs, numbers, scalar types and its
    functions on arrays (e.g. np.sin, np.pi, np.sum), and the scaling
    relations of gbkfit.relations (e.g. relations.log_mhalo_moster13).
    An expression that reads a parameter with a mode reads its decoded
    values. Expressions and a transforms function are mutually exclusive.
    No expression can read a parameter with solved elements, which have
    no values before the objective solves them (their priors are in their
    solves), and a parameter with a mode has none.

    Derived values are quantities that the models compute from the
    parameters, which no expression of them can (e.g. the circular
    velocity of a mass model at 2.2 disk scale lengths): expressions, by
    name, that can also use the functions of the models (e.g.
    'vcirc(2.2 * mass_stars_rd)'; see ParamFunctions), and the derived
    values before them. The parameters know the names of the functions;
    the functions themselves (which run parts of the models on a driver,
    see GroupPlan.functions) are given where the derived values are
    evaluated. They are evaluated after the parameters (see
    derived_values), so the elements cannot read them; the priors on
    several parameters can (see expression), and the fitters report them
    (see FitSolution).

    Parameters
    ----------
    pdescs : Mapping
        The descriptions of the parameters, by name.
    elements : Mapping
        What sets the elements of the parameters, by key: one element for
        all those of the key, or a list of one for each.
    transforms : Transforms, optional
        A function that sets the Transformed elements. It gets the values
        of all parameters (ParamValues), and sets these.
    constants : Mapping, optional
        Values that expressions can use (e.g. the radial nodes of the
        disks, from the constants() of the models), by name.
    modes : Mapping, optional
        The modes of vector parameters, by name: their elements then set
        coded values (e.g. offsets), which the mode decodes into their
        values (see ParamMode).
    priors : Sequence of JointPrior, optional
        The priors on several parameters at once, which the fitters add to
        those of the free elements.
    derived : Mapping of str to str or DerivedValue, optional
        The derived values, by name, in order: the expression of each, or
        its expression and the unit of its values.
    function_names : Collection of str, optional
        The names of the functions of the models that the derived values
        can use (e.g. ObservationGroup.function_names).

    Raises
    ------
    ValueError
        If the elements, constants, modes, transforms, priors or derived
        values are invalid (one error lists all the problems).
    """

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            pdescs: Mapping[str, ParamDesc],
            constants: Mapping[str, npt.ArrayLike] | None = None,
            function_names: Collection[str] = ()
    ) -> 'ParamSpace':
        """
        Load the parameters from their configuration: their properties,
        modes, transforms function ({file, func}), priors and derived
        values (see from_properties).

        Parameters
        ----------
        info : dict
            The configuration.
        pdescs : Mapping
            The descriptions of the parameters, by name.
        constants : Mapping, optional
            The values that expressions can use, by name.
        function_names : Collection of str, optional
            The names of the functions of the models that derived values
            can use.

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
        # message would be unreadable: their parsing explains each problem
        properties = info.pop('properties', None)
        if not isinstance(properties, Mapping):
            raise ConfigError(
                "option 'properties' must be a dict of the properties of "
                "the parameters, by key")
        priors = info.pop('priors', None)
        derived = info.pop('derived', None)
        if 'transforms' in info:
            info['transforms'] = load_option(
                _load_transforms, info, 'transforms')
        opts = parse_options_for_callable(
            info, cls.from_properties,
            ignore_params=[
                'pdescs', 'properties', 'constants', 'priors', 'derived',
                'function_names'])
        return cls.from_properties(
            pdescs, properties, **opts, constants=constants, priors=priors,
            derived=derived, function_names=function_names)

    @classmethod
    def from_properties(
            cls,
            pdescs: Mapping[str, ParamDesc],
            properties: Mapping[str, Property],
            transforms: Transforms | None = None,
            constants: Mapping[str, npt.ArrayLike] | None = None,
            modes: Mapping[str, ParamMode] | None = None,
            priors: Sequence[Mapping[str, Any]] | None = None,
            derived: Mapping[str, Any] | None = None,
            function_names: Collection[str] = ()
    ) -> 'ParamSpace':
        """
        Make the parameters of the properties of a configuration (see the
        class, and parse_properties): the language of configurations,
        whose mistakes are mistakes of the configuration.

        Parameters
        ----------
        pdescs : Mapping
            The descriptions of the parameters, by name.
        properties : Mapping
            The properties, by key.
        transforms, constants, modes, function_names
            As those of the class.
        priors : Sequence of dict, optional
            The configurations of the priors on several parameters (e.g.
            {'type': 'gauss', 'expr': 'vt * 0.9', ...}; see JointPrior).
        derived : Mapping, optional
            The derived values, by name, in order: the expression of each,
            or {expr: ..., unit: ...}.

        Returns
        -------
        ParamSpace
            The parameters.

        Raises
        ------
        ConfigError
            If the properties, priors or derived values are invalid, or
            the parameters are (as those of the class; one error lists all
            the problems).
        """
        # Imported here: joint_priors imports this module
        from gbkfit.params.joint_priors import joint_prior_parser
        elements, problems = parse_properties(properties, pdescs)
        # With those of the names, which make keys invalid (e.g. that of a
        # parameter named after a keyword)
        if problems := _name_problems(
                pdescs, constants or {}, function_names) + problems:
            raise ConfigError(problems_text(problems))
        joint = []
        for i, info in enumerate(priors or []):
            with config_path('priors', i):
                joint.append(joint_prior_parser.load(info))
        space = cls.__new__(cls)
        if problems := space._build(
                pdescs, elements, transforms, constants, modes, joint,
                parse_derived(derived or {}), function_names):
            raise ConfigError(problems_text(problems))
        return space

    def dump(self) -> dict[str, Any]:
        # Imported here: joint_priors imports this module
        from gbkfit.params.joint_priors import joint_prior_parser
        info: dict[str, Any] = dict(properties=dump_elements(self._given))
        if self._modes:
            info['modes'] = {
                name: param_mode_parser.dump(mode)
                for name, mode in self._modes.items()}
        if self._transforms is not None:
            info['transforms'] = _dump_transforms(self._transforms)
        if self._priors:
            info['priors'] = [
                joint_prior_parser.dump(prior) for prior in self._priors]
        if self._derived:
            info['derived'] = {
                name: expr if (unit := self._derived_units.get(name)) is None
                else dict(expr=expr, unit=gbkfit.units.format_unit(unit))
                for name, expr in self._derived.items()}
        return info

    def __init__(
            self,
            pdescs: Mapping[str, ParamDesc],
            elements: Elements,
            transforms: Transforms | None = None,
            constants: Mapping[str, npt.ArrayLike] | None = None,
            modes: Mapping[str, ParamMode] | None = None,
            priors: Sequence['JointPrior'] | None = None,
            derived: Mapping[str, str | DerivedValue] | None = None,
            function_names: Collection[str] = ()
    ):
        if problems := self._build(
                pdescs, elements, transforms, constants, modes, priors,
                derived, function_names):
            raise ValueError(problems_text(problems))

    def _build(
            self,
            pdescs: Mapping[str, ParamDesc],
            elements: Elements,
            transforms: Transforms | None,
            constants: Mapping[str, npt.ArrayLike] | None,
            modes: Mapping[str, ParamMode] | None,
            priors: Sequence['JointPrior'] | None,
            derived: Mapping[str, str | DerivedValue] | None,
            function_names: Collection[str]
    ) -> list[str]:
        """
        Set up the parameters (see the class), and return their problems:
        the constructor raises ValueError for them, and from_properties,
        whose arguments are of a configuration, ConfigError.

        Raises
        ------
        TypeError
            If a description is not a ParamDesc, or a prior not a
            JointPrior.
        """
        from gbkfit.params.joint_priors import JointPrior
        if bad := [n for n, p in pdescs.items()
                   if not isinstance(p, ParamDesc)]:
            raise TypeError(
                f"the descriptions of these parameters are not ParamDesc: "
                f"{bad}")
        self._pdescs = dict(pdescs)
        self._given = {
            key: list(value) if is_list(value) else value
            for key, value in elements.items()}
        self._transforms = transforms
        self._modes = dict(modes or {})
        self._priors = list(priors or [])
        if bad := [i for i, prior in enumerate(self._priors)
                   if not isinstance(prior, JointPrior)]:
            raise TypeError(f"the priors {bad} are not JointPrior")
        self._derived_given = dict(derived or {})
        self._function_names = tuple(function_names)
        # The names that the elements cannot read, and why
        self._unavailable = {
            name: "it is a derived value, evaluated after the parameters"
            for name in self._derived_given} | {
            name: "it is a function of the models, which only derived "
                  "values can use"
            for name in self._function_names}
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
        # What sets each element, and the fixed values, the free elements,
        # the solves of the solved ones, and the expressions
        self._kinds: dict[_Element, ParamKind] = {}
        self._fixed: dict[_Element, float] = {}
        self._free: dict[_Element, FreeElement] = {}
        self._solves: dict[_Element, Solve] = {}
        self._assignments: list[_Assignment] = []
        errors = _name_problems(
            self._pdescs, self._constants, self._function_names)
        errors += self._read_elements()
        errors += self._check_transforms()
        errors += self._check_modes()
        errors += self._check_fixed_ranges()
        errors += self._compile_derived()
        errors += self._check_solves()
        if not errors:
            errors += self._order_steps()
        if not errors:
            self._lay_out_buffer()
            errors += self._check_priors()
        return errors

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the descriptions of the parameters, by name."""
        return self._pdescs

    def elements(self) -> dict[str, Element | list[Element]]:
        """
        Return what sets the elements of the parameters, by key, as given
        (see the class).
        """
        return dict(self._given)

    def transforms(self) -> Transforms | None:
        """Return the transforms function, if any."""
        return self._transforms

    def modes(self) -> dict[str, ParamMode]:
        """Return the modes of the vector parameters that have one."""
        return self._modes

    def own_settings(self) -> set[str]:
        """
        Return the elements of vectors set apart from the other elements of
        their vector, by element name: by a key with a subscript (e.g.
        'v[0]'), or by a list of one element for each, but a list of free
        elements that differ only in their starting values (e.g. that of
        '*value' in a configuration, not '*min').
        """
        own = set()
        for key, value in self._given.items():
            parsed = parse_key(key, self._pdescs)
            if parsed is None or parsed.indices is None:
                continue
            if parsed.name != key.strip() or (
                    is_list(value) and not _differ_in_value_only(value)):
                own.update(element_names(parsed.name, parsed.indices))
        return own

    def priors(self) -> list['JointPrior']:
        """Return the priors on several parameters."""
        return list(self._priors)

    def derived(self) -> dict[str, str]:
        """Return the expressions of the derived values, by name."""
        return dict(self._derived)

    def unit(self, name: str) -> ParamUnit | None:
        """
        Return the unit of the values of a parameter or a derived value by
        name, or of an element (e.g. 'vpt_vt[3]'), if it has one (see
        ParamDesc.unit).
        """
        name = name.split('[', 1)[0]
        if name in self._derived_units:
            return self._derived_units[name]
        pdesc = self._pdescs.get(name)
        return None if pdesc is None else pdesc.unit()

    def function_names(self) -> tuple[str, ...]:
        """
        Return the names of the functions of the models that the derived
        values can use.
        """
        return self._function_names

    def expression(
            self,
            source: str,
            functions: ParamFunctions | None = None
    ) -> Callable[[ParamValues], np.ndarray]:
        """
        Return a function of the values of the parameters (see evaluate)
        that evaluates an expression of them, of the constants and of the
        derived values, as those of the properties (e.g. 'np.diff(posa)',
        'vt[4] * 0.9', 'v22 / 200'), into an array of floats. The derived
        values it reads are evaluated with it, with the given functions of
        the models (see derived_values).

        Raises
        ------
        ValueError
            If the expression is invalid.
        """
        unavailable = {
            name: "it is a function of the models, which only derived "
                  "values can use: derive a value with it (see the option "
                  "derived), and read that"
            for name in self._function_names} | {
            name: "it has solved elements" for name in self._solved_names()}
        try:
            expression = Expression(
                source, self._pdescs, self._constants, tuple(self._derived),
                unavailable)
        except InvalidExpressionError as e:
            raise ValueError(f"invalid expression '{source}': {e}") from None
        constants = self._constants

        def evaluate(values: ParamValues) -> np.ndarray:
            namespace = constants | dict(values)
            if expression.uses:
                namespace |= self._finite_derived_values(
                    values, expression.uses, functions)
            return np.atleast_1d(np.asarray(
                expression.evaluate(namespace), dtype=float))
        return evaluate

    def kinds(self) -> dict[str, ParamKind]:
        """
        Return what sets each element of each parameter, by element name
        (e.g. 'a', 'v[3]'), in the order of the pdescs.
        """
        return {element_name(*e): self._kinds[e] for e in self._elements}

    def free_properties(self) -> dict[str, dict[str, Any]]:
        """
        Return the properties of the free elements as a configuration
        gives them (e.g. to show them; see FreeElement.dump), by element
        name, in the order in which evaluate takes their values.
        """
        return {
            element_name(*e): self._free[e].dump()
            for e in self._free_elements}

    def free_elements(self) -> dict[str, FreeElement]:
        """
        Return the free elements (their properties for the fitters), by
        element name, in the order in which evaluate takes their values.
        """
        return {element_name(*e): self._free[e] for e in self._free_elements}

    def solves(self) -> dict[str, Solve]:
        """
        Return the solve of each solved element, by element name, in the
        order of the elements (elements of one key share one).
        """
        return {element_name(*e): self._solves[e] for e in self._solved}

    def periods(self) -> dict[str, float]:
        """
        Return the period of each element whose values repeat (see
        ParamDesc.period; e.g. 360 for an azimuth in degrees), by element
        name: of all but those of the parameters with a mode, whose values
        are coded.
        """
        return {
            element: period for element in self.kinds()
            if (name := element.split('[', 1)[0]) not in self._modes
            and (period := self._pdescs[name].period()) is not None}

    def free_valid_ranges(self) -> dict[str, ValidRange]:
        """
        Return the valid range of each free element whose value is that of
        its parameter (see ParamDesc), by element name: of all but those
        of the parameters with a mode, whose values are coded.
        """
        return {
            element_name(name, index): self._pdescs[name].valid_range()
            for name, index in self._free_elements
            if name not in self._modes}

    def evaluate(
            self,
            free: Mapping[str, float] | Sequence[float] | None = None,
            solved: Mapping[str, float] | None = None
    ) -> ParamValues:
        """
        Evaluate the parameters.

        Parameters
        ----------
        free : Mapping or Sequence, optional
            The values of the free elements: by element name, or in the
            order of free_properties. By default, their 'value' property
            (e.g. the initial value of a fit).
        solved : Mapping, optional
            The values of the solved elements, by element name (e.g. those
            of ObjectiveEvaluation.solved); those without one have their
            value without data (see Solve.default).

        Returns
        -------
        ParamValues
            The values of all parameters, by name (new arrays each time).
            The values of a parameter with a mode are decoded.

        Raises
        ------
        ValueError
            If the values of the free elements are missing or unknown, or
            values are given for elements that are not solved.
        ParamEvaluationError
            If an expression or the transforms function fails, or a value
            is not finite or outside its valid range.
        """
        buffer = self._template.copy()
        buffer[self._free_flat] = self._free_values(free)
        if solved:
            if extra := [n for n in solved if n not in self._solved_flat]:
                raise ValueError(f"these parameters are not solved: {extra}")
            for name, value in solved.items():
                buffer[self._solved_flat[name]] = value
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

    def derived_values(
            self,
            values: ParamValues,
            functions: ParamFunctions | None = None,
            names: Collection[str] | None = None
    ) -> ParamValues:
        """
        Evaluate the derived values (see the class), with the functions of
        the models at the values of the parameters.

        Parameters
        ----------
        values : ParamValues
            The values of the parameters (see evaluate).
        functions : ParamFunctions, optional
            The functions of the models (e.g. GroupPlan.functions), which
            the derived values that call them need.
        names : Collection of str, optional
            The derived values wanted (they are evaluated with those they
            read); all by default.

        Returns
        -------
        ParamValues
            The derived values, by name, in their order: a float for a
            number, an array otherwise (NaN where the models give it, e.g.
            the rotation of a disk where it has no circular orbit).

        Raises
        ------
        ValueError
            If a derived value calls a function of the models, and they
            are not given.
        ParamEvaluationError
            If an expression fails (e.g. a function of the models at radii
            outside its disk).
        """
        needed = self._needed_derived(names)
        namespace: dict[str, Any] = self._constants | dict(values)
        if calling := sorted(
                name for name in needed
                if set(self._derived_expressions[name].uses)
                & set(self._function_names)):
            if functions is None:
                raise ValueError(
                    f"the derived values {calling} call functions of the "
                    f"models: give them (see GroupPlan.functions)")
            namespace |= functions.bind(values)
        derived = {}
        for name, expression in self._derived_expressions.items():
            if name not in needed:
                continue
            try:
                result = np.asarray(
                    expression.evaluate(namespace), dtype=float)
            except Exception as e:
                raise ParamEvaluationError(
                    f"the derived value '{name}' ('{expression.source}') "
                    f"failed: {e}") from e
            namespace[name] = derived[name] = \
                float(result) if result.ndim == 0 else result
        return derived

    def derived_elements(
            self,
            values: ParamValues,
            functions: ParamFunctions | None = None
    ) -> dict[str, float]:
        """
        Return the value of each element of each derived value (see
        derived_values, with the given functions of the models), by
        element name (e.g. 'v22', 'vs[1]'; the elements of an array in its
        flat order).

        Raises
        ------
        ValueError
            If a derived value calls a function of the models, and they
            are not given.
        ParamEvaluationError
            If an expression fails.
        """
        elements = {}
        for name, value in self.derived_values(values, functions).items():
            if np.ndim(value) == 0:
                elements[name] = float(value)
                continue
            elements |= {
                element_name(name, i): float(item)
                for i, item in enumerate(np.ravel(value))}
        return elements

    def _needed_derived(self, names: Collection[str] | None) -> set[str]:
        """
        Return the derived values of the given names (all if None) and
        those they read, and so on.
        """
        if names is None:
            return set(self._derived_expressions)
        needed = set()
        pending = list(names)
        while pending:
            name = pending.pop()
            if name not in needed:
                needed.add(name)
                pending += [
                    used for used in self._derived_expressions[name].uses
                    if used in self._derived_expressions]
        return needed

    def _finite_derived_values(
            self,
            values: ParamValues,
            names: Collection[str],
            functions: ParamFunctions | None
    ) -> ParamValues:
        """
        Return the derived values that an expression reads (see
        expression), by name.

        Raises
        ------
        ParamEvaluationError
            If one fails or is not finite.
        """
        derived = self.derived_values(values, functions, names)
        if bad := [
                f"'{name}' is {derived[name]}" for name in sorted(names)
                if not np.all(np.isfinite(derived[name]))]:
            raise ParamEvaluationError(
                f"the derived values that an expression reads must be "
                f"finite: {', '.join(bad)}")
        return derived

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
            The parameters, with their other elements unchanged.

        Raises
        ------
        ValueError
            If some but not all the values of a parameter with a mode are
            given, or a key of several free elements gets values for some
            of them only, and the others have none.
        """
        starts = self._start_values(values)
        elements = {
            key: self._start_element(
                key, parse_key(key, self._pdescs), value, starts)
            for key, value in self._given.items()}
        return ParamSpace(
            self._pdescs, elements, self._transforms, self._constants,
            self._modes, self._priors, self._derived_given,
            self._function_names)

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

    def _start_element(
            self,
            key: str,
            parsed: Key,
            value: Element | list[Element],
            starts: dict[_Element, float]
    ) -> Element | list[Element]:
        """
        Return what sets the elements of a key with the free ones starting
        at the start values: one free element for each, for a key of
        several elements with start values.
        """
        if is_list(value):
            return [
                self._start_element(
                    key, Key(parsed.name, parsed.indices[i:i + 1], True),
                    item, starts)
                for i, item in enumerate(value)]
        elements = self._elements_of(parsed.name, parsed.indices)
        if not isinstance(value, FreeElement) \
                or not any(e in starts for e in elements):
            return value
        if parsed.indices is None or parsed.element or len(elements) == 1:
            return value.with_value(starts[elements[0]])
        if missing := [
                element_name(*e) for e in elements
                if e not in starts and value.value() is None]:
            raise ValueError(
                f"'{key}': no start values for {missing}, but for the "
                f"other elements of the key")
        return [
            value.with_value(starts.get(e, value.value())) for e in elements]

    def _read_elements(self) -> list[str]:
        """
        Record what sets each element, and return the problems: invalid
        keys or elements, elements set more than once or never, and
        unknown parameters.
        """
        errors = []
        keys_of = collections.defaultdict(list)
        unknown_keys = []
        for key, value in self._given.items():
            try:
                parsed = parse_key(key, self._pdescs)
            except InvalidKeyError as e:
                errors.append(f"{key!r}: {e}")
                continue
            if parsed is None:
                unknown_keys.append(key)
                continue
            # A key with an invalid element still sets its elements, so
            # that they are not also reported as missing
            for element in self._elements_of(parsed.name, parsed.indices):
                keys_of[element].append(key)
            try:
                for indices, item in self._items(parsed, value):
                    self._set(key, parsed.name, indices, item)
            except ValueError as e:
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
            errors.append(
                f"unknown parameters: "
                f"{describe_unknown(unknown_keys, self._pdescs)}")
        return errors

    @staticmethod
    def _items(
            key: Key, value: Element | Sequence[Element]
    ) -> list[tuple[np.ndarray | None, Element]]:
        """
        Split what sets the elements of a key into pairs of the indices of
        elements (None for a scalar) and one element.
        """
        if not is_list(value):
            return [(key.indices, value)]
        if key.indices is None or key.element:
            raise ValueError(
                "a list of elements needs a key of more than one element")
        if len(value) != len(key.indices):
            raise ValueError(
                f"a list of {len(value)} elements for {len(key.indices)} "
                f"elements")
        return [(key.indices[i:i + 1], v) for i, v in enumerate(value)]

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
            value: Element
    ) -> None:
        """
        Record what sets the given elements. Raise ValueError (or
        InvalidExpressionError) if the element is invalid.
        """
        elements = self._elements_of(name, indices)
        if isinstance(value, bool):
            raise ValueError(f"{value} is not a number")
        if isinstance(value, numbers.Real):
            kind = ParamKind.FIXED
            self._fixed.update(dict.fromkeys(elements, float(value)))
        elif isinstance(value, Tied):
            kind = ParamKind.TIED
            expression = Expression(
                value.expr, self._pdescs, self._constants,
                unavailable=self._unavailable)
            self._assignments.append(
                _Assignment(key, name, indices, expression))
        elif isinstance(value, Transformed):
            kind = ParamKind.TRANSFORMED
        elif isinstance(value, Solved):
            kind = ParamKind.SOLVED
            self._solves.update(dict.fromkeys(elements, value.solve))
        elif isinstance(value, FreeElement):
            kind = ParamKind.FREE
            self._free.update(dict.fromkeys(elements, value))
        else:
            raise ValueError(
                f"invalid element: {value!r} (a number, Tied, FreeElement, "
                f"Solved or Transformed)")
        self._kinds.update(dict.fromkeys(elements, kind))

    def _compile_derived(self) -> list[str]:
        """
        Compile the expressions of the derived values, each of which can
        read the parameters, the constants, the functions of the models
        and the derived values before it, and return the problems: invalid
        names or expressions.
        """
        errors = []
        self._derived: dict[str, str] = {}
        self._derived_units: dict[str, ParamUnit] = {}
        for name, value in self._derived_given.items():
            if isinstance(value, DerivedValue):
                if value.unit is not None:
                    try:
                        self._derived_units[name] = make_param_unit(
                            value.unit)
                    except ValueError as e:
                        errors.append(f"the derived value '{name}': {e}")
                value = value.expr
            self._derived[name] = value
        taken = set(self._pdescs) | set(self._constants) \
            | set(self._function_names)
        names = list(self._derived)
        self._derived_expressions: dict[str, Expression] = {}
        for i, (name, source) in enumerate(self._derived.items()):
            if reason := check_name(name):
                errors.append(
                    f"the derived value {name!r} has an invalid name: "
                    f"{reason}")
            elif name in taken:
                errors.append(
                    f"the derived value '{name}' has the name of a "
                    f"parameter, a constant or a function of the models")
            elif not isinstance(source, str):
                errors.append(
                    f"the derived value '{name}' must be an expression (a "
                    f"string); it is {source!r}")
            else:
                later = {
                    other: "derived values can read only those before them"
                    for other in names[i:]}
                try:
                    self._derived_expressions[name] = Expression(
                        source, self._pdescs, self._constants,
                        self._function_names + tuple(names[:i]), later)
                except InvalidExpressionError as e:
                    errors.append(f"the derived value '{name}': {e}")
        return errors

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
                except ValueError as e:
                    errors.append(f"the mode of '{name}': {e}")
                if any(self._kinds.get((name, i)) is ParamKind.TRANSFORMED
                       for i in range(pdesc.size())):
                    errors.append(
                        f"the transforms function cannot set elements of "
                        f"'{name}', which has a mode")
        return errors

    def _check_solves(self) -> list[str]:
        """
        Return the problems of the solved elements: those of parameters
        with a mode, those whose solves give values outside the valid range
        of their parameter, and expressions (of properties or derived
        values) that read their parameters.
        """
        errors = []
        names = self._solved_names()
        if moded := sorted(names & set(self._modes)):
            errors.append(
                f"parameters with a mode cannot have solved elements: "
                f"{moded}")
        for element, solve in self._solves.items():
            valid = self._pdescs[element[0]].valid_range()
            low, high = solve.bounds()
            if not (valid.contains(low) and valid.contains(high)):
                errors.append(
                    f"the solve of '{element_name(*element)}' gives values "
                    f"in [{low}, {high}], outside its valid range "
                    f"{self._range_text(element[0])} (e.g. give it a "
                    f"minimum)")
        readers = [(a.key, a.expression) for a in self._assignments] + [
            (f"derived value '{name}'", expression)
            for name, expression in self._derived_expressions.items()]
        for key, expression in readers:
            if read := sorted(names & set(expression.reads)):
                errors.append(
                    f"{key!r} reads {read}, which have solved elements: no "
                    f"expression can (their values are known only after "
                    f"the objective solves them)")
        return errors

    def _check_priors(self) -> list[str]:
        """
        Return the problems of the priors on several parameters: invalid
        expressions (see JointPrior.residuals).
        """
        errors = []
        for i, prior in enumerate(self._priors):
            try:
                prior.residuals(self)
            except ValueError as e:
                errors.append(f"the prior priors[{i}]: {e}")
        return errors

    def _solved_names(self) -> set[str]:
        """Return the names of the parameters with solved elements."""
        return {name for name, _ in self._solves}

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
            if e[0] not in self._modes
            and not self._pdescs[e[0]].valid_range().contains(value)]

    def _range_text(self, name: str) -> str:
        """Return the valid range of a parameter, as text."""
        return str(self._pdescs[name].valid_range())

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
        self._template = np.array([
            self._fixed[e] if e in self._fixed
            else self._solves[e].default() if e in self._solves
            else np.nan
            for e in self._elements])
        self._solved = [e for e in self._elements if e in self._solves]
        self._solved_flat = {
            element_name(*e): self._flat[e] for e in self._solved}
        self._free_elements = [e for e in self._elements if e in self._free]
        self._free_names = [element_name(*e) for e in self._free_elements]
        self._free_flat = np.array(
            [self._flat[e] for e in self._free_elements], dtype=int)
        self._ranged = [
            name for name, pdesc in self._pdescs.items()
            if pdesc.valid_range().bounded()]

    def _free_values(
            self, free: Mapping[str, float] | Sequence[float] | None
    ) -> list[float]:
        """
        Return the values of the free elements, in order (see evaluate).
        """
        if free is None:
            if missing := [
                    n for n, e in zip(self._free_names, self._free_elements)
                    if self._free[e].value() is None]:
                raise ValueError(
                    f"these free parameters have no 'value' to evaluate "
                    f"them at: {missing}")
            return [self._free[e].value() for e in self._free_elements]
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
            outside = np.logical_not(pdesc.valid_range().contains(value))
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


param_space_parser = BasicParser(ParamSpace)


def _name_problems(
        pdescs: Mapping[str, ParamDesc],
        constants: Mapping[str, Any],
        function_names: Collection[str]
) -> list[str]:
    """Return the problems of the names of parameters and constants."""
    errors = []
    for what, names in [('parameter', pdescs), ('constant', constants)]:
        for name in names:
            if reason := check_name(name):
                errors.append(
                    f"the {what} {name!r} has an invalid name: {reason}")
    if conflicts := sorted(set(constants) & set(pdescs)):
        errors.append(
            f"constants cannot have the names of parameters: {conflicts}")
    if conflicts := sorted(
            set(function_names) & (set(constants) | set(pdescs))):
        errors.append(
            f"the functions of the models have the names of parameters or "
            f"constants: {conflicts}")
    return errors


def _differ_in_value_only(elements: Sequence[Element]) -> bool:
    """
    Check whether elements are free elements that differ only in their
    starting values.
    """
    if not all(isinstance(e, FreeElement) for e in elements):
        return False
    common = [
        {k: v for k, v in e.dump().items() if k != 'value'}
        for e in elements]
    return all(c == common[0] for c in common)
