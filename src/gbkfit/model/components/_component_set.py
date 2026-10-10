from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ..base import ModelPlan, Selection
from ..mass import MassModel
from ._detail import shared_params
from .base import Component, ComponentPlan, NativeGrid
from .geometries import Geometry


__all__ = [
    'IMAGE_SPECTRAL_AXIS',
    'ComponentSet2D',
    'ComponentSet3D',
    'ComponentSetPlan2D',
    'ComponentSetPlan3D',
    'ComponentSetModelPlan'
]


# The spectral axis of an image: one channel, without a width
IMAGE_SPECTRAL_AXIS = gridutils.Grid(
    (1,), gridutils.Coords((0,), (0,), (0,), 0), 0)

# The components and the opacity components: their label in messages, the
# prefix of their parameters, and whether the parameters of the first one
# have it (see component_prefixes)
_CMP_PREFIX = ('components', 'cmp', False)
_OCMP_PREFIX = ('opacity components', 'ocmp', True)


def _with_mass_model_params(
        params: dict[str, ParamDesc],
        components: Sequence[Component],
        mass_model: MassModel | None
) -> dict[str, ParamDesc]:
    """
    Return the parameters of the components of a model, and those of its
    mass model (if any). Raise ConfigError if components take the circular
    velocity of a mass model (see Component.circular_velocity_params) and
    there is none, if there is one and none does, or if the names of the
    parameters repeat.
    """
    moved = any(cmp.circular_velocity_params() for cmp in components)
    if moved and mass_model is None:
        raise ConfigError(
            "the 'mass' velocity traits need the mass_model of their model")
    if mass_model is None:
        return params
    if not moved:
        raise ConfigError(
            "no component uses the mass_model of its model; give the "
            "components that it moves 'mass' velocity traits")
    if repeated := sorted(params.keys() & mass_model.pdescs().keys()):
        raise ConfigError(
            f"the parameters of the mass model have the names of parameters "
            f"of the components: {repeated}; rename the components")
    return params | mass_model.pdescs()


def _circular_velocities(
        components: Sequence[Component],
        mass_model: MassModel | None,
        params: dict[str, float | np.ndarray]
) -> list[dict[str, np.ndarray]]:
    """
    Return the values of the parameters of each component that are the
    circular velocity of the mass model (see
    Component.circular_velocity_params), from the parameters of the mass
    model in params.
    """
    return [
        {name: mass_model.vcirc(radii, params)
         for name, radii in cmp.circular_velocity_params().items()}
        for cmp in components]


def component_prefixes(
        components: Sequence[Component],
        label: str,
        prefix: str,
        prefix_first: bool
) -> list[str]:
    """
    Return the prefix of the parameters, constants and extra outputs of
    each component of a list: its name, or its position if the components
    have no names (e.g. 'cmp1_'; see parseutils.item_prefixes).
    """
    return parseutils.item_prefixes(
        [cmp.name() for cmp in components], label, prefix, prefix_first)


def select_components(
        components: Sequence[Component], names: Sequence[str] | None
) -> tuple[int, ...]:
    """
    Return the indices of the components of the given names, in their
    order (all of them if names is None). Raise ConfigError for unknown
    names.
    """
    if names is None:
        return tuple(range(len(components)))
    known = [component.name() for component in components]
    if not names:
        raise ConfigError("at least one component must be selected")
    if len(set(names)) != len(names):
        raise ConfigError(f"the selected components repeat names: {names}")
    if unknown := [name for name in names if name not in known]:
        raise ConfigError(
            f"there are no components named {unknown}; the named components "
            f"are {[name for name in known if name is not None]}")
    return tuple(sorted(known.index(name) for name in names))


def check_selected_lines(
        components: Sequence[Component], names: Sequence[str] | None
) -> None:
    """
    Raise ConfigError unless each of the names of lines (None for all) is
    a line of one of the components.
    """
    if names is None:
        return
    known = set()
    for component in components:
        known |= set(component.line_names())
    if unknown := [name for name in names if name not in known]:
        raise ConfigError(
            f"no component has the lines {unknown}; the lines of the "
            f"components are {sorted(known)}")


def evaluate_components(
        components: Sequence[Component],
        plans: Sequence[ComponentPlan],
        mappings: Sequence[dict[str, str]],
        params: dict[str, float | np.ndarray],
        grid: NativeGrid,
        outputs: dict[str, DeviceArray | None],
        out_extra: dict[str, Any] | None,
        out_extra_label: str,
        extra: Callable[[np.ndarray], Any],
        given: Sequence[dict[str, np.ndarray]] | None = None
) -> None:
    """
    Evaluate the components of a model through their plans, each with its
    parameters, and those its model gives it (given, a dict for each
    component, if any; see Component.circular_velocity_params). Their
    extra outputs are named after the given label and their name, or
    their index if they have none (e.g. 'opacity_dust_odata' or
    'opacity_component0_odata'), and made from their data by extra (e.g.
    with the coordinates of the grid).
    """
    if given is None:
        given = [{}] * len(components)
    for i, (component, plan, mapping, component_given) in enumerate(
            zip(components, plans, mappings, given, strict=True)):
        name = component.name()
        label = name if name is not None else f'component{i}'
        prefix = f'{out_extra_label}{label}_'
        component_params = {
            p: params[mapping[p]] for p in component.pdescs()
        } | component_given
        component_out_extra = {} if out_extra is not None else None
        plan.evaluate(component_params, grid, outputs, component_out_extra)
        if component_out_extra is not None:
            for k, v in component_out_extra.items():
                out_extra[f'{prefix}{k}'] = extra(v)


def _geometries(components: Sequence[Component]) -> list[Geometry]:
    """
    Return the geometries of the components, each once, in the order of
    the components. Raise ConfigError if two differ but have one name, or
    one has the name of a component.
    """
    geometries = list(dict.fromkeys(
        cmp.geometry() for cmp in components
        if cmp.geometry() is not None))
    names = [geometry.name() for geometry in geometries]
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        raise ConfigError(
            f"the components have different geometries of the names "
            f"{repeated}")
    if repeated := sorted(set(names) & {cmp.name() for cmp in components}):
        raise ConfigError(
            f"the geometries and the components must have different names; "
            f"repeated: {repeated}")
    return geometries


def _merge_params(
        components: Sequence[Component], prefixes: Sequence[str]
) -> tuple[dict[str, ParamDesc], tuple[dict[str, str], ...]]:
    """
    Return the parameters of the components, and for each component the
    names of its parameters in them (see iterutils.merge_with_prefixes):
    prefixed (see component_prefixes), except those they share through
    their geometries (see Geometry), which are named after their geometry
    (e.g. disk_posa), and come first. Raise ConfigError if those names are
    taken, or the components have a shared parameter of different sizes.
    """
    shared = {
        geometry: shared_params(geometry, [
            cmp for cmp in components if cmp.geometry() == geometry])
        for geometry in _geometries(components)}
    names = [
        shared[cmp.geometry()] if cmp.geometry() is not None else ()
        for cmp in components]
    params, mappings = iterutils.merge_with_prefixes(
        [{name: pdesc for name, pdesc in cmp.pdescs().items()
          if name not in cmp_names}
         for cmp, cmp_names in zip(components, names)], prefixes)
    mappings = [dict(mapping) for mapping in mappings]
    geometry_params = {}
    for cmp, cmp_names, mapping in zip(components, names, mappings):
        pdescs = cmp.pdescs()
        for name in cmp_names:
            if name not in pdescs:
                continue
            full_name = f'{cmp.geometry().name()}_{name}'
            pdesc = geometry_params.setdefault(full_name, pdescs[name])
            if pdesc.size() != pdescs[name].size():
                raise ConfigError(
                    f"the components of the geometry "
                    f"{cmp.geometry().name()!r} have {name} of different "
                    f"sizes")
            mapping[name] = full_name
    if taken := sorted(geometry_params.keys() & params.keys()):
        raise ConfigError(
            f"the parameters of the geometries have the names of "
            f"parameters of the components: {taken}; rename the geometries")
    return geometry_params | params, tuple(mappings)


def _constants(
        components: Sequence[Component], prefixes: Sequence[str]
) -> dict[str, Any]:
    """
    Return the constants of the components (see Component.constants),
    prefixed, and the radial nodes of the warped geometries (e.g.
    disk_rnodes).
    """
    constants, _ = iterutils.merge_with_prefixes(
        [cmp.constants() for cmp in components], prefixes)
    return constants | {
        f'{geometry.name()}_rnodes': geometry.rnodes()
        for geometry in _geometries(components)
        if geometry.rnodes() is not None}


class ComponentSet2D:
    """
    The components of a 2d model: their parameters, and their evaluation
    on a 2d spatial grid, the x and y axes of the data. The components get
    a 3d grid with a z axis of size 1.
    """

    def __init__(
            self,
            components: Component | Sequence[Component],
            mass_model: MassModel | None = None
    ):
        if not components:
            raise ConfigError("at least one component must be configured")
        self._components = iterutils.tuplify(components)
        self._prefixes = component_prefixes(
            self._components, *_CMP_PREFIX)
        params, self._mappings = _merge_params(
            self._components, self._prefixes)
        self._params = _with_mass_model_params(
            params, self._components, mass_model)
        self._mass_model = mass_model

    def components(self) -> tuple[Component, ...]:
        return self._components

    def geometries(self) -> list[Geometry]:
        """Return the geometries of the components, each once."""
        return _geometries(self._components)

    def mass_model(self) -> MassModel | None:
        return self._mass_model

    def mappings(self) -> tuple[dict[str, str], ...]:
        """Return the parameter names of each component, by its own names."""
        return self._mappings

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._params

    def has_weights(self) -> bool:
        return any(cmp.has_weights() for cmp in self._components)

    def constants(self) -> dict[str, Any]:
        return _constants(self._components, self._prefixes)

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection
    ) -> 'ComponentSetPlan2D':
        """
        Plan the evaluation of the components and lines of the selection
        (see Selection) on the given driver, grid of the x and y axes and
        spectral axis (gridutils.Grid, the second of one axis) and dtype,
        with spatial weights if has_weights.
        """
        return ComponentSetPlan2D(
            self, driver, grid, spectral, has_weights, dtype, selection)


class ComponentSetPlan2D:
    """
    The evaluation of a ComponentSet2D: the plans of its components and
    the spatial weights, on a grid.
    """

    def __init__(
            self,
            component_set: 'ComponentSet2D',
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection
    ):
        self._component_set = component_set
        self._driver = driver
        self._grid = grid
        self._dtype = dtype
        size = tuple(grid.size[:2])
        spec_size = spectral.size[0]
        spec_step = spectral.coords.step[0]
        spec_zero = spectral.zero()[0]
        self._native_grid = dict(
            spat_size=size + (1,),
            spat_step=tuple(grid.coords.step[:2]) + (0,),
            spat_zero=tuple(grid.zero()[:2]) + (0,),
            spat_rota=grid.coords.rota,
            spec_size=spec_size,
            spec_step=spec_step,
            spec_zero=spec_zero)
        # The selected components, their plans and their parameters
        selected = select_components(
            component_set.components(), selection.components)
        self._components = tuple(
            component_set.components()[i] for i in selected)
        self._mappings = tuple(component_set.mappings()[i] for i in selected)
        check_selected_lines(self._components, selection.lines)
        self._component_plans = tuple(
            cmp.plan(driver, spectral, dtype, selection.lines)
            for cmp in self._components)
        # The spatial weights, if weighting is requested
        self._wdata = None
        if has_weights:
            self._wdata = driver.mem_alloc_d(size[::-1], dtype)
        self._backend = driver.native_class('GModel', dtype)()

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            outputs: dict[str, DeviceArray | None],
            weights: DeviceArray | None,
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Add the components to the output array, the 'image' or the
        'scube' in outputs, and weight the data weights (if any) with the
        spatial weights.
        """
        driver = self._driver
        grid = self._grid
        dtype = self._dtype
        size = tuple(grid.size[:2])

        wdata = self._wdata
        bdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(size[::-1], dtype)
            driver.mem_fill(bdata, 0)

        # The extra outputs are images on the grid; those of the
        # components are on a grid one voxel thick
        def image(data):
            return gridutils.GridData(data, grid.coords, None)

        evaluate_components(
            self._components, self._component_plans, self._mappings,
            params, self._native_grid,
            outputs | dict(wdata=wdata, bdata=bdata),
            out_extra, '', lambda data: image(data[0]),
            _circular_velocities(
                self._components, self._component_set.mass_model(), params))

        # Weight the data with the spatial weights evaluated above
        if weights is not None:
            self._backend.wcube_evaluate(wdata[None], weights)

        if out_extra is not None:
            if wdata is not None:
                out_extra['total_wdata'] = image(driver.mem_copy_d2h(wdata))
            out_extra['total_bdata'] = image(driver.mem_copy_d2h(bdata))


class ComponentSet3D:
    """
    The components and opacity components of a 3d model: their
    parameters, and their evaluation on a 3d spatial grid, the x and y
    axes of the data and a z axis that can be configured. The opacity
    components make an opacity cube, which absorbs the brightness of the
    components.
    """

    def __init__(
            self,
            components: Component | Sequence[Component],
            opacity_components: Component | Sequence[Component] | None = None,
            size_z: int | None = None,
            step_z: float | None = None,
            zero_z: float | None = None,
            mass_model: MassModel | None = None
    ):
        if not components:
            raise ConfigError("at least one component must be configured")
        self._components = iterutils.tuplify(components)
        self._ocomponents = iterutils.tuplify(opacity_components)
        # The components and the opacity components share their names
        repeated = sorted(
            {cmp.name() for cmp in self._components}
            & {cmp.name() for cmp in self._ocomponents} - {None})
        if repeated:
            raise ConfigError(
                f"the components and the opacity components must have "
                f"different names; repeated: {repeated}")
        self._prefixes = component_prefixes(
            self._components, *_CMP_PREFIX)
        self._oprefixes = component_prefixes(
            self._ocomponents, *_OCMP_PREFIX)
        params, mappings = _merge_params(
            self._all_components(), self._prefixes + self._oprefixes)
        self._mappings = mappings[:len(self._components)]
        self._omappings = mappings[len(self._components):]
        self._params = _with_mass_model_params(
            params, self._components, mass_model)
        self._mass_model = mass_model
        # The z axis, as configured (or None to pick it per grid)
        self._size_z = size_z
        self._step_z = step_z
        self._zero_z = zero_z

    def components(self) -> tuple[Component, ...]:
        return self._components

    def opacity_components(self) -> tuple[Component, ...]:
        return self._ocomponents

    def geometries(self) -> list[Geometry]:
        """Return the geometries of the components, each once."""
        return _geometries(self._all_components())

    def mass_model(self) -> MassModel | None:
        return self._mass_model

    def size_z(self) -> int | None:
        return self._size_z

    def step_z(self) -> float | None:
        return self._step_z

    def zero_z(self) -> float | None:
        return self._zero_z

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._params

    def has_weights(self) -> bool:
        return any(cmp.has_weights() for cmp in self._components)

    def constants(self) -> dict[str, Any]:
        return _constants(
            self._all_components(), self._prefixes + self._oprefixes)

    def mappings(self) -> tuple[dict[str, str], ...]:
        """Return the parameter names of each component, by its own names."""
        return self._mappings

    def omappings(self) -> tuple[dict[str, str], ...]:
        """Return the parameter names of each opacity component."""
        return self._omappings

    def _all_components(self) -> tuple[Component, ...]:
        return self._components + self._ocomponents

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection
    ) -> 'ComponentSetPlan3D':
        """
        Plan the evaluation of the components and lines of the selection
        (see Selection) on the given driver, grid of the x and y axes and
        spectral axis (gridutils.Grid, the second of one axis) and dtype,
        with spatial weights if has_weights.
        """
        return ComponentSetPlan3D(
            self, driver, grid, spectral, has_weights, dtype, selection)


class ComponentSetPlan3D:
    """
    The evaluation of a ComponentSet3D: the plans of its components and
    opacity components, the z axis, the spatial weights and the opacity
    cube, on a grid.
    """

    def __init__(
            self,
            component_set: 'ComponentSet3D',
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            has_weights: bool,
            dtype: np.dtype,
            selection: Selection
    ):
        self._component_set = component_set
        self._driver = driver
        self._grid = grid
        self._dtype = dtype
        # A z axis that encloses the whole galaxy is hard to calculate.
        # Instead, if it is not configured, we pick the size and step of
        # the longest of the x and y axes, and place zero in the middle.
        size = grid.size
        step = grid.coords.step
        zero = grid.zero()
        longest = int(size[0] < size[1])
        size_z = component_set.size_z()
        step_z = component_set.step_z()
        zero_z = component_set.zero_z()
        size_z = size_z if size_z is not None else size[longest]
        step_z = step_z if step_z is not None else step[longest]
        zero_z = zero_z if zero_z is not None \
            else -(size_z / 2 - 0.5) * step_z
        self._size = tuple(size[:2]) + (size_z,)
        self._step = tuple(step[:2]) + (step_z,)
        self._zero = tuple(zero[:2]) + (zero_z,)
        spec_size = spectral.size[0]
        spec_step = spectral.coords.step[0]
        spec_zero = spectral.zero()[0]
        self._native_grid = dict(
            spat_size=self._size,
            spat_step=self._step,
            spat_zero=self._zero,
            spat_rota=grid.coords.rota,
            spec_size=spec_size,
            spec_step=spec_step,
            spec_zero=spec_zero)
        # The extra outputs are cubes on the grid: the x and y axes of the
        # data, and the z axis along the line of sight (from z = 0)
        self._coords = gridutils.Coords(
            self._step,
            grid.coords.rpix + (-self._zero[2] / self._step[2],),
            grid.coords.rval + (0.0,), grid.coords.rota)
        # The selected components, their plans and their parameters
        selected = select_components(
            component_set.components(), selection.components)
        self._components = tuple(
            component_set.components()[i] for i in selected)
        self._mappings = tuple(component_set.mappings()[i] for i in selected)
        check_selected_lines(self._components, selection.lines)
        self._component_plans = tuple(
            cmp.plan(driver, spectral, dtype, selection.lines)
            for cmp in self._components)
        self._ocomponent_plans = tuple(
            cmp.plan(driver, spectral, dtype, None)
            for cmp in component_set.opacity_components())
        # The spatial weights, if weighting is requested, and the opacity,
        # if there are opacity components
        self._wdata = None
        self._odata = None
        if has_weights:
            self._wdata = driver.mem_alloc_d(self._size[::-1], dtype)
        if component_set.opacity_components():
            self._odata = driver.mem_alloc_d(self._size[::-1], dtype)
        self._backend = driver.native_class('GModel', dtype)()

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            outputs: dict[str, DeviceArray | None],
            weights: DeviceArray | None,
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Add the components to the output array, the 'image' or the
        'scube' in outputs, and weight the data weights (if any) with the
        spatial weights.
        """
        driver = self._driver
        dtype = self._dtype
        component_set = self._component_set

        def cube(data):
            return gridutils.GridData(data, self._coords, None)

        wdata = self._wdata
        odata = self._odata
        bdata = None
        obdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(self._size[::-1], dtype)
            driver.mem_fill(bdata, 0)
            obdata = driver.mem_alloc_d(self._size[::-1], dtype)
            driver.mem_fill(obdata, 0)

        # The opacity components add to the opacity cube, so clear it
        if odata is not None:
            driver.mem_fill(odata, 0)
            evaluate_components(
                component_set.opacity_components(), self._ocomponent_plans,
                component_set.omappings(), params, self._native_grid,
                dict(odata=odata), out_extra, 'opacity_', cube)

        outputs = outputs | dict(
            wdata=wdata, bdata=bdata, odata=odata, obdata=obdata)
        evaluate_components(
            self._components, self._component_plans, self._mappings,
            params, self._native_grid, outputs, out_extra, '', cube,
            _circular_velocities(
                self._components, component_set.mass_model(), params))

        # Weight the data with the spatial weights evaluated above
        if weights is not None:
            self._backend.wcube_evaluate(wdata, weights)

        if out_extra is not None:
            totals = dict(wdata=wdata, odata=odata, bdata=bdata, obdata=obdata)
            for name, total in totals.items():
                if total is not None:
                    out_extra[f'total_{name}'] = cube(
                        driver.mem_copy_d2h(total))


class ComponentSetModelPlan(ModelPlan):
    """
    The evaluation of a model made of a component set: the plan of the
    set, which adds to the output of the given key ('image' or 'scube').
    """

    def __init__(
            self,
            component_set_plan: 'ComponentSetPlan2D | ComponentSetPlan3D',
            key: str
    ):
        self._component_set_plan = component_set_plan
        self._key = key

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            data: DeviceArray,
            weights: DeviceArray | None,
            out_extra: dict[str, Any] | None
    ) -> None:
        self._component_set_plan.evaluate(
            params, {self._key: data}, weights, out_extra)
