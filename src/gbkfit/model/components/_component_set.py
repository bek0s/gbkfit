"""
The components of a galaxy: their parameters (ComponentSet), and their
evaluation on the grid of an observation (ComponentSetPlan).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ..distance import Distance
from ..mass import MassContext, MassModel
from ._detail import shared_params
from .base import Component, ComponentPlan, DepthAxis, NativeGrid
from .disks._disk import DiskPlan
from .disks._profiles import DiskDensity, pressure_support
from .geometries import Geometry


__all__ = [
    'IMAGE_SPECTRAL_AXIS',
    'ComponentSet',
    'ComponentSetPlan'
]


# The spectral axis of an image: one channel, without a width
IMAGE_SPECTRAL_AXIS = gridutils.Grid(
    (1,), gridutils.Coords((0,), (0,), (0,), 0), 0)


def _with_mass_params(
        params: dict[str, ParamDesc],
        components: Sequence[Component],
        distance: Distance | None,
        mass_model: MassModel | None
) -> dict[str, ParamDesc]:
    """
    Return the parameters of the components of a galaxy, and those of its
    distance and its mass (if any). Raise ConfigError if components take
    the circular velocity of the mass (see
    Component.circular_velocity_params) and there is none, if there is
    mass and no component does, if the mass has no distance or the
    distance no mass (its only user), if a mass component follows a
    component that is not a disk of the galaxy (see
    Component.density_disk), or if the names of the parameters repeat.
    """
    moved = any(cmp.circular_velocity_params() for cmp in components)
    if moved and mass_model is None:
        raise ConfigError(
            "the 'mass' velocity traits need the mass of their galaxy")
    if mass_model is not None and not moved:
        raise ConfigError(
            "no component uses the mass of its galaxy; give the components "
            "that it moves 'mass' velocity traits")
    if mass_model is not None and distance is None:
        raise ConfigError(
            "the mass of a galaxy needs its distance (e.g. distance: "
            "{type: direct})")
    if distance is not None and mass_model is None:
        raise ConfigError(
            "the distance of a galaxy is used by its mass only; give the "
            "galaxy mass, or no distance")
    if mass_model is None:
        return params
    disks = [cmp.name() for cmp in components
             if cmp.density_disk() is not None]
    for mass_component in mass_model.components():
        for name in mass_component.followed_components():
            if name not in disks:
                raise ConfigError(
                    f"the mass component {mass_component.name()!r} follows "
                    f"the component {name!r}, which is not a disk of its "
                    f"galaxy; its disks: {disks}")
    gravity = distance.pdescs() | mass_model.pdescs()
    if repeated := sorted(params.keys() & gravity.keys()):
        raise ConfigError(
            f"the parameters of the distance and the mass have the names "
            f"of parameters of the components: {repeated}; rename the "
            f"components")
    return params | gravity


def _component_prefixes(components: Sequence[Component]) -> list[str]:
    """
    Return the prefix of the parameters, constants and extra outputs of
    each component of a galaxy: its name and '_', or none for one
    component without a name. Raise ConfigError if several components do
    not all have names (see parseutils.item_prefixes).
    """
    names = [cmp.name() for cmp in components]
    if len(names) > 1 and None in names:
        unnamed = [i for i, name in enumerate(names) if name is None]
        raise ConfigError(
            f"the components of a galaxy need names when there are "
            f"several; the components {unnamed} have none")
    return parseutils.item_prefixes(names, 'components', 'cmp', False)


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
    prefixed (see _component_prefixes), except those they share through
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


class ComponentSet:
    """
    The components of a galaxy, its distance and its mass: their
    parameters, and their evaluation on the grids of observations (see
    ComponentSetPlan).

    Raises
    ------
    ConfigError
        If there are no components, several without names, or the names
        of the components, their geometries and their parameters
        conflict, or the mass and the components do not go together (see
        _with_mass_params).
    """

    def __init__(
            self,
            components: Component | Sequence[Component],
            distance: Distance | None = None,
            mass_model: MassModel | None = None
    ):
        components = iterutils.tuplify(components)
        if not components:
            raise ConfigError("a galaxy needs at least one component")
        self._components = components
        self._prefixes = _component_prefixes(components)
        params, self._mappings = _merge_params(components, self._prefixes)
        self._params = _with_mass_params(
            params, components, distance, mass_model)
        self._distance = distance
        self._mass_model = mass_model

    def components(self) -> tuple[Component, ...]:
        return self._components

    def geometries(self) -> list[Geometry]:
        """Return the geometries of the components, each once."""
        return _geometries(self._components)

    def distance(self) -> Distance | None:
        return self._distance

    def mass_model(self) -> MassModel | None:
        return self._mass_model

    def mappings(self) -> tuple[dict[str, str], ...]:
        """Return the parameter names of each component, by its own names."""
        return self._mappings

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._params

    def constants(self) -> dict[str, Any]:
        return _constants(self._components, self._prefixes)

    def plan(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            depth: DepthAxis | None,
            dtype: np.dtype,
            emitters: Sequence[int],
            lines: tuple[str, ...] | None
    ) -> 'ComponentSetPlan':
        """
        Plan the evaluation of the components on a driver and dtype (see
        ComponentSetPlan).
        """
        return ComponentSetPlan(
            self, driver, grid, spectral, depth, dtype, emitters, lines)


class ComponentSetPlan:
    """
    The evaluation of the components of a galaxy on the grid of an
    observation: the x and y axes of its data, and a depth axis along the
    line of sight (or none: the plane of the sky), with the spectral axis
    of its data (or that of an image). In two stages: the optical depth of
    the components with opacity in each voxel (all of them: they absorb
    whatever the observation sees), then the light of the given emitters,
    dimmed by it. Each has its parameters, and those its galaxy gives it
    (the circular velocity of its mass, evaluated once for all of them).

    Parameters
    ----------
    component_set : ComponentSet
        The components.
    driver : Driver
        The driver of the evaluation.
    grid : Grid
        The grid of the x and y axes.
    spectral : Grid
        The spectral axis (one axis).
    depth : DepthAxis, optional
        The depth axis, if any.
    dtype : np.dtype
        The floating type of the evaluation.
    emitters : Sequence of int
        The indices of the components whose light is evaluated.
    lines : tuple of str, optional
        The names of the emission lines evaluated; all if None.

    Raises
    ------
    ValueError
        If there is no depth axis and components have opacity (the
        observations check that first, see LineOfSight).
    """

    def __init__(
            self,
            component_set: ComponentSet,
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            depth: DepthAxis | None,
            dtype: np.dtype,
            emitters: Sequence[int],
            lines: tuple[str, ...] | None
    ):
        components = component_set.components()
        absorbers = [
            i for i, cmp in enumerate(components) if cmp.opacity() is not None]
        if absorbers and depth is None:
            raise ValueError(
                "components with opacity need a grid with a depth axis")
        self._component_set = component_set
        self._driver = driver
        self._grid = grid
        self._dtype = dtype
        self._depth = depth
        self._native_grid = _native_grid(grid, spectral, depth)
        # The plans of the emitters, and of the absorbers that do not emit
        # here
        plans = {
            i: components[i].plan(driver, spectral, dtype, lines, True)
            for i in emitters}
        plans |= {
            i: components[i].plan(driver, spectral, dtype, lines, False)
            for i in absorbers if i not in plans}
        self._emitters = tuple(emitters)
        self._absorbers = tuple(absorbers)
        self._plans = plans
        self._followed = _followed_densities(
            component_set, plans, driver, dtype)
        self._odata = None
        if absorbers:
            self._odata = driver.mem_alloc_d(
                self._native_grid['spat_size'][::-1], dtype)
            self._gmodel = driver.native_class('GModel', dtype)()

    def evaluate(
            self,
            params: ParamValues,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Add the light of the emitters to the output array, the 'image' or
        the 'scube' in outputs. The extra outputs of each component are
        named after its name, or its position (e.g. 'disk_bdata' or
        'component0_bdata'), and the totals 'total_*'.
        """
        driver = self._driver
        dtype = self._dtype
        components = self._component_set.components()
        mappings = self._component_set.mappings()
        shape = self._native_grid['spat_size'][::-1]
        odata = self._odata
        bdata = None
        obdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(shape, dtype)
            driver.mem_fill(bdata, 0)
            if self._depth is not None:
                obdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(obdata, 0)

        evaluated = sorted(set(self._absorbers) | set(self._emitters))
        given = _circular_velocities(
            [components[i] for i in evaluated],
            [self._plans[i].disk_plan() for i in evaluated],
            [mappings[i] for i in evaluated], self._followed,
            self._component_set.distance(), self._component_set.mass_model(),
            params)
        given = dict(zip(evaluated, given))

        def component_params(i: int) -> ParamValues:
            mapping = mappings[i]
            return {p: params[mapping[p]] for p in components[i].pdescs()} \
                | given[i]

        def component_extra() -> dict[str, Any] | None:
            return {} if out_extra is not None else None

        # The absorbers add to the opacity cube, so clear it. Then it gets
        # the optical depth from the viewer to the far side of each voxel,
        # which the emitters read (its extra output keeps that of each
        # voxel).
        totals = {}
        if odata is not None:
            driver.mem_fill(odata, 0)
            for i in self._absorbers:
                extra = component_extra()
                self._plans[i].evaluate_opacity(
                    component_params(i), self._native_grid, odata, extra)
                self._add_extra(out_extra, i, extra)
            if out_extra is not None:
                totals['odata'] = driver.mem_copy_d2h(odata)
            self._gmodel.opacity_accumulate(
                spat_size=self._native_grid['spat_size'], opacity=odata)

        outputs = outputs | dict(bdata=bdata, odata=odata, obdata=obdata)
        for i in self._emitters:
            extra = component_extra()
            self._plans[i].evaluate_emission(
                component_params(i), self._native_grid, outputs, extra)
            self._add_extra(out_extra, i, extra)

        if out_extra is not None:
            for name, total in dict(bdata=bdata, obdata=obdata).items():
                if total is not None:
                    totals[name] = driver.mem_copy_d2h(total)
            for name, total in totals.items():
                out_extra[f'total_{name}'] = self._extra(total)

    def _add_extra(
            self,
            out_extra: dict[str, Any] | None,
            index: int,
            extra: dict[str, Any] | None
    ) -> None:
        """
        Add the extra outputs of the component of the given index to
        out_extra, named after its name, or its position if it has none.
        """
        if out_extra is None:
            return
        name = self._component_set.components()[index].name()
        label = name if name is not None else f'component{index}'
        for key, value in extra.items():
            out_extra[f'{label}_{key}'] = self._extra(value)

    def _extra(self, data: np.ndarray) -> gridutils.GridData:
        """
        Return an extra output on the grid: an image without a depth axis,
        a cube with one (along the line of sight, from z = 0).
        """
        grid = self._grid
        if self._depth is None:
            return gridutils.GridData(
                data.reshape(data.shape[-2:]), grid.coords, None)
        depth = self._depth
        coords = gridutils.Coords(
            tuple(grid.coords.step) + (depth.step,),
            tuple(grid.coords.rpix) + (-depth.zero / depth.step,),
            tuple(grid.coords.rval) + (0.0,), grid.coords.rota)
        return gridutils.GridData(data, coords, None)


def _native_grid(
        grid: gridutils.Grid,
        spectral: gridutils.Grid,
        depth: DepthAxis | None
) -> NativeGrid:
    """
    Return the grid of the native evaluation functions: the x and y axes
    of the grid, the depth axis (size 1 and step 0 without one) and the
    spectral axis.
    """
    if depth is None:
        depth = DepthAxis(1, 0, 0)
    return NativeGrid(
        spat_size=tuple(grid.size[:2]) + (depth.size,),
        spat_step=tuple(grid.coords.step[:2]) + (depth.step,),
        spat_zero=tuple(grid.zero()[:2]) + (depth.zero,),
        spat_rota=grid.coords.rota,
        spec_size=spectral.size[0],
        spec_step=spectral.coords.step[0],
        spec_zero=spectral.zero()[0])


def _followed_densities(
        component_set: ComponentSet,
        plans: dict[int, ComponentPlan],
        driver: Driver,
        dtype: np.dtype
) -> tuple[tuple[Component, DiskPlan, dict[str, str]], ...]:
    """
    Return the components of a galaxy that the components of its mass
    follow (see MassComponent.followed_components), with the plans of
    their disks and the names of their parameters: those of the given
    plans, or new ones for the other components (their mass counts,
    though they are not evaluated).
    """
    mass_model = component_set.mass_model()
    if mass_model is None:
        return ()
    followed = {name for mass_component in mass_model.components()
                for name in mass_component.followed_components()}
    result = []
    for i, (cmp, mapping) in enumerate(zip(
            component_set.components(), component_set.mappings())):
        if cmp.name() not in followed:
            continue
        disk_plan = plans[i].disk_plan() if i in plans \
            else cmp.density_disk().plan(driver, 0, dtype)
        result.append((cmp, disk_plan, mapping))
    return tuple(result)


def _circular_velocities(
        components: Sequence[Component],
        disk_plans: Sequence[DiskPlan | None],
        mappings: Sequence[dict[str, str]],
        followed: Sequence[tuple[Component, DiskPlan, dict[str, str]]],
        distance: Distance | None,
        mass_model: MassModel | None,
        params: ParamValues
) -> list[dict[str, np.ndarray]]:
    """
    Return the values of the parameters of each component that are the
    circular velocity of the mass of the galaxy (see
    Component.circular_velocity_params), from the parameters of the
    distance and the mass in params, and the densities of the components
    it follows (see _followed_densities and DiskDensity), with the
    pressure support of their component where its traits ask for it (see
    Disk.pressure_supported_params): NaN where their square is negative or
    infinite (there is no circular orbit). The mass is evaluated once, at
    all their radii.
    """
    radii = [cmp.circular_velocity_params() for cmp in components]
    if mass_model is None or not any(radii):
        return [{} for _ in components]
    kpc_per_arcsec = distance.kpc_per_arcsec(params)
    densities = {
        cmp.name(): DiskDensity(
            cmp.density_disk(), disk_plan,
            {p: params[mapping[p]] for p in cmp.pdescs()}, kpc_per_arcsec)
        for cmp, disk_plan, mapping in followed}
    context = MassContext(
        distance.critical_density(params), distance.omega_m(params),
        densities)
    every = [r for cmp in radii for r in cmp.values()]
    vcirc2 = mass_model.vcirc2(
        np.concatenate(every) * kpc_per_arcsec, params, context)
    ends = np.cumsum([len(r) for r in every])
    pieces = iter(vcirc2[end - len(r):end] for r, end in zip(every, ends))
    values = []
    for cmp, disk_plan, mapping, cmp_radii in zip(
            components, disk_plans, mappings, radii, strict=True):
        values.append({})
        for name, r in cmp_radii.items():
            v2 = next(pieces)
            disk = cmp.density_disk()
            if name in disk.pressure_supported_params():
                v2 = v2 + pressure_support(
                    disk, disk_plan,
                    {p: params[mapping[p]] for p in cmp.pdescs()}, r)
            values[-1][name] = np.sqrt(
                np.where(np.isfinite(v2) & (v2 >= 0), v2, np.nan))
    return values
