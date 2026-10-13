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
from ..mass import MassModel
from ._component_plans import ComponentPlans
from ._detail import shared_params
from ._transfer import TransferPlan
from .base import Component, DepthAxis, OpacityCubes
from .geometries import Geometry
from .lines import Band


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
            cubes: OpacityCubes,
            lines: tuple[str, ...] | None,
            band: Band | None
    ) -> 'ComponentSetPlan':
        """
        Plan the evaluation of the components on a driver and dtype (see
        ComponentSetPlan).
        """
        return ComponentSetPlan(
            self, driver, grid, spectral, depth, dtype, emitters, cubes,
            lines, band)

    def plan_transfer(
            self,
            driver: Driver,
            grid: gridutils.Grid,
            spectral: gridutils.Grid,
            depth: DepthAxis,
            dtype: np.dtype,
            emitters: Sequence[int],
            cubes: OpacityCubes,
            lines: tuple[str, ...] | None,
            band: Band | None,
            redshift: float,
            slab: int | None
    ) -> TransferPlan:
        """
        Plan the evaluation of the components on a driver and dtype by
        radiative transfer along each line of sight, in slabs of the given
        number of planes (see TransferPlan).
        """
        return TransferPlan(
            self, driver, grid, spectral, depth, dtype, emitters, cubes,
            lines, band, redshift, slab)


class ComponentSetPlan:
    """
    The evaluation of the components of a galaxy on the grid of an
    observation (see ComponentPlans), in two stages: the optical depth of
    the absorbers in each voxel, then the light of the emitters, dimmed by
    the optical depth from the viewer to each voxel.

    The parameters are those of ComponentPlans.
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
            cubes: OpacityCubes,
            lines: tuple[str, ...] | None,
            band: Band | None
    ):
        plans = ComponentPlans(
            component_set, driver, grid, spectral, depth, dtype, emitters,
            cubes, lines, band)
        self._plans = plans
        self._depth = depth
        # The cubes of opacity
        self._odata = None
        if plans.ncubes:
            self._odata = driver.mem_alloc_d(
                (plans.ncubes,) + plans.native_grid['spat_size'][::-1],
                dtype)
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
        plans = self._plans
        driver = plans.driver
        dtype = plans.dtype
        grid = plans.native_grid
        shape = grid['spat_size'][::-1]
        odata = self._odata
        extras = {} if out_extra is not None else None
        bdata = None
        obdata = None
        if out_extra is not None:
            bdata = driver.mem_alloc_d(shape, dtype)
            driver.mem_fill(bdata, 0)
            if self._depth is not None:
                obdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(obdata, 0)
        values = plans.params(params)

        # The absorbers write the optical depth of each voxel, which then
        # becomes the optical depth from the viewer to the far side of each
        # voxel, which the emitters read
        if odata is not None:
            plans.evaluate_opacity(values, grid, odata, extras)
            for cube in odata:
                self._gmodel.opacity_accumulate(
                    spat_size=grid['spat_size'], opacity=cube)

        outputs = outputs | dict(bdata=bdata, odata=odata, obdata=obdata)
        for i in plans.emitters:
            extra = {} if extras is not None else None
            plans.plans[i].evaluate_emission(values[i], grid, outputs, extra)
            if extras is not None:
                extras.setdefault(i, {}).update(extra)

        if out_extra is not None:
            plans.add_extras(out_extra, extras, {
                name: driver.mem_copy_d2h(total)
                for name, total in dict(bdata=bdata, obdata=obdata).items()
                if total is not None})
