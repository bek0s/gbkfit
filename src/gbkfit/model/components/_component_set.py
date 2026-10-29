"""
The components of a galaxy: their parameters (ComponentSet), and their
evaluation on the grid of an observation (ComponentSetPlan).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc, ParamValues, pdescs_with_units
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ..distance import Distance
from ..extras import Extras
from ..gravity import Capability, Gravity, GravityNewton
from ..mass import (
    ComponentDensity, DiskDensity, MassComponent, MassModel, SpheroidDensity,
    mass_component_parser)
from . import _units
from ._component_plans import ComponentPlans
from ._detail import geometry_desc, shared_params
from ._functions import GalaxyFunctions
from ._needs import Need, check_needs
from ._stage import GalaxyStage
from ._transfer import LineResolution
from ._units import DataUnit
from .base import Component, ComponentState, DepthAxis, OpacityCubes
from .disks.disk import Disk
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
        mass_model: MassModel | None,
        gravity: Gravity | None
) -> dict[str, ParamDesc]:
    """
    Return the parameters of the components of a galaxy, and those of its
    distance, its mass and its gravity (if any; Newtonian by default), with
    the mass per unit of the density of a component (see
    MassModel.mass_ratios) in the unit that the emission of the component
    gives it (see _units.mass_ratio_unit).
    Raise ConfigError if the distance or the gravity is given without mass
    (their only user), the mass has no distance, a component needs what
    the galaxy does not give (see Component.needs: e.g. the mass), a mass
    component follows a component whose density it cannot follow (see
    Component.density_kind), the gravity does not take the mass model (see
    Gravity.check_mass_model) or does not give the shares of a dark
    fraction, or the names of the parameters repeat.
    """
    if distance is not None and mass_model is None:
        raise ConfigError(
            "the distance of a galaxy is used by its mass only; give the "
            "galaxy mass, or no distance")
    if gravity is not None and mass_model is None:
        raise ConfigError(
            "the gravity of a galaxy is that of its mass; give the galaxy "
            "mass, or no gravity")
    gravity = GravityNewton() if gravity is None else gravity
    check_needs(
        [need for cmp in components for need in cmp.needs()], mass_model,
        gravity)
    if mass_model is None:
        return params
    if distance is None:
        raise ConfigError(
            "the mass of a galaxy needs its distance (e.g. distance: "
            "{type: direct})")
    kinds = {cmp.name(): cmp.density_kind() for cmp in components
             if cmp.density_kind() is not None}
    for mass_component in mass_model.components():
        for name in mass_component.followed_components():
            if kinds.get(name) is not mass_component.followed_kind():
                raise ConfigError(_follow_error(mass_component, name, kinds))
    # (of density height traits, as the densities that mass disks follow:
    # see DiskDensity.thick)
    thick = [cmp.name() for cmp in components
             if isinstance(cmp, Disk) and cmp.density_disk().traits('rht')]
    gravity.check_mass_model(mass_model, thick)
    check_needs([
        Need(Capability.SHARES,
             f"the mass component {mass_component.name()!r} takes the "
             f"shares of the components in the squared circular velocity "
             f"(its dark fraction)")
        for mass_component in mass_model.components()
        if mass_component.takes_shares()], mass_model, gravity)
    emissions = {cmp.name(): cmp.emission() for cmp in components}
    mass = pdescs_with_units(mass_model.pdescs(), {
        name: _units.mass_ratio_unit(emissions[followed])
        for name, followed in mass_model.mass_ratios().items()})
    taken = distance.pdescs() | mass | gravity.pdescs()
    if repeated := sorted(params.keys() & taken.keys()):
        raise ConfigError(
            f"the parameters of the distance, the mass and the gravity have "
            f"the names of parameters of the components: {repeated}; rename "
            f"the components")
    return params | taken


# The kinds of the densities that the mass can follow, in words
_KINDS = {DiskDensity: 'a disk', SpheroidDensity: 'a spheroid'}


def _follow_error(
        mass_component: MassComponent,
        name: str,
        kinds: dict[str, type[ComponentDensity]]
) -> str:
    """
    Return the message of the error of a mass component that follows the
    component of the given name, which is not one that it can follow,
    with the kinds of the densities of the components that the mass can
    follow, by their names.
    """
    wanted = _KINDS[mass_component.followed_kind()]
    head = (f"the mass component {mass_component.name()!r} follows the "
            f"density of {wanted}")
    if name not in kinds:
        return (f"{head}, but its galaxy has no component {name!r} whose "
                f"density the mass can follow; those it can: {list(kinds)}")
    types = [cls.type() for cls in mass_component_parser.registered_classes()
             .values() if cls.followed_kind() is kinds[name]]
    return (f"{head}, but {name!r} is {_KINDS[kinds[name]]}: the mass that "
            f"follows it is of the type {' or '.join(map(repr, types))}")


def _component_prefixes(components: Sequence[Component]) -> list[str]:
    """
    Return the prefix of the parameters and the constants of each
    component of a galaxy: its name and '_', or none for one
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


def _shared_geometries(
        components: Sequence[Component], geometry: Geometry
) -> list[Geometry | None]:
    """
    Return the geometry that each component shares (see
    Component.geometry): that of the galaxy (geometry), a named one, or
    None for its own. Raise ConfigError if the geometry of the galaxy has
    a name, or is warped and a component takes it as 'galaxy' (a warped
    geometry is given to the components themselves, whose rings it
    makes), or a component holds a geometry without a name that is not
    that of the galaxy.
    """
    if geometry.name() is not None:
        raise ConfigError(
            f"the geometry of a galaxy has no name; it is "
            f"{geometry.name()!r}")
    shared = []
    for cmp in components:
        choice = cmp.geometry()
        if choice == 'own':
            shared.append(None)
        elif choice == 'galaxy':
            if geometry.loose() or geometry.tilted():
                raise ConfigError(
                    f"the geometry of the galaxy is warped: give it to its "
                    f"component {cmp.name()!r} itself (or geometry 'own')")
            shared.append(geometry)
        elif choice.name() is None and choice != geometry:
            raise ConfigError(
                f"the component {cmp.name()!r} has a geometry without a "
                f"name that is not that of its galaxy; name it")
        else:
            shared.append(choice)
    return shared


def _geometries(shared: Sequence[Geometry | None]) -> list[Geometry]:
    """
    Return the geometries that components share (see _shared_geometries),
    each once, in the order of the components. Raise ConfigError if two
    differ but have one name.
    """
    geometries = list(dict.fromkeys(g for g in shared if g is not None))
    names = [g.name() for g in geometries]
    if repeated := sorted({n for n in names if names.count(n) > 1}):
        raise ConfigError(
            f"the components have different geometries of the names "
            f"{repeated}")
    return geometries


def _merge_params(
        components: Sequence[Component],
        prefixes: Sequence[str],
        shared: Sequence[Geometry | None]
) -> tuple[dict[str, ParamDesc], tuple[dict[str, str], ...]]:
    """
    Return the parameters of the components, and for each component the
    names of its parameters in them (see iterutils.merge_with_prefixes):
    prefixed (see _component_prefixes), except those they share through
    their geometries (see _shared_geometries), which are named after their
    geometry (e.g. disk_posa; plainly, e.g. posa, for that of the galaxy),
    and come first. Raise ConfigError if a geometry has the name of a
    component, those names are taken, or the components have a shared
    parameter of different sizes.
    """
    geometries = _geometries(shared)
    if repeated := sorted(
            ({g.name() for g in geometries} - {None})
            & {c.name() for c in components}):
        raise ConfigError(
            f"the geometries and the components must have different names; "
            f"repeated: {repeated}")
    names_of = {
        geometry: shared_params(geometry, [
            cmp for cmp, g in zip(components, shared) if g == geometry])
        for geometry in geometries}
    names = [() if g is None else names_of[g] for g in shared]
    params, mappings = iterutils.merge_with_prefixes(
        [{name: pdesc for name, pdesc in cmp.pdescs().items()
          if name not in cmp_names}
         for cmp, cmp_names in zip(components, names)], prefixes)
    mappings = [dict(mapping) for mapping in mappings]
    geometry_params = {}
    for cmp, geometry, cmp_names, mapping in zip(
            components, shared, names, mappings):
        pdescs = cmp.pdescs()
        for name in cmp_names:
            if name not in pdescs:
                continue
            full_name = name if geometry.name() is None \
                else f'{geometry.name()}_{name}'
            pdesc = geometry_params.setdefault(full_name, pdescs[name])
            if pdesc.size() != pdescs[name].size():
                raise ConfigError(
                    f"the components of {geometry_desc(geometry)} have "
                    f"{name} of different sizes")
            mapping[name] = full_name
    if taken := sorted(geometry_params.keys() & params.keys()):
        raise ConfigError(
            f"the parameters of the geometries have the names of "
            f"parameters of the components: {taken}; rename the geometries")
    return geometry_params | params, tuple(mappings)


def _constants(
        components: Sequence[Component],
        prefixes: Sequence[str],
        shared: Sequence[Geometry | None]
) -> dict[str, Any]:
    """
    Return the constants of the components (see Component.constants),
    prefixed, and the radial nodes of the warped geometries they share
    (e.g. disk_rnodes; rnodes for that of the galaxy).
    """
    constants, _ = iterutils.merge_with_prefixes(
        [cmp.constants() for cmp in components], prefixes)
    return constants | {
        'rnodes' if geometry.name() is None else f'{geometry.name()}_rnodes':
            geometry.rnodes()
        for geometry in _geometries(shared)
        if geometry.rnodes() is not None}


class ComponentSet:
    """
    The components of a galaxy, its distance, its mass and its gravity
    (Newtonian if not given): their parameters, and their evaluation on
    the grids of observations (see ComponentSetPlan).

    A galaxy of mass alone has no components.

    Raises
    ------
    ConfigError
        If there are no components and no mass, several components
        without names, or the names
        of the components, their geometries and their parameters
        conflict, the mass and the components do not go together (see
        _with_mass_params), or components give one line different units
        (see _units.check_line_units_agree).
    """

    def __init__(
            self,
            components: Component | Sequence[Component],
            distance: Distance | None = None,
            mass_model: MassModel | None = None,
            gravity: Gravity | None = None,
            geometry: Geometry | None = None
    ):
        components = iterutils.tuplify(components)
        if not components and mass_model is None:
            raise ConfigError("a galaxy needs at least one component, or mass")
        _units.check_line_units_agree(components)
        self._components = components
        self._geometry = Geometry() if geometry is None else geometry
        self._shared = _shared_geometries(components, self._geometry)
        self._prefixes = _component_prefixes(components)
        params, self._mappings = _merge_params(
            components, self._prefixes, self._shared)
        self._params = _with_mass_params(
            params, components, distance, mass_model, gravity)
        self._distance = distance
        self._mass_model = mass_model
        self._gravity = GravityNewton() if gravity is None else gravity

    def components(self) -> tuple[Component, ...]:
        return self._components

    def geometry(self) -> Geometry:
        """Return the geometry of the galaxy."""
        return self._geometry

    def geometries(self) -> list[Geometry]:
        """
        Return the geometries that the components share, each once (with
        that of the galaxy, if they share it).
        """
        return _geometries(self._shared)

    def distance(self) -> Distance | None:
        return self._distance

    def mass_model(self) -> MassModel | None:
        return self._mass_model

    def components_use_mass(self) -> bool:
        """
        Check whether a component uses the mass (see Component.uses_mass):
        e.g. it is moved or held up by it.
        """
        return any(cmp.uses_mass() for cmp in self._components)

    def gravity(self) -> Gravity:
        """Return the gravity of its mass (Newtonian if not given)."""
        return self._gravity

    def mappings(self) -> tuple[dict[str, str], ...]:
        """Return the parameter names of each component, by its own names."""
        return self._mappings

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._params

    def constants(self) -> dict[str, Any]:
        return _constants(self._components, self._prefixes, self._shared)

    def plan_stage(
            self,
            driver: Driver,
            redshift: float,
            sky_origin: tuple[float, float] | None = None
    ) -> GalaxyStage:
        """
        Plan the stage of a galaxy of the given redshift on a driver, in
        the frame of the given origin on the sky (see GalaxyStage).
        """
        return GalaxyStage(self, driver, redshift, sky_origin)

    def plan_functions(
            self,
            driver: Driver,
            redshift: float,
            sky_origin: tuple[float, float] | None = None
    ) -> GalaxyFunctions:
        """
        Plan the functions of a galaxy of the given redshift on a driver,
        in the frame of the given origin on the sky, for the derived values
        of its parameters (see GalaxyFunctions).
        """
        return GalaxyFunctions(self, driver, redshift, sky_origin)

    def function_names(self) -> tuple[str, ...]:
        """
        Return the names of the functions of a galaxy of the components
        (see plan_functions), without planning them.
        """
        return GalaxyFunctions.names_of(self)

    def function_radius_range(
            self, name: str
    ) -> tuple[float, float] | None:
        """
        Return the range of the radii (arcsec) of the function of a disk of
        the name, without planning it (see GalaxyFunctions.radius_range_of).
        """
        return GalaxyFunctions.radius_range_of(self, name)

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
            band: Band | None,
            data_unit: DataUnit | None,
            redshift: float,
            smoothing: float | None
    ) -> 'ComponentSetPlan':
        """
        Plan the evaluation of the components, of a galaxy of the given
        redshift, on a driver and dtype (see ComponentSetPlan).
        """
        return ComponentSetPlan(
            self, driver, grid, spectral, depth, dtype, emitters, cubes,
            lines, band, data_unit, redshift, smoothing)


class ComponentSetPlan:
    """
    The evaluation of the components of a galaxy on the grid of an
    observation (see ComponentPlans), in two stages: the optical depth of
    the absorbers in each voxel, then the light of the emitters, dimmed by
    the optical depth from the viewer to each voxel.

    The parameters are those of ComponentPlans, and the spectral smoothing
    of the instrument (km/s; see ComponentPlans.light_resolution; None
    for an image).
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
            band: Band | None,
            data_unit: DataUnit | None,
            redshift: float,
            smoothing: float | None
    ):
        plans = ComponentPlans(
            component_set, driver, grid, spectral, depth, dtype, emitters,
            cubes, lines, band, data_unit, redshift)
        self._plans = plans
        self._depth = depth
        self._smoothing = smoothing
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
            states: dict[int, ComponentState],
            outputs: dict[str, DeviceArray | None],
            out_extra: Extras | None
    ) -> LineResolution | None:
        """
        Add the light of the emitters to the output array, the 'image' or
        the 'scube' in outputs, with the parameters of the galaxy and what
        it gives its components, their states (see GalaxyStage.evaluate).
        The extra
        outputs of each component have its name, or its position, as the
        id 'component' (e.g. 'disk' or 'component0'), and their totals
        none (see ComponentPlans.add_extras). With extra outputs, a
        warning says where zmax shapes a disk (see
        ComponentPlans.warn_truncation), and it returns how finely it
        resolves the lines (see LineResolution: their light only); None
        otherwise.
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
        if out_extra is not None:
            plans.warn_truncation(values)

        # The absorbers write the optical depth of each voxel, which then
        # becomes the optical depth from the viewer to the far side of each
        # voxel, which the emitters read
        if odata is not None:
            driver.mem_fill(odata, 0)
            plans.evaluate_opacity(values, states, grid, odata, extras)
            for cube in odata:
                self._gmodel.opacity_accumulate(
                    spat_size=grid['spat_size'], opacity=cube)

        outputs = outputs | dict(bdata=bdata, odata=odata, obdata=obdata)
        for i in plans.emitters:
            extra = {} if extras is not None else None
            plans.plans[i].evaluate_emission(
                values[i], grid, outputs, extra, states.get(i))
            if extras is not None:
                extras.setdefault(i, {}).update(extra)

        if out_extra is None:
            return None
        plans.add_extras(out_extra, extras, {
            name: driver.mem_copy_d2h(total)
            for name, total in dict(bdata=bdata, obdata=obdata).items()
            if total is not None})
        light = 0.0 if self._smoothing is None \
            else plans.light_resolution(values, extras, self._smoothing)
        return LineResolution(0.0, 0.0, light)
