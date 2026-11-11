"""
The components of a galaxy: their parameters (ComponentSet), and their
evaluation on the grid of an observation (ComponentSetPlan).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.config import item_prefixes
from gbkfit.cosmology import Cosmology
from gbkfit.driver import DeviceArray, Driver
from gbkfit.grid import Coords, Grid
from gbkfit.params import ParamDesc, ParamValues, pdescs_with_units
from gbkfit.utils import iterutils
from ._component_plans import ComponentPlans
from ._functions import GalaxyFunctions
from ._transfer import LineResolution
from .components import _units
from .components._needs import Need, check_densities, check_needs
from .components._stage import GalaxyStage
from .components._units import DataUnit
from .components.base import Component, ComponentState, DepthAxis, OpacityCubes
from .components.disks.disk import Disk, ThickDisk
from .components.lines import Band
from .distance import Distance
from .extras import Extras
from .gravity import Capability, Gravity, GravityNewton
from .mass import (
    ComponentDensity, DiskDensity, MassComponent, MassDisk, MassModel,
    SpheroidDensity, mass_component_parser)


__all__ = [
    'IMAGE_SPECTRAL_AXIS',
    'ComponentSet',
    'ComponentSetPlan'
]


# The spectral axis of an image: one channel, without a width
IMAGE_SPECTRAL_AXIS = Grid(
    (1,), Coords((0,), (0,), (0,), 0), 0)


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
    Raise ValueError if the gravity is given without mass (its only
    user), a component needs what the galaxy does not give (see
    Component.needs: e.g. the mass, or the distance, see
    _distance_needs), a mass
    component follows a component whose density it cannot follow (see
    Component.density_kind), the gravity does not take the mass model (see
    Gravity.check_mass_model) or does not give the shares of a dark
    fraction, or the names of the parameters repeat.
    """
    if gravity is not None and mass_model is None:
        raise ValueError(
            "the gravity of a galaxy is that of its mass; give the galaxy "
            "mass, or no gravity")
    gravity = GravityNewton() if gravity is None else gravity
    check_needs(
        [need for cmp in components for need in cmp.needs()], mass_model,
        gravity)
    if mass_model is None:
        # (a distance without mass gives the length of an arcsec to the
        # derived values, e.g. radii in kpc of a disk of an arctan curve)
        return params if distance is None \
            else _with_params(params, distance.pdescs())
    check_needs(
        _distance_needs(components, mass_model), mass_model, gravity,
        distance is not None)
    kinds = {cmp.name(): cmp.density_kind() for cmp in components
             if cmp.density_kind() is not None}
    for mass_component in mass_model.components():
        for name in mass_component.followed_components():
            if kinds.get(name) is not mass_component.followed_kind():
                raise ValueError(_follow_error(mass_component, name, kinds))
    # (of density height traits, as the densities that mass disks follow:
    # see DiskDensity.thick)
    thick = [cmp.name() for cmp in components if isinstance(cmp, ThickDisk)]
    gravity.check_mass_model(mass_model, thick)
    _check_fluids(components, mass_model, thick)
    check_densities(
        [need for cmp in components for need in cmp.needs()], mass_model,
        thick)
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
    own = {} if distance is None else distance.pdescs()
    return _with_params(params, own | mass | gravity.pdescs())


def _check_fluids(
        components: Sequence[Component],
        mass_model: MassModel,
        thick: Sequence[str]
) -> None:
    """
    Check the fluids and the vertical balance of the disks of a galaxy of
    a mass model (see Disk): the mass disks that their fluids name, or the
    mass disks that follow each disk, whose surface density its
    equilibrium 'sheet' takes; that no disk balances its rotation (the
    equilibrium 'rotation') next to a mass model, where the vertical
    frequency of a disk is 2.5-3 times its angular speed; and that no mass
    disk follows the layer of a disk of the equilibrium 'galaxy' of its
    thickness (the mass would take the thickness that its own force sets:
    not yet). The galaxy has thick components of the given names.

    Raises
    ------
    ValueError
        If one of these does not hold.
    """
    layers = {
        mass.vertical().followed_component()
        for mass in mass_model.components() if isinstance(mass, MassDisk)
    } - {None}
    for cmp in components:
        if not isinstance(cmp, Disk):
            continue
        label = repr(cmp.name()) if cmp.name() is not None else ''
        equilibrium = cmp.equilibrium() if cmp.equilibrium() is not None \
            else cmp.height_equilibrium()
        gravity = equilibrium.gravity() if equilibrium is not None else None
        if gravity == 'rotation':
            raise ValueError(
                f"the vertical equilibrium 'rotation' of the component "
                f"{label} takes the pull of a spherical mass of its "
                f"rotation, which is not that of a galaxy of mass (a disk "
                f"pulls 2.5-3 times harder where it dominates): give it the "
                f"equilibrium 'galaxy' or 'sheet'")
        if gravity == 'galaxy' and cmp.height_equilibrium() is not None \
                and cmp.name() in layers:
            raise ValueError(
                f"a mass disk follows the layer of the component {label}, "
                f"whose thickness is its equilibrium in the force of the "
                f"mass: a mass that takes the thickness that its own force "
                f"sets is not supported yet; give the mass disk a vertical "
                f"structure of its own, or the disk the equilibrium 'sheet'")
        fluid = mass_model.fluid(cmp.name(), cmp.fluid())
        if gravity != 'sheet' or cmp.fluid():
            continue
        what = f"the vertical equilibrium 'sheet' of the component {label}"
        if not fluid:
            raise ValueError(
                f"{what} takes the surface density of its fluid: give it a "
                f"fluid, or a mass disk that follows it")
        check_densities(
            [Need(Capability.MASS, f"{what} takes the surface density",
                  fluid, True)], mass_model, thick)


def _distance_needs(
        components: Sequence[Component],
        mass_model: MassModel
) -> list[Need]:
    """
    Return what needs the distance of a galaxy of the components and the
    mass model (see Capability.DISTANCE): the components that take its
    mass (see Component.uses_mass), whose radii in arcsec meet its kpc,
    and the mass components that need it (see
    MassComponent.needs_distance).
    """
    needs = [
        Need(Capability.DISTANCE, (
            f"the component {cmp.name()!r} moves in the mass of its "
            f"galaxy, and so takes the length of an arcsec in it (its "
            f"distance)") if cmp.name() is not None else (
            "the component moves in the mass of its galaxy, and so takes "
            "the length of an arcsec in it (its distance)"))
        for cmp in components if cmp.uses_mass()]
    return needs + [
        Need(Capability.DISTANCE, (
            f"the mass component {mass_component.name()!r} {why}, and so "
            f"takes the length of an arcsec in its galaxy (its distance)"))
        for mass_component in mass_model.components()
        if (why := mass_component.needs_distance()) is not None]


def _with_params(
        params: dict[str, ParamDesc], taken: dict[str, ParamDesc]
) -> dict[str, ParamDesc]:
    """
    Return the parameters of the components of a galaxy with those of its
    distance, its mass and its gravity (taken), or raise ValueError if
    their names repeat.
    """
    if repeated := sorted(params.keys() & taken.keys()):
        raise ValueError(
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
    component without a name. Raise ValueError if several components do
    not all have names (see item_prefixes).
    """
    names = [cmp.name() for cmp in components]
    if len(names) > 1 and None in names:
        unnamed = [i for i, name in enumerate(names) if name is None]
        raise ValueError(
            f"the components of a galaxy need names when there are "
            f"several; the components {unnamed} have none")
    return item_prefixes(names, 'components', 'cmp', False)


# Every component has its own geometric parameters (e.g. disk_posa): the
# galaxy shares none of them between components. Components that share
# an orientation tie their parameters in the parameter space instead
# (e.g. stars_posa: disk_posa), which does the same with one mechanism.
# Shared geometries (2026-10-25 to 2026-11-04) did only that, by merging
# names here, at the cost of a geometry option on every component, named
# geometries, and rings that moved to the geometry when it was warped. A
# better form of them, if one is found, would again only map names here.
def _merge_params(
        components: Sequence[Component], prefixes: Sequence[str]
) -> tuple[dict[str, ParamDesc], tuple[dict[str, str], ...]]:
    """
    Return the parameters of the components, prefixed (see
    _component_prefixes), and for each component the names of its
    parameters in them (see iterutils.merge_with_prefixes). Raise
    ValueError if the names repeat.
    """
    params, mappings = iterutils.merge_with_prefixes(
        [cmp.pdescs() for cmp in components], list(prefixes))
    return params, mappings


class ComponentSet:
    """
    The components of a galaxy, its distance, its mass and its gravity
    (Newtonian if not given): their parameters, and their evaluation on
    the grids of observations (see ComponentSetPlan).

    A galaxy of mass alone has no components.

    Raises
    ------
    ValueError
        If there are no components and no mass, several components
        without names, or the names of the components and their
        parameters conflict, the mass and the components do not go
        together (see _with_mass_params), or components give one line
        different units (see _units.check_line_units_agree).
    """

    def __init__(
            self,
            components: Component | Sequence[Component],
            distance: Distance | None = None,
            mass_model: MassModel | None = None,
            gravity: Gravity | None = None
    ):
        components = iterutils.tuplify(components)
        if not components and mass_model is None:
            raise ValueError("a galaxy needs at least one component, or mass")
        _units.check_line_units_agree(components)
        self._components = components
        self._prefixes = _component_prefixes(components)
        params, self._mappings = _merge_params(components, self._prefixes)
        self._params = _with_mass_params(
            params, components, distance, mass_model, gravity)
        self._distance = distance
        self._mass_model = mass_model
        self._gravity = GravityNewton() if gravity is None else gravity

    def components(self) -> tuple[Component, ...]:
        return self._components

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
        constants, _ = iterutils.merge_with_prefixes(
            [cmp.constants() for cmp in self._components],
            list(self._prefixes))
        return constants

    def plan_stage(
            self,
            driver: Driver,
            redshift: float,
            sky_origin: tuple[float, float] | None = None,
            cosmology: Cosmology | None = None
    ) -> GalaxyStage:
        """
        Plan the stage of a galaxy of the given redshift on a driver, in
        the frame of the given origin on the sky and in the cosmology (see
        GalaxyStage).
        """
        return GalaxyStage(self, driver, redshift, sky_origin, cosmology)

    def plan_functions(
            self,
            driver: Driver,
            redshift: float,
            sky_origin: tuple[float, float] | None = None,
            cosmology: Cosmology | None = None
    ) -> GalaxyFunctions:
        """
        Plan the functions of a galaxy of the given redshift on a driver,
        in the frame of the given origin on the sky and in the cosmology,
        for the derived values of its parameters (see GalaxyFunctions).
        """
        return GalaxyFunctions(
            self, driver, redshift, sky_origin, cosmology)

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
            grid: Grid,
            spectral: Grid,
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
            grid: Grid,
            spectral: Grid,
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
