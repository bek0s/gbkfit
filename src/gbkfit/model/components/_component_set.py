"""
The components of a galaxy: their parameters (ComponentSet), and their
evaluation on the grid of an observation (ComponentSetPlan).
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.params import ParamDesc, ParamValues
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ..distance import Distance
from ..extras import Extras
from ..gravity import Gravity, GravityNewton
from ..mass import MassModel
from . import _units
from ._component_plans import ComponentPlans
from ._detail import shared_params
from ._functions import GalaxyFunctions
from ._gravity import GravityPlan
from ._transfer import LineResolution
from ._units import DataUnit
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
        mass_model: MassModel | None,
        gravity: Gravity | None
) -> dict[str, ParamDesc]:
    """
    Return the parameters of the components of a galaxy, and those of its
    distance, its mass and its gravity (if any; Newtonian by default).
    Raise ConfigError if components take
    the circular velocity of the mass (see
    Component.circular_velocity_params) and there is none, if there are
    components and mass but no component uses it (nor a disk in vertical
    equilibrium, see Kinematics), if a disk in a central equilibrium has neither mass
    nor rotation (tangential velocity traits) or one in another
    equilibrium no mass, if the mass has no distance or the distance no
    mass (its only user), if a mass component follows a
    component that is not a disk of the galaxy (see
    Component.density_disk), if the pressure support or the equilibrium
    of a disk takes the density of a mass component that the galaxy does
    not have, if the gravity is given without mass, if it does not give
    what a feature takes (see Gravity: the forces of a jeans equilibrium
    or of the rotation at each height, the self-gravitating sheets of a
    self_gravitating equilibrium or pressure support, the shares of a
    dark fraction), or if the names of the parameters repeat.
    """
    moved = any(cmp.circular_velocity_params() for cmp in components)
    balanced = [
        cmp for cmp in components
        if (disk := cmp.density_disk()) is not None
        and disk.equilibrium() is not None]
    if moved and mass_model is None:
        raise ConfigError(
            "the 'mass' velocity traits need the mass of their galaxy")
    if mass_model is not None and components and not moved \
            and not balanced:
        raise ConfigError(
            "no component uses the mass of its galaxy; give the components "
            "that it moves 'mass' velocity traits, or those it holds up a "
            "vertical equilibrium")
    for cmp in balanced if mass_model is None else ():
        mode = cmp.density_disk().equilibrium().mode
        if mode != 'central':
            raise ConfigError(
                f"the vertical equilibrium {mode!r} of the component "
                f"{cmp.name()!r} needs the mass of its galaxy")
        if not any(trait.direction() == 'tan'
                   for trait in cmp.density_disk().traits('vpt')):
            raise ConfigError(
                f"the vertical equilibrium of the component {cmp.name()!r} "
                f"takes the circular velocity of the mass of its galaxy, or "
                f"without mass, the rotation of its disk; it has neither: "
                f"give it tangential velocity traits, or the galaxy mass")
    if mass_model is not None and distance is None:
        raise ConfigError(
            "the mass of a galaxy needs its distance (e.g. distance: "
            "{type: direct})")
    if distance is not None and mass_model is None:
        raise ConfigError(
            "the distance of a galaxy is used by its mass only; give the "
            "galaxy mass, or no distance")
    if gravity is not None and mass_model is None:
        raise ConfigError(
            "the gravity of a galaxy is that of its mass; give the galaxy "
            "mass, or no gravity")
    if mass_model is None:
        return params
    gravity = GravityNewton() if gravity is None else gravity
    gravity.check_mass_model(mass_model)
    disks = [cmp.name() for cmp in components
             if cmp.density_disk() is not None]
    for mass_component in mass_model.components():
        for name in mass_component.followed_components():
            if name not in disks:
                raise ConfigError(
                    f"the mass component {mass_component.name()!r} follows "
                    f"the component {name!r}, which is not a disk of its "
                    f"galaxy; its disks: {disks}")
    mass_names = [component.name() for component in mass_model.components()]
    for cmp in components:
        disk = cmp.density_disk()
        supports = {} if disk is None else disk.pressure_supported_params()
        for support in supports.values():
            unknown = [
                name for name in support.density or ()
                if name not in mass_names]
            if unknown:
                raise ConfigError(
                    f"the pressure support of the component {cmp.name()!r} "
                    f"takes the density of the mass components {unknown}, "
                    f"which its galaxy does not have; its mass components: "
                    f"{mass_names}")
    for cmp in balanced:
        equilibrium = cmp.density_disk().equilibrium()
        if equilibrium.density is not None \
                and equilibrium.density not in mass_names:
            raise ConfigError(
                f"the vertical equilibrium of the component {cmp.name()!r} "
                f"takes the surface density of the mass component "
                f"{equilibrium.density!r}, which its galaxy does not have; "
                f"its mass components: {mass_names}")
        if equilibrium.mode == 'jeans':
            _check_gravity(
                gravity.check_forces, mass_model,
                what=f"the jeans equilibrium of the component "
                     f"{cmp.name()!r} takes the vertical force of the "
                     f"gravity of its galaxy")
        if equilibrium.mode == 'self_gravitating':
            _check_gravity(
                gravity.check_self_gravitating_sheets,
                what=f"the self_gravitating equilibrium of the component "
                     f"{cmp.name()!r} takes the vertical structure of "
                     f"self-gravitating sheets in Newtonian gravity")
    for cmp in components:
        disk = cmp.density_disk()
        supports = {} if disk is None else disk.pressure_supported_params()
        if any(s.mode == 'self_gravitating' for s in supports.values()):
            _check_gravity(
                gravity.check_self_gravitating_sheets,
                what=f"the self_gravitating pressure support of the "
                     f"component {cmp.name()!r} takes the vertical structure "
                     f"of self-gravitating sheets in Newtonian gravity")
        if disk is not None and disk.rotation_factor_params():
            _check_gravity(
                gravity.check_forces, mass_model,
                what=f"the rotation at each height of the component "
                     f"{cmp.name()!r} takes the radial force of the gravity "
                     f"of its galaxy off its plane")
    for mass_component in mass_model.components():
        if mass_component.takes_shares():
            _check_gravity(
                gravity.check_shares,
                what=f"the mass component {mass_component.name()!r} takes "
                     f"the shares of the components in the squared circular "
                     f"velocity (its dark fraction)")
    taken = distance.pdescs() | mass_model.pdescs() | gravity.pdescs()
    if repeated := sorted(params.keys() & taken.keys()):
        raise ConfigError(
            f"the parameters of the distance, the mass and the gravity have "
            f"the names of parameters of the components: {repeated}; rename "
            f"the components")
    return params | taken


def _check_gravity(
        check: Callable[..., None], *args: Any, what: str
) -> None:
    """
    Call a check of what a gravity gives (see Gravity) with the given
    arguments, and raise ConfigError with what a feature takes (what) if
    it does not give it.
    """
    try:
        check(*args)
    except ConfigError as e:
        raise ConfigError(
            f"{what}, which it does not give: {e.message}") from None


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
            gravity: Gravity | None = None
    ):
        components = iterutils.tuplify(components)
        if not components and mass_model is None:
            raise ConfigError("a galaxy needs at least one component, or mass")
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

    def geometries(self) -> list[Geometry]:
        """Return the geometries of the components, each once."""
        return _geometries(self._components)

    def distance(self) -> Distance | None:
        return self._distance

    def mass_model(self) -> MassModel | None:
        return self._mass_model

    def gravity(self) -> Gravity:
        """Return the gravity of its mass (Newtonian if not given)."""
        return self._gravity

    def mappings(self) -> tuple[dict[str, str], ...]:
        """Return the parameter names of each component, by its own names."""
        return self._mappings

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._params

    def constants(self) -> dict[str, Any]:
        return _constants(self._components, self._prefixes)

    def plan_gravity(self, driver: Driver, redshift: float) -> GravityPlan:
        """
        Plan the evaluation of the gravity on a driver, of a galaxy of the
        given redshift (see GravityPlan).
        """
        return GravityPlan(self, driver, redshift)

    def plan_functions(
            self, driver: Driver, redshift: float
    ) -> GalaxyFunctions:
        """
        Plan the functions of a galaxy of the given redshift on a driver,
        for the derived values of its parameters (see GalaxyFunctions).
        """
        return GalaxyFunctions(self, driver, redshift)

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
            gravity: dict[int, ParamValues],
            outputs: dict[str, DeviceArray | None],
            out_extra: Extras | None
    ) -> LineResolution | None:
        """
        Add the light of the emitters to the output array, the 'image' or
        the 'scube' in outputs, with the parameters of the galaxy and those
        that its gravity gives its components (see
        GravityPlan.evaluate). The extra outputs of each component have
        its name, or its position, as the id 'component' (e.g. 'disk' or
        'component0'), and their totals none (see
        ComponentPlans.add_extras). With extra outputs,
        a warning says where zmax shapes a disk (see
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
        values = plans.params(params, gravity)
        if out_extra is not None:
            plans.warn_truncation(values)

        # The absorbers write the optical depth of each voxel, which then
        # becomes the optical depth from the viewer to the far side of each
        # voxel, which the emitters read
        if odata is not None:
            driver.mem_fill(odata, 0)
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

        if out_extra is None:
            return None
        plans.add_extras(out_extra, extras, {
            name: driver.mem_copy_d2h(total)
            for name, total in dict(bdata=bdata, obdata=obdata).items()
            if total is not None})
        light = 0.0 if self._smoothing is None \
            else plans.light_resolution(values, extras, self._smoothing)
        return LineResolution(0.0, 0.0, light)
