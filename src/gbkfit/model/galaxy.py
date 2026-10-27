"""
Galaxies: their components and their mass, at one distance.
"""

from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.driver import Driver
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .components import _detail
from .components._component_set import ComponentSet, ComponentSetPlan
from .components._functions import GalaxyFunctions
from .components._stage import GalaxyStage
from .components._transfer import TransferMember
from .components._units import DataUnit
from .components.base import Component, DepthAxis, OpacityCubes
from .components.cones import Cone
from .components.disks import Disk
from .components.geometries import Geometry, geometry_parser
from .components.lines import Band, LineCatalogue
from .components.particles import Particles
from .components.points import Point
from .components.roles import BandEmission
from .components.spheroids import Spheroid
from .distance import Distance, distance_parser
from .gravity import Gravity, gravity_parser
from .mass import MassComponent, MassModel, mass_component_parser


__all__ = [
    'Galaxy',
    'component_parser',
    'galaxy_parser'
]


component_parser = parseutils.TypedParser(Component, [
    Disk,
    Cone,
    Spheroid,
    Point,
    Particles])


class Galaxy(parseutils.Serializable):
    """
    A galaxy: its components, and its mass at a distance. Its components
    are what observations see: each emits light (its emission), removes
    light (its opacity) or both. Its mass pulls: its components with the
    velocity trait 'mass' rotate in its gravity (see
    traits.VPTraitMass), which takes its disks as concentric and flat
    (see GalaxyStage).

    Its components share its geometry by default: one centre, systemic
    velocity and plane (see Geometry), whose parameters are its own
    ('posa'); a component can have its own instead (its option geometry
    'own') or share a named one. Its parameters are those of its
    geometries ('posa', '{geometry}_posa'), of its components, each
    prefixed by its name when there are several ('disk_pt_a'), its
    distance ('distance'), its mass ('mass_{name}_*') and its gravity
    ('gravity_*'). A name prefixes them all when there are several
    galaxies (see ObservationGroup), whose order along the line of sight
    the ObservationGroup gives (see its option depth_order).

    In configurations, its option geometries lists the named geometries
    that its components name, and its option geometry is its own (e.g.
    geometry: {tilted: true, rnodes: [...]}, which its components share
    with their option geometry 'galaxy', the default).

    Parameters
    ----------
    components : Component or Sequence of Component, optional
        Its components; they need names when there are several. A galaxy
        of mass alone has none: observations on the sky see nothing of
        it, and its mass is for what is measured of it (e.g. its rotation
        curve; see ObservationProfile).
    mass : MassComponent or Sequence of MassComponent, optional
        Its mass: the components of its mass (disks, spheroids, point
        masses, ...; a disk can follow the density of one of its disks).
    distance : Distance, optional
        Its distance, which converts arcsec to kpc for its mass (required
        with mass, and used by it only).
    gravity : Gravity, optional
        How its mass pulls (with mass only): Newtonian by default (see
        GravityNewton).
    redshift : float, optional
        Its one redshift: for what it emits and absorbs in its rest frame
        (its named lines are at their rest wavelengths times 1 + redshift,
        see Lines.factors; the dust of an image absorbs at the effective
        wavelength of its band over 1 + redshift), and for its distance by
        redshift (see DistanceRedshift); the systemic velocity of its
        components is the velocity offset from it (a relativistic radial
        velocity, the same on every spectral axis; see LinePlacement).
        With physical units, its light is dimmed by it (and by the
        systemic velocity of each component; see _units.line_factor). 0 by
        default; required (0 for none) when a component emits in bands
        with a unit and has no systemic velocity, as nothing else then
        tells how far its light is dimmed.
    name : str, optional
        Its name; required when there are several galaxies.
    geometry : Geometry, optional
        Its geometry, without a name: by default flat (not warped), of all
        the geometric parameters of the components that share it (see
        Component.geometry). The components that share a warped one hold
        it themselves (their rings are its rings).

    Raises
    ------
    ConfigError
        If there are no components and no mass, several components without
        names, the names or parameters conflict, the mass, the distance
        and the components do not go together (e.g. 'mass' velocity traits
        without mass, or a mass disk that follows no disk of the galaxy),
        or the redshift is not above -1, or missing where it is required.
    """

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> Self:
        """
        Load a galaxy from its configuration.

        Parameters
        ----------
        info : dict
            The configuration.
        catalogue : LineCatalogue, optional
            The catalogue of the lines that the emission of its components
            names (see Emission.load); that of GBKFIT by default.

        Returns
        -------
        Galaxy
            The galaxy.

        Raises
        ------
        ConfigError
            If the configuration has an error.
        """
        if info.get('geometry') is not None:
            info['geometry'] = parseutils.load_option(
                geometry_parser.load, info, 'geometry')
        _detail.load_geometries(
            info, ('components',), info.get('geometry') or Geometry())
        components = info.get('components')
        parseutils.load_option_and_update_info(
            component_parser, info, 'components',
            catalogue=[catalogue] * len(components)
            if iterutils.is_sequence(components) else catalogue)
        parseutils.load_option_and_update_info(
            mass_component_parser, info, 'mass')
        parseutils.load_option_and_update_info(
            distance_parser, info, 'distance')
        parseutils.load_option_and_update_info(
            gravity_parser, info, 'gravity')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info = dict(name=self._name) if self._name is not None else {}
        if self.geometry() != Geometry():
            info.update(geometry=geometry_parser.dump(self.geometry()))
        geometries = [g for g in self.geometries() if g.name() is not None]
        if geometries:
            info.update(geometries=geometry_parser.dump(geometries))
        if self.components():
            info.update(
                components=component_parser.dump(list(self.components())))
        if self._mass:
            info.update(mass=mass_component_parser.dump(list(self._mass)))
        if self.distance() is not None:
            info.update(distance=distance_parser.dump(self.distance()))
        if self._gravity is not None:
            info.update(gravity=gravity_parser.dump(self._gravity))
        if self._redshift is not None:
            info.update(redshift=self._redshift)
        return info

    def __init__(
            self,
            components: Component | Sequence[Component] = (),
            mass: MassComponent | Sequence[MassComponent] | None = None,
            distance: Distance | None = None,
            redshift: float | None = None,
            name: str | None = None,
            gravity: Gravity | None = None,
            geometry: Geometry | None = None
    ):
        parseutils.check_name(name)
        if redshift is not None and not redshift > -1:
            raise ConfigError(
                f"the redshift must be above -1; it is {redshift}")
        self._redshift = redshift
        mass = iterutils.tuplify(mass)
        mass_model = MassModel(mass) if mass else None
        self._component_set = ComponentSet(
            components, distance, mass_model, gravity, geometry)
        if distance is not None:
            distance.check_redshift(redshift)
        if redshift is None:
            _check_redshift_not_needed(self._component_set.components())
        self._pdescs = self._component_set.pdescs()
        self._mass = mass
        self._gravity = gravity
        self._name = name

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def components(self) -> tuple[Component, ...]:
        """Return its components."""
        return self._component_set.components()

    def component_params(self, index: int) -> dict[str, str]:
        """
        Return the names of the parameters of its component of the given
        index among its parameters, by the names of the component.
        """
        return self._component_set.mappings()[index]

    def geometry(self) -> Geometry:
        """Return its geometry, which its components share by default."""
        return self._component_set.geometry()

    def geometries(self) -> list[Geometry]:
        """
        Return the geometries that its components share, each once (with
        its own, if they share it).
        """
        return self._component_set.geometries()

    def light_amplitudes(self) -> dict[str, str | None]:
        """
        Return the parameters of its components whose elements are
        amplitudes of their light (see Component.light_amplitudes), by
        their names in the galaxy, each with None if its models are linear
        in it, or why not: also when a mass follows the light of their
        component (the light then moves its gravity).
        """
        followed = set()
        if (mass_model := self.mass_model()) is not None:
            for mass_component in mass_model.components():
                followed.update(mass_component.followed_components())
        amplitudes = {}
        for i, component in enumerate(self.components()):
            names = self.component_params(i)
            for param, reason in component.light_amplitudes().items():
                if reason is None and component.name() in followed:
                    reason = (f"a mass follows the light of the component "
                              f"{component.name()!r}")
                amplitudes[names[param]] = reason
        return amplitudes

    def mass(self) -> tuple[MassComponent, ...]:
        """Return the components of its mass (none without mass)."""
        return self._mass

    def mass_model(self) -> MassModel | None:
        """Return its mass, as a mass model, if any."""
        return self._component_set.mass_model()

    def components_use_mass(self) -> bool:
        """
        Check whether its components use its mass: moved by it ('mass'
        velocity traits) or held up by it (a vertical equilibrium).
        Observations of profiles of its functions and derived values can
        use it too (see ObservationProfile and ParamSpace).
        """
        return self._component_set.components_use_mass()

    def distance(self) -> Distance | None:
        """Return its distance, if given."""
        return self._component_set.distance()

    def gravity(self) -> Gravity:
        """Return the gravity of its mass (Newtonian if not given)."""
        return self._component_set.gravity()

    def redshift(self) -> float:
        """Return its redshift (0 if not given)."""
        return 0.0 if self._redshift is None else self._redshift

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name (see the class)."""
        return self._pdescs

    def constants(self) -> dict[str, Any]:
        """
        Return the values that parameter expressions can use (e.g. the
        radial nodes of its disks, prefixed as their parameters).
        """
        return self._component_set.constants()

    def plan_stage(
            self,
            sky_origin: tuple[float, float] | None = None,
            driver: Driver | None = None
    ) -> GalaxyStage:
        """
        Plan the stage of the galaxy on a driver, in float64, evaluated
        once for all the observations of it: its mass, and what it gives
        its components (e.g. the circular velocity of its mass at the radii
        of the components it moves; see GalaxyStage), which its plans on
        the grids of observations take.

        Parameters
        ----------
        sky_origin : tuple of float, optional
            The RA and Dec (degrees) of the origin of the frame of its
            parameters (see ObservationGroup), which its image traits need
            (see traits.RPTraitImage).
        driver : Driver, optional
            The driver where its components sample their profiles (the
            rest of the stage runs on the host); the host by default.

        Returns
        -------
        GalaxyStage
            The plan.

        Raises
        ------
        ConfigError
            If it has image traits, but there is no sky origin.
        """
        return self._component_set.plan_stage(
            _or_host(driver), self.redshift(), sky_origin)

    def plan_functions(
            self,
            sky_origin: tuple[float, float] | None = None,
            driver: Driver | None = None
    ) -> GalaxyFunctions:
        """
        Plan the functions of the galaxy on a driver, in float64, for
        the derived values of its parameters (see ParamSpace): the
        circular velocity of its mass, and the rotation and dispersion of
        its disks, at any radius (see GalaxyFunctions). They have their
        own stage and plans, apart from those of its evaluations.

        Parameters
        ----------
        sky_origin : tuple of float, optional
            The RA and Dec (degrees) of the origin of the frame of its
            parameters, which its image traits need (see plan_stage).
        driver : Driver, optional
            The driver where its components sample their profiles (see
            plan_stage); the host by default.

        Returns
        -------
        GalaxyFunctions
            The functions.

        Raises
        ------
        ConfigError
            If it has image traits, but there is no sky origin.
        """
        return self._component_set.plan_functions(
            _or_host(driver), self.redshift(), sky_origin)

    def function_names(self) -> tuple[str, ...]:
        """
        Return the names of the functions of the galaxy (see
        plan_functions), without planning them.
        """
        return self._component_set.function_names()

    def function_radius_range(
            self, name: str
    ) -> tuple[float, float] | None:
        """
        Return the range of the radii (arcsec) of the function of a disk of
        the name (e.g. rotation_gas), from its first radial node to its
        last, without planning it; None for its other functions, of any
        radius.
        """
        return self._component_set.function_radius_range(name)

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
            smoothing: float | None
    ) -> ComponentSetPlan:
        """
        Plan the evaluation of the galaxy on the grid of an observation.

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        grid : Grid
            The grid of the x and y axes.
        spectral : Grid
            The spectral axis of the outputs (one axis).
        depth : DepthAxis, optional
            The axis along the line of sight; none for a galaxy evaluated
            on the plane of the sky.
        dtype : np.dtype
            The floating type of the evaluation.
        emitters : Sequence of int
            The indices of the components whose light is evaluated (the
            opacity of all those with opacity is).
        cubes : OpacityCubes
            How the cubes of opacity of the observation hold its dust, and
            the dust that dims its light (see opacity_cubes).
        lines : tuple of str, optional
            The names of the emission lines evaluated; all if None.
        band : Band, optional
            The band of an image, if known.
        data_unit : DataUnit, optional
            The unit of the data of the observation, which the light of
            the components takes (see Component.plan); None without data.
        smoothing : float, optional
            The dispersion (km/s) by which the instrument smooths the
            spectra of the model, which the check of the resolution of
            its light takes (see LineResolution); None for an image.

        Returns
        -------
        ComponentSetPlan
            The plan.
        """
        return self._component_set.plan(
            driver, grid, spectral, depth, dtype, emitters, cubes, lines,
            band, data_unit, self.redshift(), smoothing)

    def transfer_member(
            self,
            emitters: Sequence[int],
            cubes: OpacityCubes,
            depth: DepthAxis
    ) -> TransferMember:
        """
        Return the galaxy as a march of radiative transfer evaluates it on
        the grid of an observation (see TransferPlan): the given emitters,
        cubes and its own depth axis are those of plan.
        """
        return TransferMember(
            self._component_set, tuple(emitters), cubes, self.redshift(),
            depth)


galaxy_parser = parseutils.BasicParser(Galaxy)


def _or_host(driver: Driver | None) -> Driver:
    """Return the driver, or the host driver if None."""
    if driver is not None:
        return driver
    # (imported here: the host driver needs its native module)
    from gbkfit.driver.drivers.host import DriverHost
    return DriverHost()


def _check_redshift_not_needed(components: Sequence[Component]) -> None:
    """
    Raise ConfigError if a component of a galaxy without a redshift emits
    in bands with a unit and has no systemic velocity: the redshift by
    which images see its light dimmed is then not known (see
    _units.BandFactors).
    """
    for i, component in enumerate(components):
        emission = component.emission()
        if isinstance(emission, BandEmission) \
                and emission.unit() is not None \
                and 'vsys' not in component.pdescs():
            label = component.name() or f'component{i}'
            raise ConfigError(
                f"the light of component {label!r} is in a unit, as the "
                f"galaxy emits it, which images see dimmed by its redshift: "
                f"give the galaxy its redshift (the option redshift; 0 for "
                f"none)")
