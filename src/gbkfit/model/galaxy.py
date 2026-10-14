"""
Galaxies: their components and their mass, at one distance.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Self

import numpy as np

from gbkfit.driver import Driver, driver_by_name
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .components import _detail
from .components._component_set import ComponentSet, ComponentSetPlan
from .components._gravity import GravityPlan
from .components._transfer import TransferMember
from .components.base import Component, DepthAxis, OpacityCubes
from .components.disks import MCDisk, SMDisk
from .components.geometries import Geometry, geometry_parser
from .components.lines import Band, LineCatalogue
from .components.points import Point
from .distance import Distance, distance_parser
from .mass import MassComponent, MassModel, mass_component_parser


__all__ = [
    'Galaxy',
    'component_parser',
    'galaxy_parser'
]


component_parser = parseutils.TypedParser(Component, [
    SMDisk,
    MCDisk,
    Point])


class Galaxy(parseutils.Serializable):
    """
    A galaxy: its components, and its mass at a distance. Its components
    are what observations see: each emits light (its emission), removes
    light (its opacity) or both. Its mass pulls: its components with the
    velocity trait 'mass' rotate in its gravity (see traits.VPTraitMass),
    which takes its disks as concentric and flat (see GravityPlan).

    Its parameters are those of its components, each prefixed by its name
    when there are several ('disk_pt_a'), and those of its geometries
    ('{geometry}_posa'), its distance ('distance') and its mass
    ('mass_{name}_*'). A name prefixes them all when there are several
    galaxies (see ObservationGroup), whose order along the line of sight
    the ObservationGroup gives (see its option depth_order).

    In configurations, its option geometries lists the geometries that
    its components name (see Geometry).

    Parameters
    ----------
    components : Component or Sequence of Component
        Its components; they need names when there are several.
    mass : MassComponent or Sequence of MassComponent, optional
        Its mass: the components of its mass (halos, point masses, ...,
        and trait_disk, which follows the density of one of its disks).
    distance : Distance, optional
        Its distance, which converts arcsec to kpc for its mass (required
        with mass, and used by it only).
    redshift : float, optional
        Its one redshift: for what it emits and absorbs in its rest frame
        (the dust of an image absorbs at the effective wavelength of its
        band over 1 + redshift), and for its distance by redshift (see
        DistanceRedshift); the systemic velocity of its components is the
        velocity offset from it. 0 by default.
    driver : Driver, optional
        Where its own computations run: its gravity, in float64, once for
        all the observations of it, whatever their drivers (see
        plan_gravity); by default, the host. (In its configuration, the
        name of a driver; see load.)
    name : str, optional
        Its name; required when there are several galaxies.

    Raises
    ------
    ConfigError
        If there are no components, several without names, the names or
        parameters conflict, the mass, the distance and the components do
        not go together (e.g. 'mass' velocity traits without mass, or a
        trait_disk that follows no disk of the galaxy), or the redshift is
        not above -1.
    """

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None,
            drivers: Mapping[str | None, Driver] | None = None
    ) -> Self:
        """
        Load a galaxy from its configuration.

        Its option driver is not a driver but the name of one of the
        drivers of the configuration, which declares them at its top
        level. A galaxy with mass can leave it out when there is one
        driver, and has the host when there are none; a galaxy without
        mass needs no driver.

        Parameters
        ----------
        info : dict
            The configuration.
        catalogue : LineCatalogue, optional
            The catalogue of the lines that the emission of its components
            names (see Emission.load); that of GBKFIT by default.
        drivers : Mapping of str to Driver, optional
            The drivers of the configuration, by name; the one unnamed
            driver of a configuration has the name None.

        Returns
        -------
        Galaxy
            The galaxy.

        Raises
        ------
        ConfigError
            If the configuration has an error, or the driver is not one
            of the drivers.
        """
        if 'driver' in info or info.get('mass'):
            with parseutils.config_path('driver'):
                info['driver'] = driver_by_name(
                    info.get('driver'), drivers or {}, 'galaxy')
        _detail.load_geometries(info, ('components',))
        components = info.get('components')
        parseutils.load_option_and_update_info(
            component_parser, info, 'components', required=True,
            catalogue=[catalogue] * len(components)
            if iterutils.is_sequence(components) else catalogue)
        parseutils.load_option_and_update_info(
            mass_component_parser, info, 'mass')
        parseutils.load_option_and_update_info(
            distance_parser, info, 'distance')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        info = dict(name=self._name) if self._name is not None else {}
        geometries = self.geometries()
        if geometries:
            info.update(geometries=geometry_parser.dump(geometries))
        info.update(components=component_parser.dump(list(self.components())))
        if self._mass:
            info.update(mass=mass_component_parser.dump(list(self._mass)))
        if self.distance() is not None:
            info.update(distance=distance_parser.dump(self.distance()))
        info.update(redshift=self._redshift)
        return info

    def __init__(
            self,
            components: Component | Sequence[Component],
            mass: MassComponent | Sequence[MassComponent] | None = None,
            distance: Distance | None = None,
            redshift: float = 0.0,
            driver: Driver | None = None,
            name: str | None = None
    ):
        parseutils.check_name(name)
        if not redshift > -1:
            raise ConfigError(
                f"the redshift must be above -1; it is {redshift}")
        self._redshift = redshift
        mass = iterutils.tuplify(mass)
        mass_model = MassModel(mass) if mass else None
        self._component_set = ComponentSet(components, distance, mass_model)
        self._pdescs = self._component_set.pdescs()
        if driver is None:
            # (imported here: the host driver needs its native module)
            from gbkfit.driver.drivers.host import DriverHost
            driver = DriverHost()
        self._mass = mass
        self._driver = driver
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

    def geometries(self) -> list[Geometry]:
        """Return the geometries of its components, each once."""
        return self._component_set.geometries()

    def mass(self) -> tuple[MassComponent, ...]:
        """Return the components of its mass (none without mass)."""
        return self._mass

    def mass_model(self) -> MassModel | None:
        """Return its mass, as a mass model, if any."""
        return self._component_set.mass_model()

    def distance(self) -> Distance | None:
        """Return its distance, if given."""
        return self._component_set.distance()

    def redshift(self) -> float:
        """Return its redshift."""
        return self._redshift

    def driver(self) -> Driver:
        """Return the driver of its own computations (its gravity)."""
        return self._driver

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return its parameters, by name (see the class)."""
        return self._pdescs

    def constants(self) -> dict[str, Any]:
        """
        Return the values that parameter expressions can use (e.g. the
        radial nodes of its disks, prefixed as their parameters).
        """
        return self._component_set.constants()

    def plan_gravity(self) -> GravityPlan:
        """
        Plan the evaluation of the gravity of the galaxy on its driver, in
        float64, once for all the observations of it: the circular velocity
        of its mass at the radii of the components it moves (see
        GravityPlan), which its plans on the grids of observations take.

        Returns
        -------
        GravityPlan
            The plan.
        """
        return self._component_set.plan_gravity(
            self._driver, self._redshift)

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
            band, smoothing)

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
            self._component_set, tuple(emitters), cubes, self._redshift,
            depth)


galaxy_parser = parseutils.BasicParser(Galaxy)
