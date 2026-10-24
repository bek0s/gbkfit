from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self

import astropy.units as u
import numpy as np
from astropy.table import Table

from gbkfit.dataset import Data, Dataset, DatasetProfile, check_profile_radii
from gbkfit.driver import Driver, driver_by_name
from gbkfit.model import Extras, LineCatalogue
from gbkfit.model.components._functions import GalaxyFunctions
from gbkfit.model.components._gravity import GalaxyGravity
from gbkfit.model.components._units import DataUnit
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamValues
from gbkfit.utils import gridutils, parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from ._scene import SeenGalaxies
from .likelihood import Likelihood, likelihood_parser
from .observables import HostModelData, ModelData, ObservablePlan
from .observation import Observation

if TYPE_CHECKING:
    from .noise import Noise


__all__ = [
    'ObservationProfile'
]


# The families of the functions of rotation velocities (see
# GalaxyFunctions.family), which an inclination scales
_ROTATIONS = ('vcirc', 'vnewton', 'rotation')


class ObservationProfile(Observation):
    """
    An observation of profiles of a galaxy (see Observation): the values
    of functions of the galaxy at radii (see GalaxyFunctions), e.g. its
    rotation curve, the circular velocity of its mass, compared with a
    table of them (see DatasetProfile).

    Its data items are named after the functions they measure, in their
    units: of its mass, vcirc, vnewton and vcirc_{component} (km/s),
    mass and mass_{component} (Msun), slope and slope_{component} (no
    unit); of its disks, rotation_{component} and
    dispersion_{component} (km/s; rotation and dispersion for a disk
    without a name).

    Its radii are angles or lengths (see DatasetProfile). In the galaxy,
    angles are lengths at its distance, and so are lengths measured at a
    distance (their angles are the lengths over it): e.g. the radii of a
    table in kpc, at the distance it was made at, follow a fitted
    distance. Lengths without a distance are physical radii (e.g.
    galactocentric, in the Milky Way). The functions of the mass take the
    radii in kpc, by the distance of the galaxy where they are angles;
    those of the disks in arcsec, which radii of physical lengths are not.

    A rotation curve deprojected at an inclination i0 (e.g. of a fit of
    tilted rings) can take the inclination of the galaxy as a parameter
    (see inclination): the rotation velocities of its model (vcirc,
    vnewton, vcirc_*, rotation_*) are then times sin(incl) / sin(i0),
    compared with the data and their errors as they are, as the
    chi-square of the data and their errors deprojected at incl (e.g. the
    fits of SPARC of Li et al. 2018).

    It sees one galaxy. Its model is evaluated on the host, in float64,
    and copied to its driver; its gravity is that which the observation
    group evaluates for all the observations of the galaxy (see
    GalaxyGravity).

    In its configuration (type 'profile'), the data have the form of
    DatasetProfile (no type); without data, it has its functions, its
    radii (a list) and their unit (radius_unit), and their distance (Mpc),
    if any.

    Parameters
    ----------
    functions : Sequence of str, optional
        The functions it measures (its data items); with data, those of
        the data.
    radii : Quantity, optional
        The radii; with data, those of the data.
    distance : Quantity, optional
        The distance of radii in lengths that were measured at one; with
        data, that of the data.
    inclination : float, optional
        The inclination (degrees) at which its rotation velocities were
        deprojected: its parameter '{name}_incl' is then the inclination
        of the galaxy (degrees), and its name is required.
    driver, galaxies, data, likelihood, name
        See Observation.

    Raises
    ------
    ConfigError
        If it has data and functions, radii or a distance (which the data
        give), or neither data nor functions and radii, an item is not a
        function of radii, or its data are not in its unit, the radii or
        the distance are invalid (see check_profile_radii), a slope has a
        radius of 0, it sees more than one galaxy, the inclination is not
        above 0 and at most 90, or it has no name or no rotation
        velocities, or see Observation.
    """

    @staticmethod
    def type() -> str:
        return 'profile'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            drivers: Mapping[str | None, Driver] | None = None,
            catalogue: LineCatalogue | None = None
    ) -> 'ObservationProfile':
        """
        Load a profile observation from its configuration.

        Its option driver is the name of one of the drivers of the
        configuration (see ObservationSky.load). The catalogue is not
        used (a profile has no lines).

        Parameters
        ----------
        info : dict
            The configuration.
        drivers : Mapping of str to Driver, optional
            The drivers of the configuration, by name.
        catalogue : LineCatalogue, optional
            Unused.

        Returns
        -------
        ObservationProfile
            The observation.

        Raises
        ------
        ConfigError
            If the configuration has an error, or the driver is not one
            of the drivers.
        """
        with parseutils.config_path('driver'):
            info['driver'] = driver_by_name(
                info.get('driver'), drivers or {}, 'observation')
        if info.get('data') is not None:
            with parseutils.config_path('data'):
                info['data'] = DatasetProfile.load(dict(info['data']))
        if 'radii' in info:
            with parseutils.config_path('radii'):
                unit = info.pop('radius_unit', None)
                if unit is None:
                    raise ConfigError(
                        "the radii need their unit (radius_unit, e.g. kpc)")
                info['radii'] = np.asarray(
                    info['radii'], dtype=float) * unitutils.parse_unit(unit)
        if info.get('distance') is not None:
            info['distance'] = info['distance'] * u.Mpc
        parseutils.load_option_and_update_info(
            likelihood_parser, info, 'likelihood')
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the observation to its configuration, without its driver,
        which configurations declare at their top level (see load).

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its data (see
            DatasetProfile.dump).
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.
        """
        info = {} if self._name is None else dict(name=self._name)
        if self._galaxies is not None:
            info['galaxies'] = list(self._galaxies)
        if self._inclination is not None:
            info['inclination'] = self._inclination
        if self._data is not None:
            # (the form of the data is that of a profile: no type)
            data = self._data.dump(prefix, dump_path, overwrite)
            return info | dict(
                data={k: v for k, v in data.items() if k != 'type'},
                likelihood=likelihood_parser.dump(self._likelihood))
        info |= dict(
            functions=list(self.keys()),
            radii=self._radii.value.tolist(),
            radius_unit=unitutils.format_unit(self._radii.unit))
        if self._distance is not None:
            info['distance'] = float(self._distance.to_value(u.Mpc))
        return info

    def __init__(
            self,
            functions: Sequence[str] | None = None,
            radii: u.Quantity | None = None,
            distance: u.Quantity | None = None,
            inclination: float | None = None,
            driver: Driver | None = None,
            galaxies: Sequence[str] | None = None,
            data: Dataset | None = None,
            likelihood: Likelihood | None = None,
            name: str | None = None
    ):
        if data is not None:
            if functions is not None or radii is not None \
                    or distance is not None:
                raise ConfigError(
                    "the data of a profile give its functions, its radii "
                    "and their distance")
            if not isinstance(data, DatasetProfile):
                raise ConfigError(
                    f"the data of a profile are a table of it "
                    f"(DatasetProfile); they are {type(data).__name__}")
            functions, radii, distance = \
                tuple(data.keys()), data.radii(), data.distance()
        elif functions is None or radii is None:
            raise ConfigError(
                "a profile without data needs its functions and its radii")
        keys = tuple(functions)
        for key in keys:
            if GalaxyFunctions.family(key) is None:
                raise ConfigError(
                    f"{key!r} is not a function of radii of galaxies (e.g. "
                    f"vcirc, vcirc_halo, mass, slope, rotation_gas, "
                    f"dispersion_gas; see GalaxyFunctions)")
        radii, distance = check_profile_radii(radii, distance)
        if any(radii == 0) and any(
                GalaxyFunctions.family(key) == 'slope' for key in keys):
            raise ConfigError("slopes need radii above 0")
        if galaxies is not None and len(galaxies) > 1:
            raise ConfigError(
                f"a profile is of one galaxy; it sees {list(galaxies)}")
        if inclination is not None:
            if not 0 < inclination <= 90:
                raise ConfigError(
                    f"the inclination must be above 0 and at most 90 "
                    f"degrees; it is {inclination}")
            if name is None:
                raise ConfigError(
                    "a profile with an inclination needs a name, which "
                    "names its parameter ('{name}_incl')")
            if not any(
                    GalaxyFunctions.family(key) in _ROTATIONS
                    for key in keys):
                raise ConfigError(
                    "an inclination scales rotation velocities (e.g. vcirc, "
                    "rotation_gas); the profile has none")
        if data is not None:
            _check_data(data, keys, radii, distance)
        self._radii = radii
        self._distance = distance
        self._inclination = None if inclination is None \
            else float(inclination)
        # (slopes have no values where the galaxy has no density)
        super().__init__(
            keys, driver, galaxies, data, likelihood, name,
            model_can_lack_values=any(
                GalaxyFunctions.family(key) == 'slope' for key in keys))

    def radii(self) -> u.Quantity:
        """Return the radii."""
        return self._radii

    def distance(self) -> u.Quantity | None:
        """
        Return the distance of radii in lengths that were measured at one,
        if given.
        """
        return self._distance

    def inclination(self) -> float | None:
        """
        Return the inclination at which its rotation velocities were
        deprojected (degrees), if given.
        """
        return self._inclination

    def takes_mass(self) -> bool:
        return any(GalaxyFunctions.of_mass(name) for name in self.keys())

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return its parameters: the inclination of the galaxy,
        '{name}_incl', if it has an inclination (see the class), and those
        of the fitted noise of its likelihood (see Observation.pdescs).
        """
        param = self._inclination_param()
        inclination = {} if param is None else {param: ParamScalarDesc(
            desc=f"the inclination of the galaxy (degrees), of observation "
                 f"{self._name}, whose rotation velocities were deprojected "
                 f"at {self._inclination:g} degrees")}
        return inclination | super().pdescs()

    def _inclination_param(self) -> str | None:
        """Return the name of the parameter of the inclination, if any."""
        return None if self._inclination is None else f'{self._name}_incl'

    def dataset(self, items: Mapping[str, Data]) -> Dataset:
        """
        Make the table of data items of its functions (e.g. mock data) at
        its radii (see DatasetProfile).
        """
        return DatasetProfile(items, self._radii, self._distance)

    def data_unit(self) -> DataUnit | None:
        """Return None: its data are not of light."""
        return None

    def dtype(self) -> np.dtype:
        """Return the floating type of its model: float64."""
        return np.dtype(np.float64)

    def with_data(
            self, data: Dataset, likelihood: Likelihood | None = None
    ) -> Self:
        _check_data(data, self.keys(), self._radii, self._distance)
        return type(self)(
            inclination=self._inclination, driver=self._driver,
            galaxies=self._galaxies, data=data,
            likelihood=likelihood if likelihood is not None
            else self._likelihood, name=self._name)

    def output(
            self,
            data: np.ndarray,
            unit: u.UnitBase | None = None
    ) -> Table:
        """
        Return an array of the form of its data as an output: a table of
        its radii ('radius') and the values ('value', in the unit, if
        given).
        """
        values = np.asarray(data)
        return Table(dict(
            radius=self._radii,
            value=values if unit is None else values * unit))

    def plan(self, seen: SeenGalaxies) -> ObservablePlan:
        """
        Plan the evaluation of its model (see ObservationGroup): the
        functions of the galaxy it sees at its radii (see
        GalaxyFunctions).

        Parameters
        ----------
        seen : SeenGalaxies
            The galaxy it sees.

        Returns
        -------
        ObservablePlan
            The plan.

        Raises
        ------
        ConfigError
            If the galaxy does not have a function, or a function of a
            disk is at radii of physical lengths, or outside the disk.
        """
        galaxy, mapping, label = \
            seen.galaxies[0], seen.mappings[0], seen.labels[0]
        functions = galaxy.plan_functions(seen.sky_origin)
        if missing := [k for k in self.keys() if k not in functions.names()]:
            raise ConfigError(
                f"galaxy {label} has no functions {missing}; its functions "
                f"are {list(functions.names())}")
        # The radii as angles (arcsec), or as physical lengths (kpc)
        arcsec = kpc = None
        if self._radii.unit.is_equivalent(u.arcsec):
            arcsec = self._radii.to_value(u.arcsec)
        elif self._distance is not None:
            arcsec = (self._radii / self._distance).to_value(
                u.arcsec, u.dimensionless_angles())
        else:
            kpc = self._radii.to_value(u.kpc)
        for key in self.keys():
            bounds = functions.radius_range(key)
            if bounds is None:
                continue
            if arcsec is None:
                raise ConfigError(
                    f"{key} is a function of a disk, of radii in angles; the "
                    f"radii are physical lengths (they have no distance)")
            if not np.all((bounds[0] <= arcsec) & (arcsec <= bounds[1])):
                raise ConfigError(
                    f"the radii of {key} must be on its disk, from "
                    f"{bounds[0]:g} to {bounds[1]:g} arcsec; they are "
                    f"{arcsec} arcsec")
        inclination = None
        if self._inclination is not None:
            inclination = (
                self._inclination_param(),
                np.sin(np.radians(self._inclination)),
                tuple(k for k in self.keys()
                      if GalaxyFunctions.family(k) in _ROTATIONS))
        return _ProfilePlan(
            functions, mapping, self.keys(), arcsec, kpc, inclination,
            self._driver)


def _check_data(
        dataset: Dataset,
        keys: Sequence[str],
        radii: u.Quantity,
        distance: u.Quantity | None
) -> None:
    """
    Raise ConfigError unless a dataset holds the data of a profile of the
    given functions, radii and distance: a table of it (see
    DatasetProfile), of those functions, at those radii and distance, each
    in the unit of its function (or none).
    """
    if not isinstance(dataset, DatasetProfile):
        raise ConfigError(
            f"the data of a profile are a table of it (DatasetProfile); "
            f"they are {type(dataset).__name__}")
    if set(dataset.keys()) != set(keys):
        raise ConfigError(
            f"the profile has the functions {sorted(keys)}, but the data "
            f"have {sorted(dataset.keys())}")
    same_distance = dataset.distance() is None if distance is None \
        else dataset.distance() is not None \
        and u.isclose(dataset.distance(), distance)
    if dataset.radii().unit != radii.unit \
            or not np.array_equal(dataset.radii(), radii) \
            or not same_distance:
        raise ConfigError(
            "the data are not at the radii of the profile, or of their "
            "distance")
    for key in keys:
        unit = dataset[key].unit()
        expected = GalaxyFunctions.radial_units(key)[0]
        if unit is not None and unit != expected:
            raise ConfigError(
                f"the data of {key} are in {unit}; give them in {expected}, "
                f"the unit of the function")


class _ProfilePlan(ObservablePlan):
    """
    The evaluation of a profile (see ObservationProfile): the functions of
    its galaxy, the mapping of the parameters of the galaxy to those of
    the observation group, the names of the functions, the radii as
    angles (arcsec) or as physical lengths (kpc), the inclination, if any
    (the name of its parameter, the sine of the inclination of the data
    and the functions it scales), and the driver of the model data.
    """

    def __init__(
            self,
            functions: GalaxyFunctions,
            mapping: dict[str, str],
            keys: tuple[str, ...],
            arcsec: np.ndarray | None,
            kpc: np.ndarray | None,
            inclination: tuple[str, float, tuple[str, ...]] | None,
            driver: Driver
    ):
        self._functions = functions
        self._mapping = mapping
        self._keys = keys
        self._arcsec = arcsec
        self._kpc = kpc
        self._inclination = inclination
        self._driver = driver
        size = (arcsec if kpc is None else kpc).size
        self._model = {
            key: driver.mem_alloc_d(size, np.float64) for key in keys}
        # (the grid of the noise of mock data)
        self._noise_grid = gridutils.make_grid((size,), (1,), (0,), (0,))

    def evaluate(
            self,
            params: ParamValues,
            gravity: Sequence[GalaxyGravity],
            out_extra: Extras | None
    ) -> ModelData:
        values = self._values(params, gravity)
        for key, value in values.items():
            self._driver.mem_copy_h2d(value, self._model[key])
        return {
            key: dict(d=self._model[key], m=None, w=None)
            for key in self._keys}

    def mock(
            self,
            params: ParamValues,
            gravity: Sequence[GalaxyGravity],
            noise: 'Noise',
            rng: np.random.Generator,
            ndraws: int
    ) -> tuple[list[HostModelData], dict[str, np.ndarray]]:
        values = self._values(params, gravity)
        grid = self._noise_grid
        draws = [
            {key: dict(d=value + noise.draw(rng, grid, None, value), m=None)
             for key, value in values.items()}
            for _ in range(ndraws)]
        errors = {
            key: noise.sigma(grid, value) for key, value in values.items()}
        return draws, errors

    def has_exact_errors(self) -> bool:
        return True

    def _values(
            self,
            params: ParamValues,
            gravity: Sequence[GalaxyGravity]
    ) -> dict[str, np.ndarray]:
        """
        Return the values of the functions at the radii, on the host, at
        the parameters of the observation group, in the gravity of the
        galaxy, scaled by the inclination, if any.
        """
        own = {name: params[full] for name, full in self._mapping.items()}
        bound = self._functions.bind(own, gravity[0])
        values = {}
        for key in self._keys:
            if GalaxyFunctions.radial_units(key)[1] == u.arcsec:
                radii = self._arcsec
            elif self._kpc is not None:
                radii = self._kpc
            else:
                radii = self._arcsec * bound['kpc_per_arcsec']
            values[key] = np.asarray(bound[key](radii), dtype=np.float64)
        if self._inclination is not None:
            param, sin_data, scaled = self._inclination
            factor = np.sin(np.radians(params[param])) / sin_data
            for key in scaled:
                values[key] = values[key] * factor
        return values
