"""
The observation of profiles of a galaxy: the values of functions of the
galaxy at radii (e.g. its rotation curve).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Self

import astropy.units as u
import numpy as np
from annotated_types import Gt, Le
from astropy.table import Table

import gbkfit.units
from gbkfit.config import Unit, check_options
from gbkfit.dataset import Data, Dataset, DatasetProfile, check_profile_radii
from gbkfit.driver import Driver
from gbkfit.grid import make_grid
from gbkfit.model import (
    DataUnit, Extras, GalaxyFunctions, GalaxyState)
from gbkfit.params import ParamDesc, ParamScalarDesc, ParamValues
from .observables import HostModelData, ModelData, ModelItem, ObservablePlan
from .observation import Observation, SeenGalaxies

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
    An observation of profiles of a galaxy (e.g. its rotation curve): the
    values of functions of the galaxy at radii, compared with a table of
    them. The functions are those of GalaxyFunctions (e.g. vcirc, the
    circular velocity of its mass), and the table is a DatasetProfile
    (see Observation).

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

    It sees one galaxy. Its model is evaluated in float64, from the state
    of the galaxy that is evaluated once for all the observations of the
    galaxy (see GalaxyState), with the profiles of the disks sampled on
    its driver, and copied to its driver.

    In its configuration (type 'profile'), the data have the form of
    DatasetProfile (no type; see DatasetProfile.from_file).

    The data give its functions, radii and assumed distance; it can give
    them too, as without data (e.g. a profile of mock data, which then
    takes the data), and they must then be those of the data.

    Parameters
    ----------
    functions : Sequence of str, optional
        The functions it measures (its data items); with data, those of
        the data by default.
    radii : Quantity, optional
        The radii; with data, those of the data by default.
    assumed_distance : Quantity, optional
        The distance of radii in lengths that were measured at one; with
        data, that of the data by default (when the radii are not
        given).
    inclination : float, optional
        The inclination (degrees) at which its rotation velocities were
        deprojected: its parameter '{name}_incl' is then the inclination
        of the galaxy (degrees), and its name is required.
    galaxies, data, name
        See Observation.

    Raises
    ------
    ValueError
        If it has data whose functions, radii or distance are not those it
        gives, or neither data nor functions and radii, an item is not a
        function of radii, or its data are not in a unit of the kind of
        its own (they are converted to it), the radii or the distance are
        invalid (see check_profile_radii), a slope has a
        radius of 0, it sees more than one galaxy, the inclination is not
        above 0 and at most 90, or it has no name or no rotation
        velocities, or see Observation.
    """

    @staticmethod
    def type() -> str:
        return 'profile'

    @staticmethod
    def type_title() -> str:
        return 'Radial profiles'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the observation to its configuration.

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
                data={k: v for k, v in data.items() if k != 'type'})
        info |= dict(
            functions=list(self.keys()),
            radii=dict(
                value=self._radii.value.tolist(),
                unit=gbkfit.units.format_unit(self._radii.unit)))
        if self._assumed_distance is not None:
            info['assumed_distance'] = float(
                self._assumed_distance.to_value(u.Mpc))
        return info

    def __init__(
            self,
            functions: Sequence[str] | None = None,
            radii: u.Quantity | None = None,
            assumed_distance:
            Annotated[float, Gt(0), Unit('Mpc')] | None = None,
            inclination:
            Annotated[float, Gt(0), Le(90), Unit('deg')] | None = None,
            galaxies: Sequence[str] | None = None,
            data: DatasetProfile | None = None,
            name: str | None = None
    ):
        check_options(ObservationProfile.__init__, locals())
        if assumed_distance is not None:
            assumed_distance = assumed_distance * u.Mpc
        if data is not None:
            if not isinstance(data, DatasetProfile):
                raise ValueError(
                    f"the data of a profile are a table of it "
                    f"(DatasetProfile); they are {type(data).__name__}")
            # (those it does not give are those of the data; those it
            # gives are checked against them, see _checked_data)
            if functions is None:
                functions = tuple(data.keys())
            if radii is None:
                radii = data.radii()
                if assumed_distance is None:
                    assumed_distance = data.assumed_distance()
        elif functions is None or radii is None:
            raise ValueError(
                "a profile without data needs its functions and its radii")
        keys = tuple(functions)
        for key in keys:
            if GalaxyFunctions.family(key) is None:
                raise ValueError(
                    f"{key!r} is not a function of radii of galaxies (e.g. "
                    f"vcirc, vcirc_halo, mass, slope, rotation_gas, "
                    f"dispersion_gas; see GalaxyFunctions)")
        radii, assumed_distance = check_profile_radii(
            radii, assumed_distance)
        if any(radii == 0) and any(
                GalaxyFunctions.family(key) == 'slope' for key in keys):
            raise ValueError("slopes need radii above 0")
        if galaxies is not None and len(galaxies) > 1:
            raise ValueError(
                f"a profile is of one galaxy; it sees {list(galaxies)}")
        if inclination is not None:
            if name is None:
                raise ValueError(
                    "a profile with an inclination needs a name, which "
                    "names its parameter ('{name}_incl')")
            if not any(
                    GalaxyFunctions.family(key) in _ROTATIONS
                    for key in keys):
                raise ValueError(
                    "an inclination scales rotation velocities (e.g. vcirc, "
                    "rotation_gas); the profile has none")
        if data is not None:
            data = _checked_data(data, keys, radii, assumed_distance)
        self._radii = radii
        self._assumed_distance = assumed_distance
        self._inclination = None if inclination is None \
            else float(inclination)
        # (slopes have no values where the galaxy has no density)
        super().__init__(
            keys, galaxies, data, name,
            model_can_have_missing_values=any(
                GalaxyFunctions.family(key) == 'slope' for key in keys))

    def radii(self) -> u.Quantity:
        """Return the radii."""
        return self._radii

    def assumed_distance(self) -> u.Quantity | None:
        """
        Return the distance of radii in lengths that were measured at one,
        if given.
        """
        return self._assumed_distance

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
        '{name}_incl', if it has an inclination (see the class).
        """
        param = self._inclination_param()
        inclination = {} if param is None else {param: ParamScalarDesc(
            title='Inclination',
            desc=f"the inclination of the galaxy, of observation "
                 f"{self._name}, whose rotation velocities were deprojected "
                 f"at {self._inclination:g} degrees", unit='deg',
            default=self._inclination)}
        return inclination

    def _inclination_param(self) -> str | None:
        """Return the name of the parameter of the inclination, if any."""
        return None if self._inclination is None else f'{self._name}_incl'

    def make_dataset(self, items: Mapping[str, Data]) -> Dataset:
        """
        Make the table of data items of its functions (e.g. mock data) at
        its radii (see DatasetProfile).
        """
        return DatasetProfile(
            items, self._radii,
            None if self._assumed_distance is None
            else self._assumed_distance.to_value(u.Mpc))

    def data_unit(self) -> DataUnit | None:
        """Return None: its data are not of light."""
        return None

    def dtype(self) -> np.dtype:
        """Return the floating type of its model: float64."""
        return np.dtype(np.float64)

    def with_data(self, data: Dataset) -> Self:
        return type(self)(
            inclination=self._inclination, galaxies=self._galaxies, data=data,
            name=self._name)

    def same_plan(self, other: Observation) -> bool:
        """
        Check whether its plan is also that of another observation (see
        Observation.same_plan): its plan takes the functions, the radii
        and their assumed distance of its data, and the inclination.
        """
        if not isinstance(other, ObservationProfile) \
                or not super().same_plan(other):
            return False
        # (the radii as the plan takes them: in arcsec, or in kpc)
        return self.keys() == other.keys() \
            and all(_same_radii(a, b) for a, b in zip(
                self._radii_of(), other._radii_of())) \
            and self._inclination == other._inclination

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

    def check(self, seen: SeenGalaxies) -> None:
        """
        Check that the galaxy it sees has its functions, those of a disk
        at radii in angles, in their range (see
        Galaxy.function_radius_range and Observation.check), and a
        distance for those of radii in kpc at radii in angles (or lengths
        measured at a distance).

        Raises
        ------
        ValueError
            If the galaxy does not have a function, a function of a disk is
            at radii of physical lengths or out of its range, or one of
            radii in kpc is at radii in angles and the galaxy has no
            distance.
        """
        galaxy, label = seen.galaxies[0], seen.labels[0]
        names = galaxy.function_names()
        if missing := [k for k in self.keys() if k not in names]:
            raise ValueError(
                f"galaxy {label} has no functions {missing}; its functions "
                f"are {list(names)}")
        arcsec, _ = self._radii_of()
        if arcsec is not None and galaxy.distance() is None and (kpc := [
                key for key in self.keys()
                if GalaxyFunctions.radial_units(key)[1] != u.arcsec]):
            raise ValueError(
                f"the functions {kpc} are of radii in kpc, and the radii of "
                f"the profile are angles (or lengths measured at a "
                f"distance): galaxy {label} needs a distance (e.g. "
                f"distance: {{type: parameter}})")
        for key in self.keys():
            bounds = galaxy.function_radius_range(key)
            if bounds is None:
                continue
            if arcsec is None:
                raise ValueError(
                    f"{key} is a function of a disk, of radii in angles; the "
                    f"radii are physical lengths (they have no distance)")
            if not np.all((bounds[0] <= arcsec) & (arcsec <= bounds[1])):
                raise ValueError(
                    f"the radii of {key} must be from {bounds[0]:g} to "
                    f"{bounds[1]:g} arcsec; they are {arcsec} arcsec")

    def _radii_of(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return its radii as angles (arcsec) if they are, or have a
        distance, or else as physical lengths (kpc).
        """
        if self._radii.unit.is_equivalent(u.arcsec):
            return self._radii.to_value(u.arcsec), None
        if self._assumed_distance is not None:
            return (self._radii / self._assumed_distance).to_value(
                u.arcsec, u.dimensionless_angles()), None
        return None, self._radii.to_value(u.kpc)

    def plan(self, driver: Driver, seen: SeenGalaxies) -> ObservablePlan:
        """
        Plan the evaluation of its model (see ObservationGroup.plan): the
        functions of the galaxy it sees at its radii (see
        GalaxyFunctions).

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        seen : SeenGalaxies
            The galaxy it sees.

        Returns
        -------
        ObservablePlan
            The plan.
        """
        galaxy, mapping = seen.galaxies[0], seen.mappings[0]
        functions = galaxy.plan_functions(
            seen.sky_origin, driver, seen.cosmology)
        arcsec, kpc = self._radii_of()
        inclination = None
        if self._inclination is not None:
            inclination = (
                self._inclination_param(),
                np.sin(np.radians(self._inclination)),
                tuple(k for k in self.keys()
                      if GalaxyFunctions.family(k) in _ROTATIONS))
        return _ProfilePlan(
            functions, mapping, self.keys(), arcsec, kpc, inclination, driver)


def _same_radii(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    """Return whether two arrays of radii, or None, are the same."""
    if a is None or b is None:
        return a is b
    return np.array_equal(a, b)


def _checked_data(
        dataset: Dataset,
        keys: Sequence[str],
        radii: u.Quantity,
        distance: u.Quantity | None
) -> DatasetProfile:
    """
    Return a dataset of the data of a profile of the given functions,
    radii and distance (a table of it, see DatasetProfile, of those
    functions, at those radii and distance), each in the unit of its
    function: converted from another unit of its kind (e.g. m/s), or taken
    to be in it without a unit.

    Raises
    ------
    ValueError
        If it is not, or the unit of a data item is not of the kind of
        that of its function.
    """
    if not isinstance(dataset, DatasetProfile):
        raise ValueError(
            f"the data of a profile are a table of it (DatasetProfile); "
            f"they are {type(dataset).__name__}")
    if set(dataset.keys()) != set(keys):
        raise ValueError(
            f"the profile has the functions {sorted(keys)}, but the data "
            f"have {sorted(dataset.keys())}")
    same_distance = dataset.assumed_distance() is None if distance is None \
        else dataset.assumed_distance() is not None \
        and u.isclose(dataset.assumed_distance(), distance)
    if dataset.radii().unit != radii.unit \
            or not np.array_equal(dataset.radii(), radii) \
            or not same_distance:
        raise ValueError(
            "the data are not at the radii of the profile, or of their "
            "distance")
    items = {}
    for key in keys:
        item = dataset[key]
        unit = item.unit()
        expected = GalaxyFunctions.radial_units(key)[0]
        if unit is not None and not unit.is_equivalent(expected):
            raise ValueError(
                f"the data of {key} are in {unit}; give them in a unit of "
                f"{expected}, the unit of the function")
        items[key] = item if unit is None or unit == expected \
            else item.to_unit(expected)
    if all(items[key] is dataset[key] for key in keys):
        return dataset
    return DatasetProfile(
        items, dataset.radii(),
        None if distance is None else distance.to_value(u.Mpc))


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
        self._noise_grid = make_grid((size,), (1,), (0,), (0,))

    def evaluate(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            out_extra: Extras | None
    ) -> ModelData:
        values = self._values(params, galaxies)
        for key, value in values.items():
            self._driver.mem_copy_h2d(value, self._model[key])
        return {key: ModelItem(self._model[key]) for key in self._keys}

    def mock(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            noise: 'Noise',
            rng: np.random.Generator,
            ndraws: int
    ) -> tuple[list[HostModelData], dict[str, np.ndarray]]:
        values = self._values(params, galaxies)
        grid = self._noise_grid
        draws = [
            {key: ModelItem(value + noise.draw(rng, grid, None, value))
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
            galaxies: Sequence[GalaxyState]
    ) -> dict[str, np.ndarray]:
        """
        Return the values of the functions at the radii, on the host, at
        the parameters of the observation group, of the state of the
        galaxy, scaled by the inclination, if any.
        """
        own = {name: params[full] for name, full in self._mapping.items()}
        bound = self._functions.bind(own, galaxies[0])
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
