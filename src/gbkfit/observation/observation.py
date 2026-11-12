"""
The base of the observations: data of galaxies, and how the model of
the galaxies gives them (see Observation), the galaxies an observation
sees (see SeenGalaxies), and the frame of an observation on the sky (see
SkyFrame).
"""

import abc
from collections.abc import Mapping, Sequence
from typing import NamedTuple, Self

import astropy.units
import numpy as np
from astropy.table import Table

from gbkfit.config import (
    TypedParser, TypedSerializable, check_name)
from gbkfit.cosmology import Cosmology
from gbkfit.dataset import Data, Dataset
from gbkfit.driver import Driver
from gbkfit.grid import GridData, SpectraData
from gbkfit.io import Beam
from gbkfit.model import DataUnit, DepthUnit, Galaxy
from gbkfit.params import ParamDesc
from .observables import FluxScale, ObservablePlan


__all__ = [
    'Observation',
    'SeenGalaxies',
    'SkyFrame',
    'observation_parser'
]


class SeenGalaxies(NamedTuple):
    """
    The galaxies an observation sees (see ObservationGroup): the
    galaxies, the mapping of the parameters of each to those of the
    observation group, their labels (the id 'galaxy' of their extra
    outputs), their order along the line of sight, front to back (in
    units, see DepthUnit; None if not given), the RA and Dec (degrees) of
    the origin of the frame of their parameters (that of the first
    observation on the sky; None without one), and their cosmology (see
    ObservationGroup).
    """
    galaxies: tuple[Galaxy, ...]
    mappings: tuple[dict[str, str], ...]
    labels: tuple[str, ...]
    order: tuple[DepthUnit, ...] | None
    sky_origin: tuple[float, float] | None
    cosmology: Cosmology


class SkyFrame(NamedTuple):
    """
    The frame of the model of an observation on the sky (see
    Observation.sky_frame), in which it sees the galaxies (see
    ObservationGroup).

    Attributes
    ----------
    origin : tuple of float
        The RA and Dec (degrees) of its origin, from which the centres of
        what the observation sees are measured (e.g. the reference pixel
        of the grid of its data, see Observable.sky_rval).
    offsets : tuple of (str or None)
        The names of the parameters of the offsets of the frame, of x, y
        and the velocity (see ObservationSky): None for those it does not
        fit.
    velocity_frame : str or None
        The frame of rest of its velocities (FITS SPECSYS, see FRAMES), if
        known.
    centre_params : tuple of tuple of str
        The names of the parameters of the centres (x and y) of what the
        observation adds to the galaxies it sees in the frame of the model
        (e.g. the masses of its lens), which the frame moves as the
        centres of their components (see ObservationGroup).
    shift_params : tuple of tuple of str
        The names of the parameters of the shifts (x and y) of what the
        observation adds that is placed by its RA and Dec (e.g. shifted
        deflection maps of its lens), which only the offsets of the frame
        move: the RA and Dec place it in each frame.
    """
    origin: tuple[float, float]
    offsets: tuple[str | None, str | None, str | None]
    velocity_frame: str | None
    centre_params: tuple[tuple[str, str], ...]
    shift_params: tuple[tuple[str, str], ...]


class Observation(TypedSerializable, abc.ABC):
    """
    Data of galaxies, and how the model of the galaxies gives them: what
    the plan of an observation group evaluates (see ObservationGroup.plan),
    and an objective compares with the data under the likelihood its
    setup gives it (see Likelihoods and Objective). Its kind says where
    the data are measured and how the model gives them (e.g. on the sky,
    through an instrument; see ObservationSky). Without data, its model is
    evaluated alone (e.g. to draw mock data of it, see draw_mocks).

    Its data are named items of one shape (see keys), of a dataset of its
    kind, which the kind checks before it makes it. Its kind can have
    parameters of its own, named after it (see pdescs), e.g. a flux scale
    of its model (see flux_scale).

    Parameters
    ----------
    keys : Sequence of str
        The names of its data items (e.g. 'spectra').
    galaxies : Sequence of str, optional
        The names of the galaxies it sees; required only when there are
        several (see ObservationGroup).
    data : Dataset, optional
        The data, of its kind.
    name : str, optional
        Its name, which is the id of its extra outputs instead of its
        position (see ObservationGroup.observation_label), starts the
        names of its parameters, and keys the options of setups given by
        observation (e.g. its likelihood, see Likelihoods).
    model_can_have_missing_values : bool, optional
        Whether its model can have missing values (NaN) where the data
        have some (e.g. the velocity of a spectrum below its flux_floor):
        its likelihood then needs a residual for them (see
        LikelihoodGauss). False by default.

    Raises
    ------
    ValueError
        If a name is invalid, there are no galaxies or they repeat, or the
        noise of an item is correlated where its kind of data cannot have
        it (see Dataset.noise_correlation).
    """

    def __init__(
            self,
            keys: Sequence[str],
            galaxies: Sequence[str] | None = None,
            data: Dataset | None = None,
            name: str | None = None,
            model_can_have_missing_values: bool = False
    ):
        keys = tuple(keys)
        check_name(name)
        if galaxies is not None:
            galaxies = tuple(galaxies)
            if not galaxies:
                raise ValueError("an observation sees at least one galaxy")
            if len(set(galaxies)) != len(galaxies):
                raise ValueError(f"the galaxies repeat: {list(galaxies)}")
            for galaxy in galaxies:
                check_name(galaxy)
        if data is not None:
            for key in keys:
                # (the correlation of its noise must be one its kind of data
                # can have)
                data.noise_correlation(key)
        self._keys = keys
        self._galaxies = galaxies
        self._data = data
        self._name = name
        self._model_can_have_missing_values = model_can_have_missing_values

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def galaxies(self) -> tuple[str, ...] | None:
        """Return the names of the galaxies it sees, if given."""
        return self._galaxies

    def data(self) -> Dataset | None:
        """Return the data, if any."""
        return self._data

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return its parameters, named after it: none here; kinds add their
        own (e.g. its flux scale, see ObservationSky).
        """
        return {}

    def sky_frame(self) -> SkyFrame | None:
        """
        Return the frame of its model on the sky, in which it sees the
        galaxies (see SkyFrame and ObservationGroup); here, None: its data
        are not on the sky.
        """
        return None

    def takes_mass(self) -> bool:
        """
        Check whether its model takes the mass of the galaxies it sees
        beyond what their components take of it (e.g. profiles of their
        circular velocity, see ObservationProfile); here, it does not.
        """
        return False

    def keys(self) -> tuple[str, ...]:
        """Return the names of its data items (e.g. 'spectra')."""
        return self._keys

    def model_can_have_missing_values(self) -> bool:
        """
        Check whether its model can have missing values (NaN) where the
        data have some: the likelihood then needs a residual for them (see
        LikelihoodGauss).
        """
        return self._model_can_have_missing_values

    def flux_scale(self) -> tuple[str, FluxScale] | None:
        """
        Return the parameter of the flux scale of its model and how it
        scales the model (see ObservationSky), if it has one; none here.
        """
        return None

    @abc.abstractmethod
    def make_dataset(self, items: Mapping[str, Data]) -> Dataset:
        """
        Make the dataset of data items measured as this observation (e.g.
        mock data), where its model is.

        Parameters
        ----------
        items : Mapping
            The data items, by name (see keys).

        Returns
        -------
        Dataset
            The dataset.
        """

    @abc.abstractmethod
    def data_unit(self) -> DataUnit | None:
        """
        Return the unit of the light of its data, which its model takes;
        None without data, or without data of light.
        """

    def model_unit(
            self, key: str
    ) -> tuple[astropy.units.UnitBase | None, Beam | None]:
        """
        Return the unit of the values of a data item of its model, and
        their beam, of a unit per beam: those that data of the model
        have (e.g. mock data; see draw_mocks).

        Here, those of its data, if any (None for those they lack), and
        none without data; kinds whose model has a unit without data add
        it (see ObservationSky).

        Parameters
        ----------
        key : str
            The name of the data item (see keys).

        Returns
        -------
        tuple
            The unit, or None, and the beam, or None.
        """
        if self._data is None:
            return None, None
        item = self._data[key]
        return item.unit(), item.beam()

    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Return the floating type of its model."""

    @abc.abstractmethod
    def with_data(self, data: Dataset) -> Self:
        """
        Return a copy of it with other data (e.g. mock data; see
        draw_mocks).

        Parameters
        ----------
        data : Dataset
            The data, of the observation.

        Returns
        -------
        Observation
            The copy.

        Raises
        ------
        ValueError
            If the data are not of the observation (see __init__).
        """

    def same_plan(self, other: 'Observation') -> bool:
        """
        Check whether its plan is also that of another observation, the
        same one with other data (see with_data), which can then share it
        (see GroupPlan.with_data): whether all that its plan takes from
        its data is the same in the other.

        Here, the unit of their light (see data_unit) and the floating
        type of their models, which the data can choose (see dtype);
        kinds whose plans take more from their data compare that too (see
        ObservationProfile).

        Parameters
        ----------
        other : Observation
            The other observation.

        Returns
        -------
        bool
            True if the plan is also that of the other.
        """
        return _same_unit(self.data_unit(), other.data_unit()) \
            and self.dtype() == other.dtype()

    def light_amplitudes(self, seen: SeenGalaxies) -> dict[str, str | None]:
        """
        Return the amplitudes of the light that its model takes (see
        Component.light_amplitudes), by their names in its observation
        group: those of the components whose light it sees, of the
        galaxies it sees, each with None if its model is linear in it, or
        why not (also when it is not linear in the light: see
        Observable.linear_in_light); none here, where it sees no light.
        """
        return {}

    def orbit_components(
            self, seen: SeenGalaxies
    ) -> tuple[tuple[int, int], ...]:
        """
        Return the components of the galaxies it sees whose stars it sees
        on orbits whose weights are solved (see Component.orbit_weights;
        with data, a recording of the orbits models it, see
        GroupPlan.orbit_series), each as the index of its galaxy (among
        those it sees) and its own index; none here, where it sees no
        light.
        """
        return ()

    @abc.abstractmethod
    def check(self, seen: SeenGalaxies) -> None:
        """
        Check that it can see the galaxies it sees, without a driver (see
        ObservationGroup, which calls this for each observation): every
        rule of its configuration about what it sees is raised here, before
        any plan, which then fails only for resources.

        Parameters
        ----------
        seen : SeenGalaxies
            The galaxies it sees.

        Raises
        ------
        ValueError
            If it cannot see them.
        """

    @abc.abstractmethod
    def plan(self, driver: Driver, seen: SeenGalaxies) -> ObservablePlan:
        """
        Plan the evaluation of its model (see ObservationGroup.plan).

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        seen : SeenGalaxies
            The galaxies it sees.

        Returns
        -------
        ObservablePlan
            The plan.
        """

    @abc.abstractmethod
    def output(
            self,
            data: np.ndarray,
            unit: astropy.units.UnitBase | None = None
    ) -> GridData | SpectraData | Table | np.ndarray:
        """
        Return an array of the form of its data as an output (e.g. on the
        grid of its data, with their world coordinates).

        Parameters
        ----------
        data : np.ndarray
            The array: the model, its mask or weights, or a residual.
        unit : astropy.units.UnitBase, optional
            The unit of its values, if known (e.g. that of the data, and of
            the model, which is in it), which the output takes.

        Returns
        -------
        GridData or SpectraData or Table or np.ndarray
            The output.
        """

    def images(
            self,
            driver: Driver,
            values: Mapping[str, np.ndarray],
            weights: Mapping[str, np.ndarray] | None,
            units: Mapping[str, astropy.units.UnitBase | None] | None
    ) -> dict[str, GridData]:
        """
        Return images of values of its data items, as outputs (e.g. the
        dirty images of visibilities; see Observable.images), made on a
        driver; here, none.

        Parameters
        ----------
        driver : Driver
            The driver they are made on.
        values : Mapping
            The values of each data item (e.g. those of the data, of the
            model, or of the data minus the model), NaN where they have
            none.
        weights : Mapping, optional
            The weights of the values of each data item (e.g. 1 / error^2);
            all 1 if None.
        units : Mapping, optional
            The unit of the values of each data item, if known.

        Returns
        -------
        dict
            The images, by name.
        """
        return {}

    def psf_images(
            self,
            driver: Driver,
            weights: Mapping[str, np.ndarray] | None
    ) -> dict[str, GridData]:
        """
        Return images of the PSF of its data (e.g. the dirty beam of
        visibilities; see Observable.psf_images), which its images are
        seen through, made on a driver; here, none.

        Parameters
        ----------
        driver : Driver
            The driver they are made on.
        weights : Mapping, optional
            The weights of the values of each data item (see images).

        Returns
        -------
        dict
            The images, by name.
        """
        return {}


def _same_unit(a: DataUnit | None, b: DataUnit | None) -> bool:
    """
    Return whether two units of the light of data are the same (see
    DataUnit, whose beams can be arrays, one for each spectral point).
    """
    if a is None or b is None:
        return a is b
    return all(np.array_equal(x, y) if isinstance(x, np.ndarray)
               else x == y for x, y in zip(a, b))


observation_parser = TypedParser(Observation, family='observation')
