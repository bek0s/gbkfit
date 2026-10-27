import abc
from collections.abc import Mapping, Sequence
from typing import Self

import astropy.units
import numpy as np
from astropy.table import Table

from gbkfit.dataset import Data, Dataset
from gbkfit.driver import Driver
from gbkfit.model import DataUnit
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ._scene import SeenGalaxies
from .likelihood import Likelihood, LikelihoodGaussian
from .observables import FluxScale, ObservablePlan


__all__ = [
    'Observation',
    'observation_parser'
]


class Observation(parseutils.TypedSerializable, abc.ABC):
    """
    Data of galaxies, and how the model of the galaxies gives them: what
    the plan of an observation group evaluates (see ObservationGroup.plan),
    and an objective compares with the data under its likelihood (see
    Objective). Its kind says where the data are measured and how the
    model gives them (e.g. on the sky, through an instrument; see
    ObservationSky).

    Its data are named items of one shape (see keys), of a dataset of its
    kind, which the kind checks before it makes it. Its likelihood and its
    kind can have parameters of their own, named after it (see pdescs),
    e.g. a flux scale of its model (see flux_scale).

    Parameters
    ----------
    keys : Sequence of str
        The names of its data items (e.g. 'spectra').
    galaxies : Sequence of str, optional
        The names of the galaxies it sees; required only when there are
        several (see ObservationGroup).
    data : Dataset, optional
        The data, of its kind.
    likelihood : Likelihood, optional
        How the model is compared with the data; Gaussian by default when
        there are data, and none without data.
    name : str, optional
        Its name, which is the id of its extra outputs instead of its
        position (see ObservationGroup.observation_label), and starts the
        names of its parameters.
    model_can_lack_values : bool, optional
        Whether its model can lack values (NaN) where the data have some
        (e.g. the velocity of a spectrum below its flux_floor): the
        likelihood then needs a residual for them (see
        LikelihoodGaussian). False by default.

    Raises
    ------
    ConfigError
        If a name is invalid, there are no galaxies or they repeat, there
        is a likelihood without data, its model can lack values but its
        likelihood has no residual for them (see LikelihoodGaussian), its
        likelihood fits a noise but it has no name, the noise of an item
        is correlated where its kind of data cannot have it (see
        Dataset.noise_correlation), or its likelihood does not suit it
        (see Likelihood.check).
    """

    def __init__(
            self,
            keys: Sequence[str],
            galaxies: Sequence[str] | None = None,
            data: Dataset | None = None,
            likelihood: Likelihood | None = None,
            name: str | None = None,
            model_can_lack_values: bool = False
    ):
        keys = tuple(keys)
        parseutils.check_name(name)
        if galaxies is not None:
            galaxies = tuple(galaxies)
            if not galaxies:
                raise ConfigError("an observation sees at least one galaxy")
            if len(set(galaxies)) != len(galaxies):
                raise ConfigError(f"the galaxies repeat: {list(galaxies)}")
            for galaxy in galaxies:
                parseutils.check_name(galaxy)
        if data is None and likelihood is not None:
            raise ConfigError("a likelihood needs data")
        if data is not None:
            if likelihood is None:
                likelihood = LikelihoodGaussian()
            for key in keys:
                # (the correlation of its noise must be one its kind of data
                # can have)
                data.noise_correlation(key)
            likelihood.check(keys, data)
            if name is None and likelihood.pdescs():
                raise ConfigError(
                    "an observation whose likelihood has parameters (e.g. "
                    "a fitted noise) needs a name, which names them (e.g. "
                    "'{name}_noise_scale_moment1')")
            likelihood = likelihood.with_prefix(
                '' if name is None else f'{name}_')
            if model_can_lack_values \
                    and not likelihood.takes_lacking_values():
                raise ConfigError(
                    "the model of the observation can lack values where the "
                    "data have some (e.g. the velocity of a spectrum below "
                    "its flux_floor): its likelihood needs the residual of "
                    "such a value, the option missing_residual (e.g. 3; see "
                    "LikelihoodGaussian)")
        self._keys = keys
        self._galaxies = galaxies
        self._data = data
        self._likelihood = likelihood
        self._name = name
        self._model_can_lack_values = model_can_lack_values

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def galaxies(self) -> tuple[str, ...] | None:
        """Return the names of the galaxies it sees, if given."""
        return self._galaxies

    def data(self) -> Dataset | None:
        """Return the data, if any."""
        return self._data

    def likelihood(self) -> Likelihood | None:
        """Return the likelihood, if there are data."""
        return self._likelihood

    def pdescs(self) -> dict[str, ParamDesc]:
        """
        Return its parameters: here those of the fitted noise of its
        likelihood, e.g. '{name}_noise_scale_{item}' (see
        LikelihoodGaussian); kinds add their own.
        """
        return {} if self._likelihood is None else self._likelihood.pdescs()

    def offset_params(self) -> tuple[str | None, str | None, str | None]:
        """
        Return the names of the parameters of the offsets of its frame
        on the sky, of x, y and the velocity (see ObservationSky): None
        for those it does not fit; here, none.
        """
        return None, None, None

    def centre_params(self) -> tuple[tuple[str, str], ...]:
        """
        Return the names of the parameters of the centres (x and y) of what
        it adds to the galaxies it sees in the frame of the model (e.g. the
        masses of its lens), which its frame moves as the centres of their
        components (see ObservationGroup); none here.
        """
        return ()

    def takes_mass(self) -> bool:
        """
        Check whether its model takes the mass of the galaxies it sees
        beyond what their components take of it (e.g. profiles of their
        circular velocity, see ObservationProfile); here, it does not.
        """
        return False

    def sky_origin(self) -> tuple[float, float] | None:
        """
        Return the RA and Dec (degrees) of the origin of the frame of its
        model on the sky, from which the centres of what it sees are
        measured (see ObservationGroup); here, None: its data are not on
        the sky.
        """
        return None

    def keys(self) -> tuple[str, ...]:
        """Return the names of its data items (e.g. 'spectra')."""
        return self._keys

    def model_can_lack_values(self) -> bool:
        """
        Check whether its model can lack values (NaN) where the data have
        some: the likelihood then needs a residual for them (see
        LikelihoodGaussian).
        """
        return self._model_can_lack_values

    def flux_scale(self) -> tuple[str, FluxScale] | None:
        """
        Return the parameter of the flux scale of its model and how it
        scales the model (see ObservationSky), if it has one; none here.
        """
        return None

    @abc.abstractmethod
    def dataset(self, items: Mapping[str, Data]) -> Dataset:
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

    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """Return the floating type of its model."""

    @abc.abstractmethod
    def with_data(
            self, data: Dataset, likelihood: Likelihood | None = None
    ) -> Self:
        """
        Return a copy of it with other data (e.g. mock data; see
        GroupPlan.mock).

        Parameters
        ----------
        data : Dataset
            The data, of the observation.
        likelihood : Likelihood, optional
            Their likelihood; by default its own, if it has data, or that
            of data without one (see __init__).

        Returns
        -------
        Observation
            The copy.

        Raises
        ------
        ConfigError
            If the data are not of the observation, or do not go with the
            likelihood (see __init__).
        """

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
        ConfigError
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
    ) -> gridutils.GridData | gridutils.SpectraData | Table | np.ndarray:
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
    ) -> dict[str, gridutils.GridData]:
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
    ) -> dict[str, gridutils.GridData]:
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


observation_parser = parseutils.TypedParser(Observation)
