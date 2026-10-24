import abc
import copy
from collections.abc import Mapping, Sequence
from typing import Self

import astropy.units
import numpy as np
from astropy.table import Table

from gbkfit.dataset import Data, Dataset
from gbkfit.driver import Driver
from gbkfit.model.components._units import DataUnit
from gbkfit.params import ParamDesc
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from ._scene import SeenGalaxies
from .likelihood import Likelihood, LikelihoodGaussian, correlated_pixels
from .observables import ObservablePlan


__all__ = [
    'Observation',
    'observation_parser'
]


class Observation(parseutils.TypedSerializable, abc.ABC):
    """
    Data of galaxies, and how the model of the galaxies gives them: what
    an observation group evaluates (see ObservationGroup), and an
    objective compares with the data under its likelihood (see
    Objective). Its kind says where the data are measured and how the
    model gives them (e.g. on the sky, through an instrument; see
    ObservationSky).

    Its data are named items of one shape (see keys), of a dataset of its
    kind, which the kind checks before it makes it. Its likelihood and its
    kind can have parameters of their own, named after it (see pdescs).
    A kind without spectra (no spectra_item) refuses a likelihood with
    free amplitudes (see LikelihoodGaussian).

    Parameters
    ----------
    keys : Sequence of str
        The names of its data items (e.g. 'spectra').
    driver : Driver, optional
        Where its model is evaluated; by default, the host.
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
    spectra_item : str, optional
        The name of its data item of spectra, if it has one: the spectra
        that free amplitudes scale (see LikelihoodGaussian).
    spectra_axis : int, optional
        The axis of the arrays of its spectra along which each spectrum
        runs; the first by default.
    model_can_lack_values : bool, optional
        Whether its model can lack values (NaN) where the data have some
        (e.g. the velocity of a spectrum below its mask_cutoff): the
        likelihood then needs a residual for them (see
        LikelihoodGaussian). False by default.

    Raises
    ------
    ConfigError
        If a name is invalid, there are no galaxies or they repeat, there
        is a likelihood without data, its model can lack values but its
        likelihood has no residual for them (see LikelihoodGaussian), its
        likelihood fits a noise but it has no name, its likelihood names
        items that it does not have or has amplitudes of spectra with
        upper limits on them (see _check_items), fits both the scale and
        the jitter of the noise of an item whose errors are all equal (see
        _check_noise), or the noise of an item is correlated where its
        data or its likelihood cannot take it (see _check_correlations).
    """

    def __init__(
            self,
            keys: Sequence[str],
            driver: Driver | None = None,
            galaxies: Sequence[str] | None = None,
            data: Dataset | None = None,
            likelihood: Likelihood | None = None,
            name: str | None = None,
            spectra_item: str | None = None,
            spectra_axis: int = 0,
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
        if driver is None:
            # (imported here: the host driver needs its native module)
            from gbkfit.driver.drivers.host import DriverHost
            driver = DriverHost()
        if data is None and likelihood is not None:
            raise ConfigError("a likelihood needs data")
        if data is not None:
            if likelihood is None:
                likelihood = LikelihoodGaussian()
            _check_items(likelihood, keys, spectra_item)
            _check_noise(likelihood, keys, data)
            _check_correlations(likelihood, keys, spectra_item, data)
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
                    "its mask_cutoff): its likelihood needs the residual of "
                    "such a value, the option missing_residual (e.g. 3; see "
                    "LikelihoodGaussian)")
        self._keys = keys
        self._driver = driver
        self._galaxies = galaxies
        self._data = data
        self._likelihood = likelihood
        self._name = name
        self._spectra_item = spectra_item
        self._spectra_axis = spectra_axis
        self._model_can_lack_values = model_can_lack_values

    def name(self) -> str | None:
        """Return its name, if any."""
        return self._name

    def galaxies(self) -> tuple[str, ...] | None:
        """Return the names of the galaxies it sees, if given."""
        return self._galaxies

    def driver(self) -> Driver:
        """Return the driver its model is evaluated on."""
        return self._driver

    def with_driver(self, driver: Driver) -> Self:
        """
        Return a copy of it whose model is evaluated on another driver;
        the copy shares the rest, its data too (e.g. copies of an
        objective on several devices; see Objective.with_driver).
        """
        observation = copy.copy(self)
        observation._driver = driver
        return observation

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

    def spectra_item(self) -> str | None:
        """
        Return the name of its data item of spectra, which free amplitudes
        scale (see LikelihoodGaussian), if it has one.
        """
        return self._spectra_item

    def spectra_axis(self) -> int:
        """
        Return the axis of the arrays of its data item of spectra along
        which each spectrum runs (see spectra_item).
        """
        return self._spectra_axis

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
        ObservationGroup.mock).

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
            The copy, whose model is evaluated on its driver.

        Raises
        ------
        ConfigError
            If the data are not of the observation, or do not go with the
            likelihood (see __init__).
        """

    @abc.abstractmethod
    def plan(self, seen: SeenGalaxies) -> ObservablePlan:
        """
        Plan the evaluation of its model (see ObservationGroup).

        Parameters
        ----------
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
            values: Mapping[str, np.ndarray],
            weights: Mapping[str, np.ndarray] | None,
            units: Mapping[str, astropy.units.UnitBase | None] | None
    ) -> dict[str, gridutils.GridData]:
        """
        Return images of values of its data items, as outputs (e.g. the
        dirty images of visibilities; see Observable.images), made on its
        driver; here, none.

        Parameters
        ----------
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
            self, weights: Mapping[str, np.ndarray] | None
    ) -> dict[str, gridutils.GridData]:
        """
        Return images of the PSF of its data (e.g. the dirty beam of
        visibilities; see Observable.psf_images), which its images are
        seen through, made on its driver; here, none.

        Parameters
        ----------
        weights : Mapping, optional
            The weights of the values of each data item (see images).

        Returns
        -------
        dict
            The images, by name.
        """
        return {}


def _check_items(
        likelihood: Likelihood,
        keys: Sequence[str],
        spectra_item: str | None
) -> None:
    """
    Raise ConfigError if the options of a likelihood name data items that
    an observation of the given items does not have (e.g. a typo in the
    names of the weights, the upper limits or the fitted noise; see
    LikelihoodGaussian), or it has amplitudes of the spectra and upper
    limits on them, which the amplitudes are solved without.
    """
    if unknown := sorted(likelihood.named_items() - set(keys)):
        raise ConfigError(
            f"the options of the likelihood name data items that the "
            f"observation does not have: {unknown}; its items are "
            f"{list(keys)}")
    if likelihood.amplitudes() is not None and spectra_item is not None \
            and likelihood.upper_limit(spectra_item) is not None:
        raise ConfigError(
            f"the amplitudes of the spectra '{spectra_item}' do not go with "
            f"upper limits on them: they are solved without the terms of "
            f"the limits")


def _check_noise(
        likelihood: Likelihood,
        keys: Sequence[str],
        data: Dataset
) -> None:
    """
    Raise ConfigError unless the data items whose scale and jitter of the
    noise a likelihood fits (see LikelihoodGaussian) have errors that
    differ between their values (with equal errors, or none, the two are
    one).
    """
    for key, noise in likelihood.fitted_noise().items():
        if not (noise.scale and noise.jitter):
            continue
        error = data[key].error()
        used = None if error is None else error[data[key].mask() != 0]
        if used is None or used.size == 0 or np.all(used == used[0]):
            raise ConfigError(
                f"the scale and the jitter of the noise of '{key}' cannot "
                f"both be fitted: its errors are all equal, or it has none, "
                f"so only (scale error)^2 + jitter^2 counts; fit one of them")


def _check_correlations(
        likelihood: Likelihood,
        keys: Sequence[str],
        spectra_item: str | None,
        data: Dataset
) -> None:
    """
    Raise ConfigError unless the given items of the data whose noise is
    correlated can be (see Dataset.noise_correlation), keep the pixels of
    correlated noise (see correlated_pixels), and have no upper limits or,
    for the item of spectra, amplitudes in their likelihood (see
    LikelihoodGaussian).
    """
    for key in keys:
        if data.noise_correlation(key) is None:
            continue
        correlated_pixels(key, data[key].mask())
        if likelihood.upper_limit(key) is not None:
            raise ConfigError(
                f"the noise of '{key}' is correlated, which does not go "
                f"with upper limits")
        if likelihood.amplitudes() is not None and key == spectra_item:
            raise ConfigError(
                f"the noise of '{key}' is correlated, which does not go "
                f"with amplitudes of spectra")


observation_parser = parseutils.TypedParser(Observation)
