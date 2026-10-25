import abc
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Self, TypeAlias

import astropy.units
import numpy as np

from gbkfit.dataset import Data, Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, Emission, Extras, GalaxyState
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .._scene import Scene
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'FluxScale',
    'HostModelData',
    'ModelData',
    'Observable',
    'ObservablePlan',
    'observable_parser'
]


# The model data of an observable: for each of its data items (e.g.
# 'spectra'), the arrays of the model ('d'; NaN where it has no value),
# of its mask ('m'; an output, which the objective does not use) and of
# its weights ('w') on the driver, or None
ModelData: TypeAlias = dict[str, dict[str, DeviceArray | None]]

# Model data copied to the host
HostModelData: TypeAlias = dict[str, dict[str, np.ndarray | None]]


class FluxScale(NamedTuple):
    """
    How the values of a flux scale of an observation scale its model (see
    ObservationSky): the data items of light that they multiply, the axis
    of their arrays along which each value is one (one value at each point
    of the other axes, in their order, e.g. one for each spectrum), or None
    for one value for all, and the number of values.
    """
    items: tuple[str, ...]
    axis: int | None
    size: int


class Observable(parseutils.TypedSerializable, abc.ABC):
    """
    What a dataset measures, made from a model seen through an
    instrument: the form of the data (their items, see keys) and their
    grid.

    A subclass declares the index of the spectral axis of its data
    (spectral_axis, None if none) and the class of its datasets
    (dataset_class). Its from_data makes it from data, which give some of
    its options (see options_from_data), and its dump(data) leaves those
    out. The files of its options (e.g. regions without data) are named
    with the prefix of the dump, as those of the datasets.

    Parameters
    ----------
    size : Sequence of int
        The number of pixels of each axis of the grid.
    step, rpix, rval : Sequence of float
        The world coordinates of the grid (see gridutils.Coords); rpix is
        the centre if None.
    rota : float
        The rotation of the grid on the sky.
    rest : str or Quantity, optional
        The rest of the spectral axis, if any (see gridutils.make_rest).
    medium : str, optional
        The medium of the wavelengths of a rest wavelength (see
        gridutils.make_medium).
    convention : str, optional
        The convention of the velocities of the rest (see
        gridutils.make_convention).
    """

    spectral_axis: int | None

    dataset_class: type[Dataset]

    # The emission it sees: in a band (BandEmission) for an image, lines
    # (LineEmission) for a spectral cube and what is made of one
    emission: type[Emission]

    @classmethod
    @abc.abstractmethod
    def from_data(cls, dataset: Dataset, **options: Any) -> Self:
        """
        Make the observable of a dataset.

        Parameters
        ----------
        dataset : Dataset
            The data, of the dataset class of the observable. They give its
            grid and its other options of options_from_data.
        **options
            The other options of the observable.

        Returns
        -------
        Observable
            The observable.

        Raises
        ------
        ConfigError
            If the dataset is not of the dataset class of the observable.
        """
        pass

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        """
        Return the options of the observable that a dataset gives.

        Here, the grid (and the rest, the medium and the convention of its
        spectral axis);
        observables of other data override this.

        Parameters
        ----------
        dataset : Dataset
            The data.

        Returns
        -------
        tuple of str
            The names of the options.
        """
        grid = ('size', 'step', 'rpix', 'rval', 'rota')
        return grid + (
            ('rest', 'medium', 'convention')
            if cls.spectral_axis is not None else ())

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[float],
            rpix: Sequence[float] | None,
            rval: Sequence[float],
            rota: float,
            rest: str | astropy.units.Quantity | None = None,
            medium: str | None = None,
            convention: str | None = None
    ):
        self._grid = gridutils.make_grid(
            tuple(size), step, rpix, rval, rota, self.spectral_axis, rest,
            medium, convention)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the data."""
        return self._grid

    def size(self) -> tuple[int, ...]:
        """Return the number of pixels of each axis of the grid."""
        return self._grid.size

    def step(self) -> tuple[float, ...]:
        """Return the step of each axis of the grid."""
        return self._grid.coords.step

    def rpix(self) -> tuple[float, ...]:
        """Return the reference pixel of each axis of the grid."""
        return self._grid.coords.rpix

    def rval(self) -> tuple[float, ...]:
        """Return the reference value of each axis of the grid."""
        return self._grid.coords.rval

    def rota(self) -> float:
        """Return the rotation of the grid on the sky."""
        return self._grid.coords.rota

    def sky_rval(self) -> tuple[float, float]:
        """
        Return the RA and Dec (degrees) of the origin of the frame of its
        model, from which the centres of what it sees are measured (see
        ObservationGroup): those of the reference pixel of its grid.
        """
        rval = self.rval()
        return float(rval[0]), float(rval[1])

    def rest(self) -> astropy.units.Quantity | None:
        """Return the rest of the spectral axis, if any."""
        return self._grid.coords.rest

    def medium(self) -> str | None:
        """
        Return the medium of the wavelengths of the spectral axis, if it
        has a rest wavelength.
        """
        return self._grid.coords.medium

    def convention(self) -> str | None:
        """
        Return the convention of the velocities of the spectral axis, if it
        has a rest (see gridutils.make_convention).
        """
        return self._grid.coords.convention

    @abc.abstractmethod
    def keys(self) -> tuple[str, ...]:
        """Return the names of the data items (e.g. 'spectra')."""
        pass

    @abc.abstractmethod
    def model_can_lack_values(self) -> bool:
        """
        Check whether its model can lack values (NaN) where the data have
        some (e.g. the velocity of a spectrum below its flux_floor): the
        likelihood then needs a residual for them (see LikelihoodGaussian).
        """
        pass

    @abc.abstractmethod
    def dataset(self, items: Mapping[str, Data]) -> Dataset:
        """
        Make the dataset of data items measured as this observable (e.g.
        mock data): the inverse of from_data.

        Parameters
        ----------
        items : Mapping
            The data items, by name (see keys).

        Returns
        -------
        Dataset
            The dataset, of the dataset class, where the observable is
            modelled.
        """
        pass

    @abc.abstractmethod
    def data_unit(self, dataset: Dataset | None) -> DataUnit | None:
        """
        Return the unit of the light of its data, which its model takes.

        Parameters
        ----------
        dataset : Dataset, optional
            The data.

        Returns
        -------
        DataUnit or None
            The unit of the data item of light (e.g. 'spectra' or
            'moment0'; see DataUnit), with its beam and the area of its
            pixels; None without data, or without an item of light (e.g.
            moments of order 1 and above only: velocities).
        """
        pass

    @abc.abstractmethod
    def light_items(self) -> tuple[str, ...]:
        """
        Return the names of its data items of light (e.g. 'spectra' or
        'moment0'): those that are times the flux of the model, which a
        flux scale multiplies (see flux_scales); none for data of
        velocities only.
        """
        pass

    def flux_scales(self) -> dict[str, FluxScale]:
        """
        Return the flux scales that the model of an observation of it can
        have (see ObservationSky), by kind: 'one', of its items of light,
        if any. Observables of spectra add 'per_spectrum' (or, of the lines
        of an image along an axis, 'per_row').
        """
        items = self.light_items()
        return {} if not items else dict(one=FluxScale(items, None, 1))

    def check_instrument(self, instrument: Instrument) -> None:
        """
        Check the instrument that the observable is seen through.

        It warns about the options of the observable that the instrument
        makes useless (see parseutils.warn). Here, any instrument suits;
        observables override this.

        Parameters
        ----------
        instrument : Instrument
            The instrument.

        Raises
        ------
        ConfigError
            If the observable cannot be seen through the instrument.
        """
        pass

    def require_matching(self, dataset: Dataset) -> None:
        """
        Check that a dataset holds the data of this observable.

        The dataset must be of the dataset class of the observable, have
        its data items, and have been measured where the observable is
        modelled.

        Parameters
        ----------
        dataset : Dataset
            The data.

        Raises
        ------
        ConfigError
            If the dataset does not hold the data of this observable.
        """
        self._require_dataset_class(dataset)
        if set(dataset.keys()) != set(self.keys()):
            desc = parseutils.make_typed_desc(self.__class__, 'observable')
            raise ConfigError(
                f"{desc} has the data items {sorted(self.keys())}, but the "
                f"data have {sorted(dataset.keys())}")
        self._require_matching_coordinates(dataset)

    @classmethod
    def _require_dataset_class(cls, dataset: Dataset) -> None:
        """Raise ConfigError unless the dataset is of the dataset class."""
        if not isinstance(dataset, cls.dataset_class):
            desc = parseutils.make_typed_desc(cls, 'observable')
            dataset_desc = parseutils.make_typed_desc(
                dataset.__class__, 'dataset')
            raise ConfigError(f"{desc} cannot be compared with {dataset_desc}")

    def _require_matching_coordinates(self, dataset: Dataset) -> None:
        """
        Raise ConfigError unless the data were measured on the grid of this
        observable. Observables of other data override this.
        """
        if dataset.grid() != self._grid:
            desc = parseutils.make_typed_desc(self.__class__, 'observable')
            raise ConfigError(
                f"the data are on the grid {dataset.grid()}, but {desc} "
                f"is on the grid {self._grid}")

    def output(
            self, data: np.ndarray
    ) -> gridutils.GridData | gridutils.SpectraData | np.ndarray:
        """
        Return an array of the form of the data as an output.

        Here, on the grid of the observable, with its world coordinates;
        observables of other data override this.

        Parameters
        ----------
        data : np.ndarray
            The array: the model, its mask or weights, or a residual.

        Returns
        -------
        GridData or SpectraData or np.ndarray
            The output.
        """
        return gridutils.GridData(
            data, self._grid.coords, self._grid.spectral_axis)

    def images(
            self,
            driver: Driver,
            values: Mapping[str, np.ndarray],
            weights: Mapping[str, np.ndarray] | None,
            units: Mapping[str, astropy.units.UnitBase | None] | None
    ) -> dict[str, gridutils.GridData]:
        """
        Return images of values of its data items, as outputs (e.g. the
        dirty images of visibilities).

        Here, none: observables whose data are images of the sky give
        them as they are (see output); others override this.

        Parameters
        ----------
        driver : Driver
            The driver that makes them.
        values : Mapping
            The values of each data item (e.g. those of the data, of the
            model, or of the data minus the model), NaN where they have
            none.
        weights : Mapping, optional
            The weights of the values of each data item (e.g. 1 / error^2);
            all 1 if None.
        units : Mapping, optional
            The unit of the values of each data item, if known (e.g. those
            of the data).

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
        visibilities), which its images (see images) are seen through.

        Here, none; observables with images override this.

        Parameters
        ----------
        driver : Driver
            The driver that makes them.
        weights : Mapping, optional
            The weights of the values of each data item (see images).

        Returns
        -------
        dict
            The images, by name.
        """
        return {}

    @abc.abstractmethod
    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            foreground: 'Foreground',
            instrument: Instrument,
            oversampling: Sequence[int],
            dtype: np.dtype,
            fft_dtype: np.dtype,
            pixel_weights: 'Weights | None',
            flux_scale: str | None
    ) -> 'ObservablePlan':
        """
        Plan the evaluation of a scene (what an observation sees) as this
        observable.

        Parameters
        ----------
        driver : Driver
            The driver it is evaluated on.
        scene : Scene
            What is seen: galaxies along a line of sight.
        foreground : Foreground
            What the light meets before the telescope.
        instrument : Instrument
            The instrument.
        oversampling : Sequence of int
            How many times the model is oversampled along each axis of the
            data.
        dtype : np.dtype
            The floating type of the model.
        fft_dtype : np.dtype
            The floating type of the FFT convolutions of the model (at
            least as precise as dtype).
        pixel_weights : Weights, optional
            The pixel weights of the data.
        flux_scale : str, optional
            The name of the parameter that scales the flux of the model, if
            one value (the flux scale 'one' of the observation; see
            ObservationSky).

        Returns
        -------
        ObservablePlan
            The plan.

        Raises
        ------
        ConfigError
            If it does not support pixel weights, or with the foreground (see
            DCubePlanBase).
        """
        pass


class ObservablePlan(abc.ABC):
    """The evaluation of a model as an observable."""

    @abc.abstractmethod
    def evaluate(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            out_extra: Extras | None
    ) -> ModelData:
        """
        Evaluate the model data.

        Parameters
        ----------
        params : dict
            The parameters of the model.
        galaxies : Sequence of GalaxyState
            Each galaxy it sees in the evaluation: its mass, and what it
            gives its components (see GalaxyStage.evaluate), in the order
            of the galaxies it sees.
        out_extra : Extras, optional
            Where the extra outputs go, if wanted: those of the instrument
            (e.g. the model cube), and those of the model (with the ids of
            their galaxies, and of their components).

        Returns
        -------
        ModelData
            The model data.
        """
        pass

    @abc.abstractmethod
    def mock(
            self,
            params: ParamValues,
            galaxies: Sequence[GalaxyState],
            noise: 'Noise',
            rng: np.random.Generator,
            ndraws: int
    ) -> tuple[list[HostModelData], dict[str, np.ndarray] | None]:
        """
        Draw mock data: the model data measured from the data that the
        instrument records of the model plus noise, ndraws times, each
        with its own noise, and their exact errors. The model is evaluated
        once: the noise of every draw and the errors are of its data
        without noise (the signal; see Noise).

        Parameters
        ----------
        params, galaxies
            As for evaluate.
        noise : Noise
            The noise of the recorded data.
        rng : Generator
            The random numbers of the noise.
        ndraws : int
            The number of draws.

        Returns
        -------
        list of HostModelData
            The model data of each draw, on the host: of each data item,
            the data ('d') and their mask ('m') as evaluate gives them;
            no weights.
        dict or None
            The error of each value of each data item, by name, if known
            exactly (see has_exact_errors): the dispersion of the noise of
            the signal.

        Raises
        ------
        ConfigError
            If the noise does not suit the recorded data (see
            Noise.draw).
        """
        pass

    @abc.abstractmethod
    def has_exact_errors(self) -> bool:
        """
        Check whether the errors of mock data are known exactly (see
        mock): if the data items are the recorded data plus the noise
        (e.g. spectra), the dispersion of the noise; not if they are
        measured from them (e.g. moments), when their errors are those of
        the scatter of draws.
        """
        pass


observable_parser = parseutils.TypedParser(Observable)
