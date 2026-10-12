import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, TypeAlias

import astropy.units
import numpy as np

from gbkfit.dataset import Dataset
from gbkfit.driver import DeviceArray, Driver
from gbkfit.instrument import Instrument
from gbkfit.model import Emission
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .._scene import Scene
    from ..foreground import Foreground
    from ..weights import Weights


__all__ = [
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

        Here, the grid (and the rest and the medium of its spectral axis);
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
            ('rest', 'medium') if cls.spectral_axis is not None else ())

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[float],
            rpix: Sequence[float] | None,
            rval: Sequence[float],
            rota: float,
            rest: str | astropy.units.Quantity | None = None,
            medium: str | None = None
    ):
        self._grid = gridutils.make_grid(
            tuple(size), step, rpix, rval, rota, self.spectral_axis, rest,
            medium)

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

    def rest(self) -> astropy.units.Quantity | None:
        """Return the rest of the spectral axis, if any."""
        return self._grid.coords.rest

    def medium(self) -> str | None:
        """
        Return the medium of the wavelengths of the spectral axis, if it
        has a rest wavelength.
        """
        return self._grid.coords.medium

    @abc.abstractmethod
    def keys(self) -> tuple[str, ...]:
        """Return the names of the data items (e.g. 'spectra')."""
        pass

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

    @abc.abstractmethod
    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            foreground: 'Foreground',
            instrument: Instrument,
            scale: Sequence[int],
            dtype: np.dtype,
            weights: 'Weights | None'
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
        scale : Sequence of int
            How many times the model is oversampled along each axis of the
            data.
        dtype : np.dtype
            The floating type of the model.
        weights : Weights, optional
            The spatial weights of the data.

        Returns
        -------
        ObservablePlan
            The plan.

        Raises
        ------
        ConfigError
            If it does not support weights, or with the foreground (see
            DCubePlanBase).
        """
        pass


class ObservablePlan(abc.ABC):
    """The evaluation of a model as an observable."""

    @abc.abstractmethod
    def evaluate(
            self,
            params: ParamValues,
            out_extra: dict[str, Any] | None
    ) -> ModelData:
        """
        Evaluate the model data.

        Parameters
        ----------
        params : dict
            The parameters of the model.
        out_extra : dict, optional
            Where the extra outputs go, if wanted; those of the model are
            prefixed 'model_'.

        Returns
        -------
        ModelData
            The model data.
        """
        pass


observable_parser = parseutils.TypedParser(Observable)
