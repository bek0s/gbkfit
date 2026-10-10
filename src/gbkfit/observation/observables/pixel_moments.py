from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import Dataset, DatasetPixelMoments
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model.base import Model, ModelSCube, Selection
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail, _moments
from .base import ModelData, Observable

if TYPE_CHECKING:
    from ..foreground import Foreground


__all__ = [
    'PixelMoments'
]


class PixelMoments(Observable):
    """
    Moments of the spectra on a grid of pixels: the moment maps of some
    orders (0 to 7) of a spectral cube of the model, seen through the
    instrument, as the data items moment0 to moment7.

    The moments are measured from a spectral cube with the spatial axes
    of the maps, and a spectral axis of spec_size channels of spec_step
    centred on spec_rval (km/s).

    Its configuration has the options below, but those of the grid and
    the orders when it has data, which give them (see from_data).

    Parameters
    ----------
    size : Sequence of int
        The number of pixels of the maps (nx, ny).
    step, rpix, rval : Sequence of float, optional
        The world coordinates of the maps (see gridutils.Coords); rpix is
        the centre by default.
    rota : float, optional
        The rotation of the maps on the sky.
    mask_cutoff : float, optional
        The spectra whose moment 0 is not above it are masked.
    orders : Sequence of int, optional
        The orders of the moments.
    spec_size : int, optional
        The number of channels of the spectral axis; by default, it spans
        1000 km/s.
    spec_step : float, optional
        The width of the channels (km/s).
    spec_rval : float, optional
        The centre of the spectral axis (km/s).
    spec_rest : str or Quantity, optional
        The rest wavelength or frequency of the velocities of the spectral
        axis, if known (see gridutils.make_rest).
    method : str, optional
        How the maps are measured from the spectra, as the maps of the
        data were: 'moments', or 'gaussian_fit' (the flux, centre and
        dispersion of a Gaussian fitted to each, for the orders 0 to 2).

    Raises
    ------
    ConfigError
        If the orders are not valid for the method, mask_cutoff is None or
        negative, spec_size is less than 1, or spec_step is not positive.
    """

    dataset_class = DatasetPixelMoments

    # Moment maps have no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'pixel_moments'

    @staticmethod
    def is_compatible(model: Model) -> bool:
        return isinstance(model, ModelSCube)

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The grid and the orders
        return super().options_from_data(dataset) + ('orders',)

    @classmethod
    def load(
            cls, info: dict[str, Any],
            dataset: DatasetPixelMoments | None = None
    ) -> 'PixelMoments':
        return _detail.load_observable(cls, info, dataset, 2)

    @classmethod
    def from_data(
            cls,
            dataset: DatasetPixelMoments,
            mask_cutoff: float = 1e-6,
            spec_size: int | None = None,
            spec_step: float = _moments.SPEC_STEP,
            spec_rval: float | None = None,
            spec_rest: str | astropy.units.Quantity | None = None,
            method: str = 'moments'
    ) -> 'PixelMoments':
        """
        Make the observable of moment maps, on their grid and of their
        orders.

        Parameters
        ----------
        dataset : DatasetPixelMoments
            The moment maps.
        mask_cutoff, spec_step, spec_rest, method : optional
            As those of PixelMoments.
        spec_size, spec_rval : optional
            As those of PixelMoments. By default, with moment 1, the
            spectral axis covers the velocities of the data: the range of
            moment 1, and three times the largest dispersion (moment 2,
            but at least 100 km/s) on each side.

        Returns
        -------
        PixelMoments
            The observable.

        Raises
        ------
        ConfigError
            If the dataset is not of moment maps, or the options are not
            valid.
        """
        cls._require_dataset_class(dataset)
        return cls(
            **_detail.grid_options(dataset.grid()),
            **_moments.spectral_axis_from_data(
                dataset, spec_step, spec_size, spec_rval),
            mask_cutoff=mask_cutoff, orders=dataset.orders(),
            spec_step=spec_step, spec_rest=spec_rest, method=method)

    def dump(
            self,
            data: DatasetPixelMoments | None = None,
            prefix: str = '',
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.without_options_from_data(self, dict(
            type=self.type(),
            size=self.size(),
            step=self.step(),
            rpix=self.rpix(),
            rval=self.rval(),
            rota=self.rota(),
            mask_cutoff=self._mask_cutoff,
            orders=self._orders,
            spec_size=self._spec_size,
            spec_step=self._spec_step,
            spec_rval=self._spec_rval,
            spec_rest=_detail.dump_rest(self._spec_rest),
            method=self._method), data)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[float] = (1, 1),
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] = (0, 0),
            rota: float = 0,
            mask_cutoff: float = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            spec_size: int | None = None,
            spec_step: float = _moments.SPEC_STEP,
            spec_rval: float = 0,
            spec_rest: str | astropy.units.Quantity | None = None,
            method: str = 'moments'
    ):
        super().__init__(size, step, rpix, rval, rota)
        self._orders = _moments.check_moment_options(
            orders, mask_cutoff, method, spec_size, spec_step)
        if spec_size is None:
            spec_size = _moments.default_spec_size(spec_step)
        self._mask_cutoff = mask_cutoff
        self._method = method
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval
        self._spec_rest = gridutils.make_rest(spec_rest)

    def keys(self) -> tuple[str, ...]:
        return tuple(f'moment{i}' for i in self._orders)

    def orders(self) -> tuple[int, ...]:
        return self._orders

    def mask_cutoff(self) -> float:
        return self._mask_cutoff

    def method(self) -> str:
        return self._method

    def spec_size(self) -> int:
        return self._spec_size

    def spec_step(self) -> float:
        return self._spec_step

    def spec_rval(self) -> float:
        return self._spec_rval

    def spec_rest(self) -> astropy.units.Quantity | None:
        return self._spec_rest

    def check_instrument(self, instrument: Instrument) -> None:
        _moments.warn_unmasked_noise(self._mask_cutoff, instrument)

    def plan(
            self,
            driver: Driver,
            model: Model,
            foreground: 'Foreground',
            instrument: Instrument,
            scale: Sequence[int],
            dtype: np.dtype,
            selection: Selection
    ) -> 'PixelMomentsPlan':
        if self._method == 'gaussian_fit' and model.has_weights():
            raise ConfigError(
                "the method gaussian_fit does not support models with "
                "weights (wtraits) yet")
        dcube = _moments.spectra_dcube(self, scale, instrument, dtype)
        return PixelMomentsPlan(
            self, dcube, driver, model, foreground, dtype, selection)


class PixelMomentsPlan(_detail.DCubePlanBase):

    def __init__(
            self,
            moments: PixelMoments,
            dcube: _dcube.DCube,
            driver: Driver,
            model: Model,
            foreground: 'Foreground',
            dtype: np.dtype,
            selection: Selection
    ):
        super().__init__(
            dcube, driver, model, foreground, dtype, selection)
        self._moments = _moments.MomentsPlan(
            driver, moments.size(), moments.orders(), moments.mask_cutoff(),
            moments.method(), dtype)

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            out_extra: dict[str, Any] | None
    ) -> ModelData:
        self._evaluate_cube(
            params, out_extra, _dcube.cube_extra, _dcube.cube_extra)
        # The moment maps, one mask shared by all of them, and the weight
        # map of each moment
        plan = self._dcube_plan
        return self._moments.evaluate(
            self._dcube.step(), self._dcube.zero(), plan.dcube(),
            plan.wcube())
