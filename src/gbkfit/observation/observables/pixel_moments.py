from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.dataset import Data, Dataset, DatasetPixelMoments
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import LineEmission
from gbkfit.model.components._units import DataUnit
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from . import _dcube, _detail, _moments
from .base import ModelData, Observable

if TYPE_CHECKING:
    from .._scene import Scene
    from ..foreground import Foreground
    from ..weights import Weights


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
    flux_floor : float, optional
        The spectra of the model whose moment 0 is not above this fraction
        of the largest of the maps (in absolute value) have no moments
        (they are masked): 1e-6 by default, from 0 to below 1. It keeps
        out spectra of hardly any flux, whose moments are noise (e.g. of
        the rounding of convolutions); what the data lack is for their
        mask and their likelihood (e.g. upper limits).
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
    spec_medium : str, optional
        The medium of the wavelengths of a rest wavelength: 'vacuum' (the
        default) or 'air' (see gridutils.make_medium).
    spec_convention : str, optional
        The convention of the velocities of the maps, as the data were
        measured: of a rest wavelength, 'optical' (its default) or 'log';
        of a rest frequency, 'radio'; without a rest, 'relativistic' (see
        gridutils.make_convention). A rest or the convention is required:
        none is right for all data (e.g. radio for HI, optical for lines
        of stars and ionised gas).
    method : str, optional
        How the maps are measured from the spectra, as the maps of the
        data were: 'moments'; 'gaussian_fit' (the flux, centre and
        dispersion of a Gaussian fitted to each, for the orders 0 to 2);
        or 'gauss_hermite' (of a Gauss-Hermite series fitted to each, for
        the orders 0 to 4: its flux, centre, dispersion, h3 and h4, with
        the polynomials of van der Marel & Franx 1993, as pPXF).
    median_size : int, optional
        The size (an odd number of pixels, at least 3) of the median filter
        that smoothed the cube on the sky before the maps of the data were
        measured, which smooths the spectra of the model as well (e.g. 3,
        of cubes of KMOS3D): each channel takes the median of the square of
        the size about each pixel, with zero beyond the edges (see DCube).
        The median of a sum is not the sum of the medians: the channels of
        the model (spec_step) are then those of that cube. None by default.

    Raises
    ------
    ConfigError
        If the orders are not valid for the method, flux_floor is not from
        0 to below 1, neither spec_rest nor spec_convention is given,
        spec_size is less than 1, spec_step is not positive, or
        median_size is not an odd number of at least 3.
    """

    dataset_class = DatasetPixelMoments

    emission = LineEmission

    # Moment maps have no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'pixel_moments'

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
            flux_floor: float = 1e-6,
            spec_size: int | None = None,
            spec_step: float = _moments.SPEC_STEP,
            spec_rval: float | None = None,
            spec_rest: str | astropy.units.Quantity | None = None,
            spec_medium: str | None = None,
            spec_convention: str | None = None,
            method: str = 'moments',
            median_size: int | None = None
    ) -> 'PixelMoments':
        """
        Make the observable of moment maps, on their grid and of their
        orders.

        Parameters
        ----------
        dataset : DatasetPixelMoments
            The moment maps.
        flux_floor, spec_step, spec_rest, spec_medium, spec_convention, \
                method, median_size : optional
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
            flux_floor=flux_floor, orders=dataset.orders(),
            spec_step=spec_step, spec_rest=spec_rest,
            spec_medium=spec_medium, spec_convention=spec_convention,
            method=method, median_size=median_size)

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
            flux_floor=self._flux_floor,
            orders=self._orders,
            spec_size=self._spec_size,
            spec_step=self._spec_step,
            spec_rval=self._spec_rval,
            spec_rest=_detail.dump_rest(self._spec_rest),
            spec_medium=self._spec_medium,
            spec_convention=self._spec_convention,
            method=self._method,
            median_size=self._median_size), data)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[float] = (1, 1),
            rpix: Sequence[float] | None = None,
            rval: Sequence[float] = (0, 0),
            rota: float = 0,
            flux_floor: float = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            spec_size: int | None = None,
            spec_step: float = _moments.SPEC_STEP,
            spec_rval: float = 0,
            spec_rest: str | astropy.units.Quantity | None = None,
            spec_medium: str | None = None,
            spec_convention: str | None = None,
            method: str = 'moments',
            median_size: int | None = None
    ):
        super().__init__(size, step, rpix, rval, rota)
        _detail.check_median_size(median_size)
        self._median_size = median_size
        self._orders = _moments.check_moment_options(
            orders, flux_floor, method, spec_size, spec_step)
        if spec_size is None:
            spec_size = _moments.default_spec_size(spec_step)
        self._flux_floor = flux_floor
        self._method = method
        self._spec_size = spec_size
        self._spec_step = spec_step
        self._spec_rval = spec_rval
        self._spec_rest = gridutils.make_rest(spec_rest)
        self._spec_medium = gridutils.make_medium(
            spec_medium, self._spec_rest)
        _moments.check_velocities(spec_rest, spec_convention)
        self._spec_convention = gridutils.make_convention(
            spec_convention, self._spec_rest)

    def keys(self) -> tuple[str, ...]:
        return tuple(f'moment{i}' for i in self._orders)

    def dataset(self, items: Mapping[str, Data]) -> DatasetPixelMoments:
        return DatasetPixelMoments(
            {order: items[f'moment{order}'] for order in self.orders()},
            **_detail.dataset_grid_options(self.grid()))

    def data_unit(self, dataset: Dataset | None) -> DataUnit | None:
        return _detail.data_unit(
            dataset, 'moment0' if 0 in self._orders else None,
            abs(self.step()[0] * self.step()[1]), True)

    def light_items(self) -> tuple[str, ...]:
        return ('moment0',) if 0 in self._orders else ()

    def model_can_lack_values(self) -> bool:
        # (moment 0, the flux, always has a value)
        return any(order >= 1 for order in self._orders)

    def orders(self) -> tuple[int, ...]:
        return self._orders

    def flux_floor(self) -> float:
        return self._flux_floor

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

    def spec_medium(self) -> str | None:
        return self._spec_medium

    def spec_convention(self) -> str | None:
        return self._spec_convention

    def check_instrument(self, instrument: Instrument) -> None:
        _moments.warn_unmasked_noise(self._flux_floor, instrument)

    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            foreground: 'Foreground',
            instrument: Instrument,
            scale: Sequence[int],
            dtype: np.dtype,
            fft_dtype: np.dtype,
            weights: 'Weights | None',
            flux_scale: str | None
    ) -> 'PixelMomentsPlan':
        if weights is not None and self._method != 'moments':
            raise ConfigError(
                f"the method {self._method} does not support weights yet")
        dcube = _moments.spectra_dcube(
            self, scale, instrument, dtype, fft_dtype, self._median_size)
        return PixelMomentsPlan(
            self, dcube, driver, scene, foreground, dtype, weights,
            flux_scale)


class PixelMomentsPlan(_detail.DCubePlanBase):

    def __init__(
            self,
            moments: PixelMoments,
            dcube: _dcube.DCube,
            driver: Driver,
            scene: 'Scene',
            foreground: 'Foreground',
            dtype: np.dtype,
            weights: 'Weights | None',
            flux_scale: str | None
    ):
        super().__init__(
            dcube, driver, scene, foreground, dtype, weights, flux_scale)
        self._moments = _moments.MomentsPlan(
            driver, moments.size(), moments.orders(), moments.flux_floor(),
            moments.method(), dtype)

    def _measure(self) -> ModelData:
        # The moment maps, one mask shared by all of them, and the weight
        # map of each moment
        plan = self._dcube_plan
        return self._moments.evaluate(
            self._dcube.step(), self._dcube.zero(), plan.dcube(),
            plan.wcube())
