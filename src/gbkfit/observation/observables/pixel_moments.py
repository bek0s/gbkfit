from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.dataset import Data, Dataset, DatasetPixelMoments
from gbkfit.driver import Driver
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, LightUnit, LineEmission
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError, check_options
from . import _dcube, _detail, _moments
from ._detail import MedianSize, Rota, Rval, Step
from ._moments import FluxFloor, Method
from .base import ModelData, Observable

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
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
    of the maps, and a spectral axis (spectral; see _moments.spectral_axis
    for its defaults).

    Its configuration has the options below, but those of the grid and
    the orders when it has data, which give them (see from_data).

    Parameters
    ----------
    size : Sequence of int
        The number of pixels of the maps (nx, ny).
    spectral : SpectralAxis
        The spectral axis of the cube the maps are measured from (see
        gridutils.SpectralAxis), and the convention of the velocities of
        the maps, as the data were measured, which is required (or its
        rest; see gridutils.check_convention): none is right for all data
        (e.g. radio for HI, optical for lines of ionised gas, log for the
        stars of pPXF). By default, its channels are of 1 km/s and span
        1000 km/s, centred on 0.
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
        the model (the step of spectral) are then those of that cube. None
        by default.

    Raises
    ------
    ConfigError
        If the orders are not valid for the method, flux_floor is not from
        0 to below 1, the convention of the velocities is not known, or
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
            spectral: gridutils.SpectralAxis,
            flux_floor: FluxFloor = 1e-6,
            method: Method = 'moments',
            median_size: MedianSize | None = None
    ) -> 'PixelMoments':
        """
        Make the observable of moment maps, on their grid and of their
        orders.

        Parameters
        ----------
        dataset : DatasetPixelMoments
            The moment maps.
        spectral : SpectralAxis
            As that of PixelMoments. Its size and rval by default, with
            moment 1, cover the velocities of the data: the range of moment
            1, and three times the largest dispersion (moment 2, but at
            least 100 km/s) on each side.
        flux_floor, method, median_size : optional
            As those of PixelMoments.

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
        check_options(PixelMoments.from_data, locals())
        return cls(
            **_detail.grid_options(dataset.grid()),
            spectral=_moments.spectral_axis(spectral, dataset),
            flux_floor=flux_floor, orders=dataset.orders(), method=method,
            median_size=median_size)

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
            spectral=self._spectral.dump(),
            flux_floor=self._flux_floor,
            orders=self._orders,
            method=self._method,
            median_size=self._median_size), data)

    def __init__(
            self,
            size: Sequence[int],
            spectral: gridutils.SpectralAxis,
            step: Step = (1, 1),
            rpix: Sequence[float] | None = None,
            rval: Rval = (0, 0),
            rota: Rota = 0,
            flux_floor: FluxFloor = 1e-6,
            orders: Sequence[int] = (0, 1, 2),
            method: Method = 'moments',
            median_size: MedianSize | None = None
    ):
        check_options(PixelMoments.__init__, locals())
        super().__init__(size, step, rpix, rval, rota)
        _detail.check_median_size(median_size)
        self._median_size = median_size
        self._orders = _moments.check_moment_orders(orders, method)
        self._spectral = _moments.spectral_axis(spectral)
        self._flux_floor = flux_floor
        self._method = method

    def keys(self) -> tuple[str, ...]:
        return tuple(f'moment{i}' for i in self._orders)

    def dataset(self, items: Mapping[str, Data]) -> DatasetPixelMoments:
        return DatasetPixelMoments(
            {order: items[f'moment{order}'] for order in self.orders()},
            **_detail.dataset_grid_options(self.grid()))

    def data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None = None
    ) -> DataUnit | None:
        return _detail.data_unit(
            dataset, 'moment0' if 0 in self._orders else None,
            abs(self.step()[0] * self.step()[1]), True, light)

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

    def spectral(self) -> gridutils.SpectralAxis:
        """
        Return the spectral axis of the cube the maps are measured from.
        """
        return self._spectral

    def check_observation(
            self,
            instrument: Instrument,
            oversampling: Sequence[int],
            pixel_weights: 'Weights | None'
    ) -> None:
        if pixel_weights is not None and self._method != 'moments':
            raise ConfigError(
                f"the method {self._method} does not support pixel_weights "
                f"yet")
        _moments.warn_unmasked_noise(self._flux_floor, instrument)

    def require_matching(self, dataset: Dataset) -> None:
        super().require_matching(dataset)
        _moments.check_moment_units(dataset, self._method)

    def plan(
            self,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            instrument: Instrument,
            oversampling: Sequence[int],
            dtype: np.dtype,
            fft_dtype: np.dtype,
            pixel_weights: 'Weights | None',
            flux_scale: str | None
    ) -> 'PixelMomentsPlan':
        dcube = _moments.spectra_dcube(
            self, oversampling, instrument, dtype, fft_dtype,
            self._median_size)
        return PixelMomentsPlan(
            self, dcube, driver, scene, line_of_sight, foreground, dtype,
            pixel_weights, flux_scale)


class PixelMomentsPlan(_detail.DCubePlanBase):

    def __init__(
            self,
            moments: PixelMoments,
            dcube: _dcube.DCube,
            driver: Driver,
            scene: 'Scene',
            line_of_sight: 'LineOfSight',
            foreground: 'Foreground',
            dtype: np.dtype,
            pixel_weights: 'Weights | None',
            flux_scale: str | None
    ):
        super().__init__(
            dcube, driver, scene, line_of_sight, foreground, dtype,
            pixel_weights, flux_scale)
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
