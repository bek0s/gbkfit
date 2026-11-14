"""
The observation of moments of the spectra on a grid of pixels: moment
maps.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetPixelMoments
from gbkfit.driver import Driver
from gbkfit.grid import SpectralAxis, grid_options
from gbkfit.instrument import Instrument
from gbkfit.model import Band, DataUnit, LightUnit, LineEmission
from gbkfit.moments import (
    FluxFloor, Method, MomentsPlan, check_moment_items, check_moment_orders,
    moment_item, split_moments)
from ..depth import Depth
from ..foreground import Foreground
from ..plan import ModelData
from ..weights import Weights
from . import _model_cube, _moments
from ._cube_plan import CubePlan
from ._data_units import data_unit
from ._options import (
    MedianSize, Rota, Rpix, Rval, Size, Step, check_median_size)
from .base import DType, ObservationSky, SkyOptions

if TYPE_CHECKING:
    from gbkfit.model import Scene


__all__ = [
    'ObservationPixelMoments'
]


class ObservationPixelMoments(ObservationSky):
    """
    Moments of the spectra on a grid of pixels: moment maps of a spectral
    cube of the model, seen through the instrument. The maps are of some
    orders (0 to 7), as the data items moment0 to moment7, but h3 and h4
    of the method gauss_hermite (see gbkfit.moments.moment_item).

    The moments are measured from a spectral cube with the spatial axes
    of the maps, and a spectral axis (spectral; see _moments.spectral_axis
    for its defaults).

    Its options are those below, then those of every observation on the
    sky (see ObservationSky). Its data give those of its grid and its
    orders (given too, they serve only without data); without data, size
    is required.

    Parameters
    ----------
    spectral : SpectralAxis
        The spectral axis of the cube the maps are measured from (see
        SpectralAxis), and the convention of the velocities of
        the maps, as the data were measured, which is required (or its
        rest; see check_convention): none is right for all data
        (e.g. radio for HI, optical for lines of ionised gas, log for the
        stars of pPXF). By default, its channels are of 1 km/s and span
        1000 km/s, centred on 0; with data, its size and rval cover the
        velocities of the data: the range of moment 1, and three times
        the largest dispersion (moment 2, but at least 100 km/s) on each
        side.
    size : int or Sequence of int, optional
        The number of pixels of the maps (nx, ny).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the maps (see Coords); without data, a
        step of 1, rval 0, and rpix the centre by default.
    rota : float, optional
        The rotation of the maps on the sky; 0 by default without data.
    flux_floor : float, optional
        The spectra of the model whose moment 0 is not above this fraction
        of the largest of the maps (in absolute value) have no moments
        (they are masked): 1e-6 by default, from 0 to below 1. It keeps
        out spectra of hardly any flux, whose moments are noise (e.g. of
        the rounding of convolutions); what the data lack is for their
        mask and their likelihood (e.g. upper limits).
    orders : Sequence of int, optional
        The orders of the moments; 0, 1 and 2 by default without data.
    method : str, optional
        How the maps are measured from the spectra, as the maps of the
        data were: 'moments'; 'gauss_fit' (the flux, centre and
        dispersion of a Gaussian fitted to each, for the orders 0 to 2);
        or 'gauss_hermite' (of a Gauss-Hermite series fitted to each, for
        the orders 0 to 4: its flux, centre, dispersion, h3 and h4, with
        the polynomials of van der Marel & Franx 1993, as pPXF).
    median_size : int, optional
        The size (an odd number of pixels, at least 3) of the median filter
        that smoothed the cube on the sky before the maps of the data were
        measured, which smooths the spectra of the model as well (e.g. 3,
        of cubes of KMOS3D): each channel takes the median of the square of
        the size about each pixel, with zero beyond the edges (see ModelCube).
        The median of a sum is not the sum of the medians: the channels of
        the model (the step of spectral) are then those of that cube. None
        by default.

    Raises
    ------
    ValueError
        If the orders are not valid for the method, flux_floor is not from
        0 to below 1, the convention of the velocities is not known,
        median_size is not an odd number of at least 3, or an option of
        every observation on the sky is not valid (see ObservationSky).
    """

    dataset_class = DatasetPixelMoments

    emission = LineEmission
    orbit_stars = None

    # Moment maps have no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'pixel_moments'

    @staticmethod
    def type_title() -> str:
        return 'Pixel moments (maps)'

    defaults_without_data = ObservationSky.defaults_without_data | dict(
        orders=(0, 1, 2))

    @classmethod
    def options_from_data(cls, dataset: Dataset) -> tuple[str, ...]:
        # The grid and the orders
        return super().options_from_data(dataset) + ('orders',)

    @classmethod
    def _options_from_data(
            cls, dataset: DatasetPixelMoments, options: dict[str, Any]
    ) -> dict[str, Any]:
        # (the grid and the orders of the maps, and the spectral axis of
        # the cube, which covers the velocities of the data unless its
        # size and rval are given, see _moments.spectral_axis)
        spectral = options['spectral']
        covered = {} if spectral.size() is not None \
            and spectral.rval() is not None else dict(
                spectral=_moments.spectral_axis(spectral, dataset))
        return grid_options(dataset.grid()) \
            | dict(orders=dataset.orders()) | covered

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(
            spectral=self._spectral.dump(),
            flux_floor=self._flux_floor,
            orders=self._orders,
            method=self._method,
            median_size=self._median_size)

    def _kind_options(self) -> dict[str, Any]:
        return grid_options(self.grid()) | dict(
            spectral=self._spectral, flux_floor=self._flux_floor,
            orders=self._orders, method=self._method,
            median_size=self._median_size)

    def __init__(
            self,
            spectral: SpectralAxis,
            size: Size | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            flux_floor: FluxFloor = 1e-6,
            orders: Sequence[int] | None = None,
            method: Method = 'moments',
            median_size: MedianSize | None = None,
            foreground: Foreground | None = None,
            instrument: Instrument | None = None,
            galaxies: Sequence[str] | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            band: str | Band | None = None,
            pixel_weights: Weights | None = None,
            flux_scale: str | None = None,
            offsets: Mapping[str, bool] | None = None,
            dataset: Dataset | None = None,
            oversampling: Sequence[int] | None = None,
            depth: Depth | None = None,
            dtype: DType | None = None,
            fft_dtype: DType = 'float64',
            name: str | None = None,
            unit: str | None = None
    ):
        check_options(ObservationPixelMoments.__init__, locals())
        options = self._options(dataset, dict(
            size=size, step=step, rpix=rpix, rval=rval, rota=rota,
            spectral=spectral, orders=orders))
        check_median_size(median_size)
        self._median_size = median_size
        self._orders = check_moment_orders(options.pop('orders'), method)
        self._spectral = _moments.spectral_axis(options.pop('spectral'))
        self._flux_floor = flux_floor
        self._method = method
        super().__init__(
            self._make_grid(**options), foreground, instrument, galaxies,
            components, lines, band, pixel_weights, flux_scale, offsets,
            dataset, oversampling, depth, dtype, fft_dtype, name, unit)

    def keys(self) -> tuple[str, ...]:
        return tuple(moment_item(i, self._method) for i in self._orders)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetPixelMoments:
        return DatasetPixelMoments(
            *split_moments(items), **grid_options(self.grid(), sizes=False))

    def _data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None
    ) -> DataUnit | None:
        return data_unit(
            dataset, 'moment0' if 0 in self._orders else None,
            abs(self.step()[0] * self.step()[1]), True, light)

    def light_items(self) -> tuple[str, ...]:
        return ('moment0',) if 0 in self._orders else ()

    def model_can_have_missing_values(self) -> bool:
        # (moment 0, the flux, always has a value)
        return any(order >= 1 for order in self._orders)

    def orders(self) -> tuple[int, ...]:
        """Return the orders of the moments, sorted."""
        return self._orders

    def flux_floor(self) -> float:
        """
        Return the fraction of the largest moment 0 of the model that the
        moment 0 of a spectrum must be above to have moments.
        """
        return self._flux_floor

    def method(self) -> str:
        """
        Return how the moments are measured (see
        gbkfit.moments.MOMENT_METHODS).
        """
        return self._method

    def spectral(self) -> SpectralAxis:
        """
        Return the spectral axis of the cube the maps are measured from.
        """
        return self._spectral

    def _check_options(self, options: SkyOptions) -> None:
        if options.pixel_weights is not None and self._method != 'moments':
            raise ValueError(
                f"the method {self._method} does not support pixel_weights "
                f"yet")
        _moments.warn_unmasked_noise(self._flux_floor, options.instrument)

    def _check_scene(self, scene: 'Scene') -> None:
        _moments.check_one_line(scene)
        super()._check_scene(scene)

    def check_dataset(self, dataset: Dataset) -> None:
        check_moment_items(dataset, self._method)
        super().check_dataset(dataset)

    def _plan(self, driver: Driver, scene: 'Scene') -> 'PixelMomentsPlan':
        options = self._sky_options
        model_cube = _moments.spectra_model_cube(
            self, options, self._median_size)
        return PixelMomentsPlan(self, model_cube, driver, scene, options)


class PixelMomentsPlan(CubePlan):

    def __init__(
            self,
            moments: ObservationPixelMoments,
            model_cube: _model_cube.ModelCube,
            driver: Driver,
            scene: 'Scene',
            options: SkyOptions
    ):
        super().__init__(model_cube, driver, scene, options)
        self._moments = MomentsPlan(
            driver, moments.size(), moments.orders(), moments.flux_floor(),
            moments.method(), options.dtype)

    def _measure(self, draw: bool = False) -> ModelData:
        # The moment maps, with the mask and the weight map they share: of
        # the model, or of a mock draw (data, see _measure_draw)
        plan = self._model_cube_plan
        grid = self._model_cube.grid()
        measure = self._moments.measure if draw else self._moments.evaluate
        return _moments.model_data(measure(
            grid.coords.step, grid.zero(), plan.cube(), plan.weights()))

    def _measure_draw(self) -> ModelData:
        return self._measure(draw=True)
