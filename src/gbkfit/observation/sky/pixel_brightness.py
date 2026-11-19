"""
The observation of brightness on a grid of pixels: an image.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetPixelBrightness
from gbkfit.driver import Driver
from gbkfit.grid import Grid, GridData, grid_options
from gbkfit.instrument import Instrument
from gbkfit.lensing import SourceGrid
from gbkfit.model import Band, DataUnit, LightUnit
from ..depth import Depth
from ..noise import Noise
from ..plan import ModelData, ModelItem
from ..weights import Weights
from . import _model_cube
from ._cube_plan import CubePlan, SkyDrawing
from ._data_units import data_unit
from ._options import (
    MedianSize, Rota, Rpix, Rval, Size, Step, check_median_size)
from .base import DType, ObservationSky, SkyOptions, unmasked_pixels

__all__ = [
    'ObservationPixelBrightness'
]


def _image_extra(data: np.ndarray, grid: Grid) -> GridData:
    """
    Make an extra output on a grid of ModelCube an image (its one
    channel).
    """
    return GridData(data[0], grid.spatial().coords, None)


class ObservationPixelBrightness(ObservationSky):
    """
    Brightness on a grid of pixels: an image of the model, seen through
    the instrument (which has no LSF).

    Its options are those below, then those of every observation on the
    sky (see ObservationSky). Its data give those of its grid (given too,
    they serve only without data); without data, size is required.

    Parameters
    ----------
    size : int or Sequence of int, optional
        The number of pixels of the image (nx, ny).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the image (see Coords); without data, a
        step of 1, rval 0, and rpix the centre by default.
    rota : float, optional
        The rotation of the image on the sky; 0 by default without data.
    median_size : int, optional
        The size (an odd number of pixels, at least 3) of the median filter
        that smoothed the data on the sky, which smooths the model as well:
        each pixel takes the median of the square of the size about it,
        with zero beyond the edges (see ModelCube); none by default.

    Raises
    ------
    ValueError
        If median_size is not an odd number of at least 3, or an option of
        every observation on the sky is not valid (see ObservationSky).
    """

    dataset_class = DatasetPixelBrightness

    spectral_view = False
    orbit_stars = 'light'

    # An image has no spectral axis
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'pixel_brightness'

    @staticmethod
    def type_title() -> str:
        return 'Pixel brightness (image)'

    @classmethod
    def _options_from_data(
            cls, dataset: DatasetPixelBrightness, options: dict[str, Any]
    ) -> dict[str, Any]:
        # (the grid of the image)
        return grid_options(dataset.grid())

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(median_size=self._median_size)

    def _kind_options(self) -> dict[str, Any]:
        return grid_options(self.grid()) | dict(median_size=self._median_size)

    def __init__(
            self,
            size: Size | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            median_size: MedianSize | None = None,
            source_grid: SourceGrid | None = None,
            instrument: Instrument | None = None,
            galaxies: Sequence[str] | None = None,
            components: Sequence[str] | None = None,
            lines: Sequence[str] | None = None,
            continuum: bool = False,
            band: str | Band | None = None,
            pixel_weights: Weights | None = None,
            flux_scale: str | None = None,
            offsets: Mapping[str, bool] | None = None,
            dataset: Dataset | None = None,
            noise: Noise | None = None,
            oversampling: Sequence[int] | None = None,
            depth: Depth | None = None,
            dtype: DType | None = None,
            fft_dtype: DType = 'float64',
            name: str | None = None,
            unit: str | None = None
    ):
        check_options(ObservationPixelBrightness.__init__, locals())
        grid = self._options(dataset, dict(
            size=size, step=step, rpix=rpix, rval=rval, rota=rota))
        check_median_size(median_size)
        self._median_size = median_size
        super().__init__(
            self._make_grid(**grid), source_grid, instrument, galaxies,
            components, lines, continuum, band, pixel_weights, flux_scale,
            offsets, dataset, noise, oversampling, depth, dtype, fft_dtype,
            name, unit)

    def _measured_pixels(self) -> np.ndarray:
        return unmasked_pixels(self._dataset, self._grid.size[1::-1])

    def keys(self) -> tuple[str, ...]:
        return ('brightness',)

    def make_dataset(
            self, items: Mapping[str, Data]
    ) -> DatasetPixelBrightness:
        return DatasetPixelBrightness(
            items['brightness'], **grid_options(self.grid(), sizes=False))

    def _data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None
    ) -> DataUnit | None:
        return data_unit(
            dataset, 'brightness', abs(self.step()[0] * self.step()[1]),
            False, light)

    def light_items(self) -> tuple[str, ...]:
        return ('brightness',)

    def linear_in_light(self) -> bool:
        # (its images of the light, through its instrument: linear; not
        # through a median filter, which does not add)
        return self._median_size is None

    def model_can_have_missing_values(self) -> bool:
        return False

    def _check_options(self, options: SkyOptions) -> None:
        if options.instrument.lsf() is not None:
            raise ValueError(
                "an image has no spectral axis for an LSF; remove the LSF "
                "of the instrument")

    def _plan(
            self, driver: Driver, drawing: SkyDrawing
    ) -> 'PixelBrightnessPlan':
        options = self._sky_options
        # The cube of an image has one channel (of no width)
        coords = self._grid.coords
        model_cube = _model_cube.ModelCube(
            Grid(self._grid.size + (1,), coords._replace(
                step=coords.step + (0,), rpix=coords.rpix + (0,),
                rval=coords.rval + (0,)), 2),
            oversampling=options.oversampling + (1,),
            primary_beam=options.instrument.primary_beam(),
            psf=options.instrument.psf(),
            lsf=None,
            smooth_weights=False,
            dtype=options.dtype,
            fft_dtype=options.fft_dtype,
            median_size=self._median_size)
        return PixelBrightnessPlan(model_cube, driver, drawing, options)


class PixelBrightnessPlan(CubePlan):

    def _model_grid(self) -> Grid:
        # Image models are evaluated on the x and y axes
        return self._model_cube_plan.high_res_grid().spatial()

    def has_exact_errors(self) -> bool:
        # (but through a median filter, whose noise no formula gives: then
        # the scatter of draws)
        return self._model_cube.median_size() is None

    def _exact_errors(
            self, noise: 'Noise', recorded: np.ndarray
    ) -> dict[str, np.ndarray] | None:
        if not self.has_exact_errors():
            return None
        return dict(
            brightness=noise.sigma(self._model_cube.grid(), recorded)[0])

    # (its model data are the images of the low-resolution cubes)
    _extra_lo = None
    _extra_high_res = staticmethod(_image_extra)

    def _measure(self) -> ModelData:
        plan = self._model_cube_plan
        weights = plan.weights()
        return dict(brightness=ModelItem(
            plan.cube()[0, :, :],
            weights=None if weights is None else weights[0, :, :]))
