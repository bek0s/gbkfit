"""
The observation of spectra on a grid of pixels: a spectral cube.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetPixelSpectra
from gbkfit.driver import Driver
from gbkfit.grid import SpectralAxis, grid_options
from gbkfit.instrument import Instrument
from gbkfit.lensing import SourceGrid
from gbkfit.model import Band, DataUnit, LightUnit
from ..depth import Depth
from ..noise import Noise
from ..plan import FluxScale, ModelData, ModelItem
from ..weights import Weights
from . import _model_cube
from ._cube_plan import CubePlan, SkyDrawing
from ._data_units import data_unit
from ._options import (
    MedianSize, Rota, Rpix, Rval, Size, Step, check_median_size,
    warn_unsmoothed_weights)
from .base import DType, ObservationSky, SkyOptions, unmasked_pixels

__all__ = [
    'ObservationPixelSpectra'
]


class ObservationPixelSpectra(ObservationSky):
    """
    Spectra on a grid of pixels: a spectral cube of the model, seen
    through the instrument.

    Its options are those below, then those of every observation on the
    sky (see ObservationSky). Its data give those of its grid (given too,
    they serve only without data); without data, size and spectral are
    required.

    Parameters
    ----------
    size : int or Sequence of int, optional
        The number of pixels of the cube on the sky (nx, ny).
    spectral : SpectralAxis, optional
        The spectral axis of the cube: its size, its world coordinates and
        the convention of its velocities (see make_cube_grid).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the cube on the sky (see Coords); without
        data, a step of 1, rval 0, and rpix the centre by default.
    rota : float, optional
        The rotation of the cube on the sky; 0 by default without data.
    smooth_weights : bool, optional
        Whether the weights of the model are smoothed by the PSF and the
        LSF, as the model is.
    median_size : int, optional
        The size (an odd number of pixels, at least 3) of the median filter
        that smoothed the data on the sky, which smooths the model as well
        (e.g. 3, of cubes smoothed before their maps were made): each
        channel takes the median of the square of the size about each
        pixel, with zero beyond the edges (see ModelCube); none by default.

    Raises
    ------
    ValueError
        If the spectral axis is not valid (see make_cube_grid),
        median_size is not an odd number of at least 3, or an option of
        every observation on the sky is not valid (see ObservationSky).
    """

    dataset_class = DatasetPixelSpectra

    spectral_view = True
    orbit_stars = None

    # The axes of a spectral cube: x, y and the spectral axis
    spectral_axis = 2

    config_required_without = dict(dataset=('size', 'spectral'))

    @staticmethod
    def type() -> str:
        return 'pixel_spectra'

    @staticmethod
    def type_title() -> str:
        return 'Pixel spectra (cube)'

    @classmethod
    def _options_from_data(
            cls, dataset: DatasetPixelSpectra, options: dict[str, Any]
    ) -> dict[str, Any]:
        # (the grid of the cube)
        return grid_options(dataset.grid())

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(
            smooth_weights=self._smooth_weights,
            median_size=self._median_size)

    def _kind_options(self) -> dict[str, Any]:
        return grid_options(self.grid()) | dict(
            smooth_weights=self._smooth_weights,
            median_size=self._median_size)

    def __init__(
            self,
            size: Size | None = None,
            spectral: SpectralAxis | None = None,
            step: Step | None = None,
            rpix: Rpix | None = None,
            rval: Rval | None = None,
            rota: Rota | None = None,
            smooth_weights: bool = False,
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
        check_options(ObservationPixelSpectra.__init__, locals())
        grid = self._options(dataset, dict(
            size=size, spectral=spectral, step=step, rpix=rpix, rval=rval,
            rota=rota))
        check_median_size(median_size)
        self._smooth_weights = smooth_weights
        self._median_size = median_size
        super().__init__(
            self._make_grid(**grid), source_grid, instrument, galaxies,
            components, lines, continuum, band, pixel_weights, flux_scale,
            offsets, dataset, noise, oversampling, depth, dtype, fft_dtype,
            name, unit)

    def _measured_pixels(self) -> np.ndarray:
        return unmasked_pixels(self._dataset, self._grid.size[1::-1])

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetPixelSpectra:
        return DatasetPixelSpectra(
            items['spectra'], **grid_options(self.grid(), sizes=False))

    def _data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None
    ) -> DataUnit | None:
        return data_unit(
            dataset, 'spectra', abs(self.step()[0] * self.step()[1]), False,
            light)

    def light_items(self) -> tuple[str, ...]:
        return ('spectra',)

    def flux_scale_kinds(self) -> dict[str, FluxScale]:
        # (each spectrum runs along the first axis, at each pixel)
        nspectra = self.size()[0] * self.size()[1]
        return super().flux_scale_kinds() \
            | dict(per_spectrum=FluxScale(('spectra',), 0, nspectra))

    def linear_in_light(self) -> bool:
        # (its spectra of the light, through its instrument: linear; not
        # through a median filter, which does not add)
        return self._median_size is None

    def model_can_have_missing_values(self) -> bool:
        return False

    def _check_options(self, options: SkyOptions) -> None:
        warn_unsmoothed_weights(self._smooth_weights, options.instrument)

    def _plan(self, driver: Driver, drawing: SkyDrawing) -> 'PixelSpectraPlan':
        options = self._sky_options
        model_cube = _model_cube.ModelCube(
            self.grid(),
            oversampling=options.oversampling,
            primary_beam=options.instrument.primary_beam(),
            psf=options.instrument.psf(),
            lsf=options.instrument.lsf(),
            smooth_weights=self._smooth_weights,
            dtype=options.dtype,
            fft_dtype=options.fft_dtype,
            median_size=self._median_size)
        return PixelSpectraPlan(model_cube, driver, drawing, options)


class PixelSpectraPlan(CubePlan):

    def has_exact_errors(self) -> bool:
        # (but through a median filter, whose noise no formula gives: then
        # the scatter of draws)
        return self._model_cube.median_size() is None

    def _exact_errors(
            self, noise: 'Noise', recorded: np.ndarray
    ) -> dict[str, np.ndarray] | None:
        if not self.has_exact_errors():
            return None
        return dict(spectra=noise.sigma(self._model_cube.grid(), recorded))

    # (its model data are the low-resolution cubes)
    _extra_lo = None

    def _measure(self) -> ModelData:
        plan = self._model_cube_plan
        return dict(spectra=ModelItem(plan.cube(), weights=plan.weights()))
