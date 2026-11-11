"""
The observable of spectra on a grid of pixels: a spectral cube.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetPixelSpectra
from gbkfit.driver import Driver
from gbkfit.grid import SpectralAxis
from gbkfit.instrument import Instrument
from gbkfit.model import DataUnit, LightUnit, LineEmission
from . import _model_cube
from ._cube_plan import CubeObservablePlan
from ._data_units import data_unit
from ._options import (
    MedianSize, Rota, Rpix, Rval, Size, Step, check_median_size,
    dataset_grid_options, grid_options, warn_unsmoothed_weights)
from .base import FluxScale, ModelData, Observable

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'PixelSpectra'
]


class PixelSpectra(Observable):
    """
    Spectra on a grid of pixels: a spectral cube of the model, seen
    through the instrument.

    Its configuration has the options below, but those of the grid when
    it has data, which give them (see from_data).

    Parameters
    ----------
    size : int or Sequence of int
        The number of pixels of the cube on the sky (nx, ny).
    spectral : SpectralAxis
        The spectral axis of the cube: its size, its world coordinates and
        the convention of its velocities (see make_cube_grid).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the cube on the sky (see
        Coords); rpix is the centre by default.
    rota : float, optional
        The rotation of the cube on the sky.
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
        or median_size is not an odd number of at least 3.
    """

    dataset_class = DatasetPixelSpectra

    emission = LineEmission
    orbit_stars = None

    # The axes of a spectral cube: x, y and the spectral axis
    spectral_axis = 2

    @staticmethod
    def type() -> str:
        return 'pixel_spectra'

    @staticmethod
    def type_title() -> str:
        return 'Pixel spectra (cube)'

    @classmethod
    def from_data(
            cls,
            dataset: DatasetPixelSpectra,
            smooth_weights: bool = False,
            median_size: MedianSize | None = None
    ) -> 'PixelSpectra':
        """
        Make the observable of a spectral cube, on its grid.

        Parameters
        ----------
        dataset : DatasetPixelSpectra
            The spectral cube.
        smooth_weights, median_size : optional
            As those of PixelSpectra.

        Returns
        -------
        PixelSpectra
            The observable.

        Raises
        ------
        ValueError
            If the dataset is not of spectra on pixels, or the options are
            not valid.
        """
        check_options(PixelSpectra.from_data, locals())
        return cls(
            **grid_options(dataset.grid()),
            smooth_weights=smooth_weights, median_size=median_size)

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(
            smooth_weights=self._smooth_weights,
            median_size=self._median_size)

    def __init__(
            self,
            size: Size,
            spectral: SpectralAxis,
            step: Step = (1, 1),
            rpix: Rpix | None = None,
            rval: Rval = (0, 0),
            rota: Rota = 0,
            smooth_weights: bool = False,
            median_size: MedianSize | None = None
    ):
        check_options(PixelSpectra.__init__, locals())
        super().__init__(size, step, rpix, rval, rota, spectral)
        check_median_size(median_size)
        self._smooth_weights = smooth_weights
        self._median_size = median_size

    def keys(self) -> tuple[str, ...]:
        return ('spectra',)

    def make_dataset(self, items: Mapping[str, Data]) -> DatasetPixelSpectra:
        return DatasetPixelSpectra(
            items['spectra'], **dataset_grid_options(self.grid()))

    def data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None = None
    ) -> DataUnit | None:
        return data_unit(
            dataset, 'spectra', abs(self.step()[0] * self.step()[1]), False,
            light)

    def light_items(self) -> tuple[str, ...]:
        return ('spectra',)

    def flux_scales(self) -> dict[str, FluxScale]:
        # (each spectrum runs along the first axis, at each pixel)
        nspectra = self.size()[0] * self.size()[1]
        return super().flux_scales() \
            | dict(per_spectrum=FluxScale(('spectra',), 0, nspectra))

    def linear_in_light(self) -> bool:
        # (its spectra of the light, through its instrument: linear; not
        # through a median filter, which does not add)
        return self._median_size is None

    def model_can_lack_values(self) -> bool:
        return False

    def check_observation(
            self,
            instrument: Instrument,
            oversampling: Sequence[int],
            pixel_weights: 'Weights | None'
    ) -> None:
        warn_unsmoothed_weights(self._smooth_weights, instrument)

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
    ) -> 'PixelSpectraPlan':
        model_cube = _model_cube.ModelCube(
            self.grid(),
            oversampling=tuple(oversampling),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=instrument.lsf(),
            smooth_weights=self._smooth_weights,
            dtype=dtype,
            fft_dtype=fft_dtype,
            median_size=self._median_size)
        return PixelSpectraPlan(
            model_cube, driver, scene, line_of_sight, foreground, dtype,
            pixel_weights, flux_scale)


class PixelSpectraPlan(CubeObservablePlan):

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

    def _measure(self) -> ModelData:
        plan = self._model_cube_plan
        return dict(spectra=dict(
            d=plan.cube(), m=None, w=plan.weights()))
