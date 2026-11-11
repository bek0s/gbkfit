"""
The observable of brightness on a grid of pixels: an image.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.dataset import Data, Dataset, DatasetPixelBrightness
from gbkfit.driver import Driver
from gbkfit.grid import Grid, GridData
from gbkfit.instrument import Instrument
from gbkfit.model import BandEmission, DataUnit, LightUnit
from . import _model_cube
from ._cube_plan import CubeObservablePlan
from ._data_units import data_unit
from ._options import (
    MedianSize, Rota, Rpix, Rval, Size, Step, check_median_size,
    dataset_grid_options, grid_options)
from .base import ModelData, Observable

if TYPE_CHECKING:
    from gbkfit.model import Scene
    from ..line_of_sight import LineOfSight
    from ..foreground import Foreground
    from ..noise import Noise
    from ..weights import Weights


__all__ = [
    'PixelBrightness'
]


def _image_extra(data: np.ndarray, grid: Grid) -> GridData:
    """
    Make an extra output on a grid of ModelCube an image (its one
    channel).
    """
    return GridData(data[0], grid.spatial().coords, None)


class PixelBrightness(Observable):
    """
    Brightness on a grid of pixels: an image of the model, seen through
    the instrument (which has no LSF).

    Its configuration has the options below, but those of the grid when
    it has data, which give them (see from_data).

    Parameters
    ----------
    size : int or Sequence of int
        The number of pixels of the image (nx, ny).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the image (see Coords); rpix is
        the centre by default.
    rota : float, optional
        The rotation of the image on the sky.
    median_size : int, optional
        The size (an odd number of pixels, at least 3) of the median filter
        that smoothed the data on the sky, which smooths the model as well
        (e.g. 3, of cubes smoothed before their maps were made): each
        channel takes the median of the square of the size about each
        pixel, with zero beyond the edges (see ModelCube); none by default.

    Raises
    ------
    ValueError
        If median_size is not an odd number of at least 3.
    """

    dataset_class = DatasetPixelBrightness

    emission = BandEmission
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
    def from_data(
            cls,
            dataset: DatasetPixelBrightness,
            median_size: MedianSize | None = None
    ) -> 'PixelBrightness':
        """
        Make the observable of an image, on its grid.

        Parameters
        ----------
        dataset : DatasetPixelBrightness
            The image.
        median_size : optional
            As those of PixelBrightness.

        Returns
        -------
        PixelBrightness
            The observable.

        Raises
        ------
        ValueError
            If the dataset is not of brightness, or the options are not
            valid.
        """
        check_options(PixelBrightness.from_data, locals())
        return cls(
            **grid_options(dataset.grid()), median_size=median_size)

    def _dump_options(
            self, prefix: str, dump_path: bool, overwrite: bool
    ) -> dict[str, Any]:
        return dict(median_size=self._median_size)

    def __init__(
            self,
            size: Size,
            step: Step = (1, 1),
            rpix: Rpix | None = None,
            rval: Rval = (0, 0),
            rota: Rota = 0,
            median_size: MedianSize | None = None
    ):
        check_options(PixelBrightness.__init__, locals())
        super().__init__(size, step, rpix, rval, rota)
        check_median_size(median_size)
        self._median_size = median_size

    def keys(self) -> tuple[str, ...]:
        return ('brightness',)

    def make_dataset(
            self, items: Mapping[str, Data]
    ) -> DatasetPixelBrightness:
        return DatasetPixelBrightness(
            items['brightness'], **dataset_grid_options(self.grid()))

    def data_unit(
            self,
            dataset: Dataset | None,
            light: LightUnit | None = None
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

    def check_observation(
            self,
            instrument: Instrument,
            oversampling: Sequence[int],
            pixel_weights: 'Weights | None'
    ) -> None:
        if instrument.lsf() is not None:
            raise ValueError(
                "an image has no spectral axis for an LSF; remove the LSF "
                "of the instrument")

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
    ) -> 'PixelBrightnessPlan':
        # The cube of an image has one channel (of no width)
        coords = self._grid.coords
        model_cube = _model_cube.ModelCube(
            Grid(self._grid.size + (1,), coords._replace(
                step=coords.step + (0,), rpix=coords.rpix + (0,),
                rval=coords.rval + (0,)), 2),
            oversampling=tuple(oversampling) + (1,),
            primary_beam=instrument.primary_beam(),
            psf=instrument.psf(),
            lsf=None,
            smooth_weights=False,
            dtype=dtype,
            fft_dtype=fft_dtype,
            median_size=self._median_size)
        return PixelBrightnessPlan(
            model_cube, driver, scene, line_of_sight, foreground, dtype,
            pixel_weights, flux_scale)


class PixelBrightnessPlan(CubeObservablePlan):

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

    _extra_lo = staticmethod(_image_extra)
    _extra_high_res = staticmethod(_image_extra)

    def _measure(self) -> ModelData:
        plan = self._model_cube_plan
        return dict(brightness=dict(
            d=plan.cube()[0, :, :],
            m=None,
            w=plan.weights()[0, :, :] if plan.weights() is not None else None))
