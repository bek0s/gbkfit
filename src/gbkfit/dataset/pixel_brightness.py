from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.instrument import PSF
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import check_options
from gbkfit.utils.typeutils import FitsFile
from . import _detail, _outliers, _resampling
from .base import Dataset, NoiseCorrelation
from .data import Data


__all__ = [
    'DatasetPixelBrightness'
]


# The axes of its blocks and windows (see resample_blocks and
# screen_outliers)
_AXES = ('x', 'y')


class DatasetPixelBrightness(Dataset):
    """
    Brightness on a grid of pixels: an image.

    Its configuration has the options of from_files: the files of its one
    data item (data, mask, error) and its world coordinates.

    Parameters
    ----------
    brightness : Data
        The image, of shape (ny, nx).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see gridutils.Coords); their
        defaults are those of gridutils.make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see gridutils.Coords).
    """

    # An image has no spectral axis
    ndim = 2
    spectral_axis = None

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'pixel_brightness'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'DatasetPixelBrightness':
        info = dict(info)
        _detail.load_correlations(info)
        return _detail.load_with_files(cls, info, cls.ndim)

    @classmethod
    def from_files(
            cls,
            data: FitsFile,
            mask: FitsFile | None = None,
            error: FitsFile | float | None = None,
            unit: str | None = None,
            beam: Mapping[str, Any] | None = None,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            spatial_correlation: PSF | None = None,
            outliers: Mapping[str, Any] | None = None
    ) -> 'DatasetPixelBrightness':
        """
        Read the dataset from FITS files, with the world coordinates of the
        file of its values unless given.

        Parameters
        ----------
        data : str or tuple
            The file of the values: a filename, or a filename and the HDU.
        mask : str or tuple, optional
            The file of the mask.
        error : str or tuple or float, optional
            The file of the errors, or one error for all the values.
        unit : str, optional
            The unit of the values; by default, that of the header (see
            Data.from_files).
        beam : Mapping, optional
            The options of their beam (see Beam); by default, that of the
            header.
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the grid (see gridutils.Coords); by
            default, those of the header. Either rpix or rval can be given,
            and the other comes from the header (see fitsutils.read_data).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            header.
        spatial_correlation : PSF, optional
            The correlation of the noise on the sky (see Data).
        outliers : Mapping, optional
            Screen the data for outliers by the options size and nsigma
            (see screen_outliers), which are masked.

        Returns
        -------
        DatasetPixelBrightness
            The dataset.
        """
        check_options(DatasetPixelBrightness.from_files, locals())
        item, coords = Data.from_files(
            data, mask, error, rpix, rval, unit=unit, beam=beam,
            spatial_correlation=spatial_correlation)
        dataset = cls(item, **_detail.grid_coords(coords, step, rota))
        return _detail.cleaned(dataset, outliers)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.flatten_single_item(
            _detail.dump_grid_dataset(self, prefix, dump_path, overwrite),
            'brightness')

    def __init__(
            self,
            brightness: Data,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        check_options(DatasetPixelBrightness.__init__, locals())
        super().__init__(dict(brightness=brightness))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the pixels."""
        return self._grid

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ConfigError unless the size is of blocks of pixels: a positive
        integer for each of x and y (see resample_blocks).
        """
        _resampling.block_size(size, _AXES)

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its image redrawn in blocks of pixels (see
        Dataset.resample_blocks), of the size (x, y).
        """
        return _resampling.resample_blocks(
            residuals, size, _AXES, rng)

    def screen_outliers(
            self, size: Sequence[float], nsigma: float
    ) -> 'DatasetPixelBrightness':
        """
        Return a copy of it with the outliers of its image masked (see
        Dataset.screen_outliers): of the other pixels of a window of the size
        (x, y) about each.
        """
        return _outliers.screen_grid(self, size, _AXES, nsigma)

    def noise_correlation(self, key: str) -> NoiseCorrelation | None:
        """
        Return the correlation of the noise of a data item on the
        separations of its pixels (see Dataset.noise_correlation).
        """
        return _detail.grid_noise_correlation(self, key)
