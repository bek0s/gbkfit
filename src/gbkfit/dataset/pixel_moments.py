"""
The dataset of moments of the spectra on a grid of pixels: moment maps.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.config import check_options
from gbkfit.grid import Grid
from . import _detail, _outliers, _resampling
from .base import Dataset, NoiseCorrelation
from .data import Data, DataFiles


__all__ = [
    'DatasetPixelMoments'
]


# The axes of its blocks and windows (see resample_blocks and
# screen_outliers)
_AXES = ('x', 'y')


class DatasetPixelMoments(Dataset):
    """
    Moments of the spectra on a grid of pixels: moment maps. They are
    maps of central moments of some of the orders 0 to 7, as the data
    items moment0 to moment7, and of the coefficients h3 and h4 of a
    Gauss-Hermite series, as the items h3 and h4 (see
    _detail.moment_items).

    Its configuration has those data items (the files of each: data,
    mask, error; see from_files) and the world coordinates.

    Parameters
    ----------
    moments : Mapping
        The map of the central moment of each order, of shape (ny, nx).
    hermite : Mapping, optional
        The map of the coefficient of each order (3 and 4) of a
        Gauss-Hermite series; with moments, at least one map.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see Coords); their
        defaults are those of make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see Coords).
    """

    # Moment maps have no spectral axis
    ndim = 2
    spectral_axis = None

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'pixel_moments'

    @staticmethod
    def type_title() -> str:
        return 'Pixel moments (maps)'

    @classmethod
    def from_files(
            cls,
            moment0: DataFiles | None = None,
            moment1: DataFiles | None = None,
            moment2: DataFiles | None = None,
            moment3: DataFiles | None = None,
            moment4: DataFiles | None = None,
            moment5: DataFiles | None = None,
            moment6: DataFiles | None = None,
            moment7: DataFiles | None = None,
            h3: DataFiles | None = None,
            h4: DataFiles | None = None,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            outliers: Mapping[str, Any] | None = None
    ) -> 'DatasetPixelMoments':
        """
        Read the dataset from FITS files, with the world coordinates of the
        files of the moments unless given; they must agree.

        Parameters
        ----------
        moment0, ..., moment7, h3, h4 : DataFiles, optional
            The files of the map of each moment given (at least one), and
            their options (e.g. their unit; by default, that of the
            header).
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the grid (see Coords); by
            default, those of the headers. Either rpix or rval can be
            given, and the other comes from the headers (see
            coords_from_wcs).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            headers.
        outliers : Mapping, optional
            Screen the data for outliers by the options size and nsigma
            (see screen_outliers), which are masked.

        Returns
        -------
        DatasetPixelMoments
            The dataset.
        """
        check_options(DatasetPixelMoments.from_files, locals())
        items, coords = _detail.read_moments(
            _detail.moment_options(locals()), rpix, rval)
        dataset = cls(
            *_detail.split_moments(items),
            **_detail.grid_coords(coords, step, rota))
        return _detail.cleaned(dataset, outliers)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.dump_grid_dataset(self, prefix, dump_path, overwrite)

    def __init__(
            self,
            moments: Mapping[int, Data],
            hermite: Mapping[int, Data] | None = None,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        check_options(DatasetPixelMoments.__init__, locals())
        super().__init__(_detail.moment_items(moments, hermite))
        self._grid = _detail.dataset_grid(self, step, rpix, rval, rota)

    def grid(self) -> Grid:
        """Return the grid of the pixels."""
        return self._grid

    def orders(self) -> tuple[int, ...]:
        """Return the orders of the moments."""
        return _detail.moment_orders(self)

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ValueError unless the size is of blocks of pixels: a positive
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
        Return residuals of its moment maps redrawn in blocks of pixels (see
        Dataset.resample_blocks), of the size (x, y).
        """
        return _resampling.resample_blocks(
            residuals, size, _AXES, rng)

    def screen_outliers(
            self, size: Sequence[float], nsigma: float
    ) -> 'DatasetPixelMoments':
        """
        Return a copy of it with the outliers of its moment maps masked (see
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
