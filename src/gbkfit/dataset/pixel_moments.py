from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import check_options
from . import _detail, _outliers, _resampling
from .base import Dataset, NoiseCorrelation
from .data import Data, DataFiles


__all__ = [
    'DatasetPixelMoments'
]


# The axes of its blocks and windows (see resample_blocks and
# screen_outliers)
_AXES = ('x', 'y')

# The axes along which its noise can be correlated (see Data)
_CORRELATIONS = ('spatial',)


class DatasetPixelMoments(Dataset):
    """
    Moments of the spectra on a grid of pixels: moment maps of some of the
    orders 0 to 7, as the data items moment0 to moment7.

    Its configuration has the data items moment0 to moment7 (the files of
    each: data, mask, error; see from_files) and the world coordinates.

    Parameters
    ----------
    moments : Mapping
        The moment map of each order, of shape (ny, nx); at least one.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see gridutils.Coords); their
        defaults are those of gridutils.make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see gridutils.Coords).
    """

    # Moment maps have no spectral axis
    ndim = 2
    spectral_axis = None

    config_constructors = ('from_files',)

    @staticmethod
    def type() -> str:
        return 'pixel_moments'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'DatasetPixelMoments':
        info = dict(info)
        _detail.load_moment_correlations(info)
        return _detail.load_with_files(cls, info, cls.ndim)

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
        moment0, ..., moment7 : DataFiles, optional
            The files of the moment map of each order given (at least
            one), and their options (e.g. their unit; by default, that of
            the header).
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the grid (see gridutils.Coords); by
            default, those of the headers. Either rpix or rval can be
            given, and the other comes from the headers (see
            fitsutils.read_data).
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
        dataset = cls(items, **_detail.grid_coords(coords, step, rota))
        return _detail.cleaned(dataset, outliers)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.dump_grid_dataset(self, prefix, dump_path, overwrite)

    def __init__(
            self,
            moments: Mapping[int, Data],
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        check_options(DatasetPixelMoments.__init__, locals())
        super().__init__(_detail.moment_items(moments))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the pixels."""
        return self._grid

    def orders(self) -> tuple[int, ...]:
        """Return the orders of the moments."""
        return _detail.moment_orders(self)

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
