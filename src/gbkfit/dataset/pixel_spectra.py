from collections.abc import Mapping, Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.utils import gridutils
from . import _detail, _outliers, _resampling
from .base import Dataset
from .data import Data, FitsFile


__all__ = [
    'DatasetPixelSpectra'
]


# The axes of its blocks and windows (see resample_blocks and
# screen_outliers)
_AXES = ('x', 'y', 'the channels')


class DatasetPixelSpectra(Dataset):
    """
    Spectra on a grid of pixels: a spectral cube.

    Its configuration has the options of from_files: the files of its one
    data item (data, mask, error) and its world coordinates.

    Parameters
    ----------
    spectra : Data
        The cube, of shape (nz, ny, nx): x, y and the spectral axis.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see gridutils.Coords); their
        defaults are those of gridutils.make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see gridutils.Coords).
    rest : str or Quantity, optional
        The rest of the spectral axis (see gridutils.make_rest).
    medium : str, optional
        The medium of the wavelengths of a rest wavelength (see
        gridutils.make_medium).
    """

    # The axes of a spectral cube: x, y and the spectral axis
    ndim = 3
    spectral_axis = 2

    @staticmethod
    def type() -> str:
        return 'pixel_spectra'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'DatasetPixelSpectra':
        info = dict(info)
        files = _detail.pop_item_options(info, prefix)
        return _detail.load_with_files(cls, info, cls.ndim, **files)

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
            rest: str | astropy.units.Quantity | None = None,
            medium: str | None = None
    ) -> 'DatasetPixelSpectra':
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
        rest : str or Quantity, optional
            The rest of the spectral axis (see gridutils.make_rest); by
            default, that of the header.
        medium : str, optional
            The medium of the wavelengths of a rest wavelength (see
            gridutils.make_medium); by default, that of the header (see
            fitsutils.read_data).

        Returns
        -------
        DatasetPixelSpectra
            The dataset.
        """
        item, coords = Data.from_files(
            data, mask, error, rpix, rval, rest, cls.spectral_axis, medium,
            unit, beam)
        return cls(item, **_detail.grid_coords(coords, step, rota, True))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.flatten_single_item(
            _detail.dump_grid_dataset(self, prefix, dump_path, overwrite),
            'spectra')

    def __init__(
            self,
            spectra: Data,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            rest: str | astropy.units.Quantity | None = None,
            medium: str | None = None
    ):
        super().__init__(dict(spectra=spectra))
        self._grid = _detail.make_grid(
            self, step, rpix, rval, rota, rest, medium)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the pixels and channels."""
        return self._grid

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ConfigError unless the size is of blocks of pixels and channels:
        a positive integer for each of x, y and the channels (see
        resample_blocks).
        """
        _resampling.block_size(size, _AXES)

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its cube redrawn in blocks of pixels and channels
        (see Dataset.resample_blocks), of the size (x, y, channels).
        """
        return _resampling.resample_blocks(
            residuals, size, _AXES, rng)

    def screen_outliers(
            self, size: Sequence[float], nsigma: float
    ) -> 'DatasetPixelSpectra':
        """
        Return a copy of it with the outliers of its cube masked (see
        Dataset.screen_outliers): of the other voxels of a window of the size
        (x, y, channels) about each.
        """
        return _outliers.screen_grid(self, size, _AXES, nsigma)
