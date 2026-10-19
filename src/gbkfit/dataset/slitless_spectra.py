from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gbkfit.utils import gridutils
from . import _detail, _resampling
from .base import Dataset
from .data import Data, FitsFile


__all__ = [
    'DatasetSlitlessSpectra'
]


# The axes of its blocks of residuals (see resample_blocks)
_BLOCK_AXES = ('x', 'y')


class DatasetSlitlessSpectra(Dataset):
    """
    Slitless spectra: an image of a grism (or prism) exposure, where the
    light of each wavelength lands away from its place on the sky by the
    trace of the disperser (see SlitlessSpectra).

    Its grid is that of the detector, with the world coordinates of the
    undispersed light: a pixel is at the position on the sky whose light
    would land on it without the disperser.

    Its configuration has the options of from_files: the files of its one
    data item (data, mask, error) and its world coordinates.

    Parameters
    ----------
    spectra : Data
        The image, of shape (ny, nx).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see gridutils.Coords); their
        defaults are those of gridutils.make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see gridutils.Coords).
    """

    # An image of dispersed light has no spectral axis: wavelength and
    # position share the axes of the detector
    ndim = 2
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'slitless_spectra'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'DatasetSlitlessSpectra':
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
            rota: float | None = None
    ) -> 'DatasetSlitlessSpectra':
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

        Returns
        -------
        DatasetSlitlessSpectra
            The dataset.
        """
        item, coords = Data.from_files(
            data, mask, error, rpix, rval, unit=unit, beam=beam)
        return cls(item, **_detail.grid_coords(coords, step, rota))

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
            rota: float | None = None
    ):
        super().__init__(dict(spectra=spectra))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the pixels of the detector."""
        return self._grid

    def check_blocks(self, size: Sequence[float]) -> None:
        """
        Raise ConfigError unless the size is of blocks of pixels of the
        detector: a positive integer for each of x and y (see resample_blocks).
        """
        _resampling.block_size(size, _BLOCK_AXES)

    def resample_blocks(
            self,
            residuals: Mapping[str, np.ndarray],
            size: Sequence[float],
            rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """
        Return residuals of its image redrawn in blocks of pixels of the
        detector (see Dataset.resample_blocks), of the size (x, y).
        """
        return _resampling.resample_blocks(
            residuals, size, _BLOCK_AXES, rng)
