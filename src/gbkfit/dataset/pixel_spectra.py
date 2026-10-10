from collections.abc import Sequence
from typing import Any

import astropy.units

from gbkfit.utils import gridutils
from . import _detail
from .base import Dataset
from .data import Data, FitsFile


__all__ = [
    'DatasetPixelSpectra'
]


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
        files = _detail.pop_item_files(info, prefix)
        return _detail.load_with_files(cls, info, cls.ndim, **files)

    @classmethod
    def from_files(
            cls,
            data: FitsFile,
            mask: FitsFile | None = None,
            error: FitsFile | float | None = None,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None,
            rest: str | astropy.units.Quantity | None = None
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

        Returns
        -------
        DatasetPixelSpectra
            The dataset.
        """
        item, coords = Data.from_files(
            data, mask, error, rpix, rval, rest, cls.spectral_axis)
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
            rest: str | astropy.units.Quantity | None = None
    ):
        super().__init__(dict(spectra=spectra))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota, rest)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the pixels and channels."""
        return self._grid
