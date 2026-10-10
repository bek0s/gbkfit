from collections.abc import Sequence
from typing import Any

import astropy.units

from gbkfit.utils import gridutils
from . import _detail
from .base import Dataset
from .data import Data


__all__ = [
    'DatasetPixelSpectra'
]


class DatasetPixelSpectra(Dataset):
    """
    Spectra on a grid of pixels: a spectral cube.

    Its configuration has the options of its one data item beside its own
    (e.g. data, error and step).

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
        return cls(**_detail.load_grid_dataset(
            cls, _detail.nest_single_item(info, 'spectra'), ['spectra'],
            prefix))

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
