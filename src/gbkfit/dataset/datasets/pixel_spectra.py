from collections.abc import Sequence
from numbers import Real

import astropy.units

from gbkfit.dataset.core import Dataset
from gbkfit.dataset.data import Data
from gbkfit.utils import fitsutils
from . import _detail


__all__ = [
    'DatasetPixelSpectra'
]


class DatasetPixelSpectra(Dataset):

    # The axes of a spectral cube: x, y and the spectral axis
    ndim = 3
    spectral_axis = 2

    @staticmethod
    def type():
        return 'pixel_spectra'

    @classmethod
    def load(cls, info, **kwargs):
        # The options of its one data item are given flat
        return cls(**_detail.load_grid_dataset(
            cls, _detail.nest_single_item(info, 'spectra'), ['spectra'],
            **kwargs))

    def dump(self, **kwargs):
        return _detail.flatten_single_item(
            _detail.dump_grid_dataset(self, **kwargs), 'spectra')

    def __init__(
            self,
            spectra: Data,
            step: Real | Sequence[Real] | None = None,
            rpix: Real | Sequence[Real] | None = None,
            rval: Real | Sequence[Real] | None = None,
            rota: Real | None = None,
            rest: str | astropy.units.Quantity | None = None
    ):
        """
        The world coordinates of the grid of the data (see
        fitsutils.Coords; rest is that of the spectral axis) have defaults
        (see _detail.make_grid).
        """
        super().__init__(dict(spectra=spectra))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota, rest)

    def grid(self) -> fitsutils.Grid:
        return self._grid
