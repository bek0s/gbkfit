from collections.abc import Sequence

import astropy.units

from gbkfit.dataset.base import Dataset
from gbkfit.dataset.data import Data
from gbkfit.utils import gridutils
from . import _detail


__all__ = [
    'DatasetSlitSpectra'
]


class DatasetSlitSpectra(Dataset):

    # The axes of a long-slit spectrum: the position along the slit
    # and the spectral axis
    ndim = 2
    spectral_axis = 1

    @staticmethod
    def type():
        return 'slit_spectra'

    @classmethod
    def load(cls, info, prefix=''):
        # The options of its one data item are given flat
        return cls(**_detail.load_grid_dataset(
            cls, _detail.nest_single_item(info, 'spectra'), ['spectra'],
            prefix))

    def dump(self, prefix='', dump_path=True, overwrite=False):
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
        """
        The world coordinates of the grid of the data (see
        gridutils.Coords; rest is that of the spectral axis) have defaults
        (see _detail.make_grid).
        """
        super().__init__(dict(spectra=spectra))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota, rest)

    def grid(self) -> gridutils.Grid:
        return self._grid
