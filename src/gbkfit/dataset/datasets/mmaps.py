from collections.abc import Sequence
from numbers import Real

from gbkfit.dataset.core import Dataset
from gbkfit.dataset.data import Data
from gbkfit.utils import fitsutils
from . import _detail


__all__ = [
    'DatasetMMaps'
]


class DatasetMMaps(Dataset):

    # Moment maps have no spectral axis
    _ndim = 2
    _spectral_axis = None

    @staticmethod
    def type():
        return 'mmaps'

    @classmethod
    def load(cls, info, **kwargs):
        names = [f'mmap{i}' for i in range(8)]
        return cls(**_detail.load_grid_dataset(cls, info, names, **kwargs))

    def dump(self, **kwargs):
        return _detail.dump_grid_dataset(self, **kwargs)

    def __init__(
            self,
            mmap0: Data | None = None,
            mmap1: Data | None = None,
            mmap2: Data | None = None,
            mmap3: Data | None = None,
            mmap4: Data | None = None,
            mmap5: Data | None = None,
            mmap6: Data | None = None,
            mmap7: Data | None = None,
            step: Real | Sequence[Real] | None = None,
            rpix: Real | Sequence[Real] | None = None,
            rval: Real | Sequence[Real] | None = None,
            rota: Real | None = None
    ):
        """
        The moment maps of the given orders, on one grid. The world
        coordinates of the grid (see fitsutils.Coords) have defaults (see
        _detail.make_grid).
        """
        mmaps = (mmap0, mmap1, mmap2, mmap3, mmap4, mmap5, mmap6, mmap7)
        super().__init__({
            f'mmap{order}': mmap for order, mmap in enumerate(mmaps)
            if mmap is not None})
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> fitsutils.Grid:
        return self._grid
