from collections.abc import Sequence
from numbers import Real

from gbkfit.dataset.core import Dataset
from gbkfit.dataset.data import Data
from gbkfit.utils import fitsutils
from . import _detail


__all__ = [
    'DatasetLSlit'
]


class DatasetLSlit(Dataset):

    # The axes of a long-slit spectrum: the position along the slit
    # and the spectral axis
    _ndim = 2
    _spectral_axis = 1

    @staticmethod
    def type():
        return 'lslit'

    @classmethod
    def load(cls, info, **kwargs):
        # The options of its one data item are given flat
        return cls(**_detail.load_grid_dataset(
            cls, _detail.nest_single_item(info, 'lslit'), ['lslit'],
            **kwargs))

    def dump(self, **kwargs):
        return _detail.flatten_single_item(
            _detail.dump_grid_dataset(self, **kwargs), 'lslit')

    def __init__(
            self,
            lslit: Data,
            step: Real | Sequence[Real] | None = None,
            rpix: Real | Sequence[Real] | None = None,
            rval: Real | Sequence[Real] | None = None,
            rota: Real | None = None
    ):
        """
        The world coordinates of the grid of the data (see
        fitsutils.Coords) have defaults (see _detail.make_grid).
        """
        super().__init__(dict(lslit=lslit))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> fitsutils.Grid:
        return self._grid
