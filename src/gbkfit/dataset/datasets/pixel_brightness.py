from collections.abc import Sequence
from numbers import Real

from gbkfit.dataset.base import Dataset
from gbkfit.dataset.data import Data
from gbkfit.utils import gridutils
from . import _detail


__all__ = [
    'DatasetPixelBrightness'
]


class DatasetPixelBrightness(Dataset):

    # An image has no spectral axis
    ndim = 2
    spectral_axis = None

    @staticmethod
    def type():
        return 'pixel_brightness'

    @classmethod
    def load(cls, info, prefix=''):
        # The options of its one data item are given flat
        return cls(**_detail.load_grid_dataset(
            cls, _detail.nest_single_item(info, 'brightness'), ['brightness'],
            prefix))

    def dump(self, prefix='', dump_path=True, overwrite=False):
        return _detail.flatten_single_item(
            _detail.dump_grid_dataset(self, prefix, dump_path, overwrite),
            'brightness')

    def __init__(
            self,
            brightness: Data,
            step: Real | Sequence[Real] | None = None,
            rpix: Real | Sequence[Real] | None = None,
            rval: Real | Sequence[Real] | None = None,
            rota: Real | None = None
    ):
        """
        The world coordinates of the grid of the data (see
        gridutils.Coords) have defaults (see _detail.make_grid).
        """
        super().__init__(dict(brightness=brightness))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> gridutils.Grid:
        return self._grid
