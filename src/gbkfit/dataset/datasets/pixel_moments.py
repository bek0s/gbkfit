from collections.abc import Sequence
from numbers import Real

from gbkfit.dataset.base import Dataset
from gbkfit.dataset.data import Data
from gbkfit.utils import gridutils
from . import _detail


__all__ = [
    'DatasetPixelMoments'
]


class DatasetPixelMoments(Dataset):

    # Moment maps have no spectral axis
    ndim = 2
    spectral_axis = None

    @staticmethod
    def type():
        return 'pixel_moments'

    @classmethod
    def load(cls, info, **kwargs):
        names = [f'moment{i}' for i in range(8)]
        return cls(**_detail.load_grid_dataset(cls, info, names, **kwargs))

    def dump(self, **kwargs):
        return _detail.dump_grid_dataset(self, **kwargs)

    def __init__(
            self,
            moment0: Data | None = None,
            moment1: Data | None = None,
            moment2: Data | None = None,
            moment3: Data | None = None,
            moment4: Data | None = None,
            moment5: Data | None = None,
            moment6: Data | None = None,
            moment7: Data | None = None,
            step: Real | Sequence[Real] | None = None,
            rpix: Real | Sequence[Real] | None = None,
            rval: Real | Sequence[Real] | None = None,
            rota: Real | None = None
    ):
        """
        The moment maps of the given orders, on one grid. The world
        coordinates of the grid (see gridutils.Coords) have defaults (see
        _detail.make_grid).
        """
        moments = (
            moment0, moment1, moment2, moment3, moment4, moment5, moment6,
            moment7)
        super().__init__({
            f'moment{order}': moment for order, moment in enumerate(moments)
            if moment is not None})
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> gridutils.Grid:
        return self._grid
