from collections.abc import Sequence
from typing import Any

from gbkfit.utils import gridutils
from . import _detail
from .base import Dataset
from .data import Data


__all__ = [
    'DatasetPixelMoments'
]


class DatasetPixelMoments(Dataset):
    """
    Moments of the spectra on a grid of pixels: moment maps of some of the
    orders 0 to 7, as the data items moment0 to moment7.

    Parameters
    ----------
    moment0, ..., moment7 : Data, optional
        The moment maps, each of shape (ny, nx); at least one.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see gridutils.Coords); their
        defaults are those of gridutils.make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see gridutils.Coords).
    """

    # Moment maps have no spectral axis
    ndim = 2
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'pixel_moments'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'DatasetPixelMoments':
        names = [f'moment{i}' for i in range(8)]
        return cls(**_detail.load_grid_dataset(cls, info, names, prefix))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.dump_grid_dataset(self, prefix, dump_path, overwrite)

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
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        moments = (
            moment0, moment1, moment2, moment3, moment4, moment5, moment6,
            moment7)
        super().__init__({
            f'moment{order}': moment for order, moment in enumerate(moments)
            if moment is not None})
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the pixels."""
        return self._grid
