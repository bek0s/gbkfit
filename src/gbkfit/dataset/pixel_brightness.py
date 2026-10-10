from collections.abc import Sequence
from typing import Any

from gbkfit.utils import gridutils
from . import _detail
from .base import Dataset
from .data import Data


__all__ = [
    'DatasetPixelBrightness'
]


class DatasetPixelBrightness(Dataset):
    """
    Brightness on a grid of pixels: an image.

    Its configuration has the options of its one data item beside its own
    (e.g. data, error and step).

    Parameters
    ----------
    brightness : Data
        The image, of shape (ny, nx).
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates of the grid (see gridutils.Coords); their
        defaults are those of gridutils.make_grid.
    rota : float, optional
        The rotation of the grid on the sky (see gridutils.Coords).
    """

    # An image has no spectral axis
    ndim = 2
    spectral_axis = None

    @staticmethod
    def type() -> str:
        return 'pixel_brightness'

    @classmethod
    def load(
            cls, info: dict[str, Any], prefix: str = ''
    ) -> 'DatasetPixelBrightness':
        return cls(**_detail.load_grid_dataset(
            cls, _detail.nest_single_item(info, 'brightness'), ['brightness'],
            prefix))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.flatten_single_item(
            _detail.dump_grid_dataset(self, prefix, dump_path, overwrite),
            'brightness')

    def __init__(
            self,
            brightness: Data,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        super().__init__(dict(brightness=brightness))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the pixels."""
        return self._grid
