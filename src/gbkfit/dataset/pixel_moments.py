from collections.abc import Mapping, Sequence
from typing import Any

from gbkfit.utils import gridutils
from . import _detail
from .base import Dataset
from .data import Data, FitsFile


__all__ = [
    'DatasetPixelMoments'
]


class DatasetPixelMoments(Dataset):
    """
    Moments of the spectra on a grid of pixels: moment maps of some of the
    orders 0 to 7, as the data items moment0 to moment7.

    Its configuration has the data items moment0 to moment7 (the files of
    each: data, mask, error; see from_files) and the world coordinates.

    Parameters
    ----------
    moments : Mapping
        The moment map of each order, of shape (ny, nx); at least one.
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
        info = dict(info)
        files = _detail.pop_moment_files(info, prefix)
        return _detail.load_with_files(cls, info, cls.ndim, **files)

    @classmethod
    def from_files(
            cls,
            moments: Mapping[int, FitsFile],
            masks: Mapping[int, FitsFile] | None = None,
            errors: Mapping[int, FitsFile | float] | None = None,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ) -> 'DatasetPixelMoments':
        """
        Read the dataset from FITS files, with the world coordinates of the
        files of the moments unless given; they must agree.

        Parameters
        ----------
        moments : Mapping
            The file of each moment map, by order: a filename, or a
            filename and the HDU.
        masks : Mapping, optional
            The files of the masks of some of the moment maps.
        errors : Mapping, optional
            The files of the errors of some of the moment maps, or one
            error for all the values of each.
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the grid (see gridutils.Coords); by
            default, those of the headers. Either rpix or rval can be
            given, and the other comes from the headers (see
            fitsutils.read_data).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            headers.

        Returns
        -------
        DatasetPixelMoments
            The dataset.
        """
        items, coords = _detail.read_moments(
            moments, masks, errors, rpix, rval)
        return cls(items, **_detail.grid_coords(coords, step, rota))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return _detail.dump_grid_dataset(self, prefix, dump_path, overwrite)

    def __init__(
            self,
            moments: Mapping[int, Data],
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        super().__init__(_detail.moment_items(moments))
        self._grid = _detail.make_grid(self, step, rpix, rval, rota)

    def grid(self) -> gridutils.Grid:
        """Return the grid of the pixels."""
        return self._grid

    def orders(self) -> tuple[int, ...]:
        """Return the orders of the moments."""
        return _detail.moment_orders(self)
