import abc
import os.path
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse

from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import check_options
from gbkfit.utils.typeutils import FitsFile, Unit, split_fits_file
from .apertures import Aperture, aperture_parser


__all__ = [
    'Regions',
    'RegionsApertures',
    'RegionsBins',
    'regions_parser'
]


# The world coordinates of a grid with one value per axis, x and y, or one
# for both (see gridutils.Coords), and its rotation
_PerAxis: TypeAlias = float | Sequence[float] | None
_Angle: TypeAlias = Annotated[float, Unit('deg')]


def _per_axis(value: _PerAxis) -> list[float] | None:
    """Return one value of a world coordinate per axis, x and y."""
    return [value] * 2 if isinstance(value, int | float) \
        else None if value is None else list(value)


class Regions(parseutils.TypedSerializable, abc.ABC):
    """
    The regions of the sky in which data were measured, each the weighted
    sum of the pixels of a spatial grid (see weights()): e.g. bins of
    spaxels, or apertures.
    """

    @abc.abstractmethod
    def nregions(self) -> int:
        pass

    @abc.abstractmethod
    def grid(self) -> gridutils.Grid | None:
        """
        The spatial grid on which the regions are defined, or None if they
        are defined on the sky (and can be put on any grid that covers
        them).
        """
        pass

    @abc.abstractmethod
    def centres(self) -> np.ndarray:
        """
        The centre of each region, x and y (arcsec, in the frame of the
        model, like xpos and ypos), of shape (nregions, 2): of bins, the
        mean of the positions of their pixels; of apertures, their centres
        (see Aperture.centre).
        """
        pass

    @abc.abstractmethod
    def weights(self, grid: gridutils.Grid) -> scipy.sparse.csr_array:
        """
        The weights of the pixels of a spatial grid in the regions: a
        matrix of one row for each region and one column for each pixel
        (flat index, x fastest).
        """
        pass


class RegionsBins(Regions):
    """
    Bins of the pixels of a spatial grid (e.g. Voronoi bins), given by the
    bin of each pixel (index): 0 to n - 1, or negative for the pixels in no
    bin. Each bin is the sum of its pixels.
    """

    @staticmethod
    def type():
        return 'bins'

    config_constructors = ('from_file',)

    @classmethod
    def from_file(
            cls,
            file: FitsFile,
            step: _PerAxis = None,
            rpix: _PerAxis = None,
            rval: _PerAxis = None,
            rota: _Angle | None = None
    ) -> 'RegionsBins':
        """
        Read the bins from a FITS file, with the world coordinates of its
        header unless given.

        Parameters
        ----------
        file : str or tuple
            The file of the bin of each pixel (see the class): a filename,
            or a filename and the HDU.
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the grid (see gridutils.Coords); by
            default, those of the header. Either rpix or rval can be given,
            and the other comes from the header (see fitsutils.read_data).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            header.

        Returns
        -------
        RegionsBins
            The bins.
        """
        check_options(RegionsBins.from_file, locals())
        index, coords = fitsutils.read_data(
            *split_fits_file(file), _per_axis(rpix), _per_axis(rval))
        return cls(
            index, coords.step if step is None else step, coords.rpix,
            coords.rval, coords.rota if rota is None else rota)

    def dump(self, prefix='', dump_path=True, overwrite=False):
        filename = f'{prefix}bins.fits'
        coords = self._grid.coords
        fitsutils.write_data(
            filename, self._index.astype(np.int32), coords, None, overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            step=coords.step,
            rpix=coords.rpix,
            rval=coords.rval,
            rota=coords.rota)

    def __init__(
            self,
            index: npt.ArrayLike,
            step: _PerAxis = None,
            rpix: _PerAxis = None,
            rval: _PerAxis = None,
            rota: _Angle | None = None
    ):
        """
        The world coordinates of the grid of the bins (see
        gridutils.Coords) have defaults (see gridutils.make_grid).
        """
        check_options(RegionsBins.__init__, locals())
        index = np.asarray(index)
        if index.ndim != 2:
            raise RuntimeError(
                f"the bins of the pixels must be an image; they have "
                f"{index.ndim} axes")
        if not np.all(np.isfinite(index)) or np.any(index != np.round(index)):
            raise RuntimeError("the bins of the pixels must be integers")
        index = index.astype(np.int64)
        nbins = int(index.max()) + 1
        if nbins < 1:
            raise RuntimeError("no pixel is in a bin")
        counts = np.bincount(index[index >= 0], minlength=nbins)
        if empty := np.flatnonzero(counts == 0).tolist():
            raise RuntimeError(
                f"the bins are numbered from 0 to {nbins - 1}, but these "
                f"have no pixels: {empty}")
        self._index = index
        self._nbins = nbins
        self._grid = gridutils.make_grid(
            index.shape[::-1], step, rpix, rval, rota)

    def __eq__(self, other):
        return (isinstance(other, RegionsBins)
                and self._grid == other._grid
                and np.array_equal(self._index, other._index))

    def index(self) -> np.ndarray:
        """The bin of each pixel (negative for none)."""
        return self._index

    def nregions(self):
        return self._nbins

    def grid(self):
        return self._grid

    def centres(self):
        weights = self.weights(self._grid)
        x, y = gridutils.sky_positions(self._grid)
        totals = np.asarray(weights.sum(axis=1)).ravel()
        return np.column_stack(
            [weights @ x.ravel(), weights @ y.ravel()]) / totals[:, None]

    def weights(self, grid):
        if grid != self._grid:
            raise RuntimeError(
                f"the bins are defined on the grid {self._grid}, not on "
                f"{grid}")
        index = self._index.ravel()
        pixels = np.flatnonzero(index >= 0)
        return scipy.sparse.csr_array(
            (np.ones(pixels.size), (index[pixels], pixels)),
            shape=(self._nbins, index.size))


class RegionsApertures(Regions):
    """Apertures on the sky (see Aperture), one region each."""

    @staticmethod
    def type():
        return 'apertures'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'RegionsApertures':
        parseutils.load_option_and_update_info(
            aperture_parser, info, 'apertures', required=True)
        return super().load(info)

    def dump(self, prefix='', dump_path=True, overwrite=False):
        return dict(
            type=self.type(),
            apertures=aperture_parser.dump(list(self._apertures)))

    def __init__(self, apertures: Sequence[Aperture]):
        check_options(RegionsApertures.__init__, locals())
        if not apertures:
            raise RuntimeError("at least one aperture is required")
        self._apertures = tuple(apertures)

    def __eq__(self, other):
        return (isinstance(other, RegionsApertures)
                and aperture_parser.dump(list(self._apertures))
                == aperture_parser.dump(list(other._apertures)))

    def apertures(self) -> tuple[Aperture, ...]:
        return self._apertures

    def nregions(self):
        return len(self._apertures)

    def grid(self):
        return None

    def centres(self):
        return np.array([aperture.centre() for aperture in self._apertures])

    def weights(self, grid):
        npix = int(np.prod(grid.size[:2]))
        rows, columns, values = [], [], []
        for region, aperture in enumerate(self._apertures):
            try:
                pixels, fractions = aperture.overlaps(grid)
            except RuntimeError as e:
                raise RuntimeError(f"aperture {region}: {e}") from e
            rows.append(np.full(pixels.size, region))
            columns.append(pixels)
            values.append(fractions)
        return scipy.sparse.csr_array(
            (np.concatenate(values),
             (np.concatenate(rows), np.concatenate(columns))),
            shape=(len(self._apertures), npix))


regions_parser = parseutils.TypedParser(Regions, [
    RegionsApertures,
    RegionsBins])
