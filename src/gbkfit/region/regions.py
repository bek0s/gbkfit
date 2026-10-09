import abc
import os.path
from collections.abc import Sequence

import numpy as np
import scipy.sparse

from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .apertures import Aperture, aperture_parser


__all__ = [
    'Regions',
    'RegionsApertures',
    'RegionsBins',
    'regions_parser'
]


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

    @classmethod
    def load(cls, info, prefix=''):
        """
        Load the bins from a file (a filename, or a dict with the filename
        and the HDU), with the world coordinates of its header or of the
        options step, rpix, rval and rota.
        """
        desc = parseutils.make_typed_desc(cls, 'regions')
        parseutils.sanitize_dimensional_options(info, dict(
            step=int | float, rpix=int | float, rval=int | float), 2)
        if info.get('file') is None:
            raise ConfigError(f"option 'file' of {desc} is required")
        with parseutils.config_path('file'):
            file, hdu = parseutils.parse_file(info['file'], 'bins file')
            index, coords = fitsutils.read_data(
                prefix + file, hdu, info.get('rpix'), info.get('rval'))
        info = dict(info) | dict(
            index=index,
            step=coords.step if info.get('step') is None else info['step'],
            rpix=coords.rpix,
            rval=coords.rval,
            rota=coords.rota if info.get('rota') is None else info['rota'])
        info.pop('file')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

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
            index: np.ndarray,
            step: float | Sequence[float] | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rota: float | None = None
    ):
        """
        The world coordinates of the grid of the bins (see
        gridutils.Coords) have defaults (see gridutils.make_grid).
        """
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
    def load(cls, info, prefix=''):
        desc = parseutils.make_typed_desc(cls, 'regions')
        parseutils.load_option_and_update_info(
            aperture_parser, info, 'apertures', required=True)
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self, prefix='', dump_path=True, overwrite=False):
        return dict(
            type=self.type(),
            apertures=aperture_parser.dump(list(self._apertures)))

    def __init__(self, apertures: Sequence[Aperture]):
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
