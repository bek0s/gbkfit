"""
The regions of the sky in which data were measured (see Regions): bins
of pixels, or apertures.
"""

import abc
import os.path
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse
from astropy.io import fits

from gbkfit.config import (
    FitsFile, TypedParser, TypedSerializable, Unit, check_options,
    split_fits_file)
from gbkfit.grid import Grid, make_grid, sky_positions
from gbkfit.io import read_data, write_data
from .apertures import Aperture, aperture_parser


__all__ = [
    'Regions',
    'RegionsApertures',
    'RegionsBins',
    'regions_parser'
]


# The world coordinates of a grid with one value per axis, x and y, or one
# for both (see Coords), and its rotation
_PerAxis: TypeAlias = float | Sequence[float] | None
_Angle: TypeAlias = Annotated[float, Unit('deg')]


def _per_axis(value: _PerAxis) -> list[float] | None:
    """Return one value of a world coordinate per axis, x and y."""
    return [value] * 2 if isinstance(value, int | float) \
        else None if value is None else list(value)


class Regions(TypedSerializable, abc.ABC):
    """
    The regions of the sky in which data were measured, each the weighted
    sum of the pixels of a spatial grid (see weights()): e.g. bins of
    spaxels, or apertures.
    """

    @abc.abstractmethod
    def nregions(self) -> int:
        pass

    @abc.abstractmethod
    def grid(self) -> Grid | None:
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
    def weights(self, grid: Grid) -> scipy.sparse.csr_array:
        """
        The weights of the pixels of a spatial grid in the regions: a
        matrix of one row for each region and one column for each pixel
        (flat index, x fastest).
        """
        pass

    def valid(self) -> np.ndarray | None:
        """
        Which pixels of its grid count in each channel of spectra: 1 or
        0, of shape (channels, ny, nx), e.g. of a spectrum of a bin that
        sums the valid pixels of each channel of a cube (see RegionsBins);
        None (here) if all do.
        """
        return None


class RegionsBins(Regions):
    """
    Bins of the pixels of a spatial grid (as Voronoi bins), given by the
    bin of each pixel. The bin of each pixel (index) is 0 to n - 1, or
    negative for the pixels in no bin. Each bin is the sum of its pixels;
    of spectra, optionally of those valid in each channel (valid: 1 or 0,
    of shape (channels, ny, nx), e.g. those of a cube of which some
    spaxels are masked in some channels; see prep_region_spectra), which
    the model sums too.
    """

    @staticmethod
    def type() -> str:
        return 'bins'

    @staticmethod
    def type_title() -> str:
        return 'Bins of pixels'

    config_constructors = ('from_file',)

    @classmethod
    def from_file(
            cls,
            file: FitsFile,
            step: _PerAxis = None,
            rpix: _PerAxis = None,
            rval: _PerAxis = None,
            rota: _Angle | None = None,
            valid: FitsFile | None = None
    ) -> 'RegionsBins':
        """
        Read the bins from a FITS file, with the world coordinates of its
        header unless given, and which pixels are valid in each channel
        from another, if given.

        Parameters
        ----------
        file : str or tuple
            The file of the bin of each pixel (see the class): a filename,
            or a filename and the HDU.
        step, rpix, rval : float or Sequence of float, optional
            The world coordinates of the grid (see Coords); by
            default, those of the header. Either rpix or rval can be given,
            and the other comes from the header (see read_data).
        rota : float, optional
            The rotation of the grid on the sky; by default, that of the
            header.
        valid : str or tuple, optional
            The file of which pixels are valid in each channel (see the
            class; e.g. prep_<cube>_valid.fits of prep_region_spectra).

        Returns
        -------
        RegionsBins
            The bins.
        """
        check_options(RegionsBins.from_file, locals())
        index, coords = read_data(
            *split_fits_file(file), _per_axis(rpix), _per_axis(rval))
        if valid is not None:
            valid, _ = read_data(*split_fits_file(valid))
        return cls(
            index, coords.step if step is None else step, coords.rpix,
            coords.rval, coords.rota if rota is None else rota, valid)

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}bins.fits'
        coords = self._grid.coords
        write_data(
            filename, self._index.astype(np.int32), coords, None, overwrite)
        info = dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            step=coords.step,
            rpix=coords.rpix,
            rval=coords.rval,
            rota=coords.rota)
        if self._valid is not None:
            # (an array without world coordinates, in the orientation of
            # the model: the bins give them)
            valid = f'{prefix}bins_valid.fits'
            fits.writeto(valid, self._valid, overwrite=overwrite)
            info['valid'] = valid if dump_path else os.path.basename(valid)
        return info

    def __init__(
            self,
            index: npt.ArrayLike,
            step: _PerAxis = None,
            rpix: _PerAxis = None,
            rval: _PerAxis = None,
            rota: _Angle | None = None,
            valid: npt.ArrayLike | None = None
    ):
        """
        The world coordinates of the grid of the bins (see
        Coords) have defaults (see make_grid).
        """
        check_options(RegionsBins.__init__, locals())
        index = np.asarray(index)
        if index.ndim != 2:
            raise ValueError(
                f"the bins of the pixels must be an image; they have "
                f"{index.ndim} axes")
        if not np.all(np.isfinite(index)) or np.any(index != np.round(index)):
            raise ValueError("the bins of the pixels must be integers")
        index = index.astype(np.int64)
        nbins = int(index.max()) + 1
        if nbins < 1:
            raise ValueError("no pixel is in a bin")
        counts = np.bincount(index[index >= 0], minlength=nbins)
        if empty := np.flatnonzero(counts == 0).tolist():
            raise ValueError(
                f"the bins are numbered from 0 to {nbins - 1}, but these "
                f"have no pixels: {empty}")
        if valid is not None:
            valid = np.asarray(valid)
            if valid.ndim != 3 or valid.shape[1:] != index.shape:
                raise ValueError(
                    f"the valid pixels of the bins must be a cube (channels, "
                    f"{index.shape[0]}, {index.shape[1]}), of the pixels of "
                    f"the bins; they have the shape {valid.shape}")
            valid = (np.nan_to_num(valid) != 0).astype(np.uint8)
        self._index = index
        self._nbins = nbins
        self._valid = valid
        self._grid = make_grid(
            index.shape[::-1], step, rpix, rval, rota)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, RegionsBins)
                and self._grid == other._grid
                and np.array_equal(self._index, other._index)
                and np.array_equal(self._valid, other._valid))

    def index(self) -> np.ndarray:
        """The bin of each pixel (negative for none)."""
        return self._index

    def valid(self) -> np.ndarray | None:
        return self._valid

    def nregions(self) -> int:
        return self._nbins

    def grid(self) -> Grid:
        return self._grid

    def centres(self) -> np.ndarray:
        weights = self.weights(self._grid)
        x, y = sky_positions(self._grid)
        totals = np.asarray(weights.sum(axis=1)).ravel()
        return np.column_stack(
            [weights @ x.ravel(), weights @ y.ravel()]) / totals[:, None]

    def weights(self, grid: Grid) -> scipy.sparse.csr_array:
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
    """
    Apertures on the sky (e.g. the circles of fibres), one region each.
    Their shapes are those of Aperture.
    """

    @staticmethod
    def type() -> str:
        return 'apertures'

    @staticmethod
    def type_title() -> str:
        return 'Apertures on the sky'

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            apertures=aperture_parser.dump(list(self._apertures)))

    def __init__(self, apertures: Sequence[Aperture]):
        check_options(RegionsApertures.__init__, locals())
        if not apertures:
            raise ValueError("at least one aperture is required")
        self._apertures = tuple(apertures)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, RegionsApertures)
                and aperture_parser.dump(list(self._apertures))
                == aperture_parser.dump(list(other._apertures)))

    def apertures(self) -> tuple[Aperture, ...]:
        return self._apertures

    def nregions(self) -> int:
        return len(self._apertures)

    def grid(self) -> None:
        return None

    def centres(self) -> np.ndarray:
        return np.array([aperture.centre() for aperture in self._apertures])

    def weights(self, grid: Grid) -> scipy.sparse.csr_array:
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


regions_parser = TypedParser(Regions, [
    RegionsApertures,
    RegionsBins])
