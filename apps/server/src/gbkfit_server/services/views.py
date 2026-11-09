"""
Views of the cubes of files of data of the workspace, to look at them: a
channel, the spectrum of a pixel, and their quick moment maps, by the
tools of the library (read_oriented, prep.signal_mask,
analysis.moment_maps). A cube is read once, in the orientation of the
model, and kept for the next views (the latest few cubes). Its spectral
axis is that of its header, in a unit to read (km/s, GHz or micron),
whatever the model makes of it: a cube of wavelengths is seen before it
is made one of velocities. Cubes made here (e.g. a render of a galaxy, see
Views.put) are kept the same way, at paths of their own ('@render/1').
"""

import collections
import dataclasses
import os
import threading
import warnings
from typing import Literal

import astropy.units as u
import astropy.wcs
import numpy as np

import gbkfit.analysis
import gbkfit.dataset
from gbkfit import prep
from gbkfit.config import ConfigError
from gbkfit.io import read_oriented

from .errors import Invalid
from .workspace import Workspace


__all__ = [
    'MADE_PREFIX',
    'Cube',
    'Views'
]


# The cubes kept
_KEPT = 2

# The cubes made here that are kept (the latest), and the start of their
# paths
_MADE = 8
MADE_PREFIX = '@render/'

# The units of the spectral axes shown, by the kind of their coordinates
# (the first four letters of their type)
_SPECTRAL_UNITS = dict(
    VELO='km/s', VRAD='km/s', VOPT='km/s', FELO='km/s', FREQ='GHz',
    WAVE='micron', AWAV='micron')

# The signal-to-noise ratio to which the mask of the moments grows from
# its seeds, as a fraction of that of the seeds, and the smallest region
# of the mask kept, as a fraction of the largest (see prep.signal_mask)
_GROW = 0.5
_REGIONS = 0.1

# The values sampled for the range of the colours of a cube, and the
# percentiles of the range: from the median (of the noise, in a cube of
# lines; below it, the colour of the background), up to its brightest
_SAMPLES = 1_000_000
_PERCENTILES = (50, 99.9)


@dataclasses.dataclass(frozen=True)
class Cube:
    """
    A cube in the orientation of the model: its values (channels, ny,
    nx), the coordinates of its channels along its spectral axis and
    their unit (see _SPECTRAL_UNITS; None if its header does not give
    them), the range of its values for colours (percentiles of its
    finite values, see cube_range), and the least and the greatest of its
    finite values (None if it has none; see cube_extent).
    """
    values: np.ndarray
    spectral: np.ndarray | None
    unit: str | None
    range: tuple[float, float]
    extent: tuple[float, float] | None

    @classmethod
    def of(
            cls,
            values: np.ndarray,
            spectral: np.ndarray | None = None,
            unit: str | None = None
    ) -> 'Cube':
        """Return the cube of values (with the coordinates of channels)."""
        return cls(
            values, spectral, unit, cube_range(values), cube_extent(values))

    def velocities(self) -> np.ndarray | None:
        """Return the velocities of its channels (km/s), if they are."""
        return self.spectral if self.unit == 'km/s' else None


class Views:
    """
    The views of the cubes of a workspace.

    Parameters
    ----------
    workspace : Workspace
        The workspace.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace
        self._cubes: collections.OrderedDict[tuple, Cube] = \
            collections.OrderedDict()
        # (the moment maps of the cubes kept, by their signal-to-noise
        # ratios and methods)
        self._moments: dict[
            tuple, dict[tuple[float, str], dict[str, np.ndarray]]] = {}
        self._lock = threading.Lock()
        # (the cubes made here, by their paths, and how many were made)
        self._made: collections.OrderedDict[str, Cube] = \
            collections.OrderedDict()
        self._count = 0

    def put(self, cube: Cube) -> str:
        """
        Keep a cube made here (not of a file), and return its path, which
        the views take as that of a file (its array 0); the latest few are
        kept.
        """
        with self._lock:
            self._count += 1
            path = f'{MADE_PREFIX}{self._count}'
            self._made[path] = cube
            while len(self._made) > _MADE:
                old, _ = self._made.popitem(last=False)
                self._moments.pop((old, 0, 0), None)
        return path

    def cube(self, path: str, hdu: int) -> Cube:
        """
        Return a cube: an array of a file of data of the workspace.

        Raises
        ------
        NotFound
            If the file does not exist.
        Invalid
            If the array is not a cube, or a cube made here is no longer
            kept.
        """
        if path.startswith(MADE_PREFIX):
            with self._lock:
                if path in self._made and hdu == 0:
                    return self._made[path]
            raise Invalid(f"the cube {path} is no longer kept; make it again")
        key = self._key(path, hdu)
        with self._lock:
            if key in self._cubes:
                self._cubes.move_to_end(key)
                return self._cubes[key]
        cube = _read(self._workspace.file(path), hdu)
        with self._lock:
            self._cubes[key] = cube
            while len(self._cubes) > _KEPT:
                old, _ = self._cubes.popitem(last=False)
                self._moments.pop(old, None)
        return cube

    def channel(self, path: str, hdu: int, channel: int) -> np.ndarray:
        """
        Return a channel of a cube (ny, nx).

        Raises
        ------
        Invalid
            If the cube has no such channel (or as cube).
        """
        values = self.cube(path, hdu).values
        if not 0 <= channel < values.shape[0]:
            raise Invalid(
                f"the cube has {values.shape[0]} channels; not {channel}")
        return values[channel]

    def spectrum(
            self, path: str, hdu: int, x: int | None = None,
            y: int | None = None,
            region: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        """
        Return the spectrum of a pixel (x, y) of a cube (its channels), or,
        without one, its total spectrum: the sum of the values of its
        pixels in each channel (NaN left out; NaN where all are), of all
        its pixels or of those of a region (x0, x1, y0, y1, inclusive).

        Raises
        ------
        Invalid
            If the pixel or the region is not in the cube (or as cube).
        """
        values = self.cube(path, hdu).values
        _, ny, nx = values.shape
        if x is not None or y is not None:
            if x is None or y is None or not (0 <= x < nx and 0 <= y < ny):
                raise Invalid(
                    f"the cube has {nx} x {ny} pixels; not ({x}, {y})")
            return values[:, y, x]
        x0, x1, y0, y1 = (0, nx - 1, 0, ny - 1) if region is None else region
        if not (0 <= x0 <= x1 < nx and 0 <= y0 <= y1 < ny):
            raise Invalid(
                f"the cube has {nx} x {ny} pixels; not the region "
                f"({x0}-{x1}, {y0}-{y1})")
        pixels = values[:, y0:y1 + 1, x0:x1 + 1]
        missing = np.isnan(pixels).all(axis=(1, 2))
        return np.where(missing, np.nan, np.nansum(pixels, axis=(1, 2)))

    def moments(
            self, path: str, hdu: int, snr: float,
            method: Literal['moments', 'gauss_hermite'] = 'moments',
            rest: str | None = None, velocity: str | None = None,
            spectral_type: str | None = None
    ) -> dict[str, np.ndarray]:
        """
        Return the quick moment maps of a cube, by name, of the voxels of
        the mask of its signal grown from seeds of a signal-to-noise ratio
        (see prep.signal_mask; 0: of all of them): measured as moments,
        its flux, velocity, dispersion, skewness and kurtosis; or of a
        Gauss-Hermite series fitted to each spectrum (slower), its h3 and
        h4 (see gbkfit.analysis.moment_maps). Channels of wavelengths or
        frequencies are of the velocities of a rest, as a dataset of the
        cube with it, the convention of velocities its header leaves
        ambiguous and the type of an axis FITS does not define reads them
        (see Workspace.describe).

        Raises
        ------
        Invalid
            If its channels are not of velocities, of the rest given if any
            (or as cube).
        """
        cube = self.cube(path, hdu)
        key = self._key(path, hdu)
        kept = (snr, method, rest, velocity, spectral_type)
        with self._lock:
            if kept in self._moments.get(key, {}):
                return self._moments[key][kept]
        velocities = cube.velocities()
        if velocities is None and not path.startswith(MADE_PREFIX):
            velocities = self._velocities(
                path, hdu, cube, rest, velocity, spectral_type)
        if velocities is None:
            raise Invalid(
                f"the channels of '{path}' are not of velocities: its "
                f"moments need them (of wavelengths or frequencies, the "
                f"rest of a line)")
        if snr == 0:
            values, blank = cube.values, None
        else:
            mask = prep.signal_mask(cube.values, snr, _GROW * snr, _REGIONS)
            # (a voxel left out has no flux: a missing value (NaN) would
            # mask the moments of its whole spectrum; see moment_maps);
            # the pixels with none kept have no moments
            values = np.where(mask, cube.values, 0)
            blank = ~mask.any(axis=0)
        maps = gbkfit.analysis.moment_maps(
            values, velocities, orders=(0, 1, 2, 3, 4), method=method)
        result = dict(
            flux=maps.data['moment0'], velocity=maps.data['moment1'],
            dispersion=maps.data['moment2'], skewness=maps.skewness(),
            kurtosis=maps.kurtosis()) if method == 'moments' \
            else dict(h3=maps.data['h3'], h4=maps.data['h4'])
        if blank is not None:
            result = {
                name: np.where(blank, np.nan, values)
                for name, values in result.items()}
        with self._lock:
            if key in self._cubes:
                self._moments.setdefault(key, {})[kept] = result
        return result

    def _velocities(
            self, path: str, hdu: int, cube: Cube, rest: str | None,
            velocity: str | None, spectral_type: str | None
    ) -> np.ndarray | None:
        """
        Return the velocities of the channels of a cube of a file as a
        dataset of it reads them, with a rest, a convention and a type, if
        given (see gbkfit.dataset.describe_fits; none if they are not read
        so), in the order of the channels of the cube.
        """
        if cube.spectral is None:
            return None
        try:
            hdus = gbkfit.dataset.describe_fits(
                self._workspace.file(path), velocity, rest, spectral_type)
        except OSError:
            return None
        coords = next((h.coords for h in hdus if h.index == hdu), None)
        if coords is None or len(coords.step) < 3:
            return None
        channels = np.arange(cube.values.shape[0])
        velocities = \
            coords.rval[2] + (channels - coords.rpix[2]) * coords.step[2]
        # (those of the dataset go to greater velocities, as the model's
        # channels do; those of the cube, as its file has them, to longer
        # wavelengths or lower frequencies, or the other way)
        rising = cube.spectral[-1] > cube.spectral[0]
        return velocities if rising == (cube.unit == 'micron') \
            else velocities[::-1]

    def _key(self, path: str, hdu: int) -> tuple:
        """
        Return the key of a cube: its file and array, and when the file
        changed (a file written again is read again); of a cube made here,
        its path.
        """
        if path.startswith(MADE_PREFIX):
            return path, hdu, 0
        filename = self._workspace.file(path)
        return filename, hdu, os.stat(filename).st_mtime_ns


def _read(filename: str, hdu: int) -> Cube:
    """
    Read a cube: an array of a FITS file, as read_oriented reads
    it, with the coordinates of its channels if its header gives them; a
    map (e.g. a moment map) is read as a cube of one channel, without them.

    Raises
    ------
    Invalid
        If the file cannot be read, or the array is neither a cube nor a
        map.
    """
    try:
        values, wcs = read_oriented((filename, hdu))
    except (OSError, IndexError, ConfigError) as e:
        raise Invalid(f"cannot read the array {hdu} of {filename}: {e}") \
            from e
    if values.ndim not in (2, 3):
        raise Invalid(
            f"the array {hdu} of {filename} is neither a cube nor a map: "
            f"its shape is {values.shape}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        return Cube.of(values[np.newaxis])
    spectral, unit = _spectral(wcs, values.shape[0])
    return Cube.of(values, spectral, unit)


def _spectral(
        wcs: astropy.wcs.WCS | None, nchannels: int
) -> tuple[np.ndarray | None, str | None]:
    """
    Return the coordinates of the channels of a cube along its spectral
    axis, and their unit (see _SPECTRAL_UNITS), if its world coordinates
    give them.
    """
    if wcs is None or wcs.wcs.spec < 0:
        return None, None
    axis = wcs.sub([wcs.wcs.spec + 1])
    unit = _SPECTRAL_UNITS.get(str(axis.wcs.ctype[0])[:4])
    if unit is None:
        return None, None
    try:
        values = axis.all_pix2world(np.arange(nchannels), 0)[0]
        return (values * u.Unit(axis.wcs.cunit[0])).to_value(unit), unit
    except (ValueError, u.UnitsError):
        return None, None


def cube_range(values: np.ndarray) -> tuple[float, float]:
    """
    Return the range of the values of a cube for colours: percentiles of
    a sample of its finite values ((0, 1) if it has none).
    """
    sample = values.ravel()[::max(1, values.size // _SAMPLES)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(sample, _PERCENTILES)
    return float(low), float(high)


def cube_extent(values: np.ndarray) -> tuple[float, float] | None:
    """
    Return the least and the greatest of the finite values of a cube
    (None if it has none).
    """
    with warnings.catch_warnings():
        # (of no values: NaN)
        warnings.simplefilter('ignore', RuntimeWarning)
        low, high = np.nanmin(values), np.nanmax(values)
    if not (np.isfinite(low) and np.isfinite(high)):
        # (no values, or infinities among them: of the finite ones)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        low, high = finite.min(), finite.max()
    return float(low), float(high)
