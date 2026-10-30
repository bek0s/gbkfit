"""
Views of the cubes of files of data of the workspace, to look at them: a
channel, the spectrum of a pixel, and their quick moment maps. A cube is
read once, in the orientation of the model, and kept for the next views
(the latest few cubes). Its spectral axis is that of its header, in a
unit to read (km/s, GHz or micron), whatever the model makes of it: a
cube of wavelengths is seen before it is made one of velocities.
"""

import collections
import dataclasses
import os
import threading
import warnings

import astropy.units as u
import astropy.wcs
import numpy as np
from astropy.io import fits

import gbkfit.analysis
from gbkfit.utils import fitsutils

from .errors import Invalid
from .workspace import Workspace


__all__ = [
    'Cube',
    'Views'
]


# The cubes kept
_KEPT = 2

# The units of the spectral axes shown, by the kind of their coordinates
# (the first four letters of their type)
_SPECTRAL_UNITS = dict(
    VELO='km/s', VRAD='km/s', VOPT='km/s', FELO='km/s', FREQ='GHz',
    WAVE='micron', AWAV='micron')

# The values sampled for the range of the colours of a cube, and the
# percentiles of the range
_SAMPLES = 1_000_000
_PERCENTILES = (0.5, 99.5)


@dataclasses.dataclass(frozen=True)
class Cube:
    """
    A cube in the orientation of the model: its values (channels, ny,
    nx), the coordinates of its channels along its spectral axis and
    their unit (see _SPECTRAL_UNITS; None if its header does not give
    them), and the range of its values for colours (percentiles of its
    finite values).
    """
    values: np.ndarray
    spectral: np.ndarray | None
    unit: str | None
    range: tuple[float, float]

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
        self._moments: dict[tuple, gbkfit.analysis.MomentMaps] = {}
        self._lock = threading.Lock()

    def cube(self, path: str, hdu: int) -> Cube:
        """
        Return a cube: an array of a file of data of the workspace.

        Raises
        ------
        NotFound
            If the file does not exist.
        Invalid
            If the array is not a cube.
        """
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

    def spectrum(self, path: str, hdu: int, x: int, y: int) -> np.ndarray:
        """
        Return the spectrum of a pixel of a cube (its channels).

        Raises
        ------
        Invalid
            If the pixel is not in the cube (or as cube).
        """
        values = self.cube(path, hdu).values
        _, ny, nx = values.shape
        if not (0 <= x < nx and 0 <= y < ny):
            raise Invalid(f"the cube has {nx} x {ny} pixels; not ({x}, {y})")
        return values[:, y, x]

    def moments(self, path: str, hdu: int) -> gbkfit.analysis.MomentMaps:
        """
        Return the quick moment maps of a cube: its flux, velocity and
        dispersion (see gbkfit.analysis.moment_maps).

        Raises
        ------
        Invalid
            If the header does not give the velocities of its channels (or
            as cube).
        """
        cube = self.cube(path, hdu)
        key = self._key(path, hdu)
        with self._lock:
            if key in self._moments:
                return self._moments[key]
        if (velocities := cube.velocities()) is None:
            raise Invalid(
                f"the channels of '{path}' are not of velocities: its "
                f"moments need them")
        maps = gbkfit.analysis.moment_maps(cube.values, velocities)
        with self._lock:
            if key in self._cubes:
                self._moments[key] = maps
        return maps

    def _key(self, path: str, hdu: int) -> tuple:
        """
        Return the key of a cube: its file and array, and when the file
        changed (a file written again is read again).
        """
        filename = self._workspace.file(path)
        return filename, hdu, os.stat(filename).st_mtime_ns


def _read(filename: str, hdu: int) -> Cube:
    """
    Read a cube: an array of a FITS file, in the orientation of the model
    if its header allows (see fitsutils.to_model_axes), without its axes
    of length 1 (e.g. Stokes), with the coordinates of its channels if
    its header gives them.

    Raises
    ------
    Invalid
        If the file cannot be read, or the array is not a cube.
    """
    try:
        with fits.open(filename, memmap=True) as hdus:
            header = hdus[hdu].header
            values = hdus[hdu].data
            if values is not None:
                values, wcs = _model_axes(values, header)
                values = np.array(values, dtype=np.float32)
    except (OSError, IndexError) as e:
        raise Invalid(f"cannot read the array {hdu} of {filename}: {e}") \
            from e
    if values is None or values.squeeze().ndim != 3:
        raise Invalid(
            f"the array {hdu} of {filename} is not a cube: its shape is "
            f"{None if values is None else values.shape}")
    keep = tuple(i for i, size in enumerate(values.shape) if size > 1)
    values = values.reshape([values.shape[i] for i in keep])
    spectral, unit = _spectral(wcs, values.shape[0])
    return Cube(values, spectral, unit, _range(values))


def _model_axes(
        values: np.ndarray, header: fits.Header
) -> tuple[np.ndarray, astropy.wcs.WCS | None]:
    """
    Return an array and its world coordinates in the orientation of the
    model; the array as it is, and no coordinates, if they cannot be
    read.
    """
    try:
        # (the fixes of astropy to the header are not of the orientation)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', astropy.wcs.FITSFixedWarning)
            wcs = astropy.wcs.WCS(
                fitsutils.centre_missing_crpix(header, values.shape))
        return fitsutils.to_model_axes(values, wcs)
    except Exception:
        # (any error of astropy: the array is shown as it is)
        return values, None


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


def _range(values: np.ndarray) -> tuple[float, float]:
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
