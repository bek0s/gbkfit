"""
Grids of pixels with world coordinates, and data on them.

The axes of a grid are in FITS order: x, y and, for spectral cubes, the
velocity. The model measures the positions on the sky in arcsec from the
reference pixel of its grid (see sky_positions); other grids (e.g. that of
the image of a primary beam) are placed by their RA and Dec (see
pixels_on).
"""

import numbers
from collections.abc import Sequence
from typing import NamedTuple

import astropy.units
import numpy as np

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Coords',
    'Grid',
    'GridData',
    'SpectraData',
    'check_overlap',
    'make_grid',
    'make_rest',
    'pixels_on',
    'sky_positions',
    'sky_to_pixel'
]


# Arcsec per radian
_ARCSEC = np.degrees(1) * 3600


class Coords(NamedTuple):
    """
    The world coordinates of the pixels of data, in the units of the
    model, for each axis in FITS order.

    Axes without a known type have the values of their header.

    Attributes
    ----------
    step : tuple of float
        The size of the pixels: arcsec (x and y) and km/s (the velocity).
    rpix : tuple of float
        The reference pixel (0-based).
    rval : tuple of float
        The world position at rpix: RA and Dec (degrees) and the velocity
        (km/s). The model measures the x and y axes from rpix.
    rota : float
        The rotation of the pixel grid on the sky: the position angle
        (degrees, north through east) of the +y axis.
    rest : Quantity or None
        The rest wavelength or frequency that the velocities of the
        spectral axis refer to (in m or Hz; see make_rest). The velocities
        of a rest wavelength are optical (VOPT), and those of a rest
        frequency radio (VRAD).
    """
    step: tuple[float, ...]
    rpix: tuple[float, ...]
    rval: tuple[float, ...]
    rota: float
    rest: astropy.units.Quantity | None = None

    def axes(self, *indices: int) -> 'Coords':
        """
        Return the world coordinates of some of the axes.

        Parameters
        ----------
        *indices : int
            The indices of the axes.

        Returns
        -------
        Coords
            The world coordinates of the axes, with the same rota and rest.
        """
        def pick(values):
            return tuple(values[i] for i in indices)
        return Coords(
            pick(self.step), pick(self.rpix), pick(self.rval), self.rota,
            self.rest)


class Grid(NamedTuple):
    """
    A grid of pixels with world coordinates.

    Attributes
    ----------
    size : tuple of int
        The number of pixels of each axis, in FITS order.
    coords : Coords
        The world coordinates.
    spectral_axis : int or None
        The index of the spectral axis, if any.
    """
    size: tuple[int, ...]
    coords: Coords
    spectral_axis: int | None

    def zero(self) -> tuple[float, ...]:
        """
        Return the world position of the first pixel of each axis.

        Returns
        -------
        tuple of float
            The positions, in model units: the x and y axes are measured
            from the reference pixel, and the spectral axis from its world
            value there.
        """
        step, rpix, rval = self.coords.step, self.coords.rpix, self.coords.rval
        return tuple(
            (rval[axis] if axis == self.spectral_axis else 0)
            - rpix[axis] * step[axis]
            for axis in range(len(self.size)))

    def spatial(self) -> 'Grid':
        """
        Return the grid of the x and y axes.

        Returns
        -------
        Grid
            The grid, without a spectral axis.
        """
        return Grid(
            self.size[:2], self.coords.axes(0, 1)._replace(rest=None), None)

    def spectral(self) -> 'Grid':
        """
        Return the grid of the spectral axis.

        Returns
        -------
        Grid
            The grid of one axis, not rotated.
        """
        axis = self.spectral_axis
        return Grid(
            (self.size[axis],), self.coords.axes(axis)._replace(rota=0), 0)


def make_grid(
        size: Sequence[int],
        step: float | Sequence[float] | None = None,
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None,
        rota: float | None = None,
        spectral_axis: int | None = None,
        rest: str | astropy.units.Quantity | None = None
) -> Grid:
    """
    Make a grid.

    Parameters
    ----------
    size : Sequence of int
        The number of pixels of each axis, in FITS order.
    step, rpix, rval : float or Sequence of float, optional
        The world coordinates (see Coords): a value for all axes, or one
        for each. By default, step 1, the reference pixel at the centre and
        reference value 0.
    rota : float, optional
        The rotation of the grid on the sky (see Coords); by default 0.
    spectral_axis : int, optional
        The index of the spectral axis, if any.
    rest : str or Quantity, optional
        The rest of the spectral axis (see make_rest).

    Returns
    -------
    Grid
        The grid.

    Raises
    ------
    ConfigError
        If the sizes are not positive integers, the world coordinates do
        not have a value for each axis, a step is not positive, or the
        spectral axis is not an axis of the grid, or if a grid without a
        spectral axis has a rest.
    """
    ndim = len(size)
    if not all(isinstance(n, numbers.Integral) and not isinstance(n, bool)
               and n > 0 for n in size):
        raise ConfigError(
            f"the sizes of a grid must be positive integers; they are "
            f"{tuple(size)}")
    if step is None:
        step = 1
    if rpix is None:
        rpix = tuple((np.asarray(size) / 2 - 0.5).tolist())
    if rval is None:
        rval = 0
    if rota is None:
        rota = 0
    step, rpix, rval = (
        (value,) * ndim if np.ndim(value) == 0 else tuple(value)
        for value in (step, rpix, rval))
    for name, value in dict(step=step, rpix=rpix, rval=rval).items():
        if len(value) != ndim:
            raise ConfigError(
                f"the grid has {ndim} axes, but {name} has {len(value)} "
                f"values")
    if not all(value > 0 for value in step):
        raise ConfigError(f"step must be positive; it is {step}")
    if spectral_axis is not None and spectral_axis not in range(ndim):
        raise ConfigError(
            f"the grid has {ndim} axes; its spectral axis cannot be "
            f"{spectral_axis}")
    if rest is not None and spectral_axis is None:
        raise ConfigError("a grid without a spectral axis has no rest")
    return Grid(
        tuple(size), Coords(step, rpix, rval, rota, make_rest(rest)),
        spectral_axis)


def make_rest(
        value: str | astropy.units.Quantity | None
) -> astropy.units.Quantity | None:
    """
    Make the rest wavelength or frequency of a spectral axis.

    Parameters
    ----------
    value : str or Quantity or None
        The rest, with units (e.g. '6562.8 Angstrom', '1420.405752 MHz').

    Returns
    -------
    Quantity or None
        The rest, in m or Hz, or None for None.

    Raises
    ------
    ConfigError
        If the value is not a positive wavelength or frequency.
    """
    if value is None:
        return None
    error = ConfigError(
        f"the rest of a spectral axis must be a positive wavelength or "
        f"frequency with units (e.g. '6562.8 Angstrom'); it is {value!r}")
    try:
        rest = astropy.units.Quantity(value)
    except (TypeError, ValueError) as e:
        raise error from e
    for unit in (astropy.units.m, astropy.units.Hz):
        if rest.unit.is_equivalent(unit) and rest.isscalar and rest.value > 0:
            return rest.to(unit)
    raise error


def sky_positions(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the positions on the sky of the pixels of a grid.

    The positions are in the frame of the model: arcsec from the reference
    pixel, with x and y like xpos and ypos (+y to the north, +x to the
    west), so the pixel grid is rotated in it by rota (see Coords).

    Parameters
    ----------
    grid : Grid
        The grid; only its x and y axes are used.

    Returns
    -------
    tuple of ndarray
        The x and y of the centres of the pixels, each of shape (ny, nx).
    """
    size_x, size_y = grid.size[:2]
    zero_x, zero_y = grid.zero()[:2]
    step_x, step_y = grid.coords.step[:2]
    j, i = np.mgrid[0:size_y, 0:size_x]
    x = zero_x + i * step_x
    y = zero_y + j * step_y
    rota = np.radians(grid.coords.rota)
    return (x * np.cos(rota) - y * np.sin(rota),
            x * np.sin(rota) + y * np.cos(rota))


def sky_to_pixel(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the map from the sky to the pixels of a grid.

    The map is the inverse of sky_positions: pixel = matrix @ sky + offset,
    with the sky in the frame of the model.

    Parameters
    ----------
    grid : Grid
        The grid; only its x and y axes are used.

    Returns
    -------
    tuple of ndarray
        The matrix (2, 2) and the offset (2,).
    """
    rota = np.radians(grid.coords.rota)
    rotation = np.array([
        [np.cos(rota), np.sin(rota)],
        [-np.sin(rota), np.cos(rota)]])
    matrix = rotation / np.asarray(grid.coords.step[:2], float)[:, None]
    return matrix, np.asarray(grid.coords.rpix[:2], float)


def _to_sky(x, y, rval):
    """
    Return the RA and Dec (radians) of positions in the frame of a grid
    (arcsec; see sky_positions) with the reference RA and Dec rval
    (degrees), as a gnomonic (TAN) projection.
    """
    ra0, dec0 = np.radians(rval[:2])
    east, north = -np.asarray(x) / _ARCSEC, np.asarray(y) / _ARCSEC
    denominator = np.cos(dec0) - north * np.sin(dec0)
    ra = ra0 + np.arctan2(east, denominator)
    dec = np.arctan2(
        np.sin(dec0) + north * np.cos(dec0), np.hypot(east, denominator))
    return ra, dec


def _from_sky(ra, dec, rval):
    """The inverse of _to_sky."""
    ra0, dec0 = np.radians(rval[:2])
    cos_distance = (
        np.sin(dec0) * np.sin(dec)
        + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0))
    east = np.cos(dec) * np.sin(ra - ra0) / cos_distance
    north = (
        np.cos(dec0) * np.sin(dec)
        - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_distance
    return -east * _ARCSEC, north * _ARCSEC


def pixels_on(grid: Grid, other: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the pixel coordinates, on another grid, of the pixels of a grid,
    by their RA and Dec.

    The positions on the sky of both grids are gnomonic (TAN) projections
    around their reference pixels.

    Parameters
    ----------
    grid : Grid
        The grid whose pixels are placed; only its x and y axes are used.
    other : Grid
        The grid of the pixel coordinates (e.g. that of an image to sample
        at the pixels of grid); only its x and y axes are used.

    Returns
    -------
    tuple of ndarray
        The x and y pixel coordinates (0-based) on other of the centres of
        the pixels of grid, each of shape (ny, nx) of grid.
    """
    ra, dec = _to_sky(*sky_positions(grid), grid.coords.rval)
    x, y = _from_sky(ra, dec, other.coords.rval)
    matrix, rpix = sky_to_pixel(other)
    return (matrix[0, 0] * x + matrix[0, 1] * y + rpix[0],
            matrix[1, 0] * x + matrix[1, 1] * y + rpix[1])


def check_overlap(grid: Grid, other: Grid, desc: str) -> None:
    """
    Check that another grid covers some of the pixels of a grid on the sky.

    Parameters
    ----------
    grid : Grid
        The grid whose pixels must be covered.
    other : Grid
        The grid that must cover them.
    desc : str
        What other is, for messages (e.g. 'the primary beam image').

    Raises
    ------
    RuntimeError
        If other covers none of the pixels of grid (e.g. because one of
        them has the wrong RA and Dec).
    """
    pixel_x, pixel_y = pixels_on(grid, other)
    size_x, size_y = other.size[:2]
    if not np.any(
            (pixel_x >= -0.5) & (pixel_x <= size_x - 0.5)
            & (pixel_y >= -0.5) & (pixel_y <= size_y - 0.5)):
        x, y = _from_sky(
            *np.radians(other.coords.rval[:2]), grid.coords.rval)
        raise RuntimeError(
            f"{desc} does not cover any of the pixels it is used on: by "
            f"their RA and Dec, its reference pixel is at x = {x:.6g}, "
            f"y = {y:.6g} arcsec from theirs; check the world coordinates "
            f"of both, or give its rval")


class GridData(NamedTuple):
    """
    Data on a grid with world coordinates (see fitsutils.write_data).

    Attributes
    ----------
    data : ndarray
        The data.
    coords : Coords
        The world coordinates of its axes, in FITS order.
    spectral_axis : int or None
        The index of its spectral axis (FITS order), if any.
    """
    data: np.ndarray
    coords: Coords
    spectral_axis: int | None


class SpectraData(NamedTuple):
    """
    Spectra of regions of the sky, with the world coordinates of their
    spectral axis (see fitsutils.write_spectra).

    Attributes
    ----------
    data : ndarray
        The spectra, of shape (nchannels, nregions): in FITS, the regions
        along x and the velocity along y.
    coords : Coords
        The world coordinates of the spectral axis (one axis).
    """
    data: np.ndarray
    coords: Coords
