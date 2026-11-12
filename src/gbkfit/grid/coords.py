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
from typing import Any, NamedTuple, TypeVar

import astropy.units
import numpy as np
import numpy.typing as npt

import gbkfit.units
from .spectral import SpectralAxis, make_convention, make_medium, make_rest


__all__ = [
    'Coords',
    'Grid',
    'GridData',
    'SpectraData',
    'check_overlap',
    'frame_to_pixels',
    'grid_options',
    'make_cube_grid',
    'make_grid',
    'per_axis',
    'pixels_on',
    'sky_positions',
    'sky_to_pixel',
    'x_axis_pa',
    'x_axis_rota'
]


# Arcsec per radian
_ARCSEC = np.degrees(1) * 3600

_T = TypeVar('_T')


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
        spectral axis refer to (in m or Hz; see make_rest).
    medium : str or None
        The medium of the wavelengths of the velocities and of their
        rest: 'vacuum' or 'air' (see make_medium); None without a rest
        wavelength.
    convention : str or None
        The convention of the velocities of the spectral axis (see
        make_convention): 'optical' (VOPT), 'log' (WAVE-LOG), 'radio'
        (VRAD) or 'relativistic' (VELO); None: the default of the rest
        (optical of a rest wavelength, radio of a rest frequency, or
        relativistic without a rest).
    frame : str or None
        The frame of rest of the velocities (FITS SPECSYS, see FRAMES; e.g.
        'BARYCENT' or 'LSRK'), if known.
    """
    step: tuple[float, ...]
    rpix: tuple[float, ...]
    rval: tuple[float, ...]
    rota: float
    rest: astropy.units.Quantity | None = None
    medium: str | None = None
    convention: str | None = None
    frame: str | None = None

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
            The world coordinates of the axes, with the same rota, rest and
            medium.
        """
        def pick(values: tuple[float, ...]) -> tuple[float, ...]:
            return tuple(values[i] for i in indices)
        return self._replace(
            step=pick(self.step), rpix=pick(self.rpix), rval=pick(self.rval))


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
            self.size[:2],
            self.coords.axes(0, 1)._replace(
                rest=None, medium=None, convention=None), None)

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


def per_axis(
        option: str,
        value: _T | Sequence[_T],
        naxes: int
) -> tuple[_T, ...]:
    """
    Return the values of an option of a grid with a value per axis (e.g.
    size, step), given one value for all axes, or one for each.

    Parameters
    ----------
    option : str
        The name of the option, for messages.
    value : Any or Sequence
        One value, or one for each axis.
    naxes : int
        The number of axes.

    Returns
    -------
    tuple
        The value of each axis.

    Raises
    ------
    ValueError
        If the option has neither one value nor one for each axis.
    """
    if np.ndim(value) == 0:
        return (value,) * naxes
    if len(value) != naxes:
        raise ValueError(
            f"option '{option}' must have one value for all {naxes} axes, "
            f"or one for each; it is {list(value)}")
    return tuple(value)


def make_grid(
        size: Sequence[int],
        step: float | Sequence[float] | None = None,
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None,
        rota: float | None = None,
        spectral_axis: int | None = None,
        rest: str | astropy.units.Quantity | None = None,
        medium: str | None = None,
        convention: str | None = None,
        frame: str | None = None
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
    medium : str, optional
        The medium of the wavelengths of a rest wavelength (see
        make_medium).
    convention : str, optional
        The convention of the velocities of the rest (see
        make_convention).
    frame : str, optional
        The frame of rest of the velocities (see Coords).

    Returns
    -------
    Grid
        The grid.

    Raises
    ------
    ValueError
        If the sizes are not positive integers, the world coordinates do
        not have a value for each axis, a step is not positive, or the
        spectral axis is not an axis of the grid, if a grid without a
        spectral axis has a rest, or if the medium or the convention is
        not valid.
    """
    ndim = len(size)
    if not all(isinstance(n, numbers.Integral) and not isinstance(n, bool)
               and n > 0 for n in size):
        raise ValueError(
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
        per_axis(name, value, ndim)
        for name, value in dict(step=step, rpix=rpix, rval=rval).items())
    if not all(value > 0 for value in step):
        raise ValueError(f"step must be positive; it is {step}")
    if spectral_axis is not None and spectral_axis not in range(ndim):
        raise ValueError(
            f"the grid has {ndim} axes; its spectral axis cannot be "
            f"{spectral_axis}")
    if rest is not None and spectral_axis is None:
        raise ValueError("a grid without a spectral axis has no rest")
    rest = make_rest(rest)
    return Grid(
        tuple(size),
        Coords(step, rpix, rval, rota, rest, make_medium(medium, rest),
               make_convention(convention, rest), frame),
        spectral_axis)


def make_cube_grid(
        size: Sequence[int],
        step: float | Sequence[float] | None,
        rpix: float | Sequence[float] | None,
        rval: float | Sequence[float] | None,
        rota: float | None,
        spectral: 'SpectralAxis | None'
) -> Grid:
    """
    Make a grid of axes on the sky (as make_grid) and a spectral axis
    after them: of the spectral axis, step 1, rval 0 and rpix the centre
    by default.

    Parameters
    ----------
    size : Sequence of int
        The number of pixels of each axis on the sky, in FITS order.
    step, rpix, rval : float or Sequence of float or None
        The world coordinates of the axes on the sky (see make_grid).
    rota : float or None
        The rotation of the grid on the sky (see make_grid).
    spectral : SpectralAxis
        The spectral axis, with its number of channels.

    Returns
    -------
    Grid
        The grid.

    Raises
    ------
    ValueError
        If the spectral axis is not given or has no size, or the grid is not
        valid (see make_grid).
    """
    if spectral is None or spectral.size() is None:
        raise ValueError(
            "the spectral axis needs its number of channels: spectral: "
            "{size: ...}")
    coords = make_grid(tuple(size), step, rpix, rval, rota).coords
    nchannels = spectral.size()
    return make_grid(
        tuple(size) + (nchannels,),
        coords.step + (1.0 if spectral.step() is None else spectral.step(),),
        coords.rpix + (
            nchannels / 2 - 0.5 if spectral.rpix() is None
            else spectral.rpix(),),
        coords.rval + (0.0 if spectral.rval() is None else spectral.rval(),),
        coords.rota, len(size), spectral.rest(), spectral.medium(),
        spectral.convention(), spectral.frame())


def grid_options(grid: Grid, sizes: bool = True) -> dict[str, Any]:
    """
    Return the options that make a grid: those of make_grid of a grid
    without a spectral axis, or of make_cube_grid of one with a spectral
    axis after its axes on the sky.

    Parameters
    ----------
    grid : Grid
        The grid.
    sizes : bool, optional
        Whether the options have the numbers of pixels of its axes (size,
        and that of the spectral axis): not for data, which give them.

    Returns
    -------
    dict
        The options: size (if sizes), step, rpix and rval, each of the
        axes on the sky (before the spectral axis), rota and, of a grid
        with a spectral axis, spectral, its SpectralAxis.
    """
    coords = grid.coords
    axis = grid.spectral_axis
    # (the axes on the sky: all of them without a spectral axis)
    sky = slice(axis)
    options = dict(
        size=grid.size[sky], step=coords.step[sky], rpix=coords.rpix[sky],
        rval=coords.rval[sky], rota=coords.rota)
    if not sizes:
        del options['size']
    if axis is not None:
        spectral = SpectralAxis.of(grid)
        if not sizes:
            spectral = SpectralAxis(**{
                key: value for key, value in spectral.dump().items()
                if key != 'size'})
        options.update(spectral=spectral)
    return options


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


def x_axis_rota(position_angle: float) -> float:
    """
    Return the rotation (rota, the position angle of its +y axis; see
    Coords) of a grid whose +x axis is at the given position angle
    (degrees, north through east; e.g. a slit along x, whose position
    angle is quoted so): 90 degrees more, as +x is 90 degrees clockwise of
    +y on a sky of north up and east to the left (west, of a grid of rota
    0; see sky_positions).
    """
    return position_angle + 90


def x_axis_pa(rota: float) -> float:
    """
    Return the position angle (degrees, north through east) of the +x
    axis of a grid of the given rotation (see x_axis_rota).
    """
    return rota - 90


def _to_sky(
        x: npt.ArrayLike, y: npt.ArrayLike, rval: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
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


def _from_sky(
        ra: npt.ArrayLike, dec: npt.ArrayLike, rval: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
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
    return frame_to_pixels(*sky_positions(grid), grid.coords.rval, other)


def frame_to_pixels(
        x: np.ndarray,
        y: np.ndarray,
        rval: Sequence[float],
        other: Grid
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the pixel coordinates, on a grid, of positions in the frame of
    a reference RA and Dec, by their RA and Dec.

    The frame and the positions on the sky of the grid are gnomonic (TAN)
    projections around their references.

    Parameters
    ----------
    x, y : np.ndarray
        The positions (arcsec from the reference, +x to the west and +y to
        the north, as xpos and ypos; see sky_positions).
    rval : Sequence of float
        The RA and Dec (degrees) of the reference of the frame.
    other : Grid
        The grid of the pixel coordinates; only its x and y axes are used.

    Returns
    -------
    tuple of ndarray
        The x and y pixel coordinates (0-based) on other of the positions,
        of their shape.
    """
    ra, dec = _to_sky(x, y, rval)
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
    ValueError
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
        raise ValueError(
            f"{desc} does not cover any of the pixels it is used on: by "
            f"their RA and Dec, its reference pixel is at x = {x:.6g}, "
            f"y = {y:.6g} arcsec from theirs; check the world coordinates "
            f"of both, or give its rval")


class GridData(NamedTuple):
    """
    Data on a grid with world coordinates (see write_data).

    Attributes
    ----------
    data : ndarray
        The data.
    coords : Coords
        The world coordinates of its axes, in FITS order.
    spectral_axis : int or None
        The index of its spectral axis (FITS order), if any.
    unit : AnyUnit or None
        The unit of its values, if known.
    """
    data: np.ndarray
    coords: Coords
    spectral_axis: int | None
    unit: gbkfit.units.AnyUnit | None = None


class SpectraData(NamedTuple):
    """
    Spectra of regions of the sky, with the world coordinates of their
    spectral axis (see write_spectra).

    Attributes
    ----------
    data : ndarray
        The spectra, of shape (nchannels, nregions): in FITS, the regions
        along x and the velocity along y.
    coords : Coords
        The world coordinates of the spectral axis (one axis).
    unit : AnyUnit or None
        The unit of the spectra, if known.
    """
    data: np.ndarray
    coords: Coords
    unit: gbkfit.units.AnyUnit | None = None
