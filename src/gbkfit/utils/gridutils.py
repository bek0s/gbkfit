import typing

import astropy.units
import numpy as np

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Coords',
    'Grid',
    'GridData',
    'SpectraData',
    'make_grid',
    'make_rest',
    'sky_positions',
    'sky_to_pixel'
]


class Coords(typing.NamedTuple):
    """
    The world coordinates of the pixels of data, in the units of the
    model, for each axis in FITS order (x, y and, for spectral cubes, the
    velocity):

    - step: arcsec per pixel (spatial axes), km/s per channel (velocity)
    - rpix: the reference pixel (0-based)
    - rval: the world position at rpix: RA and Dec (degrees), and the
      velocity (km/s). The model measures the spatial axes from rpix.
    - rota: the rotation of the pixel grid on the sky: the position angle
      (degrees, north through east) of the +y axis.
    - rest: the rest wavelength or frequency that the velocities of the
      spectral axis refer to (an astropy Quantity in m or Hz; see
      make_rest), or None. The velocities of a rest wavelength are optical
      (VOPT), and those of a rest frequency radio (VRAD).

    Axes without a known type have the values of their header.
    """
    step: tuple[float, ...]
    rpix: tuple[float, ...]
    rval: tuple[float, ...]
    rota: float
    rest: astropy.units.Quantity | None = None

    def axes(self, *indices: int) -> 'Coords':
        """The coordinates of the given axes."""
        def pick(values):
            return tuple(values[i] for i in indices)
        return Coords(
            pick(self.step), pick(self.rpix), pick(self.rval), self.rota,
            self.rest)


class Grid(typing.NamedTuple):
    """
    A grid of pixels: its size and world coordinates (see Coords) for each
    axis in FITS order, and the index of its spectral axis (or None).
    """
    size: tuple[int, ...]
    coords: Coords
    spectral_axis: int | None

    def zero(self) -> tuple[float, ...]:
        """
        The world position of the first pixel on each axis, in model units:
        the spatial axes are measured from the reference pixel, and the
        spectral axis from its world value there.
        """
        step, rpix, rval = self.coords.step, self.coords.rpix, self.coords.rval
        return tuple(
            (rval[axis] if axis == self.spectral_axis else 0)
            - rpix[axis] * step[axis]
            for axis in range(len(self.size)))

    def spatial(self) -> 'Grid':
        """The grid of the x and y axes."""
        return Grid(
            self.size[:2], self.coords.axes(0, 1)._replace(rest=None), None)

    def spectral(self) -> 'Grid':
        """The grid of the spectral axis (one axis, not rotated)."""
        axis = self.spectral_axis
        return Grid(
            (self.size[axis],), self.coords.axes(axis)._replace(rota=0), 0)


def make_grid(
        size: typing.Sequence[int],
        step: float | typing.Sequence[float] | None = None,
        rpix: float | typing.Sequence[float] | None = None,
        rval: float | typing.Sequence[float] | None = None,
        rota: float | None = None,
        spectral_axis: int | None = None,
        rest: typing.Any = None
) -> Grid:
    """
    A grid of the given size (FITS order) with the given world
    coordinates (see Coords), each a value or one per axis, or their
    defaults: step 1, the reference pixel at the centre, reference value 0
    and no rotation. rest (see make_rest) needs a spectral axis.
    """
    ndim = len(size)
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
            raise RuntimeError(
                f"the grid has {ndim} axes, but {name} has {len(value)} "
                f"values")
    if not all(value > 0 for value in step):
        raise RuntimeError(f"step must be positive; it is {step}")
    if rest is not None and spectral_axis is None:
        raise RuntimeError("a grid without a spectral axis has no rest")
    return Grid(
        tuple(size), Coords(step, rpix, rval, rota, make_rest(rest)),
        spectral_axis)


def make_rest(value: typing.Any) -> astropy.units.Quantity | None:
    """
    The rest wavelength or frequency of a spectral axis (see Coords), in m
    or Hz, from a Quantity or a string with units (e.g. '6562.8 Angstrom',
    '1420.405752 MHz'), or None.
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
    The positions on the sky of the centres of the pixels of the x and y
    axes of a grid, in the frame of the model (arcsec from the reference
    pixel, x and y like xpos and ypos): two arrays of shape (ny, nx). The
    pixel grid is rotated on the sky by rota (see Coords).
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
    The affine map from the sky (the frame of the model, see
    sky_positions) to the pixel coordinates of the x and y axes of a grid:
    pixel = matrix @ sky + offset (the inverse of sky_positions).
    """
    rota = np.radians(grid.coords.rota)
    rotation = np.array([
        [np.cos(rota), np.sin(rota)],
        [-np.sin(rota), np.cos(rota)]])
    matrix = rotation / np.asarray(grid.coords.step[:2], float)[:, None]
    return matrix, np.asarray(grid.coords.rpix[:2], float)


class GridData(typing.NamedTuple):
    """
    Data on a grid with world coordinates (see Coords and
    fitsutils.write_data).
    spectral_axis is the index of its spectral axis (FITS order), or None.
    """
    data: np.ndarray
    coords: Coords
    spectral_axis: int | None


class SpectraData(typing.NamedTuple):
    """
    Spectra of regions of the sky (data of shape (nchannels, nregions);
    FITS: the regions along x, the velocity along y), with the world
    coordinates of their spectral axis (a Coords of one axis; see
    fitsutils.write_spectra).
    """
    data: np.ndarray
    coords: Coords
