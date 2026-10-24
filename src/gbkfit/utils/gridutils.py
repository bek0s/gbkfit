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
from typing import NamedTuple, TypeVar

import astropy.constants
import astropy.units
import numpy as np
import numpy.typing as npt

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Coords',
    'Grid',
    'GridData',
    'SpectraData',
    'VelocityConvention',
    'air_refractive_index',
    'check_overlap',
    'doppler',
    'make_grid',
    'make_convention',
    'make_medium',
    'make_rest',
    'frame_to_pixels',
    'pixels_on',
    'sky_positions',
    'sky_to_pixel'
]


# Arcsec per radian
_ARCSEC = np.degrees(1) * 3600

# The speed of light (km/s)
_C = astropy.constants.c.to_value('km/s')

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
        The convention of the velocities of a rest (see make_convention):
        'optical' (VOPT) or 'log' of a rest wavelength, 'radio' (VRAD) of
        a rest frequency; None: the default of the rest (optical, radio,
        or relativistic, VELO, without a rest).
    """
    step: tuple[float, ...]
    rpix: tuple[float, ...]
    rval: tuple[float, ...]
    rota: float
    rest: astropy.units.Quantity | None = None
    medium: str | None = None
    convention: str | None = None

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
        def pick(values):
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


def make_grid(
        size: Sequence[int],
        step: float | Sequence[float] | None = None,
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None,
        rota: float | None = None,
        spectral_axis: int | None = None,
        rest: str | astropy.units.Quantity | None = None,
        medium: str | None = None,
        convention: str | None = None
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

    Returns
    -------
    Grid
        The grid.

    Raises
    ------
    ConfigError
        If the sizes are not positive integers, the world coordinates do
        not have a value for each axis, a step is not positive, or the
        spectral axis is not an axis of the grid, if a grid without a
        spectral axis has a rest, or if the medium or the convention is
        not valid.
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
    rest = make_rest(rest)
    return Grid(
        tuple(size),
        Coords(step, rpix, rval, rota, rest, make_medium(medium, rest),
               make_convention(convention, rest)),
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


def make_convention(
        convention: str | None, rest: astropy.units.Quantity | None
) -> str | None:
    """
    Make the convention of the velocities of a spectral axis of the given
    rest (see make_rest): of a rest wavelength w0, 'optical', c (w / w0 -
    1), the default, or 'log', c ln(w / w0) (exactly linear on an axis
    uniform in ln w, as FITS WAVE-LOG); of a rest frequency, 'radio', c (1
    - f / f0); without a rest, 'relativistic' if given, or none (true
    radial velocities either way; none for axes that are not spectral).

    Raises
    ------
    ConfigError
        If the convention is not one of those of the rest.
    """
    if rest is None:
        allowed, default = (None, 'relativistic'), None
    elif rest.unit.is_equivalent(astropy.units.m):
        allowed, default = ('optical', 'log'), 'optical'
    else:
        allowed, default = ('radio',), 'radio'
    convention = default if convention is None else convention
    if convention not in allowed:
        raise ConfigError(
            f"the convention of the velocities of the rest {rest} must be "
            f"one of {list(allowed)}; it is {convention!r}")
    return convention


def make_medium(
        medium: str | None, rest: astropy.units.Quantity | None
) -> str | None:
    """
    Make the medium of the wavelengths of a spectral axis of optical
    velocities, those of the given rest (see make_rest).

    Parameters
    ----------
    medium : str or None
        'vacuum' or 'air'; vacuum by default (the medium of the
        wavelengths of FITS).
    rest : Quantity or None
        The rest of the spectral axis.

    Returns
    -------
    str or None
        The medium, or None without a rest wavelength.

    Raises
    ------
    ConfigError
        If the medium is not one of those, or there is a medium without a
        rest wavelength.
    """
    if rest is None or rest.unit != astropy.units.m:
        if medium is not None:
            raise ConfigError(
                f"only the wavelengths of a rest wavelength have a medium; "
                f"the rest is {rest}")
        return None
    if medium is None:
        return 'vacuum'
    if medium not in ('vacuum', 'air'):
        raise ConfigError(
            f"the medium must be 'vacuum' or 'air'; it is {medium!r}")
    return medium


def doppler(velocity: float | np.ndarray) -> float | np.ndarray:
    """
    Return the Doppler factor of a radial velocity (km/s, relativistic):
    the 1 + z of its light, sqrt((1 + v / c) / (1 - v / c)).
    """
    beta = velocity / _C
    return np.sqrt((1 + beta) / (1 - beta))


def air_refractive_index(wavelength: float) -> float:
    """
    Return the refractive index of standard air at a wavelength in vacuum
    (m), as the IAU adopted it (Morton 2000, ApJS 130, 403, eq. 8; from
    Edlen 1966 and Birch & Downs 1994): air wavelengths are those in
    vacuum over it.

    Raises
    ------
    ConfigError
        Below 2000 Angstrom, where air wavelengths are not used.
    """
    if not wavelength >= 2e-7:
        raise ConfigError(
            f"wavelengths in air are used above 2000 Angstrom only; a line "
            f"is at {wavelength * 1e10:.1f} Angstrom (in vacuum)")
    # The wavenumber (1 / um), squared
    s2 = (1e-6 / wavelength) ** 2
    return 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)


class VelocityConvention:
    """
    The convention of the velocities of a spectral axis, with the rest
    they refer to and the medium of its wavelengths (see Coords): which
    light is at each velocity. A velocity v of the axis is that of light
    of 1 + z = D (its wavelength over the rest wavelength, or the rest
    frequency over its frequency), by the convention:

    - 'optical' (a rest wavelength): v = c (D - 1);
    - 'log' (a rest wavelength; an axis uniform in ln w, as FITS
      WAVE-LOG): v = c ln D;
    - 'radio' (a rest frequency): v = c (1 - 1 / D);
    - none (an axis without a rest, VELO): relativistic, v = c (D^2 - 1)
      / (D^2 + 1).

    The wavelengths of the axis and of its rest are in its medium, its
    frequencies in vacuum: the refractive index of air is taken at the
    rest wavelength (it changes by about 1e-8 across 1000 km/s).

    Parameters
    ----------
    rest : Quantity, optional
        The rest wavelength or frequency (see make_rest).
    medium : str, optional
        The medium of the wavelengths of a rest wavelength (see
        make_medium).
    convention : str, optional
        The convention of the velocities of the rest (see
        make_convention).

    Raises
    ------
    ConfigError
        If the convention or the medium is not one of those of the rest.
    """

    @classmethod
    def of(cls, coords: Coords) -> 'VelocityConvention':
        """Return the convention of the spectral axis of coordinates."""
        return cls(coords.rest, coords.medium, coords.convention)

    def __init__(
            self,
            rest: astropy.units.Quantity | None = None,
            medium: str | None = None,
            convention: str | None = None
    ):
        self._name = make_convention(convention, rest)
        self._medium = make_medium(medium, rest)
        self._rest = rest
        # The rest frequency (Hz, in vacuum), and the rest wavelength (m,
        # in the medium; of a rest frequency, in vacuum)
        self._rest_frequency = self._rest_wavelength = None
        if rest is not None:
            self._rest_frequency = rest.to_value(
                astropy.units.Hz, astropy.units.spectral())
            self._rest_wavelength = rest.to_value(
                astropy.units.m, astropy.units.spectral())
            if self._medium == 'air':
                self._rest_frequency /= air_refractive_index(
                    self._rest_wavelength)

    def name(self) -> str | None:
        """
        Return its name: 'optical', 'log', 'radio', or 'relativistic' (or
        None: relativistic too, not given).
        """
        return self._name

    def rest(self) -> astropy.units.Quantity | None:
        """Return the rest of the velocities, if any."""
        return self._rest

    def factor(self, velocity: npt.ArrayLike) -> np.ndarray:
        """Return the 1 + z of the light at velocities of the axis (km/s)."""
        v = np.asarray(velocity, dtype=float)
        if self._name == 'optical':
            return 1 + v / _C
        if self._name == 'log':
            return np.exp(v / _C)
        if self._name == 'radio':
            return 1 / (1 - v / _C)
        return doppler(v)

    def velocity(self, factor: npt.ArrayLike) -> np.ndarray:
        """Return the velocities of the axis (km/s) of light of 1 + z."""
        d = np.asarray(factor, dtype=float)
        if self._name == 'optical':
            return _C * (d - 1)
        if self._name == 'log':
            return _C * np.log(d)
        if self._name == 'radio':
            return _C * (1 - 1 / d)
        return _C * (d ** 2 - 1) / (d ** 2 + 1)

    def log_slope(self, velocity: npt.ArrayLike) -> np.ndarray:
        """
        Return d ln D / dv at velocities of the axis (1 / (km/s)): the
        relative change of the wavelength (or of the frequency, with the
        sign changed) per km/s of the axis.
        """
        v = np.asarray(velocity, dtype=float)
        if self._name == 'optical':
            return 1 / (_C + v)
        if self._name == 'log':
            return np.full_like(v, 1 / _C)
        if self._name == 'radio':
            return 1 / (_C - v)
        return 1 / (_C * (1 - (v / _C) ** 2))

    def stretch(self, velocity: npt.ArrayLike) -> np.ndarray:
        """
        Return the km/s of the axis that a km/s of motion (in the frame of
        the light) spans at velocities of the axis: 1 + v / c (optical),
        1 (log), 1 - v / c (radio), or 1 - (v / c)^2 (relativistic).
        """
        return 1 / (_C * self.log_slope(velocity))

    def frequency(self, velocity: npt.ArrayLike) -> np.ndarray:
        """
        Return the frequencies (Hz, in vacuum) at velocities of the axis.

        Raises
        ------
        ConfigError
            If the axis has no rest.
        """
        return self._require_rest(self._rest_frequency) \
            / self.factor(velocity)

    def wavelength(self, velocity: npt.ArrayLike) -> np.ndarray:
        """
        Return the wavelengths (m; in the medium of a rest wavelength, in
        vacuum of a rest frequency) at velocities of the axis.

        Raises
        ------
        ConfigError
            If the axis has no rest.
        """
        return self._require_rest(self._rest_wavelength) \
            * self.factor(velocity)

    def velocity_at(self, frequency: npt.ArrayLike) -> np.ndarray:
        """
        Return the velocities of the axis (km/s) at frequencies (Hz, in
        vacuum).

        Raises
        ------
        ConfigError
            If the axis has no rest.
        """
        return self.velocity(
            self._require_rest(self._rest_frequency)
            / np.asarray(frequency, dtype=float))

    def equivalency(self) -> astropy.units.Equivalency:
        """
        Return the astropy equivalency between the velocities of the axis
        and wavelengths or frequencies (those of the rest, ignoring the
        medium: e.g. the points of a table of a varying PSF).

        Raises
        ------
        ConfigError
            If the axis has no rest.
        """
        rest = self._require_rest(self._rest)
        if self._name == 'optical':
            return astropy.units.doppler_optical(rest)
        if self._name == 'radio':
            return astropy.units.doppler_radio(rest)
        wavelength = self._rest_wavelength
        kms = astropy.units.km / astropy.units.s
        c = astropy.constants.c.si.value
        return astropy.units.Equivalency([
            (kms, astropy.units.m,
             lambda v: wavelength * np.exp(v / _C),
             lambda w: _C * np.log(w / wavelength)),
            (kms, astropy.units.Hz,
             lambda v: c / (wavelength * np.exp(v / _C)),
             lambda f: _C * np.log(c / (f * wavelength)))], 'doppler_log')

    def _require_rest(self, value: _T) -> _T:
        """Return a value of the rest, or raise if the axis has none."""
        if self._rest is None:
            raise ConfigError(
                "velocities and wavelengths or frequencies are converted "
                "with the rest of the spectral axis (RESTWAV or RESTFRQ in "
                "the header of the data, or the option rest), which is not "
                "given")
        return value


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
    unit : str or None
        The unit of its values (FITS BUNIT), if known.
    """
    data: np.ndarray
    coords: Coords
    spectral_axis: int | None
    unit: str | None = None


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
    unit : str or None
        The unit of the spectra (FITS BUNIT), if known.
    """
    data: np.ndarray
    coords: Coords
    unit: str | None = None
