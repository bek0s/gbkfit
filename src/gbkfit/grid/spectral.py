"""
The spectral axes of grids: their channels, of velocities of a
convention, and what light is at each velocity: the rest wavelength or
frequency that the velocities refer to, the medium of the wavelengths,
and the frame of rest.
"""

import typing
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

import astropy.constants
import astropy.units
import numpy as np
import numpy.typing as npt
from annotated_types import Gt

from gbkfit.config import (
    BasicParser, Choices, Serializable, check_options)
from gbkfit.utils import unitutils

if typing.TYPE_CHECKING:
    from .coords import Coords, Grid


__all__ = [
    'CONVENTIONS',
    'FRAMES',
    'SPECTRAL_TYPES',
    'Rest',
    'SpectralAxis',
    'VelocityConvention',
    'air_refractive_index',
    'check_convention',
    'doppler',
    'make_convention',
    'make_medium',
    'make_rest',
    'rest_of_convention',
    'spectral_axis_parser',
    'vacuum_wavelength'
]


# The speed of light (km/s)
_C = astropy.constants.c.to_value('km/s')

_T = TypeVar('_T')


# The conventions of the velocities of spectral axes (see
# VelocityConvention)
CONVENTIONS = ('optical', 'log', 'radio', 'relativistic')

# The frames of rest of velocities (FITS WCS Paper III, SPECSYS): of the
# observer, the centre of the Earth, the barycentre of the solar system,
# the centre of the Sun, the kinematic and the dynamical local standard of
# rest, the centre of the Galaxy, the local group, the CMB dipole, and the
# source
FRAMES = (
    'TOPOCENT', 'GEOCENTR', 'BARYCENT', 'HELIOCEN', 'LSRK', 'LSRD',
    'GALACTOC', 'LOCALGRP', 'CMBDIPOL', 'SOURCE')

# The types (FITS 4.0, Table 25, and WCS Paper III) that the spectral
# axis of data whose header gives it a type FITS does not define (e.g.
# CTYPE3 = 'WAVELENGTH') is read as (see SpectralAxis, and
# gbkfit.io.with_spectral_type): linear wavelengths in vacuum or in air,
# logarithmic wavelengths in vacuum or in air, frequencies, and radio or
# optical velocities
SPECTRAL_TYPES = (
    'WAVE', 'AWAV', 'WAVE-LOG', 'AWAV-LOG', 'FREQ', 'VRAD', 'VOPT')

# The options of a spectral axis: the convention of its velocities, the
# medium of its wavelengths, its frame of rest and its type (see
# SpectralAxis)
_Convention: TypeAlias = Annotated[Literal[CONVENTIONS], Choices(
    optical='Optical', log='Logarithmic (as pPXF)', radio='Radio',
    relativistic='Relativistic')]
_Medium: TypeAlias = Annotated[
    Literal['vacuum', 'air'], Choices(vacuum='Vacuum', air='Air')]
_Frame: TypeAlias = Annotated[Literal[FRAMES], Choices(
    TOPOCENT='Topocentric (observer)', GEOCENTR='Geocentric',
    BARYCENT='Barycentric', HELIOCEN='Heliocentric',
    LSRK='Kinematic LSR (LSRK)', LSRD='Dynamical LSR (LSRD)',
    GALACTOC='Galactocentric', LOCALGRP='Local Group',
    CMBDIPOL='CMB dipole', SOURCE='Rest frame of the source')]
_Type: TypeAlias = Annotated[Literal[SPECTRAL_TYPES], Choices({
    'WAVE': 'Wavelengths in vacuum', 'AWAV': 'Wavelengths in air',
    'WAVE-LOG': 'Logarithmic wavelengths in vacuum',
    'AWAV-LOG': 'Logarithmic wavelengths in air',
    'FREQ': 'Frequencies', 'VRAD': 'Radio velocities',
    'VOPT': 'Optical velocities'})]


def check_convention(spectral: 'SpectralAxis') -> None:
    """
    Raise ValueError unless the convention of the velocities of a
    spectral axis is known: given, or that of its rest. None is right for
    all data (e.g. radio for HI, optical for lines of ionised gas, log for
    the stars of pPXF; at 5000 km/s, the rotation of the others differs by
    1.6%).
    """
    if spectral.convention() is None and spectral.rest() is None:
        raise ValueError(
            "the velocities of a spectral axis need their convention: give "
            "its convention ('radio' for HI, 'optical' for lines of ionised "
            "gas, 'log' for the stars of pPXF, or 'relativistic'), or its "
            "rest: spectral: {convention: ...}, in the options of the data "
            "if they have a spectral axis of another type than VRAD, VOPT, "
            "VELO or WAVE-LOG")


# The rest of a spectral axis, as it is given (see make_rest)
Rest: TypeAlias = str | astropy.units.Quantity | None


def make_rest(value: Rest) -> astropy.units.Quantity | None:
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
    ValueError
        If the value is not a positive wavelength or frequency.
    """
    if value is None:
        return None
    error = ValueError(
        f"the rest of a spectral axis must be a positive wavelength or "
        f"frequency with units (e.g. '6562.8 Angstrom'); it is {value!r}")
    try:
        rest = unitutils.parse_quantity(value) if isinstance(value, str) \
            else astropy.units.Quantity(value)
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
    rest (see make_rest and VelocityConvention): of a rest wavelength w0,
    'optical', c (w / w0 - 1), the default, or 'log', c ln(w / w0)
    (exactly linear on an axis uniform in ln w, as FITS WAVE-LOG); of a
    rest frequency, 'radio', c (1 - f / f0). Without a rest, any of
    CONVENTIONS, or 'relativistic', the true radial velocities: each maps
    velocities to 1 + z by itself; none if not given (relativistic too,
    or an axis that is not spectral).

    Raises
    ------
    ValueError
        If the convention is not one of those of the rest.
    """
    if rest is None:
        allowed, default = (None,) + CONVENTIONS, None
    elif rest.unit.is_equivalent(astropy.units.m):
        allowed, default = ('optical', 'log'), 'optical'
    else:
        allowed, default = ('radio',), 'radio'
    convention = default if convention is None else convention
    if convention not in allowed:
        of = 'without a rest' if rest is None else f"of the rest {rest}"
        raise ValueError(
            f"the convention of the velocities {of} must be one of "
            f"{[c for c in allowed if c is not None]}; it is {convention!r}")
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
    ValueError
        If the medium is not one of those, or there is a medium without a
        rest wavelength.
    """
    if rest is None or rest.unit != astropy.units.m:
        if medium is not None:
            raise ValueError(
                f"only the wavelengths of a rest wavelength have a medium; "
                f"the rest is {rest}")
        return None
    if medium is None:
        return 'vacuum'
    if medium not in ('vacuum', 'air'):
        raise ValueError(
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
    ValueError
        Below 2000 Angstrom, where air wavelengths are not used.
    """
    if not wavelength >= 2e-7:
        raise ValueError(
            f"wavelengths in air are used above 2000 Angstrom only; a line "
            f"is at {wavelength * 1e10:.1f} Angstrom (in vacuum)")
    # The wavenumber (1 / um), squared
    s2 = (1e-6 / wavelength) ** 2
    return 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)


def vacuum_wavelength(air: float) -> float:
    """
    Return the wavelength in vacuum (m) of one in air: the inverse of
    dividing by air_refractive_index, by fixed point (the refractive index
    of air changes by about 1e-9 between the two, so that three steps
    leave rounding).
    """
    vacuum = air
    for _ in range(3):
        vacuum = air * air_refractive_index(vacuum)
    return vacuum


def rest_of_convention(
        rest: astropy.units.Quantity | None,
        medium: str | None,
        convention: str
) -> tuple[astropy.units.Quantity | None, str | None]:
    """
    Return the rest of a line for velocities of a convention (see
    make_convention), and the medium of its wavelength: a rest wavelength
    for optical or log velocities (c / f0 in vacuum, of a rest frequency
    f0), a rest frequency for radio ones (c over the rest wavelength in
    vacuum), and none for relativistic ones, which have no rest.

    Parameters
    ----------
    rest : Quantity or None
        The rest (see make_rest).
    medium : str or None
        The medium of the wavelength of a rest wavelength (see
        make_medium).
    convention : str
        The convention.

    Returns
    -------
    Quantity or None
        The rest, in m or Hz.
    str or None
        The medium of its wavelength ('vacuum' or 'air'), or None for a
        frequency or no rest.
    """
    if rest is None or convention == 'relativistic':
        return None, None
    is_wavelength = rest.unit.is_equivalent(astropy.units.m)
    if convention == 'radio':
        if not is_wavelength:
            return rest, None
        wavelength = rest.to_value(astropy.units.m)
        if medium == 'air':
            wavelength = vacuum_wavelength(wavelength)
        return (wavelength * astropy.units.m).to(
            astropy.units.Hz, astropy.units.spectral()), None
    if is_wavelength:
        return rest, medium
    return rest.to(astropy.units.m, astropy.units.spectral()), 'vacuum'


class VelocityConvention:
    """
    The convention of the velocities of a spectral axis, with the rest
    they refer to and the medium of its wavelengths (see Coords): which
    light is at each velocity. A velocity v of the axis is that of light
    of 1 + z = D (its wavelength over the rest wavelength, or the rest
    frequency over its frequency), by the convention:

    - 'optical' (VOPT; of a rest wavelength, if any): v = c (D - 1);
    - 'log' (of a rest wavelength, if any; an axis uniform in ln w, as
      FITS WAVE-LOG and pPXF): v = c ln D;
    - 'radio' (VRAD; of a rest frequency, if any): v = c (1 - 1 / D);
    - 'relativistic' or none (VELO, without a rest): v = c (D^2 - 1) /
      (D^2 + 1).

    None needs the rest to map velocities to D; wavelengths and
    frequencies do (e.g. of lines of other rests).

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
    ValueError
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
        ValueError
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
        ValueError
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
        ValueError
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
        ValueError
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
            raise ValueError(
                "velocities and wavelengths or frequencies are converted "
                "with the rest of the spectral axis (RESTWAV or RESTFRQ in "
                "the header of the data, or its option rest), which is not "
                "given")
        return value


class SpectralAxis(Serializable):
    """
    The spectral axis of an observable or of data: its channels, of
    velocities (km/s) of a convention, and what light is at each velocity
    (see VelocityConvention). The options not given are those of the
    data, or the defaults of the observable.

    Parameters
    ----------
    size : int, optional
        The number of channels.
    step : float, optional
        The width of the channels (km/s).
    rpix : float, optional
        The channel of rval (from 0); by default, the centre.
    rval : float, optional
        The velocity of the channel rpix (km/s).
    convention : str, optional
        The convention of the velocities (see make_convention): 'optical',
        'log' (as pPXF), 'radio' or 'relativistic'; that of the rest by
        default.
    rest : str or Quantity, optional
        The rest wavelength or frequency of the velocities (see
        make_rest): where the wavelengths or frequencies of the channels
        are needed (e.g. of lines of other rests, or of the medium air).
    medium : str, optional
        The medium of the wavelengths of a rest wavelength (see
        make_medium).
    frame : str, optional
        The frame of rest of the velocities (FITS SPECSYS, see FRAMES):
        velocities of one galaxy in two frames differ (by up to some 30
        km/s), which a joint fit of them must know (see
        ObservationGroup).
    type : str, optional
        What the spectral axis of data is (see SPECTRAL_TYPES: e.g.
        'AWAV', linear wavelengths in air), where their header gives it a
        type that FITS does not define (e.g. CTYPE3 = 'WAVELENGTH', as
        CALIFA writes): such a header does not say what its values are,
        and data of one are not read without it (see
        gbkfit.io.with_spectral_type).

    Raises
    ------
    ValueError
        If size is not a positive integer, step is not positive, or the
        rest, the medium, the convention, the frame or the type is not
        valid.
    """

    @classmethod
    def of(cls, grid: Grid) -> 'SpectralAxis':
        """Return the spectral axis of a grid (that of a spectral axis)."""
        axis = grid.spectral_axis
        coords = grid.coords
        return cls(
            grid.size[axis], coords.step[axis], coords.rpix[axis],
            coords.rval[axis], coords.convention, coords.rest,
            coords.medium, coords.frame)

    def dump(self) -> dict[str, Any]:
        options = dict(
            size=self._size, step=self._step, rpix=self._rpix,
            rval=self._rval, convention=self._convention,
            rest=None if self._rest is None else str(self._rest),
            medium=self._medium, frame=self._frame, type=self._type)
        return {k: v for k, v in options.items() if v is not None}

    def __init__(
            self,
            size: Annotated[int, Gt(0)] | None = None,
            step: Annotated[float, Gt(0)] | None = None,
            rpix: float | None = None,
            rval: float | None = None,
            convention: _Convention | None = None,
            rest: str | astropy.units.Quantity | None = None,
            medium: _Medium | None = None,
            frame: _Frame | None = None,
            type: _Type | None = None
    ):
        check_options(SpectralAxis.__init__, locals())
        self._rest = make_rest(rest)
        make_medium(medium, self._rest)
        make_convention(convention, self._rest)
        self._size = None if size is None else int(size)
        self._step = None if step is None else float(step)
        self._rpix = None if rpix is None else float(rpix)
        self._rval = None if rval is None else float(rval)
        self._convention = convention
        self._medium = medium
        self._frame = frame
        self._type = type

    def size(self) -> int | None:
        """Return the number of channels, if given."""
        return self._size

    def step(self) -> float | None:
        """Return the width of the channels (km/s), if given."""
        return self._step

    def rpix(self) -> float | None:
        """Return the channel of rval, if given."""
        return self._rpix

    def rval(self) -> float | None:
        """Return the velocity of the channel rpix (km/s), if given."""
        return self._rval

    def convention(self) -> str | None:
        """Return the convention of the velocities, if given."""
        return self._convention

    def rest(self) -> astropy.units.Quantity | None:
        """Return the rest of the velocities (m or Hz), if given."""
        return self._rest

    def medium(self) -> str | None:
        """Return the medium of the wavelengths, if given."""
        return self._medium

    def frame(self) -> str | None:
        """Return the frame of rest of the velocities, if given."""
        return self._frame

    def type(self) -> str | None:
        """
        Return what the spectral axis of data is, where their header does
        not say (see SPECTRAL_TYPES), if given.
        """
        return self._type

    def given(self) -> tuple[str, ...]:
        """Return the names of its options that are given."""
        return tuple(self.dump())

    def replaced(self, other: 'SpectralAxis') -> 'SpectralAxis':
        """Return it with the options that another one gives instead."""
        return SpectralAxis(**self.dump() | other.dump())


spectral_axis_parser = BasicParser(SpectralAxis)
