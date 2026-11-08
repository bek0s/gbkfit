"""
The spectral axes of FITS headers: their types and the conventions of
their velocities (FITS WCS Paper III, and those of AIPS and MIRIAD),
their rests and the medium of their wavelengths, and their reading as
velocities of a rest. An axis of a type that FITS does not define is
read as the type given (see _with_spectral_type).

A rest wavelength is in the medium of the wavelengths of its axis, as
the model takes it (see VelocityConvention), and in vacuum in headers
(RESTWAV, as FITS defines it, whatever the medium of the axis): it is
turned from the one to the other where headers are read and written
(see _set_rest and _rest_in_medium).
"""

import astropy.constants
import astropy.io.fits
import astropy.units as u
import astropy.wcs
import numpy as np

from gbkfit.config import warn
from gbkfit.grid import (
    FRAMES, Coords, Rest, SPECTRAL_TYPES, air_refractive_index,
    make_convention, make_medium, make_rest, vacuum_wavelength)
from .headers import has_wcs, ignoring_wcs_fixes


__all__ = [
    'nonstandard_spectral_axis',
    'velocity_is_ambiguous'
]


# The spectral axis types that are velocities (FITS WCS Paper III)
_VELOCITY_TYPES = ('VRAD', 'VOPT', 'VELO')

# The conventions of the velocities of each type of spectral axis, and
# the types of the conventions (as written; log velocities are written as
# logarithmic wavelengths, or without a rest wavelength as a linear axis
# of the type VLOG, which FITS does not define)
_CONVENTIONS_OF_TYPES = dict(
    VRAD='radio', VOPT='optical', VELO='relativistic', VLOG='log')
_TYPES_OF_CONVENTIONS = {
    convention: type_ for type_, convention in _CONVENTIONS_OF_TYPES.items()}

# The types of spectral axes of logarithmic wavelengths (FITS WCS Paper
# III) by the medium of their wavelengths, and the medium of each
_LOG_TYPES = dict(vacuum='WAVE-LOG', air='AWAV-LOG')
_MEDIUMS_OF_LOG_TYPES = {type_: medium for medium, type_ in _LOG_TYPES.items()}

# The values of the header keyword AIRORVAC of each medium of
# wavelengths, as make_medium names them, and the medium of each
_HEADER_MEDIUMS = dict(vacuum='vac', air='air')
_MEDIUMS_OF_HEADER = {
    value: medium for medium, value in _HEADER_MEDIUMS.items()}

# The frames of the velocity axes of the AIPS convention (VELO-LSR, ...)
# and their codes in VELREF, which adds 256 for radio velocities
_AIPS_FRAMES = dict(LSR=1, HEL=2, OBS=3)

# The conventions of velocities (see _with_velocity_convention)
_VELOCITY_CONVENTIONS = ('radio', 'optical')

# The speed of light (km/s), and the unit of velocities
_C = astropy.constants.c.to_value('km/s')
_KM_S = u.km / u.s

# The units of spectral axes: of wavelengths, frequencies and velocities
_SPECTRAL_UNITS = (u.m, u.Hz, _KM_S)


def nonstandard_spectral_axis(
        header: astropy.io.fits.Header,
        shape: tuple[int, ...] | None = None
) -> tuple[int, str] | None:
    """
    Return the axis of a header whose type (CTYPE) is not one FITS
    defines and whose unit is one of a spectral axis (of wavelengths,
    frequencies or velocities), and that type (e.g. 2 and 'WAVELENGTH',
    in Angstrom, as CALIFA writes the spectral axis of its cubes); None if
    no axis is one.

    The type of such an axis says nothing of what its values are
    (astropy and wcslib read it as an axis of no kind): FITS defines the
    spectral types (FITS 4.0, Table 25), and any other is a name alone
    (e.g. 'WAVELENGTH' says neither air or vacuum, nor linear or
    logarithmic). Its unit says that it is spectral, not what it is.
    Other axes of types FITS does not define (e.g. LINEAR or OFFSET, of
    positions along a slit, in angles or of no unit) are positions, as
    they are read, and VLOG is GBKFIT's own (its log velocities without a
    rest, see _CONVENTIONS_OF_TYPES); axes without a type, and axes of
    length 1 of an array of the given shape (numpy order) or beyond its
    axes, which reading leaves out, are not looked at.

    Parameters
    ----------
    header : Header
        The header.
    shape : tuple of int, optional
        The shape of its array (numpy order).

    Returns
    -------
    tuple of int and str, or None
        The axis (FITS order, from 0) and its type.
    """
    if not has_wcs(header):
        return None
    try:
        with ignoring_wcs_fixes():
            wcs = astropy.wcs.WCS(header)
    except ValueError:
        # (the errors of astropy's WCS: the world coordinates are reported
        # as invalid where they are read, see parse_wcs)
        return None
    for axis, kind in enumerate(wcs.wcs.axis_types):
        ctype = str(wcs.wcs.ctype[axis]).strip()
        if kind != 0 or not ctype or ctype in _CONVENTIONS_OF_TYPES:
            continue
        if shape is not None and (
                axis >= len(shape) or shape[len(shape) - 1 - axis] == 1):
            continue
        unit = wcs.wcs.cunit[axis]
        if any(unit.is_equivalent(of) for of in _SPECTRAL_UNITS):
            return axis, ctype
    return None


def _with_spectral_type(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...] | None,
        spectral_type: str | None
) -> astropy.io.fits.Header:
    """
    Return the header with the type of its spectral axis whose type is not
    one FITS defines (see nonstandard_spectral_axis) replaced by the type
    given, or as it is if it has no such axis.

    Such an axis cannot be read without its type: its name (e.g.
    'WAVELENGTH') does not say whether its values are wavelengths in
    vacuum or in air, linear or logarithmic, and nothing in the header
    has to; the type is given, never taken from its name.

    Parameters
    ----------
    source : str
        The name of the file of the header, for messages.
    header : Header
        The header.
    shape : tuple of int, optional
        The shape of its array (numpy order; see
        nonstandard_spectral_axis).
    spectral_type : str, optional
        The type the axis is read as: one of SPECTRAL_TYPES (e.g. 'AWAV'
        for linear wavelengths in air).

    Raises
    ------
    ValueError
        If the type given is not one of SPECTRAL_TYPES, or the header has
        an axis of a type FITS does not define and no type is given.
    """
    if spectral_type is not None and spectral_type not in SPECTRAL_TYPES:
        raise ValueError(
            f"the type of the spectral axis must be one of "
            f"{list(SPECTRAL_TYPES)}; it is {spectral_type!r}")
    found = nonstandard_spectral_axis(header, shape)
    if found is None:
        return header
    axis, ctype = found
    if spectral_type is None:
        unit = str(header.get(f'CUNIT{axis + 1}', '')).strip()
        raise ValueError(
            f"{source}: the type of the axis {axis + 1}, {ctype!r}"
            f"{f' (in {unit})' if unit else ''}, is not one FITS defines: "
            f"it does not say what its values are; give the type of the "
            f"spectral axis (spectral: {{type: ...}}), one of "
            f"{', '.join(SPECTRAL_TYPES)}")
    header = header.copy()
    header[f'CTYPE{axis + 1}'] = spectral_type
    return header


def _with_velocity_convention(
        source: str,
        header: astropy.io.fits.Header,
        velocity: str | None
) -> astropy.io.fits.Header:
    """
    Return the header with the convention of the velocities of its
    spectral axis stated, if it is one of the AIPS convention (VELO-LSR,
    VELO-HEL or VELO-OBS) without the keyword VELREF that states it.

    Such an axis is ambiguous: AIPS takes its velocities as optical, and
    so does astropy (wcslib), but MIRIAD writes them radio ("VELO means
    radio velocity (VRAD), and FELO is optical velocity", MIRIAD, velsw;
    e.g. the cubes of LVHIS). Taking the wrong one is off by 2 vsys / c
    in the widths of the lines (see LinePlacement), and by the frequency
    of the lines. Its convention ('radio' or 'optical') must be given:
    the header gets it as VELREF (the code of the frame, plus 256 for
    radio velocities), as AIPS defines it.

    Parameters
    ----------
    source : str
        The name of the file of the header, for messages.
    header : Header
        The header.
    velocity : str, optional
        The convention of the velocities of an axis of the AIPS convention
        without VELREF: 'radio' or 'optical'; ignored for other axes.

    Returns
    -------
    Header
        The header (a copy, with VELREF, if its axis is of the AIPS
        convention without it).

    Raises
    ------
    ValueError
        If the axis is of the AIPS convention without VELREF, and the
        convention is not given, or is not 'radio' or 'optical'.
    """
    if not velocity_is_ambiguous(header):
        return header
    frame = _aips_frame(header)
    if velocity is None:
        raise ValueError(
            f"{source}: the velocities of the spectral axis (VELO-{frame}"
            f", without VELREF) may be radio or optical: AIPS reads them "
            f"as optical, but MIRIAD writes radio ones (e.g. LVHIS). Say "
            f"which, where the data are read (the spectral convention of "
            f"a dataset, spectral: {{convention: radio}}; the velocity "
            f"of prep), or add VELREF to the header "
            f"({_AIPS_FRAMES[frame]} for optical, "
            f"{_AIPS_FRAMES[frame] + 256} for radio)")
    if velocity not in _VELOCITY_CONVENTIONS:
        raise ValueError(
            f"the convention of velocities must be one of "
            f"{list(_VELOCITY_CONVENTIONS)}; it is {velocity!r}")
    header = header.copy()
    radio = 256 if velocity == 'radio' else 0
    header['VELREF'] = _AIPS_FRAMES[frame] + radio
    return header


def velocity_is_ambiguous(header: astropy.io.fits.Header) -> bool:
    """
    Return whether the velocities of the spectral axis of a header may be
    radio or optical: an axis of the AIPS convention (VELO-LSR, VELO-HEL
    or VELO-OBS) without the keyword VELREF that states which (see
    _with_velocity_convention).
    """
    return _aips_frame(header) is not None and 'VELREF' not in header


def _stated_header(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        spectral_type: str | None,
        velocity: str | None
) -> astropy.io.fits.Header:
    """
    Return a header (of an array of the given shape, numpy order) whose
    spectral axis states what its values are: of the type given, if its
    own is not one FITS defines (see _with_spectral_type), and of the
    convention of velocities given, if the header leaves it ambiguous
    (see _with_velocity_convention); with a warning if AIRORVAC
    contradicts its type (see _check_header_medium).
    """
    header = _with_spectral_type(source, header, shape, spectral_type)
    _check_header_medium(source, header)
    return _with_velocity_convention(source, header, velocity)


def _aips_frame(header: astropy.io.fits.Header) -> str | None:
    """
    Return the frame of the spectral axis of the AIPS convention of a
    header (e.g. 'HEL'), if it has one.
    """
    # An axis of one pixel (e.g. that left in a moment map) has no
    # velocities to read
    ctypes = [
        str(header.get(f'CTYPE{n}', ''))
        for n in range(1, header.get('NAXIS', 0) + 1)
        if header.get(f'NAXIS{n}', 1) > 1]
    return next((
        ctype[5:].strip() for ctype in ctypes
        if ctype[:5] == 'VELO-' and ctype[5:].strip() in _AIPS_FRAMES), None)


def _axis_of_rest(
        source: str,
        header: astropy.io.fits.Header,
        wcs: astropy.wcs.WCS,
        shape: tuple[int, ...],
        rest: Rest,
        medium: str | None
) -> tuple[astropy.wcs.WCS, dict[str, str]]:
    """
    Return the world coordinates of a header (of an array of the given
    shape, numpy order, with an axis for each of theirs) with the
    spectral axis in velocities of a rest, if given (see _velocity_axis;
    an axis of one pixel takes the rest alone, see _with_flat_axis_rest),
    and the cards that the header then takes: the medium of the
    wavelengths of an axis of wavelengths (AIRORVAC). The medium of an
    axis of optical velocities is that AIRORVAC states, or the given one
    (see _stated_medium).
    """
    if rest is None or wcs.wcs.spec < 0:
        return wcs, {}
    stated = _stated_medium(source, header, medium)
    if shape[len(shape) - 1 - wcs.wcs.spec] == 1:
        wcs, wavelengths = _with_flat_axis_rest(wcs, rest, stated)
    else:
        wcs, wavelengths = _velocity_axis(source, wcs, rest, stated)
    if wavelengths is None:
        return wcs, {}
    return wcs, {'AIRORVAC': _HEADER_MEDIUMS[wavelengths]}


def _velocity_axis(
        source: str,
        wcs: astropy.wcs.WCS,
        rest: Rest,
        medium: str | None = None
) -> tuple[astropy.wcs.WCS, str | None]:
    """
    Return world coordinates with the spectral axis in velocities of a
    rest (in the medium of the wavelengths of the axis), and the medium of
    its wavelengths ('vacuum' or 'air', see make_medium), if it is of
    wavelengths (see model_header). The medium of the wavelengths of an
    axis of optical velocities (VOPT) is that given (that its header
    states, AIRORVAC), in vacuum by default.

    Raises
    ------
    ValueError
        If the spectral axis is not linear in its coordinate (but for
        logarithmic wavelengths), has relativistic velocities (VELO),
        which have no rest, or is not of wavelengths, frequencies, or
        optical or radio velocities.
    """
    rest = make_rest(rest)
    spec = wcs.wcs.spec
    wcs = wcs.deepcopy()
    ctype = wcs.wcs.ctype[spec]
    kind = ctype[:4]
    rest_wavelength = rest.to_value(u.m, u.spectral())
    rest_frequency = rest.to_value(u.Hz, u.spectral())
    if ctype in _MEDIUMS_OF_LOG_TYPES:
        # Read as the log velocities of the rest (see coords_from_header)
        log_medium = _MEDIUMS_OF_LOG_TYPES[ctype]
        _set_rest(wcs, wavelength=rest_wavelength, medium=log_medium)
        wcs.wcs.set()
        return wcs, log_medium
    if not _is_linear_type(ctype):
        raise ValueError(
            f"{source}: the spectral axis ({ctype}) is not linear in its "
            f"coordinate")
    if kind == 'VELO':
        raise _relativistic_rest_error(source)
    if kind not in ('WAVE', 'AWAV', 'FREQ', 'VOPT', 'VRAD'):
        raise ValueError(
            f"{source}: the spectral axis ({ctype}) is not a wavelength, "
            f"frequency, or optical or radio velocity")
    # The axis becomes the velocities v = crval + scale x (km/s) of the
    # rest, x its intermediate world coordinate, exactly: all are linear
    # in the pixels
    crval = wcs.wcs.crval[spec]
    unit = u.Unit(str(wcs.wcs.cunit[spec]) or '1')
    wavelengths = None
    if kind in ('WAVE', 'AWAV'):
        crval, scale = _optical_velocities(crval, unit, rest_wavelength)
        kind, wavelengths = 'VOPT', 'air' if kind == 'AWAV' else 'vacuum'
    elif kind == 'FREQ':
        crval, scale = _radio_velocities(crval, unit, rest_frequency)
        kind = 'VRAD'
    else:
        # (the rest of the axis, if any, in the medium of its wavelengths)
        current = _rest_in_medium(_spectral_rest(source, wcs, None), medium)
        crval, scale = (None, None) if current is None else _velocities_of(
            kind, crval, unit, current.value / (
                rest_wavelength if kind == 'VOPT' else rest_frequency))
    if crval is not None:
        wcs.wcs.ctype[spec] = kind
        wcs.wcs.cunit[spec] = 'km/s'
        wcs.wcs.crval[spec] = crval
        _scale_axis(wcs, spec, scale)
    if kind == 'VOPT':
        _set_rest(
            wcs, wavelength=rest_wavelength, medium=wavelengths or medium)
    else:
        _set_rest(wcs, frequency=rest_frequency)
    wcs.wcs.set()
    return wcs, wavelengths


def _optical_velocities(
        crval: float, unit: u.UnitBase, rest_wavelength: float
) -> tuple[float, float]:
    """
    Return the optical velocities c (w / rest - 1) of a linear axis of
    wavelengths w (of a value crval at its reference pixel, in a unit) as
    those of a linear axis: its value at the reference pixel (km/s), and
    the factor of its steps.
    """
    ratio = unit.to(u.m) / rest_wavelength
    return _C * (crval * ratio - 1), _C * ratio


def _radio_velocities(
        crval: float, unit: u.UnitBase, rest_frequency: float
) -> tuple[float, float]:
    """
    Return the radio velocities c (1 - f / rest) of a linear axis of
    frequencies f as those of a linear axis (see _optical_velocities).
    """
    ratio = unit.to(u.Hz) / rest_frequency
    return _C * (1 - crval * ratio), -_C * ratio


def _velocities_of(
        kind: str, crval: float, unit: u.UnitBase, ratio: float
) -> tuple[float, float]:
    """
    Return the velocities of a linear axis of optical (VOPT) or radio
    (VRAD) velocities v of another rest as those of this one, of the ratio
    of the rests (the other over this one: of the wavelengths for optical
    velocities, of the frequencies for radio ones): c (r (1 + v / c) - 1)
    or c (1 - r (1 - v / c)), as those of a linear axis (see
    _optical_velocities).
    """
    if kind == 'VOPT':
        crval = _C * (ratio - 1) + ratio * crval * unit.to(_KM_S)
    else:
        crval = _C * (1 - ratio) + ratio * crval * unit.to(_KM_S)
    return crval, ratio * unit.to(_KM_S)


def _with_flat_axis_rest(
        wcs: astropy.wcs.WCS, rest: Rest, medium: str | None = None
) -> tuple[astropy.wcs.WCS, str | None]:
    """
    Return world coordinates with the rest of a spectral axis of one pixel
    (e.g. that of a moment map, which reading leaves out, see
    model_header), as _velocity_axis gives it to other axes, and the medium
    of its wavelengths, if it is of wavelengths. An axis of one pixel has
    no velocities to read: a rest is no error whatever its type, and one
    of a type that takes none (e.g. relativistic velocities, VELO) is left
    as it is. The medium of an axis of optical velocities is that given
    (see _velocity_axis).
    """
    rest = make_rest(rest)
    spec = wcs.wcs.spec
    wcs = wcs.deepcopy()
    ctype = wcs.wcs.ctype[spec]
    kind = ctype[:4]
    wavelengths = _type_medium(ctype)
    if wavelengths is not None or kind == 'VOPT':
        _set_rest(
            wcs, wavelength=rest.to_value(u.m, u.spectral()),
            medium=wavelengths or medium)
    elif kind in ('FREQ', 'VRAD'):
        _set_rest(wcs, frequency=rest.to_value(u.Hz, u.spectral()))
    wcs.wcs.set()
    return wcs, wavelengths


def _log_velocity_axis(
        source: str, wcs: astropy.wcs.WCS
) -> tuple[astropy.wcs.WCS, str | None]:
    """
    Return the world coordinates with a spectral axis of logarithmic
    wavelengths (WAVE-LOG in vacuum, AWAV-LOG in air: w = S exp(x / S) of
    the intermediate coordinate x, S at the reference pixel) as the
    linear axis of their log velocities c ln(w / rest) (km/s; as VOPT, of
    the rest), exactly, and the medium of its wavelengths if it was one
    ('vacuum' or 'air'), or else None; others unchanged. The rest is that
    of the WCS (RESTWAV, in vacuum: in the medium of the axis, see
    _rest_in_medium).

    Raises
    ------
    ValueError
        If an axis of logarithmic wavelengths has no rest wavelength.
    """
    spec = wcs.wcs.spec
    if spec < 0 or wcs.wcs.ctype[spec] not in _MEDIUMS_OF_LOG_TYPES:
        return wcs, None
    log_medium = _MEDIUMS_OF_LOG_TYPES[wcs.wcs.ctype[spec]]
    if not wcs.wcs.restwav > 0:
        raise ValueError(
            f"{source}: the spectral axis of logarithmic wavelengths "
            f"({wcs.wcs.ctype[spec]}) needs a rest wavelength (RESTWAV, or "
            f"the rest)")
    vacuum = float(wcs.wcs.restwav)
    rest = _rest_in_medium(vacuum * u.m, log_medium)
    wcs = wcs.deepcopy()
    unit = u.Unit(str(wcs.wcs.cunit[spec]) or 'm')
    crval = float(wcs.wcs.crval[spec])
    _scale_axis(wcs, spec, _C / crval)
    wcs.wcs.crval[spec] = _C * np.log(crval / rest.to_value(unit))
    wcs.wcs.ctype[spec] = 'VOPT'
    wcs.wcs.cunit[spec] = 'km/s'
    _set_rest(wcs, wavelength=vacuum)
    wcs.wcs.set()
    return wcs, log_medium


def _spectral_convention(wcs: astropy.wcs.WCS) -> str | None:
    """
    Return the convention of the velocities of the spectral axis (see
    Coords) of its type: radio (VRAD), optical (VOPT), relativistic (VELO)
    or log (VLOG, a linear axis for FITS); None without one.
    """
    types = [ctype[:4] for ctype in wcs.wcs.ctype]
    if wcs.wcs.spec >= 0:
        types = [types[wcs.wcs.spec]]
    return next(
        (_CONVENTIONS_OF_TYPES[t] for t in types
         if t in _CONVENTIONS_OF_TYPES), None)


def _spectral_rest(
        source: str,
        wcs: astropy.wcs.WCS,
        rest: u.Quantity | None
) -> u.Quantity | None:
    """
    Return the rest of the spectral axis (see Coords): that given (rest,
    if not None), or that of the WCS (RESTWAV or RESTFRQ), as a
    wavelength for optical velocities (VOPT) and as a frequency for radio
    velocities (VRAD). Without a spectral axis type, a given rest keeps
    its kind. Relativistic velocities (VELO) have none: their lines are
    not at linear offsets, so a given rest is an error.
    """
    if wcs.wcs.spec < 0:
        return rest
    kind = wcs.wcs.ctype[wcs.wcs.spec][:4]
    if kind == 'VELO':
        if rest is not None:
            raise _relativistic_rest_error(source)
        return None
    # A WCS can have both: that of the convention of the axis first
    wavelength = wcs.wcs.restwav * u.m
    frequency = wcs.wcs.restfrq * u.Hz
    candidates = (
        (frequency, wavelength) if kind == 'VRAD' else (wavelength, frequency))
    if rest is None:
        rest = next((value for value in candidates if value.value > 0), None)
    if rest is None:
        return None
    unit = u.m if kind == 'VOPT' else u.Hz
    return rest.to(unit, u.spectral())


def _spectral_terms(
        source: str,
        header: astropy.io.fits.Header,
        wcs: astropy.wcs.WCS,
        rest: Rest,
        medium: str | None,
        log: str | None
) -> tuple[u.Quantity | None, str | None, str | None, str | None]:
    """
    Return the rest, the medium of the wavelengths, the convention of the
    velocities and the frame of rest of the spectral axis of the world
    coordinates of a header in the orientation of the model (see Coords,
    and coords_from_header): the rest given, or that of the header
    (RESTWAV, in vacuum, in the medium of the axis); the medium given, or
    that the header states (log: that of the type of an axis of
    logarithmic wavelengths, see _log_velocity_axis; or AIRORVAC), in
    vacuum by default; the convention of the type of the axis, or 'log';
    and the frame of SPECSYS (see _frame).

    Raises
    ------
    ValueError
        If a rest is given for relativistic velocities, or the medium is
        not valid or not the one the header states.
    """
    given = make_rest(rest)
    rest = _spectral_rest(source, wcs, given)
    wavelengths = make_medium(medium, rest)
    if wavelengths is not None:
        # The medium the header states: that of the type of an axis of
        # logarithmic wavelengths (FITS), or AIRORVAC (which model_header
        # writes from the type of an axis of wavelengths)
        stated = log or _header_medium(source, header)
        if medium is None:
            wavelengths = stated or wavelengths
        elif stated is not None and wavelengths != stated:
            raise _medium_contradiction(source, wavelengths, stated)
    if given is None:
        # (the rest of the header, RESTWAV, is in vacuum)
        rest = _rest_in_medium(rest, wavelengths)
    convention = 'log' if log else _spectral_convention(wcs)
    return rest, wavelengths, convention, _frame(source, wcs)


def _frame(source: str, wcs: astropy.wcs.WCS) -> str | None:
    """
    Return the frame of rest of the velocities of a WCS (SPECSYS, see
    FRAMES), if it gives one of FITS; a warning says when it gives
    another (see warn), which is left out.
    """
    frame = wcs.wcs.specsys.strip().upper()
    if frame and frame not in FRAMES:
        warn(
            f"{source}: SPECSYS is {frame!r}, not a frame of FITS "
            f"({', '.join(FRAMES)}); it is left out")
    return frame if frame in FRAMES else None


def _header_medium(
        source: str,
        header: astropy.io.fits.Header
) -> str | None:
    """
    Return the medium of the wavelengths of a header (its keyword
    AIRORVAC, in any case, see make_medium), if it has one.
    """
    value = header.get('AIRORVAC')
    if value is None:
        return None
    medium = _MEDIUMS_OF_HEADER.get(str(value).strip().lower())
    if medium is None:
        raise ValueError(
            f"{source}: AIRORVAC must be one of {list(_MEDIUMS_OF_HEADER)}; "
            f"it is {value!r}")
    return medium


def _stated_medium(
        source: str,
        header: astropy.io.fits.Header,
        medium: str | None = None
) -> str | None:
    """
    Return the medium of the wavelengths of optical velocities of a header
    and of their rest: that its keyword AIRORVAC states (see
    _header_medium), else the given one ('vacuum' or 'air'), if any.

    Raises
    ------
    ValueError
        If the given medium is not one of those, or is not the one that
        AIRORVAC states.
    """
    stated = _header_medium(source, header)
    if medium is None or medium == stated:
        return stated
    if medium not in _HEADER_MEDIUMS:
        raise ValueError(
            f"the medium must be 'vacuum' or 'air'; it is {medium!r}")
    if stated is not None:
        raise _medium_contradiction(source, medium, stated)
    return medium


def _medium_contradiction(source: str, given: str, stated: str) -> ValueError:
    """
    Return the error of a medium of wavelengths given for a header that
    states another.
    """
    return ValueError(
        f"{source}: the medium of the wavelengths given ({given}) is not "
        f"the one the header states ({stated}); leave it out")


def _type_medium(ctype: str) -> str | None:
    """
    Return the medium of the wavelengths of a spectral axis of a type
    (CTYPE), as FITS defines it: 'air' for AWAV (and AWAV-LOG), 'vacuum'
    for WAVE (and WAVE-LOG); None for others.
    """
    return dict(WAVE='vacuum', AWAV='air').get(ctype[:4])


def _check_header_medium(source: str, header: astropy.io.fits.Header) -> None:
    """
    Warn if AIRORVAC contradicts the medium of the wavelengths that the
    type of the spectral axis of a header states (see _type_medium): the
    type is what FITS defines, so that it is the one read.
    """
    stated = _header_medium(source, header)
    for n in range(1, header.get('NAXIS', 0) + 1):
        ctype = str(header.get(f'CTYPE{n}', ''))
        medium = _type_medium(ctype)
        if stated is not None and medium is not None and medium != stated:
            warn(
                f"{source}: AIRORVAC says its wavelengths are in {stated}, "
                f"but the type of its spectral axis, {ctype}, says {medium} "
                f"(FITS): they are read as in {medium}")


def _is_linear_type(ctype: str) -> bool:
    """
    Return whether a spectral axis type (CTYPE) is linear in its
    coordinate: it has no algorithm code (e.g. not WAVE-LOG or VRAD-F2V).
    """
    return not ctype[4:].strip('-').strip()


def _read_as_velocities(ctype: str) -> bool:
    """
    Return whether a spectral axis of a type (CTYPE) is read as
    velocities: velocities (see _VELOCITY_TYPES), and logarithmic
    wavelengths, as log velocities (see _log_velocity_axis).
    """
    return ctype[:4] in _VELOCITY_TYPES or ctype in _MEDIUMS_OF_LOG_TYPES


def _velocity_scale(source: str, wcs: astropy.wcs.WCS) -> float:
    """
    Return the factor from the unit of the spectral axis of world
    coordinates in the orientation of the model to km/s.

    Raises
    ------
    ValueError
        If the axis is not of velocities (a rest turns one of wavelengths
        or frequencies into velocities, see model_header), or is not
        linear in velocity.
    """
    ctype = wcs.wcs.ctype[wcs.wcs.spec]
    if ctype[:4] not in _VELOCITY_TYPES:
        # (the rest of the header, if any, which is not taken unless given)
        stated = (
            f" (its header states '{wcs.wcs.restfrq:.10g} Hz', RESTFRQ)"
            if wcs.wcs.restfrq else
            f" (its header states '{wcs.wcs.restwav:.10g} m', RESTWAV)"
            if wcs.wcs.restwav else "")
        raise ValueError(
            f"{source}: the spectral axis is of type '{ctype}'; only "
            f"velocity axes ({', '.join(_VELOCITY_TYPES)}) are supported: "
            f"give the rest frequency or wavelength of its line (spectral: "
            f"{{rest: ...}}), which turns it into velocities{stated}")
    if not _is_linear_type(ctype):
        raise ValueError(
            f"{source}: the spectral axis ({ctype}) is not linear in "
            f"velocity; only linear velocity axes are supported")
    return u.Unit(wcs.wcs.cunit[wcs.wcs.spec]).to(_KM_S)


def _scale_axis(wcs: astropy.wcs.WCS, axis: int, factor: float) -> None:
    """
    Scale the steps of a world axis of a WCS by a factor (its row of CD,
    or its CDELT), in place.
    """
    if wcs.wcs.has_cd():
        cd = wcs.wcs.cd.copy()
        cd[axis, :] *= factor
        wcs.wcs.cd = cd
    else:
        wcs.wcs.cdelt[axis] *= factor


def _rest_in_medium(
        rest: u.Quantity | None, medium: str | None
) -> u.Quantity | None:
    """
    Return a rest of a header (RESTWAV, in vacuum, or RESTFRQ) in the
    medium of the wavelengths of its axis: a rest wavelength in air for
    wavelengths in air (see air_refractive_index), in m; others as they
    are.
    """
    if (rest is None or medium != 'air'
            or rest.unit.physical_type == 'frequency'):
        return rest
    vacuum = rest.to_value(u.m)
    return vacuum / air_refractive_index(vacuum) * u.m


def _set_rest(
        wcs: astropy.wcs.WCS,
        wavelength: float = 0.0,
        frequency: float = 0.0,
        medium: str | None = None
) -> None:
    """
    Set the rest of the spectral axis of a WCS, in place: a wavelength (m,
    in the medium of the wavelengths of the axis, given) as RESTWAV, in
    vacuum, or a frequency (Hz) as RESTFRQ; the other 0 (none).
    """
    if wavelength and medium == 'air':
        wavelength = vacuum_wavelength(wavelength)
    wcs.wcs.restwav, wcs.wcs.restfrq = wavelength, frequency


def _relativistic_rest_error(source: str) -> ValueError:
    """Return the error of a rest given to relativistic velocities."""
    return ValueError(
        f"{source}: the spectral axis has relativistic velocities (VELO), "
        f"which have no rest; convert it to optical (VOPT) or radio (VRAD) "
        f"velocities")


def _velocity_axis_keywords(
        coords: Coords,
        n: int,
        index: int | None = None
) -> dict[str, float | str]:
    """
    Return the header keywords of the spectral axis, FITS axis n, whose
    world coordinates are those of axis index of coords (by default
    n - 1): velocities of the type of their convention (VOPT, VRAD or
    VELO; that of the rest by default), with the rest, if any, and the
    logarithmic wavelengths (WAVE-LOG, or AWAV-LOG in air) of log
    velocities of a rest wavelength (without one, VLOG, see
    _CONVENTIONS_OF_TYPES). A rest wavelength is written in vacuum
    (RESTWAV), whatever the medium of the axis (AIRORVAC).
    """
    index = n - 1 if index is None else index
    rest = coords.rest
    medium = make_medium(coords.medium, rest)
    convention = make_convention(coords.convention, rest) or 'relativistic'
    vacuum = None if medium is None else (
        vacuum_wavelength(rest.value) if medium == 'air' else rest.value)
    if convention == 'log' and rest is not None:
        crval = rest.value * np.exp(coords.rval[index] / _C)
        return {
            f'CTYPE{n}': _LOG_TYPES[medium], f'CUNIT{n}': 'm',
            f'CDELT{n}': coords.step[index] * crval / _C,
            f'CRPIX{n}': coords.rpix[index] + 1, f'CRVAL{n}': crval,
            'RESTWAV': vacuum, 'AIRORVAC': _HEADER_MEDIUMS[medium]}
    header = {
        f'CTYPE{n}': _TYPES_OF_CONVENTIONS[convention], f'CUNIT{n}': 'km/s',
        f'CDELT{n}': coords.step[index],
        f'CRPIX{n}': coords.rpix[index] + 1,
        f'CRVAL{n}': coords.rval[index]}
    if rest is not None and rest.unit == u.m:
        header.update({
            'RESTWAV': vacuum,
            'AIRORVAC': _HEADER_MEDIUMS[medium]})
    elif rest is not None:
        header.update(RESTFRQ=rest.value)
    return header
