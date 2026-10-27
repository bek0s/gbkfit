"""
Reading and writing data with world coordinates in FITS files.

Data are read in the orientation of the model, whatever that of their
file (see to_model_axes): the celestial axes first, longitude (e.g. RA)
then latitude, and the spectral axis last; east to the left of north;
and velocities increasing along the spectral axis. Their world
coordinates become those of the model (see coords_from_wcs).
"""

import typing

import astropy.constants
import astropy.coordinates
import astropy.io.fits
import astropy.table
import astropy.units
import astropy.wcs
import astropy.wcs.utils
import numpy as np

from gbkfit.utils.gridutils import (
    Coords, make_convention, make_medium, make_rest)
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'VELOCITY_TYPES',
    'casa_beam_table',
    'centre_missing_crpix',
    'coords_from_wcs',
    'flip_data',
    'flip_wcs',
    'model_axis_flips',
    'model_axis_order',
    'read_beam',
    'read_companion',
    'read_data',
    'read_unit',
    'reorder_data',
    'reorder_wcs',
    'to_model_axes',
    'with_velocity_convention',
    'write_data',
    'write_spectra'
]


# The spectral axis types that are velocities (FITS WCS Paper III)
VELOCITY_TYPES = ('VRAD', 'VOPT', 'VELO')

# The conventions of the velocities of each type of spectral axis, and
# the types of the conventions (as written; log velocities are written as
# WAVE-LOG, or without a rest wavelength as a linear axis of the type
# VLOG, which FITS does not define)
_CONVENTIONS_OF_TYPES = dict(
    VRAD='radio', VOPT='optical', VELO='relativistic', VLOG='log')
_TYPES_OF_CONVENTIONS = {
    convention: type_ for type_, convention in _CONVENTIONS_OF_TYPES.items()}

_KM_S = astropy.units.km / astropy.units.s

# The speed of light (km/s)
_C = astropy.constants.c.to_value('km/s')

# The values of the header keyword AIRORVAC of each medium of wavelengths
# (see gridutils.make_medium)
_HEADER_MEDIUMS = dict(vacuum='vac', air='air')

# The frames of the velocity axes of the AIPS convention (VELO-LSR, ...)
# and their codes in VELREF, which adds 256 for radio velocities
_AIPS_FRAMES = dict(LSR=1, HEL=2, OBS=3)

# The conventions of velocities (see with_velocity_convention)
_VELOCITY_CONVENTIONS = ('radio', 'optical')


def read_unit(filename: str, hdu: int | str = 0) -> str | None:
    """
    Return the unit of the data of a FITS file, as its header gives it
    (BUNIT), if it does.
    """
    unit = astropy.io.fits.getheader(filename, hdu).get('BUNIT')
    if unit is None or not str(unit).strip():
        return None
    return str(unit).strip()


def read_beam(
        filename: str, hdu: int | str = 0
) -> dict[str, typing.Any] | None:
    """
    Return the beam of the data of a FITS file, if it has one: the
    Gaussian restoring beam of a radio image, as the options bmaj, bmin
    (arcsec) and bpa (degrees): of the header (BMAJ, BMIN and BPA, in
    degrees), or for each channel of a cube of CASA (its table BEAMS; see
    casa_beam_table), as arrays with its spectral points ('spectral', a
    Quantity).
    """
    header = astropy.io.fits.getheader(filename, hdu)
    if 'BMAJ' in header and 'BMIN' in header:
        return dict(
            bmaj=float(header['BMAJ']) * 3600,
            bmin=float(header['BMIN']) * 3600,
            bpa=float(header.get('BPA', 0.0)))
    with astropy.io.fits.open(filename) as hdus:
        has_table = 'BEAMS' in hdus
    if not has_table:
        return None
    table = casa_beam_table(filename)
    return dict(
        bmaj=table['bmaj'].to_value('arcsec'),
        bmin=table['bmin'].to_value('arcsec'),
        bpa=table['bpa'].to_value('deg'),
        spectral=table.columns[0].quantity)


def casa_beam_table(filename: str) -> astropy.table.QTable:
    """
    Return the beams of the channels of a cube of CASA (its table BEAMS)
    as a table of the options of a Gaussian beam PSF (bmaj, bmin, bpa)
    against the points of its spectral axis (frequency, wavelength or
    velocity). Only the first polarisation is used.

    Raises
    ------
    ConfigError
        If the cube has no spectral axis.
    """
    u = astropy.units
    beams = astropy.table.Table.read(filename, hdu='BEAMS')
    if 'POL' in beams.colnames:
        beams = beams[beams['POL'] == 0]
    wcs = astropy.wcs.WCS(astropy.io.fits.getheader(filename, 0))
    if wcs.wcs.spec < 0:
        raise ConfigError(f"{filename}: the cube has no spectral axis")
    spectral = wcs.sub([wcs.wcs.spec + 1])
    unit = u.Unit(spectral.wcs.cunit[0])
    points = spectral.pixel_to_world_values(np.asarray(beams['CHAN']))
    axis = ('frequency' if unit.is_equivalent(u.Hz) else
            'wavelength' if unit.is_equivalent(u.m) else 'velocity')

    def column(name: str, to: str) -> u.Quantity:
        values = beams[name].quantity if beams[name].unit \
            else beams[name] * u.Unit(to)
        return values.to(to)
    return astropy.table.QTable({
        axis: np.asarray(points) * unit, 'bmaj': column('BMAJ', 'arcsec'),
        'bmin': column('BMIN', 'arcsec'), 'bpa': column('BPA', 'deg')})


def read_data(
        filename: str,
        hdu: int | str = 0,
        rpix: typing.Sequence[float] | None = None,
        rval: typing.Sequence[float] | None = None,
        rest: typing.Any = None,
        spectral_axis: int | None = None,
        medium: str | None = None
) -> tuple[np.ndarray, Coords]:
    """
    Read the data of a FITS file and its world coordinates.

    The data are in the orientation of the model (see to_model_axes), and
    their world coordinates in its units (see coords_from_wcs). The
    wavelengths of a spectral axis of optical velocities are in vacuum
    (FITS), or in the medium of the keyword AIRORVAC ('air' or 'vac') of
    the header, or in the given medium.

    Parameters
    ----------
    filename : str
        The name of the file.
    hdu : int or str
        The HDU with the data (e.g. 'SCI'); by default the first.
    rpix, rval, rest : optional
        World coordinates that replace those of the header (see
        coords_from_wcs), for the axes in the orientation of the model.
    spectral_axis : int, optional
        The index (FITS order, from 0) that the spectral axis of the data
        (if any) must have in the orientation of the model.
    medium : str, optional
        The medium of the wavelengths of the spectral axis, instead of
        that of the header (see gridutils.make_medium).

    Returns
    -------
    np.ndarray
        The data.
    Coords
        Their world coordinates.

    Raises
    ------
    ConfigError
        If the HDU has no data, the header has invalid world coordinates,
        or coordinates the model cannot represent (see coords_from_wcs),
        or does not say whether its velocities are radio or optical (see
        with_velocity_convention), or the medium is not valid.
    """
    data, header = _read_hdu(filename, hdu)
    header = with_velocity_convention(header, None, filename)
    wcs, log = _file_wcs(filename, header, data.shape, rest)
    data, wcs = to_model_axes(data, wcs)
    spec = wcs.wcs.spec
    if spectral_axis is not None and spec >= 0 and spec != spectral_axis:
        raise ConfigError(
            f"{filename}: the spectral axis must be the axis "
            f"{spectral_axis + 1}; it is the axis {spec + 1}")
    coords = coords_from_wcs(filename, wcs, rpix, rval, rest, medium)
    if medium is None and coords.medium is not None:
        coords = coords._replace(
            medium=_header_medium(filename, header) or coords.medium)
    if log:
        coords = coords._replace(convention='log')
    return np.ascontiguousarray(data), coords


def read_companion(
        filename: str,
        hdu: int | str,
        data_file: tuple[str, int | str],
        rest: typing.Any = None
) -> np.ndarray:
    """
    Read a companion of the data of a FITS file (e.g. their errors or
    mask), in the orientation of the model as the data are (see
    read_data): by its own world coordinates, if its header has any; else
    by those of the data, as an array in the axes of the file of the data
    (a file without world coordinates cannot say how its axes lie, and
    taken as it is, its values would sit on the wrong channels of a
    reversed spectral axis).

    Parameters
    ----------
    filename : str
        The name of the file.
    hdu : int or str
        The HDU with its values.
    data_file : tuple of str and (int or str)
        The file and the HDU of the data.
    rest : optional
        The rest that replaces that of the header of the data (see
        read_data).

    Raises
    ------
    ConfigError
        If the HDU has no values, or, without world coordinates, its shape
        is not that of the data.
    """
    values, header = _read_hdu(filename, hdu)
    if any(f'CTYPE{i}' in header for i in range(1, values.ndim + 1)):
        return read_data(filename, hdu, rest=rest)[0]
    data_name, data_hdu = data_file
    with astropy.io.fits.open(data_name) as hdulist:
        data_header = hdulist[data_hdu].header
    shape = tuple(
        data_header.get(f'NAXIS{i}', 0)
        for i in range(data_header.get('NAXIS', 0), 0, -1))
    if values.shape != shape:
        raise ConfigError(
            f"{filename}: without world coordinates, its values must be of "
            f"the shape of the data, {shape}, in the axes of their file; "
            f"they are of {values.shape}")
    header = with_velocity_convention(data_header, None, data_name)
    wcs, _ = _file_wcs(data_name, header, shape, rest)
    return np.ascontiguousarray(to_model_axes(values, wcs)[0])


def _read_hdu(
        filename: str, hdu: int | str
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the array and the header of an HDU of a FITS file, or raise
    ConfigError if it has no array.
    """
    with astropy.io.fits.open(filename) as hdulist:
        data = hdulist[hdu].data
        header = hdulist[hdu].header
    if data is None:
        raise ConfigError(
            f"{filename}: HDU {hdu} has no data; choose the HDU with the "
            f"data (e.g. hdu: SCI)")
    return data, header


def _file_wcs(
        filename: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        rest: typing.Any
) -> tuple[astropy.wcs.WCS, bool]:
    """
    Return the world coordinates of the header of a FITS file of an array
    of the given shape, in the axes of the file (with a missing reference
    pixel at the centre, and an axis of logarithmic wavelengths as one of
    log velocities, see _log_velocity_axis), and whether it has such an
    axis.

    Raises
    ------
    ConfigError
        If the world coordinates are invalid, or not of the axes of the
        array.
    """
    header = centre_missing_crpix(header, shape)
    try:
        wcs = astropy.wcs.WCS(header)
    except Exception as e:
        raise ConfigError(
            f"{filename}: invalid world coordinates: {e}") from e
    if wcs.naxis != len(shape):
        raise ConfigError(
            f"{filename}: the header has world coordinates for {wcs.naxis} "
            f"axes, but the data has {len(shape)}")
    return _log_velocity_axis(filename, wcs, rest)


def _log_velocity_axis(
        source: str,
        wcs: astropy.wcs.WCS,
        rest: str | astropy.units.Quantity | None
) -> tuple[astropy.wcs.WCS, bool]:
    """
    Return the world coordinates with a spectral axis of logarithmic
    wavelengths (WAVE-LOG, in vacuum: w = S exp(x / S) of the intermediate
    coordinate x, S at the reference pixel) as the linear axis of their
    log velocities c ln(w / rest) (km/s; as VOPT, of the rest), exactly,
    and whether it was; others unchanged. The rest is that given, or that
    of the WCS (RESTWAV).

    Raises
    ------
    ConfigError
        If an axis of logarithmic wavelengths has no rest wavelength.
    """
    s = wcs.wcs.spec
    if s < 0 or wcs.wcs.ctype[s] != 'WAVE-LOG':
        return wcs, False
    rest = make_rest(rest)
    if rest is None and wcs.wcs.restwav > 0:
        rest = wcs.wcs.restwav * astropy.units.m
    if rest is None or not rest.unit.is_equivalent(astropy.units.m):
        raise ConfigError(
            f"{source}: the spectral axis of logarithmic wavelengths "
            f"({wcs.wcs.ctype[s]}) needs a rest wavelength (RESTWAV, or the "
            f"rest)")
    wcs = wcs.deepcopy()
    unit = astropy.units.Unit(str(wcs.wcs.cunit[s]) or 'm')
    crval = float(wcs.wcs.crval[s])
    factor = _C / crval
    if wcs.wcs.has_cd():
        cd = wcs.wcs.cd.copy()
        cd[s, :] *= factor
        wcs.wcs.cd = cd
    else:
        wcs.wcs.cdelt[s] *= factor
    wcs.wcs.crval[s] = _C * np.log(crval / rest.to_value(unit))
    wcs.wcs.ctype[s] = 'VOPT'
    wcs.wcs.cunit[s] = 'km/s'
    wcs.wcs.restwav, wcs.wcs.restfrq = rest.to_value(astropy.units.m), 0
    wcs.wcs.set()
    return wcs, True


def coords_from_wcs(
        source: str,
        wcs: astropy.wcs.WCS,
        rpix: typing.Sequence[float] | None = None,
        rval: typing.Sequence[float] | None = None,
        rest: typing.Any = None,
        medium: str | None = None
) -> Coords:
    """
    Return the world coordinates of the model (see Coords) of a WCS.

    rpix is the reference pixel (CRPIX - 1), and rval the world position
    there (CRVAL). Either can be given instead, for each axis, and the
    other is computed from the WCS (the axes are uncoupled, but the
    celestial ones, which take the same one). The rest of the spectral
    axis is that of the WCS (RESTWAV or RESTFRQ), or rest, if given, and
    the convention of its velocities that of its type (see
    _spectral_convention). Celestial axes of another
    frame than ICRS (e.g. Galactic, or FK4) are turned into it: the RA and
    Dec of the reference pixel, and the rotation of the pixel grid from
    north of ICRS there (exact at it; the frames differ by more, a hair
    away from it, only for fields of degrees).

    Parameters
    ----------
    source : str
        The name of the source of the WCS (e.g. a file), for messages.
    wcs : astropy.wcs.WCS
        World coordinates, with the axes in the orientation of the model
        (see to_model_axes).
    rpix, rval : Sequence[float or None], optional
        The reference pixel or value, in the units of the model, a value
        for each axis (None: not given) or one for all.
    rest : str or Quantity, optional
        The rest of the spectral axis (see gridutils.make_rest).
    medium : str, optional
        The medium of the wavelengths of a rest wavelength (see
        gridutils.make_medium).

    Returns
    -------
    Coords
        The world coordinates.

    Raises
    ------
    ConfigError
        For coordinates the model cannot represent: a skewed pixel grid,
        celestial axes coupled to other axes, and spectral axes that are
        not velocities or are not linear in velocity; or if the medium is
        not valid.
    ValueError
        If the axes are not in the orientation of the model.
    """
    naxis = wcs.naxis
    if model_axis_order(wcs) != tuple(range(naxis)) or model_axis_flips(wcs):
        raise ValueError(
            f"{source}: the axes of the world coordinates are not in the "
            f"orientation of the model (see to_model_axes)")
    lng, lat = wcs.wcs.lng, wcs.wcs.lat
    linear = _linear(wcs)
    # The factor from the world units of each axis to model units
    scale = [_axis_scale(source, wcs, axis) for axis in range(naxis)]
    rota = _celestial_rotation(source, linear) if lng >= 0 else 0.0
    _check_uncoupled(source, wcs, linear)
    step = [
        np.hypot(*linear[:, axis][[lng, lat]]) * 3600
        if axis in (lng, lat)
        else linear[axis, axis] * scale[axis]
        for axis in range(naxis)]
    # (NaN where not given; an axis takes the other one of the WCS, at the
    # values of the header on the other axes, which do not change it)
    rpix, rval = (
        np.full(naxis, np.nan) if value is None
        else np.broadcast_to(np.asarray(value, float), naxis)
        if np.ndim(value) == 0
        else np.array([np.nan if x is None else x for x in value], float)
        for value in (rpix, rval))
    header_rpix = wcs.wcs.crpix - 1
    header_rval = np.multiply(wcs.wcs.crval, scale)
    pixel = np.where(np.isnan(rpix), header_rpix, rpix)
    world = np.where(np.isnan(rval), wcs.wcs.crval, np.divide(rval, scale))
    rval_at_rpix = np.ravel(wcs.pixel_to_world_values(*pixel)) * scale
    rpix_at_rval = np.ravel(wcs.world_to_pixel_values(*world))
    rpix, rval = (
        np.where(~np.isnan(rpix), rpix, np.where(
            np.isnan(rval), header_rpix, rpix_at_rval)).tolist(),
        np.where(~np.isnan(rval), rval, np.where(
            np.isnan(rpix), header_rval, rval_at_rpix)).tolist())
    if lng >= 0:
        rval, rota = _to_icrs(source, wcs, list(rval), rota)
    rest = _spectral_rest(source, wcs, make_rest(rest))
    return Coords(
        tuple(float(x) for x in step),
        tuple(float(x) for x in rpix),
        tuple(float(x) for x in rval),
        float(rota),
        rest, make_medium(medium, rest), _spectral_convention(wcs))


def model_axis_order(wcs: astropy.wcs.WCS) -> tuple[int, ...]:
    """
    Return the axes of a WCS in the order of the model.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        World coordinates.

    Returns
    -------
    tuple[int, ...]
        Its axes (FITS order, from 0): the celestial axes first,
        longitude (e.g. RA) then latitude, the spectral axis last, and
        the other axes between them, in their order.
    """
    lng, lat, spec = wcs.wcs.lng, wcs.wcs.lat, wcs.wcs.spec
    first = (lng, lat) if lng >= 0 else ()
    last = (spec,) if spec >= 0 else ()
    middle = tuple(
        axis for axis in range(wcs.naxis) if axis not in first + last)
    return first + middle + last


def model_axis_flips(wcs: astropy.wcs.WCS) -> tuple[int, ...]:
    """
    Return the axes of a WCS that the model needs reversed.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        World coordinates, with the axes in the order of the model (see
        model_axis_order).

    Returns
    -------
    tuple[int, ...]
        The axes to reverse (FITS order, from 0): the longitude axis of a
        mirrored image (east to the right of north), and a spectral axis
        of velocities that decrease along it.
    """
    lng, lat, spec = wcs.wcs.lng, wcs.wcs.lat, wcs.wcs.spec
    linear = _linear(wcs)
    flips = []
    if lng >= 0 and np.linalg.det(linear[np.ix_([lng, lat], [lng, lat])]) > 0:
        flips.append(lng)
    is_velocity = spec >= 0 and wcs.wcs.ctype[spec][:4] in VELOCITY_TYPES
    if is_velocity and linear[spec, spec] < 0:
        flips.append(spec)
    return tuple(flips)


def reorder_data(data: np.ndarray, order: typing.Sequence[int]) -> np.ndarray:
    """
    Return data with their axes in the given order.

    Parameters
    ----------
    data : np.ndarray
        The data.
    order : Sequence[int]
        Their axes (FITS order, from 0) in their new order (see
        model_axis_order).

    Returns
    -------
    np.ndarray
        The data, as a view.
    """
    n = data.ndim
    # (numpy axes are in the reverse of the FITS order)
    return np.transpose(data, [n - 1 - order[n - 1 - i] for i in range(n)])


def reorder_wcs(
        wcs: astropy.wcs.WCS,
        order: typing.Sequence[int]
) -> astropy.wcs.WCS:
    """
    Return world coordinates with their axes in the given order.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        World coordinates.
    order : Sequence[int]
        Their axes (FITS order, from 0) in their new order (see
        model_axis_order).

    Returns
    -------
    astropy.wcs.WCS
        The world coordinates of the reordered axes.
    """
    return wcs.sub([axis + 1 for axis in order])


def flip_data(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Return data reversed along an axis.

    Parameters
    ----------
    data : np.ndarray
        The data.
    axis : int
        The axis (FITS order, from 0).

    Returns
    -------
    np.ndarray
        The data, as a view.
    """
    return np.flip(data, data.ndim - 1 - axis)


def flip_wcs(
        wcs: astropy.wcs.WCS,
        axis: int,
        size: int
) -> astropy.wcs.WCS:
    """
    Return world coordinates with a pixel axis reversed.

    Each pixel keeps its world coordinates when its data are reversed
    along the axis (see flip_data).

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        World coordinates.
    axis : int
        The axis (FITS order, from 0).
    size : int
        The number of pixels along the axis.

    Returns
    -------
    astropy.wcs.WCS
        The world coordinates of the reversed axis.
    """
    wcs = wcs.deepcopy()
    # Reversing a pixel axis negates its column of the linear
    # transformation (CD, or CDELT times the rows of PC). For PC, negating
    # CDELT and the row of PC cancel out, leaving CDELT positive.
    if wcs.wcs.has_cd():
        cd = wcs.wcs.cd.copy()
        cd[:, axis] *= -1
        wcs.wcs.cd = cd
    else:
        pc = wcs.wcs.get_pc().copy()
        pc[axis, :] *= -1
        pc[:, axis] *= -1
        wcs.wcs.pc = pc
        wcs.wcs.cdelt[axis] *= -1
    wcs.wcs.crpix[axis] = size + 1 - wcs.wcs.crpix[axis]
    wcs.wcs.set()
    return wcs


def to_model_axes(
        data: np.ndarray,
        wcs: astropy.wcs.WCS
) -> tuple[np.ndarray, astropy.wcs.WCS]:
    """
    Return data and their world coordinates in the orientation of the
    model.

    The axes are reordered (see model_axis_order), and those that the
    model needs reversed are reversed (see model_axis_flips).

    Parameters
    ----------
    data : np.ndarray
        The data.
    wcs : astropy.wcs.WCS
        Their world coordinates.

    Returns
    -------
    np.ndarray
        The data, as a view.
    astropy.wcs.WCS
        Their world coordinates.
    """
    order = model_axis_order(wcs)
    if order != tuple(range(wcs.naxis)):
        data = reorder_data(data, order)
        wcs = reorder_wcs(wcs, order)
    for axis in model_axis_flips(wcs):
        data = flip_data(data, axis)
        wcs = flip_wcs(wcs, axis, data.shape[data.ndim - 1 - axis])
    return data, wcs


def centre_missing_crpix(
        header: astropy.io.fits.Header,
        shape: tuple[int, ...]
) -> astropy.io.fits.Header:
    """
    Return a header with the reference pixel of each axis that has none
    at its centre.

    This is the convention of the model; FITS would put it at 0.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The header.
    shape : tuple[int, ...]
        The shape of its data (numpy order).

    Returns
    -------
    astropy.io.fits.Header
        A copy of the header, with the missing CRPIXn.
    """
    header = header.copy()
    for n, size in enumerate(shape[::-1], start=1):
        if f'CRPIX{n}' not in header:
            header[f'CRPIX{n}'] = size / 2 + 0.5
    return header


def write_data(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        spectral_axis: int | None = None,
        overwrite: bool = False,
        unit: str | None = None,
        cards: typing.Mapping[str, typing.Any] | None = None
) -> None:
    """
    Write data with the world coordinates of the model to a FITS file.

    The axes of the data are the x and y axes of the sky (RA and Dec, TAN
    projection, rotated with a PC matrix), followed by the spectral axis
    (spectral_axis = 2, a velocity) or by a position along the line of
    sight (spectral_axis = None, an offset in arcsec from its reference
    pixel); or a position along a slit followed by the spectral axis
    (spectral_axis = 1); or the spectral axis alone (spectral_axis = 0).

    Parameters
    ----------
    filename : str
        The name of the file.
    data : np.ndarray
        The data.
    coords : Coords
        Their world coordinates.
    spectral_axis : int, optional
        The index of their spectral axis (FITS order, from 0), if any.
    overwrite : bool
        Whether to overwrite an existing file.
    unit : str, optional
        The unit of the data (BUNIT), if known.
    cards : Mapping, optional
        Further header cards, by keyword (e.g. provenance).

    Raises
    ------
    ValueError
        For data of another layout.
    """
    header = astropy.io.fits.Header()
    if spectral_axis == 0 and data.ndim == 1:
        header.update(_velocity_header(coords, 1))
    elif spectral_axis is None and data.ndim == 2:
        header.update(_sky_header(coords))
    elif spectral_axis == 2 and data.ndim == 3:
        header.update(_sky_header(coords))
        header.update(_velocity_header(coords, 3))
    elif spectral_axis is None and data.ndim == 3:
        header.update(_sky_header(coords))
        header.update(_offset_header(
            3, coords.step[2], coords.rpix[2], coords.rval[2]))
    elif spectral_axis == 1 and data.ndim == 2:
        # The position along the slit is an offset from its reference
        header.update(_offset_header(
            1, coords.step[0], coords.rpix[0], 0.0))
        header.update(_velocity_header(coords, 2))
    else:
        raise ValueError(
            f"unsupported data: {data.ndim} axes, of which the spectral "
            f"axis is {spectral_axis}")
    if unit is not None:
        header['BUNIT'] = unit
    header.update(cards or {})
    astropy.io.fits.writeto(
        filename, data, header,
        output_verify='exception', overwrite=overwrite, checksum=True)


def write_spectra(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        overwrite: bool = False,
        unit: str | None = None,
        cards: typing.Mapping[str, typing.Any] | None = None
) -> None:
    """
    Write spectra of regions to a FITS file.

    The regions are along x (an index, without world coordinates), and
    the velocity along y (see gridutils.SpectraData).

    Parameters
    ----------
    filename : str
        The name of the file.
    data : np.ndarray
        The spectra, of shape (nchannels, nregions).
    coords : Coords
        The world coordinates of their spectral axis (of one axis).
    overwrite : bool
        Whether to overwrite an existing file.
    unit : str, optional
        The unit of the spectra (BUNIT), if known.
    cards : Mapping, optional
        Further header cards, by keyword (e.g. provenance).
    """
    header = astropy.io.fits.Header(_velocity_header(coords, 2, 0))
    if unit is not None:
        header['BUNIT'] = unit
    header.update(cards or {})
    astropy.io.fits.writeto(
        filename, data, header,
        output_verify='exception', overwrite=overwrite, checksum=True)


def _linear(wcs):
    """
    Return the linear transformation from pixels to intermediate world
    coordinates: one row for each world axis, one column for each pixel
    axis.
    """
    return wcs.wcs.get_cdelt()[:, None] * wcs.wcs.get_pc()


def _sky_header(coords):
    """Return the header keywords of the x and y axes on the sky."""
    step_x, step_y = coords.step[:2]
    rota = np.radians(coords.rota)
    # The +y axis points to the position angle rota, and the +x axis to
    # rota - 90 (east is to the left of north). Intermediate world x
    # points east and y north. The rows of the PC matrix are scaled by
    # CDELT, so PC is a rotation only for square pixels.
    cd = np.array([
        [-step_x * np.cos(rota), step_y * np.sin(rota)],
        [step_x * np.sin(rota), step_y * np.cos(rota)]]) / 3600
    cdelt = np.array([-step_x, step_y]) / 3600
    pc = cd / cdelt[:, None]
    return dict(
        CTYPE1='RA---TAN', CUNIT1='deg', CDELT1=cdelt[0],
        CRPIX1=coords.rpix[0] + 1, CRVAL1=coords.rval[0],
        CTYPE2='DEC--TAN', CUNIT2='deg', CDELT2=cdelt[1],
        CRPIX2=coords.rpix[1] + 1, CRVAL2=coords.rval[1], RADESYS='ICRS',
        PC1_1=pc[0, 0], PC1_2=pc[0, 1],
        PC2_1=pc[1, 0], PC2_2=pc[1, 1])


def _offset_header(n, step, rpix, rval):
    """Return the header keywords of an offset in arcsec, FITS axis n."""
    return {
        f'CTYPE{n}': 'OFFSET', f'CUNIT{n}': 'arcsec', f'CDELT{n}': step,
        f'CRPIX{n}': rpix + 1, f'CRVAL{n}': rval}


def _velocity_header(coords, n, index=None):
    """
    Return the header keywords of the spectral axis, FITS axis n, whose
    world coordinates are those of axis index of coords (by default
    n - 1): velocities of the type of their convention (VOPT, VRAD or
    VELO; that of the rest by default), with the rest, if any, and the
    logarithmic wavelengths (WAVE-LOG) of log velocities of a rest
    wavelength (without one, VLOG, see _CONVENTIONS_OF_TYPES).
    """
    index = n - 1 if index is None else index
    rest = coords.rest
    convention = make_convention(coords.convention, rest) or 'relativistic'
    if convention == 'log' and rest is not None:
        crval = rest.value * np.exp(coords.rval[index] / _C)
        return {
            f'CTYPE{n}': 'WAVE-LOG', f'CUNIT{n}': 'm',
            f'CDELT{n}': coords.step[index] * crval / _C,
            f'CRPIX{n}': coords.rpix[index] + 1, f'CRVAL{n}': crval,
            'RESTWAV': rest.value, 'AIRORVAC': _HEADER_MEDIUMS[coords.medium]}
    header = {
        f'CTYPE{n}': _TYPES_OF_CONVENTIONS[convention], f'CUNIT{n}': 'km/s',
        f'CDELT{n}': coords.step[index],
        f'CRPIX{n}': coords.rpix[index] + 1,
        f'CRVAL{n}': coords.rval[index]}
    if rest is not None and rest.unit == astropy.units.m:
        header.update({
            'RESTWAV': rest.value,
            'AIRORVAC': _HEADER_MEDIUMS[coords.medium]})
    elif rest is not None:
        header.update(RESTFRQ=rest.value)
    return header


def _header_medium(source, header):
    """
    Return the medium of the wavelengths of a header (its keyword
    AIRORVAC, see gridutils.make_medium), if it has one.
    """
    value = header.get('AIRORVAC')
    if value is None:
        return None
    mediums = {v: k for k, v in _HEADER_MEDIUMS.items()}
    if value not in mediums:
        raise ConfigError(
            f"{source}: AIRORVAC must be one of {list(mediums)}; it is "
            f"{value!r}")
    return mediums[value]


def with_velocity_convention(
        header: astropy.io.fits.Header,
        velocity: str | None,
        source: str
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
    header : Header
        The header.
    velocity : str, optional
        The convention of the velocities: 'radio' or 'optical'; only for
        an axis of the AIPS convention without VELREF.
    source : str
        The name of the file of the header, for the errors.

    Returns
    -------
    Header
        The header (a copy, with VELREF, if the convention is given).

    Raises
    ------
    ConfigError
        If the convention is ambiguous and not given, or it is given and
        the header states it, or it is not 'radio' or 'optical'.
    """
    if velocity is not None and velocity not in _VELOCITY_CONVENTIONS:
        raise ConfigError(
            f"the convention of velocities must be one of "
            f"{list(_VELOCITY_CONVENTIONS)}; it is {velocity!r}")
    # (an axis of one pixel, e.g. that left in a moment map, has no
    # velocities to read)
    ctypes = [
        str(header.get(f'CTYPE{n}', ''))
        for n in range(1, header.get('NAXIS', 0) + 1)
        if header.get(f'NAXIS{n}', 1) > 1]
    frame = next((
        ctype[5:].strip() for ctype in ctypes
        if ctype[:5] == 'VELO-' and ctype[5:].strip() in _AIPS_FRAMES), None)
    ambiguous = frame is not None and 'VELREF' not in header
    if velocity is None:
        if ambiguous:
            raise ConfigError(
                f"{source}: the velocities of the spectral axis (VELO-{frame}"
                f", without VELREF) may be radio or optical: AIPS reads them "
                f"as optical, but MIRIAD writes radio ones (e.g. LVHIS). Say "
                f"which: prepare the data with gbkfit-cli prep and the option "
                f"--velocity radio or optical, or add VELREF to the header "
                f"({_AIPS_FRAMES[frame]} for optical, "
                f"{_AIPS_FRAMES[frame] + 256} for radio)")
        return header
    if not ambiguous:
        raise ConfigError(
            f"{source}: the convention of velocities is for a spectral axis "
            f"of the AIPS convention without VELREF (e.g. VELO-HEL); this "
            f"header states its own")
    header = header.copy()
    header['VELREF'] = _AIPS_FRAMES[frame] + (256 if velocity == 'radio' else 0)
    return header


def _spectral_convention(wcs):
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


def _spectral_rest(source, wcs, rest):
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
            raise ConfigError(
                f"{source}: the spectral axis has relativistic velocities "
                f"(VELO), which have no rest; convert it to optical (VOPT) "
                f"or radio (VRAD) velocities")
        return None
    # A WCS can have both: that of the convention of the axis first
    wavelength = wcs.wcs.restwav * astropy.units.m
    frequency = wcs.wcs.restfrq * astropy.units.Hz
    candidates = (
        (frequency, wavelength) if kind == 'VRAD' else (wavelength, frequency))
    if rest is None:
        rest = next((value for value in candidates if value.value > 0), None)
    if rest is None:
        return None
    unit = astropy.units.m if kind == 'VOPT' else astropy.units.Hz
    return rest.to(unit, astropy.units.spectral())


def _axis_scale(source, wcs, axis):
    """
    Return the factor from the world units of an axis to model units:
    celestial axes stay in degrees (only their steps become arcsec),
    velocity axes become km/s, other axes in units of angle (e.g. the
    position along a slit, OFFSET) become arcsec, and the other axes keep
    their units.
    """
    if axis != wcs.wcs.spec:
        unit = astropy.units.Unit(wcs.wcs.cunit[axis])
        celestial = axis in (wcs.wcs.lng, wcs.wcs.lat)
        if not celestial and unit.physical_type == 'angle':
            return unit.to(astropy.units.arcsec)
        return 1.0
    ctype = wcs.wcs.ctype[axis]
    if ctype[:4] not in VELOCITY_TYPES:
        raise ConfigError(
            f"{source}: the spectral axis is of type '{ctype}'; only "
            f"velocity axes ({', '.join(VELOCITY_TYPES)}) are supported")
    if ctype[4:].strip('-').strip():
        raise ConfigError(
            f"{source}: the spectral axis ({ctype}) is not linear in "
            f"velocity; only linear velocity axes are supported")
    return astropy.units.Unit(wcs.wcs.cunit[axis]).to(_KM_S)


def _celestial_rotation(source, linear):
    """
    Return the position angle (degrees, north through east) of the +y
    pixel axis, from the linear transformation of the celestial axes (the
    first two). Intermediate world x points east and y north.
    """
    column_x, column_y = linear[:2, :2].T
    cosine = column_x @ column_y / np.hypot(*column_x) / np.hypot(*column_y)
    if abs(cosine) > 1e-6:
        raise ConfigError(
            f"{source}: the pixel grid is skewed (its axes are not "
            f"perpendicular on the sky), which the model does not support")
    return float(np.degrees(np.arctan2(column_y[0], column_y[1])))


def _to_icrs(source, wcs, rval, rota):
    """
    Return the world position of the reference pixel (rval, of the
    celestial axes in degrees) and the position angle of the +y pixel
    axis (rota) of a WCS in ICRS: as they are in it, else turned from its
    celestial frame, by the position angle of its north in ICRS at the
    reference pixel.
    """
    try:
        frame = astropy.wcs.utils.wcs_to_celestial_frame(wcs)
    except ValueError as e:
        raise ConfigError(
            f"{source}: the celestial frame of the world coordinates is "
            f"not known: {e}") from e
    if isinstance(frame, astropy.coordinates.ICRS):
        return rval, rota
    lng, lat = wcs.wcs.lng, wcs.wcs.lat
    reference = astropy.coordinates.SkyCoord(
        rval[lng], rval[lat], unit='deg', frame=frame)
    north = reference.directional_offset_by(
        0 * astropy.units.deg, 1 * astropy.units.arcsec)
    reference, north = reference.icrs, north.icrs
    rval[lng], rval[lat] = reference.ra.deg, reference.dec.deg
    turn = reference.position_angle(north).deg
    return rval, float((rota + turn + 180) % 360 - 180)


def _check_uncoupled(source, wcs, linear):
    """
    Raise ConfigError if a pixel axis other than the celestial pair is
    rotated into another world axis (e.g. velocity varying along x).
    """
    celestial = {wcs.wcs.lng, wcs.wcs.lat} - {-1}
    coupled = linear.copy()
    for i in celestial:
        for j in celestial:
            coupled[i, j] = 0
    np.fill_diagonal(coupled, 0)
    if np.any(coupled):
        raise ConfigError(
            f"{source}: the world coordinates couple axes that the model "
            f"keeps independent (e.g. the velocity varies along a spatial "
            f"axis)")
