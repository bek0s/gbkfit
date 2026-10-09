import typing

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np

from gbkfit.utils.gridutils import Coords, make_rest
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'VELOCITY_TYPES',
    'centre_missing_crpix',
    'read_data',
    'write_data',
    'write_spectra'
]


# The spectral axis types that are velocities (FITS WCS Paper III)
VELOCITY_TYPES = ('VRAD', 'VOPT', 'VELO')

_KM_S = astropy.units.km / astropy.units.s


def read_data(
        filename: str,
        hdu: int | str = 0,
        rpix: typing.Sequence[float] | None = None,
        rval: typing.Sequence[float] | None = None,
        rest: typing.Any = None,
        spectral_axis: int | None = None
) -> tuple[np.ndarray, Coords]:
    """
    The data of a FITS file (from the given HDU) and its world
    coordinates in the units of the model (see Coords). Its celestial
    axes (if any) must be the first two (x, y), longitude (e.g. RA)
    first, and its spectral axis (if any) the axis spectral_axis (FITS
    order, from 0), if given.

    rpix is CRPIX - 1 (the centre of the axes without CRPIX), and rval is
    CRVAL, the world position at rpix. Either can be given instead (in
    model units), and the other is computed from the header. The rest of
    the spectral axis (see _spectral_rest) is that of the header (RESTWAV
    or RESTFRQ), or rest if given.

    Raise ConfigError for coordinates the model cannot represent: a
    header it cannot read, a mirrored (east to the right of north) or
    skewed pixel grid, celestial axes coupled to other axes, and
    spectral axes that are not velocities.
    """
    with astropy.io.fits.open(filename) as hdulist:
        data = hdulist[hdu].data
        header = hdulist[hdu].header
    if data is None:
        raise ConfigError(
            f"{filename}: HDU {hdu} has no data; choose the HDU with the "
            f"data (e.g. hdu: SCI)")
    header = centre_missing_crpix(header, data.shape)
    try:
        wcs = astropy.wcs.WCS(header)
    except Exception as e:
        raise ConfigError(
            f"{filename}: invalid world coordinates: {e}") from e
    if wcs.naxis != data.ndim:
        raise ConfigError(
            f"{filename}: the header has world coordinates for {wcs.naxis} "
            f"axes, but the data has {data.ndim}")
    lng, lat, spec = wcs.wcs.lng, wcs.wcs.lat, wcs.wcs.spec
    if lng >= 0 and (lng, lat) != (0, 1):
        raise ConfigError(
            f"{filename}: the celestial axes must be the first two, "
            f"longitude (e.g. RA) first; they are the axes {lng + 1} "
            f"(longitude) and {lat + 1}")
    if spectral_axis is not None and spec >= 0 and spec != spectral_axis:
        raise ConfigError(
            f"{filename}: the spectral axis must be the axis "
            f"{spectral_axis + 1}; it is the axis {spec + 1}")
    # The linear transformation from pixels to intermediate world
    # coordinates: one row for each world axis, one column for each
    # pixel axis
    linear = wcs.wcs.get_cdelt()[:, None] * wcs.wcs.get_pc()
    # The factor from the world units of each axis to model units
    scale = [_axis_scale(filename, wcs, axis) for axis in range(wcs.naxis)]
    rota = 0.0
    if wcs.wcs.lng >= 0:
        rota = _celestial_rotation(filename, linear, wcs.wcs.lng, wcs.wcs.lat)
    _check_uncoupled(filename, wcs, linear)
    step = [
        np.hypot(*linear[:, axis][[wcs.wcs.lng, wcs.wcs.lat]]) * 3600
        if axis in (wcs.wcs.lng, wcs.wcs.lat)
        else linear[axis, axis] * scale[axis]
        for axis in range(wcs.naxis)]
    if wcs.wcs.spec >= 0 and step[wcs.wcs.spec] < 0:
        raise ConfigError(
            f"{filename}: the velocity decreases along the spectral axis; "
            f"reverse the axis with gbkfit-cli prep")
    if rpix is not None:
        rpix = np.broadcast_to(np.asarray(rpix, float), wcs.naxis)
    if rval is not None:
        rval = np.broadcast_to(np.asarray(rval, float), wcs.naxis)
    if rpix is None and rval is None:
        rpix = (wcs.wcs.crpix - 1).tolist()
        rval = np.multiply(wcs.wcs.crval, scale).tolist()
    elif rpix is None:
        world = np.divide(rval, scale)
        rpix = np.ravel(wcs.world_to_pixel_values(*world)).tolist()
    elif rval is None:
        world = np.ravel(wcs.pixel_to_world_values(*rpix))
        rval = np.multiply(world, scale).tolist()
    coords = Coords(
        tuple(float(x) for x in step),
        tuple(float(x) for x in rpix),
        tuple(float(x) for x in rval),
        float(rota),
        _spectral_rest(filename, wcs, make_rest(rest)))
    return data, coords


def centre_missing_crpix(
        header: astropy.io.fits.Header,
        shape: tuple[int, ...]
) -> astropy.io.fits.Header:
    """
    A copy of the header of data of the given shape (numpy order) in which
    the axes without a reference pixel (CRPIXn) have it at their centre.
    This is the convention of the model; FITS would put it at 0.
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
        overwrite: bool = False
) -> None:
    """
    Write data with the world coordinates of the model (see Coords) to a
    FITS file. The axes of the data are the x and y axes of the sky (RA
    and Dec, TAN projection, rotated with a PC matrix), followed by the
    spectral axis (spectral_axis = 2, a radio velocity) or by a position
    along the line of sight (spectral_axis = None, an offset in arcsec
    from its reference pixel), or a position along a slit followed by the
    spectral axis (spectral_axis = 1), or the spectral axis alone
    (spectral_axis = 0).
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
    astropy.io.fits.writeto(
        filename, data, header,
        output_verify='exception', overwrite=overwrite, checksum=True)


def write_spectra(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        overwrite: bool = False
) -> None:
    """
    Write spectra of regions (see gridutils.SpectraData) to a FITS file: the regions
    along x (an index, without world coordinates), and the velocity along
    y, with the world coordinates of a spectral axis (a Coords of one
    axis).
    """
    header = astropy.io.fits.Header(_velocity_header(coords, 2, 0))
    astropy.io.fits.writeto(
        filename, data, header,
        output_verify='exception', overwrite=overwrite, checksum=True)


def _sky_header(coords):
    """The header keywords of the x and y axes on the sky."""
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
        CRPIX2=coords.rpix[1] + 1, CRVAL2=coords.rval[1],
        PC1_1=pc[0, 0], PC1_2=pc[0, 1],
        PC2_1=pc[1, 0], PC2_2=pc[1, 1])


def _offset_header(n, step, rpix, rval):
    """The header keywords of an offset in arcsec, FITS axis n."""
    return {
        f'CTYPE{n}': 'OFFSET', f'CUNIT{n}': 'arcsec', f'CDELT{n}': step,
        f'CRPIX{n}': rpix + 1, f'CRVAL{n}': rval}


def _velocity_header(coords, n, index=None):
    """
    The header keywords of the spectral axis, FITS axis n, whose world
    coordinates are those of axis index of coords (by default n - 1):
    optical velocities with a rest wavelength, radio velocities with a
    rest frequency or without a rest.
    """
    index = n - 1 if index is None else index
    rest = coords.rest
    header = {
        f'CTYPE{n}': 'VRAD', f'CUNIT{n}': 'km/s',
        f'CDELT{n}': coords.step[index],
        f'CRPIX{n}': coords.rpix[index] + 1,
        f'CRVAL{n}': coords.rval[index]}
    if rest is not None and rest.unit == astropy.units.m:
        header.update({f'CTYPE{n}': 'VOPT', 'RESTWAV': rest.value})
    elif rest is not None:
        header.update(RESTFRQ=rest.value)
    return header


def _spectral_rest(filename, wcs, rest):
    """
    The rest of the spectral axis (see Coords): that given (rest, if not
    None), or that of the header (RESTWAV or RESTFRQ), as a wavelength for
    optical velocities (VOPT) and as a frequency for radio velocities
    (VRAD). Without a spectral axis type, a given rest keeps its kind.
    Relativistic velocities (VELO) have none: their lines are not at
    linear offsets, so a given rest is an error.
    """
    if wcs.wcs.spec < 0:
        return rest
    kind = wcs.wcs.ctype[wcs.wcs.spec][:4]
    if kind == 'VELO':
        if rest is not None:
            raise ConfigError(
                f"{filename}: the spectral axis has relativistic velocities "
                f"(VELO), which have no rest; convert it to optical (VOPT) "
                f"or radio (VRAD) velocities")
        return None
    # A header can have both: that of the convention of the axis first
    wavelength = wcs.wcs.restwav * astropy.units.m
    frequency = wcs.wcs.restfrq * astropy.units.Hz
    headers = (
        (frequency, wavelength) if kind == 'VRAD' else (wavelength, frequency))
    if rest is None:
        rest = next((value for value in headers if value.value > 0), None)
    if rest is None:
        return None
    unit = astropy.units.m if kind == 'VOPT' else astropy.units.Hz
    return rest.to(unit, astropy.units.spectral())


def _axis_scale(filename, wcs, axis):
    """
    The factor from the world units of an axis to model units: celestial
    axes stay in degrees (only their steps become arcsec), velocity axes
    become km/s, and the other axes keep their units.
    """
    if axis != wcs.wcs.spec:
        return 1.0
    ctype = wcs.wcs.ctype[axis]
    if ctype[:4] not in VELOCITY_TYPES:
        raise ConfigError(
            f"{filename}: the spectral axis is of type '{ctype}'; only "
            f"velocity axes ({', '.join(VELOCITY_TYPES)}) are supported")
    if ctype[4:].strip('-').strip():
        raise ConfigError(
            f"{filename}: the spectral axis ({ctype}) is not linear in "
            f"velocity; only linear velocity axes are supported")
    return astropy.units.Unit(wcs.wcs.cunit[axis]).to(_KM_S)


def _celestial_rotation(filename, linear, lng, lat):
    """
    The position angle (degrees, north through east) of the +y pixel axis,
    from the linear transformation of the celestial axes. Intermediate
    world x points east and y north.
    """
    matrix = linear[np.ix_([lng, lat], [lng, lat])]
    column_x, column_y = matrix.T
    if np.linalg.det(matrix) > 0:
        raise ConfigError(
            f"{filename}: the image is mirrored (east is to the right of "
            f"north), which the model does not support")
    cosine = column_x @ column_y / np.hypot(*column_x) / np.hypot(*column_y)
    if abs(cosine) > 1e-6:
        raise ConfigError(
            f"{filename}: the pixel grid is skewed (its axes are not "
            f"perpendicular on the sky), which the model does not support")
    return float(np.degrees(np.arctan2(column_y[0], column_y[1])))


def _check_uncoupled(filename, wcs, linear):
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
            f"{filename}: the world coordinates couple axes that the model "
            f"keeps independent (e.g. the velocity varies along a spatial "
            f"axis)")
