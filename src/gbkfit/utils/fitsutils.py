import typing

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Coords',
    'read_data',
    'write_data'
]


# The spectral axis types that are velocities (FITS WCS Paper III)
_VELOCITY_TYPES = ('VRAD', 'VOPT', 'VELO')

_KM_S = astropy.units.km / astropy.units.s


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

    Axes without a known type have the values of their header.
    """
    step: tuple[float, ...]
    rpix: tuple[float, ...]
    rval: tuple[float, ...]
    rota: float


def read_data(
        filename: str,
        hdu: int | str = 0,
        rpix: typing.Sequence[float] | None = None,
        rval: typing.Sequence[float] | None = None
) -> tuple[np.ndarray, Coords]:
    """
    The data of a FITS file (from the given HDU) and its world
    coordinates in the units of the model (see Coords).

    rpix is CRPIX - 1 (the centre of the axes without CRPIX), and rval is
    CRVAL, the world position at rpix. Either can be given instead (in
    model units), and the other is computed from the header.

    Raise ConfigError for coordinates the model cannot represent: a
    header it cannot read, a mirrored (east to the right of north) or
    skewed pixel grid, celestial axes coupled to other axes, and
    spectral axes that are not velocities.
    """
    with astropy.io.fits.open(filename) as hdulist:
        data = hdulist[hdu].data
        header = hdulist[hdu].header
    try:
        wcs = astropy.wcs.WCS(header)
    except Exception as e:
        raise ConfigError(
            f"{filename}: invalid world coordinates: {e}") from e
    if wcs.naxis != data.ndim:
        raise ConfigError(
            f"{filename}: the header has world coordinates for {wcs.naxis} "
            f"axes, but the data has {data.ndim}")
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
    if rpix is not None:
        rpix = np.broadcast_to(np.asarray(rpix, float), wcs.naxis)
    if rval is not None:
        rval = np.broadcast_to(np.asarray(rval, float), wcs.naxis)
    # The axes without a reference pixel have it at their centre
    for n, size in enumerate(data.shape[::-1], start=1):
        if f'CRPIX{n}' not in header:
            wcs.wcs.crpix[n - 1] = size / 2 + 0.5
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
        float(rota))
    return data, coords


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
    spectral axis (spectral_axis = 2, a radio velocity), or a position
    along a slit followed by the spectral axis (spectral_axis = 1), or
    the spectral axis alone (spectral_axis = 0).
    """
    header = astropy.io.fits.Header()
    if spectral_axis == 0 and data.ndim == 1:
        header.update(_velocity_header(coords, 1))
    elif spectral_axis is None and data.ndim == 2:
        header.update(_sky_header(coords))
    elif spectral_axis == 2 and data.ndim == 3:
        header.update(_sky_header(coords))
        header.update(_velocity_header(coords, 3))
    elif spectral_axis == 1 and data.ndim == 2:
        # The position along the slit is an offset from its reference
        header.update(
            CTYPE1='OFFSET', CUNIT1='arcsec', CDELT1=coords.step[0],
            CRPIX1=coords.rpix[0] + 1, CRVAL1=0.0)
        header.update(_velocity_header(coords, 2))
    else:
        raise ValueError(
            f"unsupported data: {data.ndim} axes, of which the spectral "
            f"axis is {spectral_axis}")
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


def _velocity_header(coords, n):
    """The header keywords of the spectral axis, FITS axis n."""
    return {
        f'CTYPE{n}': 'VRAD', f'CUNIT{n}': 'km/s',
        f'CDELT{n}': coords.step[n - 1],
        f'CRPIX{n}': coords.rpix[n - 1] + 1,
        f'CRVAL{n}': coords.rval[n - 1]}


def _axis_scale(filename, wcs, axis):
    """
    The factor from the world units of an axis to model units: celestial
    axes stay in degrees (only their steps become arcsec), velocity axes
    become km/s, and the other axes keep their units.
    """
    if axis != wcs.wcs.spec:
        return 1.0
    ctype = wcs.wcs.ctype[axis]
    if ctype[:4] not in _VELOCITY_TYPES:
        raise ConfigError(
            f"{filename}: the spectral axis is of type '{ctype}'; only "
            f"velocity axes ({', '.join(_VELOCITY_TYPES)}) are supported")
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
