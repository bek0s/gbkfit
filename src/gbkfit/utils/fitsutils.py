import typing

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Coords',
    'Grid',
    'GridData',
    'VELOCITY_TYPES',
    'centre_missing_crpix',
    'read_data',
    'write_data'
]


# The spectral axis types that are velocities (FITS WCS Paper III)
VELOCITY_TYPES = ('VRAD', 'VOPT', 'VELO')

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

    def axes(self, *indices: int) -> 'Coords':
        """The coordinates of the given axes."""
        def pick(values):
            return tuple(values[i] for i in indices)
        return Coords(
            pick(self.step), pick(self.rpix), pick(self.rval), self.rota)


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
        return Grid(self.size[:2], self.coords.axes(0, 1), None)


class GridData(typing.NamedTuple):
    """
    Data on a grid with world coordinates (see Coords and write_data).
    spectral_axis is the index of its spectral axis (FITS order), or None.
    """
    data: np.ndarray
    coords: Coords
    spectral_axis: int | None


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
        float(rota))
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
    if ctype[:4] not in VELOCITY_TYPES:
        raise ConfigError(
            f"{filename}: the spectral axis is of type '{ctype}'; only "
            f"velocity axes ({', '.join(VELOCITY_TYPES)}) are supported")
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
