"""
Reading and writing data with world coordinates in FITS files.

Data are read in the orientation of the model, whatever that of their
file (see to_model_axes): the celestial axes first, longitude (e.g. RA)
then latitude, and the spectral axis last; east to the left of north;
and velocities increasing along the spectral axis. Their world
coordinates become those of the model (see coords_from_wcs).
"""

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
    'coords_from_wcs',
    'flip_data',
    'flip_wcs',
    'model_axis_flips',
    'model_axis_order',
    'read_data',
    'reorder_data',
    'reorder_wcs',
    'to_model_axes',
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
    Read the data of a FITS file and its world coordinates.

    The data are in the orientation of the model (see to_model_axes), and
    their world coordinates in its units (see coords_from_wcs).

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
        or coordinates the model cannot represent (see coords_from_wcs).
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
    data, wcs = to_model_axes(data, wcs)
    spec = wcs.wcs.spec
    if spectral_axis is not None and spec >= 0 and spec != spectral_axis:
        raise ConfigError(
            f"{filename}: the spectral axis must be the axis "
            f"{spectral_axis + 1}; it is the axis {spec + 1}")
    coords = coords_from_wcs(filename, wcs, rpix, rval, rest)
    return np.ascontiguousarray(data), coords


def coords_from_wcs(
        source: str,
        wcs: astropy.wcs.WCS,
        rpix: typing.Sequence[float] | None = None,
        rval: typing.Sequence[float] | None = None,
        rest: typing.Any = None
) -> Coords:
    """
    Return the world coordinates of the model (see Coords) of a WCS.

    rpix is the reference pixel (CRPIX - 1), and rval the world position
    there (CRVAL). Either can be given instead, and the other is computed
    from the WCS. The rest of the spectral axis is that of the WCS
    (RESTWAV or RESTFRQ), or rest, if given.

    Parameters
    ----------
    source : str
        The name of the source of the WCS (e.g. a file), for messages.
    wcs : astropy.wcs.WCS
        World coordinates, with the axes in the orientation of the model
        (see to_model_axes).
    rpix, rval : Sequence[float], optional
        The reference pixel or value, in the units of the model, a value
        for each axis or one for all.
    rest : str or Quantity, optional
        The rest of the spectral axis (see gridutils.make_rest).

    Returns
    -------
    Coords
        The world coordinates.

    Raises
    ------
    ConfigError
        For coordinates the model cannot represent: a skewed pixel grid,
        celestial axes coupled to other axes, and spectral axes that are
        not velocities or are not linear in velocity.
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
    if rpix is not None:
        rpix = np.broadcast_to(np.asarray(rpix, float), naxis)
    if rval is not None:
        rval = np.broadcast_to(np.asarray(rval, float), naxis)
    if rpix is None and rval is None:
        rpix = (wcs.wcs.crpix - 1).tolist()
        rval = np.multiply(wcs.wcs.crval, scale).tolist()
    elif rpix is None:
        world = np.divide(rval, scale)
        rpix = np.ravel(wcs.world_to_pixel_values(*world)).tolist()
    elif rval is None:
        world = np.ravel(wcs.pixel_to_world_values(*rpix))
        rval = np.multiply(world, scale).tolist()
    return Coords(
        tuple(float(x) for x in step),
        tuple(float(x) for x in rpix),
        tuple(float(x) for x in rval),
        float(rota),
        _spectral_rest(source, wcs, make_rest(rest)))


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
        overwrite: bool = False
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
    """
    header = astropy.io.fits.Header(_velocity_header(coords, 2, 0))
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
        CRPIX2=coords.rpix[1] + 1, CRVAL2=coords.rval[1],
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
    n - 1): optical velocities with a rest wavelength, radio velocities
    with a rest frequency or without a rest.
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
    velocity axes become km/s, and the other axes keep their units.
    """
    if axis != wcs.wcs.spec:
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
