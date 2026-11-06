"""
The world coordinates of the model (see Coords) of headers and WCSs in
its orientation (see model_header): steps in arcsec and km/s, the
reference pixel and its world position in ICRS, the rotation of the
pixel grid on the sky, and the rest, medium, convention and frame of the
spectral axis.
"""

from collections.abc import Sequence

import astropy.coordinates
import astropy.io.fits
import astropy.units as u
import astropy.wcs
import astropy.wcs.utils
import numpy as np

from gbkfit.config import warn
from gbkfit.grid import Coords, FRAMES, Rest, make_medium, make_rest
from .headers import (
    centre_missing_crpix, header_shape, linear_transform, parse_wcs)
from .orientation import model_axis_flips, model_axis_order
from .spectral import (
    VELOCITY_TYPES, header_medium, is_linear_type, log_velocity_axis,
    medium_contradiction, rest_in_medium, spectral_convention, spectral_rest)


__all__ = [
    'coords_from_header',
    'coords_from_wcs'
]


# How small a coupling of axes is taken as none, relative to the largest
# step of its world axis: rounding left by the tools that compute the
# matrices of headers (e.g. 1e-17). Over 1e5 pixels it is still 1e-5 of
# a pixel.
_COUPLING_TOLERANCE = 1e-10


def coords_from_header(
        source: str,
        header: astropy.io.fits.Header,
        rpix: Sequence[float] | None = None,
        rval: Sequence[float] | None = None,
        spectral_axis: int | None = None,
        medium: str | None = None,
        rest: Rest = None
) -> Coords:
    """
    Return the world coordinates of the model (see Coords) of a header in
    its orientation (see model_header).

    A spectral axis of logarithmic wavelengths (WAVE-LOG) is that of their
    log velocities, of the convention 'log' (see log_velocity_axis). The
    wavelengths of a spectral axis of optical velocities are in vacuum
    (FITS), or in the medium of the keyword AIRORVAC ('air' or 'vac') of
    the header, or in the given medium.

    Parameters
    ----------
    source : str
        The name of the file of the header, for messages.
    header : astropy.io.fits.Header
        The header.
    rpix, rval : optional
        World coordinates that replace those of the header (see
        coords_from_wcs).
    spectral_axis : int, optional
        The index (FITS order, from 0) that the spectral axis (if any)
        must have.
    medium : str, optional
        The medium of the wavelengths of the spectral axis, if its header
        does not state it (by the type of the axis, e.g. AWAV, or by
        AIRORVAC; see make_medium).
    rest : str or Quantity, optional
        The rest of the spectral axis (see coords_from_wcs): that of the
        header, after model_header gave it, or of a spectral axis that the
        header does not have (e.g. of data without world coordinates).

    Returns
    -------
    Coords
        The world coordinates.

    Raises
    ------
    ValueError
        If the world coordinates are invalid or ones the model cannot
        represent (see coords_from_wcs), the spectral axis is not at
        spectral_axis, or the medium is not valid or not the one the
        header states.
    """
    # A header without world coordinates has its reference pixels at the
    # centres (see centre_missing_crpix)
    wcs = parse_wcs(
        source, centre_missing_crpix(header, header_shape(header)))
    wcs, log = log_velocity_axis(source, wcs)
    spec = wcs.wcs.spec
    if spectral_axis is not None and spec >= 0 and spec != spectral_axis:
        raise ValueError(
            f"{source}: the spectral axis must be the axis "
            f"{spectral_axis + 1}; it is the axis {spec + 1}")
    coords = coords_from_wcs(source, wcs, rpix, rval, rest, medium)
    if coords.medium is not None:
        # The medium the header states: that of the type of an axis of
        # logarithmic wavelengths (FITS), or AIRORVAC (which model_header
        # writes from the type of an axis of wavelengths)
        stated = log or header_medium(source, header)
        if medium is None:
            coords = coords._replace(medium=stated or coords.medium)
            if rest is None:
                # (the rest of the header, RESTWAV, is in vacuum)
                coords = coords._replace(
                    rest=rest_in_medium(coords.rest, coords.medium))
        elif stated is not None and coords.medium != stated:
            raise medium_contradiction(source, coords.medium, stated)
    if log:
        coords = coords._replace(convention='log')
    return coords


def coords_from_wcs(
        source: str,
        wcs: astropy.wcs.WCS,
        rpix: Sequence[float] | None = None,
        rval: Sequence[float] | None = None,
        rest: Rest = None,
        medium: str | None = None
) -> Coords:
    """
    Return the world coordinates of the model (see Coords) of a WCS.

    rpix is the reference pixel (CRPIX - 1), and rval the world position
    there (CRVAL). Either can be given instead, for each axis, and the
    other is computed from the WCS (the axes are uncoupled, but the
    celestial ones, which take the same one: both or neither are given).
    The rest of the spectral axis is that of the WCS (RESTWAV or RESTFRQ),
    or rest, if given, and the convention of its velocities that of its
    type (see spectral_convention).

    The celestial positions are of ICRS: the RA and Dec of the reference
    pixel, given or of the WCS (of another frame, e.g. Galactic or FK4,
    turned into ICRS). The rotation of the pixel grid (rota) is the
    position angle of its +y axis at the reference pixel, from north of
    ICRS there, whatever the projection and the frame of the WCS (on the
    sky, it changes away from the reference pixel, by a hair only within
    fields of a degree or so).

    Parameters
    ----------
    source : str
        The name of the source of the WCS (e.g. a file), for messages.
    wcs : astropy.wcs.WCS
        World coordinates, with the axes in the orientation of the model
        (see model_header).
    rpix, rval : Sequence[float or None], optional
        The reference pixel or value, in the units of the model (of the
        celestial axes, the RA and Dec of ICRS, in degrees), a value for
        each axis (None: not given) or one for all.
    rest : str or Quantity, optional
        The rest of the spectral axis (see make_rest).
    medium : str, optional
        The medium of the wavelengths of a rest wavelength (see
        make_medium).

    Returns
    -------
    Coords
        The world coordinates.

    Raises
    ------
    ValueError
        For coordinates the model cannot represent: a skewed pixel grid,
        celestial axes coupled to other axes, and spectral axes that are
        not velocities or are not linear in velocity; if the medium is not
        valid; if the reference of only one of the celestial axes is given;
        or if the axes are not in the orientation of the model.
    """
    naxis = wcs.naxis
    if model_axis_order(wcs) != tuple(range(naxis)) or model_axis_flips(wcs):
        raise ValueError(
            f"{source}: the axes of the world coordinates are not in the "
            f"orientation of the model (see model_header)")
    lng, lat = wcs.wcs.lng, wcs.wcs.lat
    linear = linear_transform(wcs)
    # The factor from the world units of each axis to model units
    scale = [_axis_scale(source, wcs, axis) for axis in range(naxis)]
    if lng >= 0:
        _check_unskewed(source, linear)
    _check_uncoupled(source, wcs, linear)
    step = [
        np.hypot(*linear[:, axis][[lng, lat]]) * 3600
        if axis in (lng, lat)
        else linear[axis, axis] * scale[axis]
        for axis in range(naxis)]
    # NaN where not given: an axis takes the other one of the WCS, at the
    # values of the header on the other axes, which do not change it
    rpix, rval = (
        np.full(naxis, np.nan) if value is None
        else np.broadcast_to(np.asarray(value, float), naxis)
        if np.ndim(value) == 0
        else np.array([np.nan if x is None else x for x in value], float)
        for value in (rpix, rval))
    given_rval = ~np.isnan(rval)
    native = rval.copy()
    if lng >= 0:
        _check_celestial_pairs(source, rpix, rval, lng, lat)
        if given_rval[lng]:
            # (a given position is of ICRS; the WCS is of its own frame)
            native[lng], native[lat] = _from_icrs(
                source, wcs, rval[lng], rval[lat])
    header_rpix = wcs.wcs.crpix - 1
    header_rval = np.multiply(wcs.wcs.crval, scale)
    pixel = np.where(np.isnan(rpix), header_rpix, rpix)
    world = np.where(
        np.isnan(native), wcs.wcs.crval, np.divide(native, scale))
    rval_at_rpix = np.ravel(wcs.pixel_to_world_values(*pixel)) * scale
    rpix_at_rval = np.ravel(wcs.world_to_pixel_values(*world))
    rpix, rval = (
        np.where(~np.isnan(rpix), rpix, np.where(
            np.isnan(rval), header_rpix, rpix_at_rval)).tolist(),
        np.where(~np.isnan(rval), rval, np.where(
            np.isnan(rpix), header_rval, rval_at_rpix)).tolist())
    rota = 0.0
    if lng >= 0:
        ra, dec, rota = _sky_reference(source, wcs, rpix)
        if not given_rval[lng]:
            rval[lng], rval[lat] = ra, dec
    given = make_rest(rest)
    rest = spectral_rest(source, wcs, given)
    medium = make_medium(medium, rest)
    if given is None:
        # (the rest of the WCS, RESTWAV, is in vacuum)
        rest = rest_in_medium(rest, medium)
    return Coords(
        tuple(float(x) for x in step),
        tuple(float(x) for x in rpix),
        tuple(float(x) for x in rval),
        float(rota),
        rest, medium, spectral_convention(wcs),
        _frame(source, wcs))


def _frame(source: str, wcs: astropy.wcs.WCS) -> str | None:
    """
    Return the frame of rest of the velocities of a WCS (SPECSYS, see
    FRAMES), if it gives one of FITS; a warning says when it
    gives another (see warn), which is left out.
    """
    frame = wcs.wcs.specsys.strip().upper()
    if frame and frame not in FRAMES:
        warn(
            f"{source}: SPECSYS is {frame!r}, not a frame of FITS "
            f"({', '.join(FRAMES)}); it is left out")
    return frame if frame in FRAMES else None


def _axis_scale(source: str, wcs: astropy.wcs.WCS, axis: int) -> float:
    """
    Return the factor from the world units of an axis to model units:
    celestial axes stay in degrees (only their steps become arcsec),
    velocity axes become km/s, other axes in units of angle (e.g. the
    position along a slit, OFFSET) become arcsec, and the other axes keep
    their units.
    """
    if axis != wcs.wcs.spec:
        unit = u.Unit(wcs.wcs.cunit[axis])
        celestial = axis in (wcs.wcs.lng, wcs.wcs.lat)
        if not celestial and unit.physical_type == 'angle':
            return unit.to(u.arcsec)
        return 1.0
    ctype = wcs.wcs.ctype[axis]
    if ctype[:4] not in VELOCITY_TYPES:
        raise ValueError(
            f"{source}: the spectral axis is of type '{ctype}'; only "
            f"velocity axes ({', '.join(VELOCITY_TYPES)}) are supported: "
            f"give the rest frequency or wavelength of its line (spectral: "
            f"{{rest: ...}}), which turns it into velocities")
    if not is_linear_type(ctype):
        raise ValueError(
            f"{source}: the spectral axis ({ctype}) is not linear in "
            f"velocity; only linear velocity axes are supported")
    return u.Unit(wcs.wcs.cunit[axis]).to(u.km / u.s)


def _check_unskewed(source: str, linear: np.ndarray) -> None:
    """
    Raise ValueError if the pixel grid is skewed on the sky: its axes, of
    the linear transformation of the celestial axes (the first two), are
    not perpendicular.
    """
    column_x, column_y = linear[:2, :2].T
    cosine = column_x @ column_y / np.hypot(*column_x) / np.hypot(*column_y)
    if abs(cosine) > 1e-6:
        raise ValueError(
            f"{source}: the pixel grid is skewed (its axes are not "
            f"perpendicular on the sky), which the model does not support")


def _check_celestial_pairs(
        source: str,
        rpix: np.ndarray,
        rval: np.ndarray,
        lng: int,
        lat: int
) -> None:
    """
    Raise ValueError if the reference pixel, or its value, is given for
    one of the celestial axes alone (NaN: not given): on the sky, either
    axis of a pixel depends on both.
    """
    for name, values in (('rpix', rpix), ('rval', rval)):
        if np.isnan(values[lng]) != np.isnan(values[lat]):
            raise ValueError(
                f"{source}: the reference of the celestial axes ({name}) "
                f"must be given for both of them or for neither; it is for "
                f"one")


def _celestial_frame(
        source: str, wcs: astropy.wcs.WCS
) -> astropy.coordinates.BaseCoordinateFrame:
    """Return the celestial frame of a WCS, or raise ValueError."""
    try:
        return astropy.wcs.utils.wcs_to_celestial_frame(wcs)
    except ValueError as e:
        raise ValueError(
            f"{source}: the celestial frame of the world coordinates is "
            f"not known: {e}") from e


def _from_icrs(
        source: str, wcs: astropy.wcs.WCS, ra: float, dec: float
) -> tuple[float, float]:
    """
    Return the celestial position (degrees) in the frame of a WCS of a
    position of ICRS.
    """
    frame = _celestial_frame(source, wcs)
    if isinstance(frame, astropy.coordinates.ICRS):
        return ra, dec
    sky = astropy.coordinates.SkyCoord(
        ra, dec, unit='deg', frame='icrs').transform_to(frame)
    return float(sky.spherical.lon.deg), float(sky.spherical.lat.deg)


def _sky_reference(
        source: str, wcs: astropy.wcs.WCS, rpix: list[float]
) -> tuple[float, float, float]:
    """
    Return the RA and Dec (degrees, of ICRS) of the reference pixel of a
    WCS (rpix, from 0, of each axis), and the position angle (degrees,
    north of ICRS through east) of the +y pixel axis there: the mean of
    the directions to the pixels half a pixel along it on each side,
    which leaves out the first order of the curve of the axis on the sky.
    """
    frame = _celestial_frame(source, wcs)
    lng, lat = wcs.wcs.lng, wcs.wcs.lat
    half = np.zeros(wcs.naxis)
    half[lat] = 0.5

    def sky(pixel: np.ndarray) -> astropy.coordinates.SkyCoord:
        world = np.ravel(wcs.pixel_to_world_values(*pixel))
        return astropy.coordinates.SkyCoord(
            world[lng], world[lat], unit='deg', frame=frame).icrs
    centre = sky(np.asarray(rpix))
    ahead = centre.position_angle(sky(np.add(rpix, half))).deg
    behind = centre.position_angle(sky(np.subtract(rpix, half))).deg - 180
    rota = ahead + ((behind - ahead + 180) % 360 - 180) / 2
    return (
        float(centre.ra.deg), float(centre.dec.deg),
        float((rota + 180) % 360 - 180))


def _check_uncoupled(
        source: str,
        wcs: astropy.wcs.WCS,
        linear: np.ndarray
) -> None:
    """
    Raise ValueError if a pixel axis other than the celestial pair is
    rotated into another world axis (e.g. velocity varying along x),
    beyond rounding (see _COUPLING_TOLERANCE).
    """
    celestial = {wcs.wcs.lng, wcs.wcs.lat} - {-1}
    coupled = linear.copy()
    for i in celestial:
        for j in celestial:
            coupled[i, j] = 0
    np.fill_diagonal(coupled, 0)
    scale = np.abs(linear).max(axis=1, keepdims=True)
    if np.any(np.abs(coupled) > _COUPLING_TOLERANCE * scale):
        raise ValueError(
            f"{source}: the world coordinates couple axes that the model "
            f"keeps independent (e.g. the velocity varies along a spatial "
            f"axis)")
