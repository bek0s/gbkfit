"""
The world coordinates of the model (see Coords) of headers and WCSs in
its orientation (see model_header): steps in arcsec and km/s, the
reference pixel and its world position in ICRS, the rotation of the
pixel grid on the sky, and the rest, medium, convention and frame of the
spectral axis; and the headers of world coordinates of the model, of
data written to files (see write_data).
"""

from collections.abc import Sequence

import astropy.coordinates
import astropy.io.fits
import astropy.units as u
import astropy.wcs
import astropy.wcs.utils
import numpy as np

from gbkfit.grid import Coords, Rest
from .headers import (
    _linear_transform, centre_missing_crpix, header_shape, parse_wcs)
from .orientation import _model_axis_flips, model_axis_order
from .spectral import (
    _log_velocity_axis, _spectral_terms, _velocity_axis_keywords,
    _velocity_scale)


__all__ = [
    'coords_from_header'
]


# How small a coupling of axes is taken as none, relative to the largest
# step of its world axis: rounding left by the tools that compute the
# matrices of headers (e.g. 1e-17). Over 1e5 pixels it is still 1e-5 of
# a pixel.
_COUPLING_TOLERANCE = 1e-10

# How skewed the pixel grid may be on the sky, as the cosine of the angle
# of its axes (0 if they are perpendicular): rounding left by the tools
# that compute the matrices of headers
_SKEW_TOLERANCE = 1e-6


def coords_from_header(
        source: str,
        header: astropy.io.fits.Header,
        *,
        rpix: Sequence[float] | None = None,
        rval: Sequence[float] | None = None,
        rest: Rest = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> Coords:
    """
    Return the world coordinates of the model (see Coords) of a header in
    its orientation (see model_header).

    The reference pixel (rpix, CRPIX - 1) or its world position there
    (rval, CRVAL) can be given instead of those of the header, for each
    axis, and the other is computed from the header (the axes are
    uncoupled, but the celestial ones, which take the same one: both or
    neither are given). The celestial positions are of ICRS: the RA and
    Dec of the reference pixel, given or of the header (of another frame,
    e.g. Galactic or FK4, turned into ICRS). The rotation of the pixel
    grid (rota) is the position angle of its +y axis at the reference
    pixel, from north of ICRS there, whatever the projection and the
    frame of the header (on the sky, it changes away from the reference
    pixel, by a hair only within fields of a degree or so).

    The rest of the spectral axis is that of the header (RESTWAV or
    RESTFRQ), or rest, if given, and the convention of its velocities
    that of its type (see _spectral_convention). A spectral axis of
    logarithmic wavelengths (WAVE-LOG) is that of their log velocities,
    of the convention 'log' (see _log_velocity_axis). The wavelengths of
    a spectral axis of optical velocities are in vacuum (FITS), or in the
    medium of the keyword AIRORVAC ('air' or 'vac') of the header, or in
    the given medium.

    Parameters
    ----------
    source : str
        The name of the file of the header, for messages.
    header : astropy.io.fits.Header
        The header.
    rpix, rval : Sequence[float or None], optional
        The reference pixel or its world position, in the units of the
        model (of the celestial axes, the RA and Dec of ICRS, in
        degrees): a value for each axis (None: not given) or one for all.
    rest : str or Quantity, optional
        The rest of the spectral axis (see make_rest): that of the
        header, after model_header gave it, or of a spectral axis that the
        header does not have (e.g. of data without world coordinates).
    medium : str, optional
        The medium of the wavelengths of the spectral axis, if its header
        does not state it (by the type of the axis, e.g. AWAV, or by
        AIRORVAC; see make_medium).
    spectral_axis : int, optional
        The index (FITS order, from 0) that the spectral axis (if any)
        must have.

    Returns
    -------
    Coords
        The world coordinates.

    Raises
    ------
    ValueError
        If the world coordinates are invalid, or ones the model cannot
        represent: a skewed pixel grid, celestial axes coupled to other
        axes, or a spectral axis that is not of velocities or is not
        linear in velocity; if the axes are not in the orientation of the
        model; if the reference of only one of the celestial axes is
        given; if the spectral axis is not at spectral_axis; or if the
        medium is not valid or not the one the header states.
    """
    # A header without world coordinates has its reference pixels at the
    # centres (see centre_missing_crpix)
    wcs = parse_wcs(
        source, centre_missing_crpix(header, header_shape(header)))
    wcs, log = _log_velocity_axis(source, wcs)
    spec = wcs.wcs.spec
    if spectral_axis is not None and spec >= 0 and spec != spectral_axis:
        raise ValueError(
            f"{source}: the spectral axis must be the axis "
            f"{spectral_axis + 1}; it is the axis {spec + 1}")
    step, rpix, rval, rota = _grid_of_wcs(source, wcs, rpix=rpix, rval=rval)
    return Coords(
        step, rpix, rval, rota,
        *_spectral_terms(source, header, wcs, rest, medium, log))


def _grid_of_wcs(
        source: str,
        wcs: astropy.wcs.WCS,
        *,
        rpix: Sequence[float] | None = None,
        rval: Sequence[float] | None = None
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    """
    Return the steps, the reference pixel, its world position and the
    rotation of the grid (see Coords) of a WCS in the orientation of the
    model, with the references given (see coords_from_header).
    """
    naxis = wcs.naxis
    if model_axis_order(wcs) != tuple(range(naxis)) or _model_axis_flips(wcs):
        raise ValueError(
            f"{source}: the axes of the world coordinates are not in the "
            f"orientation of the model (see model_header)")
    lng, lat = wcs.wcs.lng, wcs.wcs.lat
    linear = _linear_transform(wcs)
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
    rpix, rval = _per_axis(rpix, naxis), _per_axis(rval, naxis)
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
    return (
        tuple(float(x) for x in step), tuple(float(x) for x in rpix),
        tuple(float(x) for x in rval), float(rota))


def _per_axis(
        value: Sequence[float | None] | float | None, naxis: int
) -> np.ndarray:
    """
    Return a value given for each of naxis axes, or one for all (see
    coords_from_header), as floats: NaN where it is not given.
    """
    if value is None:
        return np.full(naxis, np.nan)
    if np.ndim(value) == 0:
        return np.full(naxis, float(value))
    return np.array([np.nan if x is None else x for x in value], float)


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
    return _velocity_scale(source, wcs)


def _check_unskewed(source: str, linear: np.ndarray) -> None:
    """
    Raise ValueError if the pixel grid is skewed on the sky: its axes, of
    the linear transformation of the celestial axes (the first two), are
    not perpendicular.
    """
    column_x, column_y = linear[:2, :2].T
    cosine = column_x @ column_y / np.hypot(*column_x) / np.hypot(*column_y)
    if abs(cosine) > _SKEW_TOLERANCE:
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


def _header_from_coords(
        coords: Coords, ndim: int, spectral_axis: int | None
) -> astropy.io.fits.Header:
    """
    Return the header of the world coordinates of the model (see Coords)
    of data of ndim axes, of which the spectral axis is spectral_axis (see
    write_data), with the frame of their velocities (SPECSYS).

    Raises
    ------
    ValueError
        For data of another layout.
    """
    header = astropy.io.fits.Header()
    if spectral_axis == 0 and ndim == 1:
        header.update(_velocity_axis_keywords(coords, 1))
    elif spectral_axis is None and ndim == 2:
        header.update(_sky_keywords(coords))
    elif spectral_axis == 2 and ndim == 3:
        header.update(_sky_keywords(coords))
        header.update(_velocity_axis_keywords(coords, 3))
    elif spectral_axis is None and ndim == 3:
        header.update(_sky_keywords(coords))
        header.update(_offset_keywords(
            3, coords.step[2], coords.rpix[2], coords.rval[2]))
    elif spectral_axis == 1 and ndim == 2:
        # The position along the slit is an offset from its reference
        header.update(_offset_keywords(
            1, coords.step[0], coords.rpix[0], 0.0))
        header.update(_velocity_axis_keywords(coords, 2))
    else:
        raise ValueError(
            f"unsupported data: {ndim} axes, of which the spectral "
            f"axis is {spectral_axis}")
    return _with_frame(header, coords)


def _spectra_header_from_coords(coords: Coords) -> astropy.io.fits.Header:
    """
    Return the header of spectra of regions (see write_spectra): of the
    world coordinates of their spectral axis (of one axis), along y, with
    the frame of their velocities (SPECSYS).
    """
    return _with_frame(
        astropy.io.fits.Header(_velocity_axis_keywords(coords, 2, 0)), coords)


def _with_frame(
        header: astropy.io.fits.Header, coords: Coords
) -> astropy.io.fits.Header:
    """
    Return a header with the frame of the velocities of world coordinates
    (SPECSYS), if they have one.
    """
    if coords.frame is not None:
        header['SPECSYS'] = coords.frame
    return header


def _sky_keywords(coords: Coords) -> dict[str, float | str]:
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


def _offset_keywords(
        n: int,
        step: float,
        rpix: float,
        rval: float
) -> dict[str, float | str]:
    """Return the header keywords of an offset in arcsec, FITS axis n."""
    return {
        f'CTYPE{n}': 'OFFSET', f'CUNIT{n}': 'arcsec', f'CDELT{n}': step,
        f'CRPIX{n}': rpix + 1, f'CRVAL{n}': rval}
