"""
The FITS header of a linear grid that stands for world coordinates of
any kind (of the APE 14 interface of astropy, e.g. the gwcs of an ASDF
file), when they are those of a grid the model represents (see
linear_header).
"""

import astropy.coordinates
import astropy.io.fits
import astropy.units as u
import astropy.wcs
import astropy.wcs.utils
import numpy as np
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS

import gbkfit.units
from ..spectral import _SPECTRAL_UNITS


__all__ = [
    'linear_header'
]


# How far (in pixels) world coordinates of any kind may be from those of
# the grid of the header that stands for them
_LINEAR_TOLERANCE = 1e-3


def linear_header(
        source: str,
        wcs: BaseLowLevelWCS | BaseHighLevelWCS,
        shape: tuple[int, ...]
) -> astropy.io.fits.Header:
    """
    Return the FITS header of world coordinates of any kind (of the APE
    14 interface of astropy, e.g. the gwcs of an ASDF file) of an array,
    if they are those of a grid that the model represents: celestial axes
    of a grid on the plane tangent to the sky, and a spectral axis linear
    in wavelength, frequency or velocity, or in the logarithm of the
    wavelength (WAVE-LOG).

    The header is that of the grid at the centre of the array, and must
    give the pixels of the world coordinates within _LINEAR_TOLERANCE at
    its corners, the centres of its edges and its centre.

    Parameters
    ----------
    source : str
        The name of the file of the world coordinates, for messages.
    wcs : BaseLowLevelWCS or BaseHighLevelWCS
        The world coordinates.
    shape : tuple of int
        The shape of the array (numpy order).

    Returns
    -------
    astropy.io.fits.Header
        The header.

    Raises
    ------
    ValueError
        If the world coordinates are not of the axes of the array, have
        axes that are not celestial or spectral, or couple the spectral
        axis to others, or are not those of such a grid (e.g. distortions,
        or a spectral axis linear in none of those): the array must then
        be resampled first.
    """
    low = getattr(wcs, 'low_level_wcs', wcs)
    n = len(shape)
    if low.pixel_n_dim != n or low.world_n_dim != n:
        raise ValueError(
            f"{source}: the world coordinates have {low.pixel_n_dim} pixel "
            f"and {low.world_n_dim} world axes; the array has {n}")
    sizes = tuple(shape[::-1])
    centre = np.array([(size - 1) / 2 for size in sizes])
    components = low.world_axis_object_components
    classes = low.world_axis_object_classes
    correlation = np.asarray(low.axis_correlation_matrix)
    linear = astropy.wcs.WCS(naxis=n)
    ctype, cunit = [''] * n, [''] * n
    crval, cd = np.zeros(n), np.zeros((n, n))
    # The world axis of each pixel axis, and the factor from its unit to
    # that of the header
    axes, factors = [-1] * n, [1.0] * n
    sky = [
        j for j, (key, *_) in enumerate(components)
        if classes[key][0] is astropy.coordinates.SkyCoord]
    for j in range(n):
        if j in sky:
            continue
        unit = u.Unit(low.world_axis_units[j])
        pixels = np.flatnonzero(correlation[j])
        if not any(unit.is_equivalent(x) for x in _SPECTRAL_UNITS):
            raise ValueError(
                f"{source}: the world axis "
                f"{low.world_axis_physical_types[j]!r} is neither celestial "
                f"nor spectral")
        if len(pixels) != 1:
            raise ValueError(
                f"{source}: the spectral axis varies along more than one "
                f"pixel axis")
        k = int(pixels[0])
        axes[k] = j
        ctype[k], cunit[k], crval[k], cd[k, k] = _linear_spectral_axis(
            source, low, j, k, centre, sizes[k])
    if sky:
        pixels = np.flatnonzero(correlation[sky].any(axis=0))
        if len(sky) != 2 or len(pixels) != 2:
            raise ValueError(
                f"{source}: the celestial axes must be two, of two pixel "
                f"axes")
        lon, lat = sorted(sky, key=lambda j: components[j][1])
        frame = astropy.wcs.utils.celestial_frame_to_wcs(
            classes[components[lon][0]][2]['frame'], projection='TAN')
        a, b = (int(k) for k in pixels)
        axes[a], axes[b] = lon, lat
        world = _world_values(low, centre)
        ctype[a], ctype[b] = frame.wcs.ctype
        cunit[a] = cunit[b] = 'deg'
        crval[a], crval[b] = world[lon], world[lat]
        linear.wcs.radesys, linear.wcs.equinox = (
            frame.wcs.radesys, frame.wcs.equinox)
        # The step along each of their pixel axes, on the plane tangent to
        # the sky at the centre (a WCS whose pixels are degrees of it)
        tangent = frame.deepcopy()
        tangent.wcs.crval = crval[a], crval[b]
        tangent.wcs.crpix = 1, 1
        tangent.wcs.cdelt = 1, 1
        tangent.wcs.set()
        for k in (a, b):
            half = np.zeros(n)
            half[k] = 0.5
            ends = [_world_values(low, centre + sign * half)
                    for sign in (1, -1)]
            plus, minus = (
                tangent.wcs_world2pix([[end[lon], end[lat]]], 0)[0]
                for end in ends)
            cd[[a, b], k] = plus - minus
    linear.wcs.ctype = ctype
    linear.wcs.cunit = cunit
    linear.wcs.crval = crval
    linear.wcs.crpix = centre + 1
    linear.wcs.cd = cd
    linear.wcs.set()
    for k in range(n):
        factors[k] = u.Unit(low.world_axis_units[axes[k]]).to(
            linear.world_axis_units[k])
    # The pixels of the world coordinates at the corners, the centres of
    # the edges and the centre, by the header
    points = np.array(np.meshgrid(*[
        [0, (size - 1) / 2, size - 1] for size in sizes])).reshape(n, -1)
    world = _world_values(low, points)
    pixels = linear.world_to_pixel_values(*[
        world[axes[k]] * factors[k] for k in range(n)])
    pixels = np.array(pixels).reshape(points.shape)
    if not np.allclose(pixels, points, rtol=0, atol=_LINEAR_TOLERANCE):
        raise ValueError(
            f"{source}: the world coordinates are not those of a grid the "
            f"model represents (e.g. they have distortions, or a spectral "
            f"axis linear in none of wavelength, frequency, velocity or log "
            f"wavelength): resample the data first")
    return linear.to_header()


def _world_values(
        low: BaseLowLevelWCS, pixels: np.ndarray
) -> list[np.ndarray]:
    """Return the values of each world axis of APE 14 coordinates."""
    values = low.pixel_to_world_values(*pixels)
    if low.world_n_dim == 1:
        values = [values]
    return [np.asarray(value, dtype=float) for value in values]


def _linear_spectral_axis(
        source: str,
        low: BaseLowLevelWCS,
        axis: int,
        pixel: int,
        centre: np.ndarray,
        size: int
) -> tuple[str, str, float, float]:
    """
    Return the type, unit, value at the centre and step (CTYPE, CUNIT,
    CRVAL and CDELT) of the spectral axis of APE 14 coordinates (world
    axis axis, along the pixel axis pixel), of the kind it is the most
    linear in (see linear_header; e.g. a short axis of log wavelengths is
    nearly linear in wavelength too); its step at the centre for an axis
    of one pixel.
    """
    unit = u.Unit(low.world_axis_units[axis])
    if size > 1:
        points = np.repeat(centre[:, None], size, axis=1)
        points[pixel] = np.arange(size)
        values = _world_values(low, points)[axis]
    else:
        half = np.zeros_like(centre)
        half[pixel] = 0.5
        values = np.array([
            _world_values(low, centre + sign * half)[axis][()]
            for sign in (-1, 1)])
    if unit.is_equivalent(u.m):
        kinds = [('WAVE', values), ('WAVE-LOG', np.log(values))]
    elif unit.is_equivalent(u.Hz):
        kinds = [('FREQ', values)]
    else:
        physical = low.world_axis_physical_types[axis] or ''
        kind = (
            'VRAD' if physical.endswith('.radio')
            else 'VOPT' if physical.endswith('.opt') else 'VELO')
        kinds = [(kind, values)]
    x = np.arange(len(values)) - (len(values) - 1) / 2
    lines = []
    for kind, y in kinds:
        step, middle = np.polyfit(x, y, 1)
        # The largest distance from the line, in pixels
        distance = np.max(np.abs(y - (middle + step * x))) / abs(step)
        lines.append((distance, kind, step, middle))
    distance, kind, step, middle = min(lines)
    if not distance <= _LINEAR_TOLERANCE:
        raise ValueError(
            f"{source}: the spectral axis is linear in none of "
            f"{', '.join(kind for kind, _ in kinds)}: resample the data "
            f"first")
    if kind == 'WAVE-LOG':
        # w = S exp(x / S), with S at the reference pixel
        middle = np.exp(middle)
        step *= middle
    return kind, gbkfit.units.format_unit(unit), float(middle), float(step)
