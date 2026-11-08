"""
Writing data as FITS files with world coordinates: data on the grid of
the model (see write_data), and spectra of regions, of the model (see
write_spectra) or of data read from files (see spectra_header). The
headers are made here, and the FITS format writes them (see
formats.fits.write_image).
"""

from collections.abc import Mapping
from typing import Any

import astropy.io.fits
import astropy.wcs
import numpy as np

import gbkfit.units
from gbkfit.grid import Coords
from .formats import write_image
from .headers import _with_shape, parse_wcs
from .spectral import _velocity_axis_keywords


__all__ = [
    'spectra_header',
    'write_data',
    'write_spectra'
]


# The axis of the regions of spectra of regions, along x: their index,
# from 0
_REGIONS_AXIS = dict(CTYPE1='LINEAR', CRPIX1=1.0, CDELT1=1.0, CRVAL1=0.0)


def write_data(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        spectral_axis: int | None = None,
        *,
        unit: gbkfit.units.AnyUnit | None = None,
        cards: Mapping[str, Any] | None = None,
        checksum: bool = False,
        overwrite: bool = False
) -> None:
    """
    Write data with the world coordinates of the model to a FITS file.

    The axes of the data are the x and y axes of the sky (RA and Dec, TAN
    projection, rotated with a PC matrix), alone (spectral_axis = None)
    or followed by the spectral axis (spectral_axis = 2, a velocity) or
    by a position along the line of sight (spectral_axis = None, in
    arcsec); or a position along a slit (in arcsec from its reference
    pixel) followed by the spectral axis (spectral_axis = 1); or the
    spectral axis alone (spectral_axis = 0).

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
    unit, cards, checksum, overwrite : optional
        As write_image.

    Raises
    ------
    ValueError
        For data of another layout.
    """
    write_image(
        filename, data, _header_from_coords(coords, data.ndim, spectral_axis),
        unit=unit, cards=cards, checksum=checksum, overwrite=overwrite)


def write_spectra(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        *,
        unit: gbkfit.units.AnyUnit | None = None,
        cards: Mapping[str, Any] | None = None,
        checksum: bool = False,
        overwrite: bool = False
) -> None:
    """
    Write spectra of regions to a FITS file.

    The regions are along x (their index, from 0), and the velocity along
    y (see SpectraData).

    Parameters
    ----------
    filename : str
        The name of the file.
    data : np.ndarray
        The spectra, of shape (nchannels, nregions).
    coords : Coords
        The world coordinates of their spectral axis (of one axis).
    unit, cards, checksum, overwrite : optional
        As write_image.
    """
    write_image(
        filename, data, _spectra_header_from_coords(coords), unit=unit,
        cards=cards, checksum=checksum, overwrite=overwrite)


def spectra_header(
        source: str, header: astropy.io.fits.Header, nregions: int
) -> astropy.io.fits.Header:
    """
    Return the header of the spectra of regions of the data of a header,
    as write_spectra writes them: the regions along x (their index, from
    0), and the spectral axis of the data along y, as the header states
    it (e.g. of logarithmic wavelengths, WAVE-LOG), with its rest, its
    frame and the medium of its wavelengths.

    Raise ValueError if the data have no spectral axis.
    """
    wcs = parse_wcs(source, header)
    if wcs.wcs.spec < 0:
        raise ValueError(f"{source}: the data have no spectral axis")
    spectral = wcs.sub([wcs.wcs.spec + 1])
    spectra = astropy.wcs.WCS(
        astropy.io.fits.Header({'WCSAXES': 2, **_REGIONS_AXIS}))
    spectra.wcs.ctype[1] = spectral.wcs.ctype[0]
    spectra.wcs.cunit[1] = spectral.wcs.cunit[0]
    spectra.wcs.crpix[1] = spectral.wcs.crpix[0]
    spectra.wcs.crval[1] = spectral.wcs.crval[0]
    spectra.wcs.cdelt[1] = spectral.pixel_scale_matrix[0, 0]
    spectra.wcs.restfrq, spectra.wcs.restwav = (
        spectral.wcs.restfrq, spectral.wcs.restwav)
    spectra.wcs.specsys = spectral.wcs.specsys
    result = spectra.to_header()
    if 'AIRORVAC' in header:
        result['AIRORVAC'] = header['AIRORVAC']
    nchannels = int(header[f'NAXIS{wcs.wcs.spec + 1}'])
    return _with_shape(result, (nchannels, nregions))


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
    Return the header of spectra of regions (see write_spectra): the
    regions along x (see _REGIONS_AXIS), and the world coordinates of
    their spectral axis (of one axis) along y, with the frame of their
    velocities (SPECSYS).
    """
    return _with_frame(
        astropy.io.fits.Header(
            {**_REGIONS_AXIS, **_velocity_axis_keywords(coords, 2, 0)}),
        coords)


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
