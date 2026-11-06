"""
The FITS format: reading the arrays of FITS files, their headers and the
tables of the beams of their channels (of cubes of CASA), and writing
data with the world coordinates of the model as FITS files.
"""

from collections.abc import Mapping
from typing import Any

import astropy.io.fits
import astropy.table
import astropy.units as u
import astropy.wcs
import numpy as np

from gbkfit.grid import Coords
from ..headers import parse_wcs
from ..spectral import velocity_axis_keywords


__all__ = [
    'casa_beam_table',
    'read_hdu',
    'write_data',
    'write_spectra'
]


def read_image(
        filename: str, key: int | str | None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the image and the header of an HDU of a FITS file (key: its
    index or name, or None for the first with an image; see read_hdu).
    """
    return read_hdu(filename, key)


def read_header(
        filename: str, key: int | str | None
) -> astropy.io.fits.Header:
    """
    Return the header of an HDU of a FITS file (key: its index or name, or
    None for the first with an image, as read_hdu finds it), without its
    data.
    """
    if key is not None:
        return astropy.io.fits.getheader(filename, key)
    with astropy.io.fits.open(filename) as hdus:
        return _first_image(filename, hdus).header.copy()


def read_channel_beams(
        filename: str, header: astropy.io.fits.Header
) -> astropy.table.QTable | None:
    """
    Return the beams of the channels of a cube of a FITS file of the given
    header, if the file has a table of them (BEAMS, of CASA; see
    casa_beam_table).
    """
    # (the file opened once: of a compressed file, each opening reads it
    # through to the table, at its end)
    with astropy.io.fits.open(filename) as hdus:
        if 'BEAMS' not in hdus:
            return None
        beams = astropy.table.Table.read(hdus['BEAMS'])
    return _channel_beams(filename, beams, header)


def read_hdu(
        filename: str, hdu: int | str | None = None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the image and the header of an HDU of a FITS file: the one
    given, or else the first with an image (e.g. the extension after an
    empty primary HDU), whatever tables come before it.

    Raises
    ------
    ValueError
        If the file has no such HDU, or it is not an image (e.g. a table),
        or it has no data.
    """
    with astropy.io.fits.open(filename) as hdus:
        if hdu is None:
            found = _first_image(filename, hdus)
        else:
            try:
                found = hdus[hdu]
            except (IndexError, KeyError) as e:
                raise ValueError(
                    f"{filename}: it has no HDU {hdu!r}") from e
            if not found.is_image:
                raise ValueError(
                    f"{filename}: HDU {hdu!r} is not an image (e.g. a "
                    f"table); choose the HDU of the image (e.g. hdu: SCI)")
            if found.data is None:
                raise ValueError(
                    f"{filename}: HDU {hdu!r} has no data; choose the HDU "
                    f"with the data (e.g. hdu: SCI)")
        return np.array(found.data), found.header.copy()


def _first_image(
        filename: str, hdus: astropy.io.fits.HDUList
) -> astropy.io.fits.PrimaryHDU | astropy.io.fits.ImageHDU:
    """
    Return the first HDU of a FITS file with an image, whatever tables
    come before it, or raise ValueError.
    """
    # (by the header, so that the data of the HDUs passed by are not read)
    found = next((
        hdu for hdu in hdus
        if hdu.is_image and hdu.header.get('NAXIS', 0) > 0), None)
    if found is None:
        raise ValueError(f"{filename}: no HDU has an image")
    return found


def casa_beam_table(
        filename: str, header: astropy.io.fits.Header | None = None
) -> astropy.table.QTable:
    """
    Return the beams of the channels of a cube of CASA (its table BEAMS)
    as a table of the options of a Gaussian beam PSF (bmaj, bmin, bpa)
    against the points of its spectral axis (frequency, wavelength or
    velocity), in the order of its channels: of the world coordinates of
    the header of the cube, if given, or else of its first HDU. Only the
    first polarisation is used.

    Raises
    ------
    ValueError
        If the cube has no spectral axis.
    """
    if header is None:
        header = astropy.io.fits.getheader(filename, 0)
    return _channel_beams(
        filename, astropy.table.Table.read(filename, hdu='BEAMS'), header)


def _channel_beams(
        filename: str,
        beams: astropy.table.Table,
        header: astropy.io.fits.Header
) -> astropy.table.QTable:
    """
    Return the table BEAMS of a cube of CASA of the given header as the
    table of casa_beam_table.
    """
    if 'POL' in beams.colnames:
        beams = beams[beams['POL'] == 0]
    beams.sort('CHAN')
    wcs = parse_wcs(filename, header)
    if wcs.wcs.spec < 0:
        raise ValueError(f"{filename}: the cube has no spectral axis")
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


def write_data(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        spectral_axis: int | None = None,
        overwrite: bool = False,
        unit: str | None = None,
        cards: Mapping[str, Any] | None = None
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
        header.update(velocity_axis_keywords(coords, 1))
    elif spectral_axis is None and data.ndim == 2:
        header.update(_sky_keywords(coords))
    elif spectral_axis == 2 and data.ndim == 3:
        header.update(_sky_keywords(coords))
        header.update(velocity_axis_keywords(coords, 3))
    elif spectral_axis is None and data.ndim == 3:
        header.update(_sky_keywords(coords))
        header.update(_offset_keywords(
            3, coords.step[2], coords.rpix[2], coords.rval[2]))
    elif spectral_axis == 1 and data.ndim == 2:
        # The position along the slit is an offset from its reference
        header.update(_offset_keywords(
            1, coords.step[0], coords.rpix[0], 0.0))
        header.update(velocity_axis_keywords(coords, 2))
    else:
        raise ValueError(
            f"unsupported data: {data.ndim} axes, of which the spectral "
            f"axis is {spectral_axis}")
    _write(filename, data, header, coords, overwrite, unit, cards)


def write_spectra(
        filename: str,
        data: np.ndarray,
        coords: Coords,
        overwrite: bool = False,
        unit: str | None = None,
        cards: Mapping[str, Any] | None = None
) -> None:
    """
    Write spectra of regions to a FITS file.

    The regions are along x (an index, without world coordinates), and
    the velocity along y (see SpectraData).

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
    header = astropy.io.fits.Header(velocity_axis_keywords(coords, 2, 0))
    _write(filename, data, header, coords, overwrite, unit, cards)


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


def _write(
        filename: str,
        data: np.ndarray,
        header: astropy.io.fits.Header,
        coords: Coords,
        overwrite: bool,
        unit: str | None,
        cards: Mapping[str, Any] | None
) -> None:
    """
    Write data to a FITS file with a header of their axes, and the frame
    of their velocities (SPECSYS), their unit (BUNIT) and further cards
    (see write_data).
    """
    if coords.frame is not None:
        header['SPECSYS'] = coords.frame
    if unit is not None:
        header['BUNIT'] = unit
    header.update(cards or {})
    astropy.io.fits.writeto(
        filename, data, header,
        output_verify='exception', overwrite=overwrite, checksum=True)
