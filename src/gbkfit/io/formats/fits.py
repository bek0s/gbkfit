"""
The FITS format: reading the arrays of FITS files, their headers and the
tables of the beams of their channels (of cubes of CASA); and writing
arrays as FITS files, with the world coordinates of the model or a
header of their own (gbkfit writes FITS files only).
"""

from collections.abc import Mapping
from typing import Any

import astropy.io.fits
import astropy.table
import astropy.units as u
import astropy.wcs
import numpy as np

import gbkfit.units
from gbkfit.config import DataFile, split_data_file
from gbkfit.grid import Coords
from ..coords import _header_from_coords, _spectra_header_from_coords
from ..headers import parse_wcs


__all__ = [
    'casa_beam_table',
    'write_data',
    'write_image',
    'write_spectra'
]


def read_image(
        filename: str, key: int | str | None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the image of an HDU of a FITS file and its header (see
    _image_hdu).
    """
    with astropy.io.fits.open(filename) as hdus:
        found = _image_hdu(filename, hdus, key)
        return np.array(found.data), found.header.copy()


def read_header(
        filename: str, key: int | str | None
) -> astropy.io.fits.Header:
    """
    Return the header of the HDU of an image of a FITS file (see
    _image_hdu), without its data.
    """
    with astropy.io.fits.open(filename) as hdus:
        return _image_hdu(filename, hdus, key).header.copy()


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


def _image_hdu(
        filename: str, hdus: astropy.io.fits.HDUList, key: int | str | None
) -> astropy.io.fits.PrimaryHDU | astropy.io.fits.ImageHDU:
    """
    Return the HDU of an image of a FITS file: the one given (key: its
    index or name), or else the first with an image (e.g. the extension
    after an empty primary HDU), whatever tables come before it.

    Raises
    ------
    ValueError
        If the file has no such HDU, or it is not an image (e.g. a table),
        or it has no data.
    """
    # (by their headers, so that the data of the HDUs are not read)
    if key is None:
        found = next((
            hdu for hdu in hdus
            if hdu.is_image and hdu.header.get('NAXIS', 0) > 0), None)
        if found is None:
            raise ValueError(f"{filename}: no HDU has an image")
        return found
    try:
        found = hdus[key]
    except (IndexError, KeyError) as e:
        raise ValueError(f"{filename}: it has no HDU {key!r}") from e
    if not found.is_image:
        raise ValueError(
            f"{filename}: HDU {key!r} is not an image (e.g. a table); "
            f"choose the HDU of the image (e.g. hdu: SCI)")
    if found.header.get('NAXIS', 0) == 0:
        raise ValueError(
            f"{filename}: HDU {key!r} has no data; choose the HDU with the "
            f"data (e.g. hdu: SCI)")
    return found


def casa_beam_table(file: DataFile) -> astropy.table.QTable:
    """
    Return the beams of the channels of a cube of CASA (the table BEAMS
    of its file) as a table of the options of a Gaussian beam PSF (bmaj,
    bmin, bpa) against the points of its spectral axis (frequency,
    wavelength or velocity), in the order of its channels. Only the first
    polarisation is used.

    Raises
    ------
    ValueError
        If the file has no such cube (see read_image), or the cube has no
        spectral axis.
    """
    filename, key = split_data_file(file)
    return _channel_beams(
        filename, astropy.table.Table.read(filename, hdu='BEAMS'),
        read_header(filename, key))


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
        values = (
            beams[name].quantity if beams[name].unit
            else beams[name] * u.Unit(to))
        return values.to(to)
    return astropy.table.QTable({
        axis: np.asarray(points) * unit, 'bmaj': column('BMAJ', 'arcsec'),
        'bmin': column('BMIN', 'arcsec'), 'bpa': column('BPA', 'deg')})


def write_image(
        filename: str,
        data: np.ndarray,
        header: astropy.io.fits.Header | None = None,
        *,
        unit: gbkfit.units.AnyUnit | None = None,
        cards: Mapping[str, Any] | None = None,
        checksum: bool = False,
        overwrite: bool = False
) -> None:
    """
    Write an array to a FITS file as it is, with a header (e.g. of its
    world coordinates), its unit and further cards.

    Parameters
    ----------
    filename : str
        The name of the file.
    data : np.ndarray
        The array.
    header : astropy.io.fits.Header, optional
        Its header; its BUNIT is replaced by the unit, if given.
    unit : AnyUnit, optional
        The unit of its values, written as BUNIT (see
        gbkfit.units.format_unit).
    cards : Mapping, optional
        Further header cards, by keyword (e.g. provenance).
    checksum : bool, optional
        Whether to add the checksums of FITS (CHECKSUM and DATASUM; the
        file then has the time it was written, and differs each time).
    overwrite : bool, optional
        Whether to overwrite an existing file.
    """
    header = astropy.io.fits.Header() if header is None else header.copy()
    if unit is not None:
        header['BUNIT'] = gbkfit.units.format_unit(unit)
    header.update(cards or {})
    astropy.io.fits.writeto(
        filename, data, header,
        output_verify='exception', overwrite=overwrite, checksum=checksum)


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
    unit, cards, checksum, overwrite : optional
        As write_image.
    """
    write_image(
        filename, data, _spectra_header_from_coords(coords), unit=unit,
        cards=cards, checksum=checksum, overwrite=overwrite)
