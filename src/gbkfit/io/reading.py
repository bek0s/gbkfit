"""
Reading the arrays of files of any format (see formats) in the
orientation of the model: with the world coordinates of the model (see
read_data), or as they are, to look at them (see read_array); and their
world coordinates, unit and beam alone.
"""

from collections.abc import Sequence
from typing import Any

import astropy.io.fits
import astropy.wcs
import numpy as np

from gbkfit.grid import Coords, Rest
from .coords import coords_from_header
from .formats import read_channel_beams, read_header, read_image
from .headers import has_wcs, header_shape, header_unit, ignoring_wcs_fixes
from .orientation import model_header, oriented_header, squeezed_shape


__all__ = [
    'header_beam',
    'read_array',
    'read_beam',
    'read_coords',
    'read_data',
    'read_unit'
]


def read_data(
        filename: str,
        hdu: int | str | None = None,
        rpix: Sequence[float] | None = None,
        rval: Sequence[float] | None = None,
        rest: Rest = None,
        spectral_axis: int | None = None,
        medium: str | None = None,
        velocity: str | None = None,
        spectral_type: str | None = None
) -> tuple[np.ndarray, Coords]:
    """
    Read the data of a file and its world coordinates.

    The data are in the orientation of the model (see model_header),
    without their axes of length 1 (e.g. an axis of Stokes, or the
    spectral axis of a moment map), and their world coordinates in its
    units (see coords_from_header).

    Parameters
    ----------
    filename : str
        The name of the file.
    hdu : int or str, optional
        Where the data are in the file (see formats.read_image): of a FITS
        file, their HDU (e.g. 'SCI'); by default the first with data (see
        read_hdu).
    rpix, rval : optional
        World coordinates that replace those of the header (see
        coords_from_wcs), for the axes in the orientation of the model.
    rest : str or Quantity, optional
        The rest of the spectral axis (see model_header).
    spectral_axis : int, optional
        The index (FITS order, from 0) that the spectral axis of the data
        (if any) must have in the orientation of the model: the data have
        an axis of their own before it for each, of length 1 too (see
        model_header).
    medium : str, optional
        The medium of the wavelengths of the spectral axis and of the
        rest, if the header does not state it (see model_header and
        coords_from_header).
    velocity : str, optional
        The convention of the velocities of a spectral axis that the
        header leaves ambiguous (see model_header).
    spectral_type : str, optional
        The type of an axis whose header gives one that FITS does not
        define (see with_spectral_type).

    Returns
    -------
    np.ndarray
        The data.
    Coords
        Their world coordinates.

    Raises
    ------
    ValueError
        If the HDU has no data, or the header has invalid world
        coordinates or ones the model cannot represent (see model_header
        and coords_from_header).
    """
    data, header = read_image(filename, hdu)
    # (the axes before the spectral axis are the data's own, also those of
    # length 1: e.g. of one region)
    ndim = None if spectral_axis is None else spectral_axis + 1
    header, orientation = model_header(
        filename, header, data.shape, rest, velocity, spectral_type, medium,
        ndim)
    coords = coords_from_header(
        filename, header, rpix, rval, spectral_axis, medium, rest)
    return orientation.apply(data), coords


def read_array(
        filename: str, hdu: int | str | None = None
) -> tuple[np.ndarray, astropy.wcs.WCS | None]:
    """
    Read the array of a file in the orientation of the model (see
    model_header), without its axes of length 1, with its world
    coordinates in that orientation, to look at it: unlike read_data, its
    coordinates need not be ones the model can represent (e.g. of
    wavelengths, or of velocities whose convention the header does not
    state).

    Parameters
    ----------
    filename : str
        The name of the file.
    hdu : int or str, optional
        Where the array is in the file (see read_data); by default the
        first with data.

    Returns
    -------
    np.ndarray
        The array; as it is in the file (without its axes of length 1) if
        its header has no world coordinates that astropy reads.
    astropy.wcs.WCS or None
        Its world coordinates, if any.

    Raises
    ------
    ValueError
        If the HDU has no array.
    """
    data, header = read_image(filename, hdu)
    try:
        # (the fixes of astropy to the header are not of the orientation)
        with ignoring_wcs_fixes():
            header, orientation = oriented_header(filename, header, data.shape)
            wcs = astropy.wcs.WCS(header) if has_wcs(header) else None
    except ValueError:
        # (world coordinates that astropy cannot read, or that cannot be
        # put in the orientation of the model, leave the array as it is;
        # the errors of astropy's WCS are ValueErrors)
        return data.reshape(squeezed_shape(data.shape)), None
    return orientation.apply(data), wcs


def read_coords(
        filename: str,
        hdu: int | str | None = None,
        velocity: str | None = None,
        rest: Rest = None,
        spectral_type: str | None = None
) -> Coords:
    """
    Read the world coordinates of the data of a file, as read_data does,
    from its header alone (the data are not read).

    Parameters
    ----------
    filename : str
        The name of the file.
    hdu : int or str, optional
        Where the data are in the file (see read_data); by default the
        first with data.
    velocity : str, optional
        The convention of the velocities of an ambiguous spectral axis
        (see read_data).
    rest : str or Quantity, optional
        The rest of the spectral axis, which turns an axis of wavelengths
        or frequencies into velocities (see model_header).
    spectral_type : str, optional
        The type of an axis whose header gives one that FITS does not
        define (see with_spectral_type).

    Returns
    -------
    Coords
        The world coordinates.

    Raises
    ------
    ValueError
        As read_data, for the world coordinates of the header.
    """
    header = read_header(filename, hdu)
    header, _ = model_header(
        filename, header, header_shape(header), rest, velocity,
        spectral_type)
    return coords_from_header(filename, header, rest=rest)


def read_unit(filename: str, hdu: int | str | None = None) -> str | None:
    """
    Return the unit of the data of a file (where they are in it: see
    read_data; by default the first with data), as its header gives it
    (BUNIT), if it does.
    """
    return header_unit(read_header(filename, hdu))


def read_beam(
        filename: str, hdu: int | str | None = None
) -> dict[str, Any] | None:
    """
    Return the beam of the data of a file (where they are in it: see
    read_data; by default the first with data), if it has one (see
    header_beam).
    """
    return header_beam(read_header(filename, hdu), filename)


def header_beam(
        header: astropy.io.fits.Header, filename: str | None
) -> dict[str, Any] | None:
    """
    Return the beam of the data of a header, if it has one: the Gaussian
    restoring beam of a radio image, as the options bmaj, bmin (arcsec)
    and bpa (degrees): of the header (BMAJ, BMIN and BPA, in degrees), or
    else of each channel of a cube, from the table of the beams of the
    channels of its file (filename, if given; e.g. the table BEAMS of a
    cube of CASA, see read_channel_beams), as arrays with their spectral
    points ('spectral', a Quantity), in the order of the channels of the
    file.
    """
    if 'BMAJ' in header and 'BMIN' in header:
        return dict(
            bmaj=float(header['BMAJ']) * 3600,
            bmin=float(header['BMIN']) * 3600,
            bpa=float(header.get('BPA', 0.0)))
    table = None if filename is None else read_channel_beams(
        filename, header)
    if table is None:
        return None
    return dict(
        bmaj=table['bmaj'].to_value('arcsec'),
        bmin=table['bmin'].to_value('arcsec'),
        bpa=table['bpa'].to_value('deg'),
        spectral=table.columns[0])
