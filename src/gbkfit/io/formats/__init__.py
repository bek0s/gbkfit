"""
The formats of files of data. Each reads the arrays of its files, and
states what a file says of an array as a FITS header: its world
coordinates, unit and beam in the keywords of FITS, whatever the format
(see headers). gbkfit.io reads all formats alike through read_image,
read_header and _read_channel_beams; it writes FITS files only (see
fits.write_image).

A format is a module with the functions read_image, read_header and
read_channel_beams (see fits and asdf). Within gbkfit.io, another one is
a module of its own and an entry of _FORMATS.
"""

import importlib
from types import ModuleType

import astropy.io.fits
import astropy.table
import numpy as np

from gbkfit.config import DataFile, is_asdf_file, split_data_file
from . import fits
from .fits import write_image


__all__ = [
    'read_header',
    'read_image',
    'write_image'
]


# The formats other than FITS, each with how its files are told and the
# module that reads them (imported when a file of it is read: its
# libraries are slow to import); FITS is the format of any other file
_FORMATS = (
    (is_asdf_file, 'asdf'),
)


def read_image(file: DataFile) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Read the array of a file as the file stores it, and the FITS header
    of what the file states of it (see DataFile: by default, of a FITS
    file, its first HDU with an image).

    Raises
    ------
    ValueError
        If the file has no such array (e.g. no such HDU, or one that is
        not an image).
    """
    filename, key = split_data_file(file)
    return _format(filename).read_image(filename, key)


def read_header(file: DataFile) -> astropy.io.fits.Header:
    """
    Read the FITS header of what a file states of its array (see
    read_image), without the array where the format can.
    """
    filename, key = split_data_file(file)
    return _format(filename).read_header(filename, key)


def _read_channel_beams(
        filename: str, header: astropy.io.fits.Header
) -> astropy.table.QTable | None:
    """
    Return the beams of the channels of a cube of a file, of the given
    header, if the file has a table of them (e.g. the table BEAMS of a
    cube of CASA, see casa_beam_table).
    """
    return _format(filename).read_channel_beams(filename, header)


def _format(filename: str) -> ModuleType:
    """Return the module of the format of a file."""
    name = next(
        (name for is_format, name in _FORMATS if is_format(filename)), None)
    return fits if name is None else importlib.import_module(
        f'.{name}', __name__)
