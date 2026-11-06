"""
The formats of files of data. Each reads the arrays of its files, and
states what a file says of an array as a FITS header: its world
coordinates, unit and beam in the keywords of FITS, whatever the format
(see headers). The rest of gbkfit.io reads all formats alike through
read_image, read_header and read_channel_beams.

A format is a module with these three functions (see fits and asdf).
Within gbkfit.io, another one is a module of its own and an entry of
_FORMATS.
"""

import importlib
from types import ModuleType

import astropy.io.fits
import astropy.table
import numpy as np

from gbkfit.config import DataFile, is_asdf_file
from . import fits
from .fits import casa_beam_table, read_hdu, write_data, write_spectra


__all__ = [
    'casa_beam_table',
    'read_hdu',
    'write_data',
    'write_spectra'
]


# The formats other than FITS, each with how its files are told and the
# module that reads them (imported when a file of it is read: its
# libraries are slow to import); FITS is the format of any other file
_FORMATS = (
    (is_asdf_file, 'asdf'),
)


def split_file(file: DataFile) -> tuple[str, int | str | None]:
    """
    Return the name of a file and the key of its array in it (e.g. the
    HDU of a FITS file, or the path of the array in the tree of an ASDF
    file), None if not given.
    """
    return (file, None) if isinstance(file, str) else (file[0], file[1])


def read_image(
        filename: str, key: int | str | None = None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return an array of a file and the FITS header of what the file states
    of it (key: where the array is in the file; None for the first, if the
    format has a first).
    """
    return _format(filename).read_image(filename, key)


def read_header(
        filename: str, key: int | str | None = None
) -> astropy.io.fits.Header:
    """
    Return the FITS header of what a file states of an array of it,
    without the array if the format can (see read_image).
    """
    return _format(filename).read_header(filename, key)


def read_channel_beams(
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
