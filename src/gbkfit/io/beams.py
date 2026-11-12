"""
The beams of data: the Gaussian restoring beams of radio images, from
their headers, the tables of the beams of the channels of their files,
or their history (see header_beam); those of the channels of a cube in
their order in the orientation of the model (see _oriented_beam); and
the table of the beams of the channels of a cube of CASA (see
casa_beam_table).
"""

import re
from typing import NamedTuple

import astropy.io.fits
import astropy.table
import astropy.units
import astropy.wcs
import numpy as np

from gbkfit.config import DataFile, split_data_file
from .formats import _read_channel_beams, read_header
from .orientation import Orientation


__all__ = [
    'Beam',
    'beam_area',
    'casa_beam_table',
    'header_beam'
]


# The beam that AIPS writes in the history of an image it cleaned, in
# degrees (e.g. 'AIPS   CLEAN BMAJ=  3.1753E-03 BMIN=  2.6007E-03 BPA=
# -80.43', of THINGS)
_AIPS_BEAM = re.compile(
    r'AIPS\s+CLEAN\s+BMAJ=\s*(\S+)\s+BMIN=\s*(\S+)\s+BPA=\s*(\S+)')


class Beam(NamedTuple):
    """
    The beam of data in units per beam (e.g. Jy/beam): the Gaussian
    restoring beam of a radio image, of the full widths at half maximum
    bmaj and bmin (arcsec) and the position angle bpa (degrees). A cube
    can have one for each channel (e.g. of CASA): its values are then
    arrays, against the spectral points of the channels (spectral: a
    Quantity of frequencies, wavelengths or velocities).
    """
    bmaj: float | np.ndarray
    bmin: float | np.ndarray
    bpa: float | np.ndarray = 0.0
    spectral: astropy.units.Quantity | None = None

    def area(self) -> float | np.ndarray:
        """Return its solid angle (arcsec^2; see beam_area)."""
        return beam_area(self.bmaj, self.bmin)


def beam_area(
        bmaj: float | np.ndarray, bmin: float | np.ndarray
) -> float | np.ndarray:
    """
    Return the solid angle of a Gaussian beam of the full widths at half
    maximum bmaj and bmin: pi / (4 ln 2) bmaj bmin, in the square of
    their unit.
    """
    return np.pi / (4 * np.log(2)) * np.multiply(bmaj, bmin)


def header_beam(
        header: astropy.io.fits.Header, filename: str | None
) -> Beam | None:
    """
    Return the beam of the data of a header, if it has one (see Beam). It
    is the first of:

    - that of the header (BMAJ, BMIN and BPA, in degrees);
    - that of each channel of a cube, from the table of the beams of the
      channels of its file (filename, if given; e.g. the table BEAMS of
      a cube of CASA, see casa_beam_table): arrays in the order of
      the channels of the file, with their spectral points;
    - the last that AIPS wrote in the history of the header (see
      _AIPS_BEAM).
    """
    if 'BMAJ' in header and 'BMIN' in header:
        return Beam(
            float(header['BMAJ']) * 3600, float(header['BMIN']) * 3600,
            float(header.get('BPA', 0.0)))
    table = None if filename is None else _read_channel_beams(
        filename, header)
    if table is None:
        return _history_beam(header)
    return Beam(
        table['bmaj'].to_value('arcsec'), table['bmin'].to_value('arcsec'),
        table['bpa'].to_value('deg'), table.columns[0])


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
        If the file has no such cube (see read_image), the cube has no
        spectral axis, or the file has no table of the beams of its
        channels.
    """
    filename = split_data_file(file)[0]
    table = _read_channel_beams(filename, read_header(file))
    if table is None:
        raise ValueError(
            f"{filename}: the file has no table of the beams of the "
            f"channels of its cube (BEAMS, of CASA)")
    return table


def _oriented_beam(
        filename: str,
        header: astropy.io.fits.Header,
        model: astropy.io.fits.Header,
        orientation: Orientation
) -> Beam | None:
    """
    Return the beam of the values of a file of the given header (see
    header_beam): of a cube of CASA, the beams of its channels, in their
    order in the orientation of the model (of the header model).

    Raises
    ------
    ValueError
        If the beams of a cube are not one for each channel.
    """
    beam = header_beam(header, filename)
    if beam is None or beam.spectral is None:
        return beam
    # A cube with a table BEAMS has a spectral axis (see casa_beam_table).
    # The axes that the orientation reverses are those of the model, where
    # the spectral axis can be at another index than in the file (e.g. of
    # RA, Dec, Stokes and frequency, once the Stokes axis of 1 is dropped).
    axis = astropy.wcs.WCS(model).wcs.spec
    if axis < 0:
        # (a spectral axis of one channel, left out: the beam of it)
        if len(beam.bmaj) != 1:
            raise ValueError(
                f"{filename}: the cube has 1 channel, but its table BEAMS "
                f"has beams for {len(beam.bmaj)}")
        return Beam(
            float(beam.bmaj[0]), float(beam.bmin[0]), float(beam.bpa[0]))
    nchannels = int(model[f'NAXIS{axis + 1}'])
    if len(beam.bmaj) != nchannels:
        raise ValueError(
            f"{filename}: the cube has {nchannels} channels, but its table "
            f"BEAMS has beams for {len(beam.bmaj)}")
    if axis not in orientation.flips:
        return beam
    return Beam(*(value[::-1] for value in beam))


def _history_beam(header: astropy.io.fits.Header) -> Beam | None:
    """
    Return the beam of the last cleaning of AIPS in the history of a
    header (see _AIPS_BEAM), as header_beam does, if it has one.
    """
    for card in reversed(header.get('HISTORY', [])):
        if match := _AIPS_BEAM.search(str(card)):
            bmaj, bmin, bpa = map(float, match.groups())
            return Beam(bmaj * 3600, bmin * 3600, bpa)
    return None
