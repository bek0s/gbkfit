"""
Descriptions of FITS files of data from their headers alone (see
describe_fits): what each HDU holds, and what a dataset of it refuses
or needs given, so that the user can say what each holds before making
a dataset of it.
"""

import dataclasses
import warnings

import astropy.io.fits
import astropy.units
import astropy.wcs

from gbkfit.utils import fitsutils, gridutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from .data import Beam


__all__ = [
    'HDUDescription',
    'describe_fits'
]


@dataclasses.dataclass(frozen=True)
class HDUDescription:
    """
    What an HDU of a FITS file holds (see describe_fits).

    Attributes
    ----------
    index : int
        Its index in the file.
    name : str
        Its name (EXTNAME; 'PRIMARY' for the first).
    shape : tuple of int
        The number of pixels along each axis, in the orientation of the
        model (x, y, then the spectral axis; see fitsutils.to_model_axes),
        or in that of the file if its world coordinates are invalid.
    axes : tuple of str
        The type of each axis (CTYPE, e.g. 'RA---SIN' or 'FREQ'; '' if
        none), in the order of shape.
    unit : str or None
        The unit of its values, as the header writes it (BUNIT).
    beam : Beam or None
        Its beam (see fitsutils.read_beam).
    coords : Coords or None
        The world coordinates of its data in the model (see
        fitsutils.read_coords); None if they cannot be read.
    problems : tuple of str
        What a dataset of it refuses or needs given (e.g. the convention
        of velocities that the header does not state, or the beam of a
        unit per beam).
    """
    index: int
    name: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    unit: str | None
    beam: Beam | None
    coords: gridutils.Coords | None
    problems: tuple[str, ...]


def describe_fits(filename: str) -> list[HDUDescription]:
    """
    Describe the HDUs of a FITS file that hold an array (e.g. data, their
    errors and their mask), from their headers alone: their arrays are
    not read.

    Parameters
    ----------
    filename : str
        The name of the file.

    Returns
    -------
    list of HDUDescription
        The description of each HDU with an array, in the order of the
        file.

    Raises
    ------
    OSError
        If the file cannot be read as a FITS file.
    """
    with astropy.io.fits.open(filename) as hdus:
        headers = [
            (index, hdu.name, hdu.header.copy())
            for index, hdu in enumerate(hdus)
            if hdu.is_image and hdu.header.get('NAXIS', 0) > 0]
    return [_describe(filename, *header) for header in headers]


def _describe(
        filename: str,
        index: int,
        name: str,
        header: astropy.io.fits.Header
) -> HDUDescription:
    """Return the description of an HDU of a FITS file with an array."""
    problems = []
    order = _model_order(header)
    sizes = fitsutils.header_shape(header)[::-1]
    shape = tuple(sizes[axis] for axis in order)
    axes = tuple(
        str(header.get(f'CTYPE{axis + 1}', '')).strip() for axis in order)
    if flat := [axes[i] or f'axis {i + 1}'
                for i, size in enumerate(shape) if size == 1]:
        problems.append(
            f"axes of length 1 ({', '.join(flat)}): remove them with "
            f"gbkfit-cli prep")
    coords = None
    try:
        coords = fitsutils.read_coords(filename, index)
    except ConfigError as e:
        problems.append(str(e))
    unit = fitsutils.read_unit(filename, index)
    beam = None
    try:
        options = fitsutils.read_beam(filename, index)
        beam = None if options is None else Beam(**options)
    except ConfigError as e:
        problems.append(str(e))
    if unit is not None:
        try:
            per_beam = unitutils.spatial_unit(
                unitutils.parse_unit(unit)) == astropy.units.beam
        except ConfigError as e:
            problems.append(f"{e}: give the unit (option unit of the data)")
        else:
            if per_beam and beam is None:
                problems.append(
                    f"the unit, {unit}, is per beam, but the header has no "
                    f"beam (BMAJ and BMIN): give it (option beam of the "
                    f"data)")
    return HDUDescription(
        index=index, name=name, shape=shape, axes=axes, unit=unit,
        beam=beam, coords=coords, problems=tuple(problems))


def _model_order(header: astropy.io.fits.Header) -> tuple[int, ...]:
    """
    Return the axes of the header in the order of the model (see
    fitsutils.model_axis_order), or in their own if its world coordinates
    are invalid.
    """
    shape = fitsutils.header_shape(header)
    try:
        # (the fixes of astropy to the header are for the coords)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', astropy.wcs.FITSFixedWarning)
            wcs = astropy.wcs.WCS(
                fitsutils.centre_missing_crpix(header, shape))
    except Exception:
        # (any error of astropy: the problem is reported with the coords)
        return tuple(range(len(shape)))
    return fitsutils.model_axis_order(wcs)
