"""
Descriptions of FITS files of data from their headers alone (see
describe_fits): what each HDU holds, and what a dataset of it refuses
or needs given, so that the user can say what each holds before making
a dataset of it.
"""

import dataclasses
import warnings
from typing import Any

import astropy.io.fits
import astropy.units
import astropy.wcs

from gbkfit.grid import Coords
from gbkfit.io import (
    centre_missing_crpix, coords_from_header, header_beam, header_shape,
    header_unit, model_axis_order, model_header, nonstandard_spectral_axis,
    velocity_is_ambiguous)
from gbkfit.utils import unitutils
from .data import Beam


__all__ = [
    'HDUDescription',
    'describe_fits'
]


# The keywords of headers that state the noise of the values (see
# HDUDescription.rms), in order
_RMS_KEYS = ('RMS', 'NOISE')

# What an HDU of errors holds, as its header states it in the classes of
# HDUs of ESO's science data products (e.g. MUSE, and MaNGA): HDUCLAS2 =
# 'ERROR', and HDUCLAS3 one of these (see HDUDescription.holds)
_ERROR_CLASSES = dict(
    RMSE='sigma', MSE='variance', INVMSE='inverse_variance')


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
        model (x, y, then the spectral axis; see to_model_axes),
        or in that of the file if its world coordinates are invalid; with
        its axes of length 1, which reading leaves out.
    axes : tuple of str
        The type of each axis (CTYPE, e.g. 'RA---SIN' or 'FREQ'; '' if
        none), in the order of shape.
    unit : str or None
        The unit of its values, as the header writes it (BUNIT).
    beam : Beam or None
        Its beam (see read_beam).
    coords : Coords or None
        The world coordinates of its data in the model (see
        read_coords); None if they cannot be read, or if its
        velocities are ambiguous and their convention is not given.
    rms : float or None
        The noise of its values that the header states (keyword RMS or
        NOISE, in the unit of the values; e.g. that of the cubes of
        LVHIS), if any.
    ambiguous_velocity : bool
        Whether the header leaves it open whether the velocities of its
        spectral axis are radio or optical (see
        velocity_is_ambiguous): a dataset of it gives their
        convention, and so does a description of it (see describe_fits);
        until then, this is not one of its problems.
    problems : tuple of str
        What a dataset of it refuses or needs given (e.g. the convention
        of velocities that the header does not state, or the beam of a
        unit per beam).
    holds : str or None
        What its values are, as errors, if its header states it (HDUCLAS2
        = 'ERROR', and HDUCLAS3 'RMSE', 'MSE' or 'INVMSE'): 'sigma'
        (standard deviations), 'variance' or 'inverse_variance' (the
        error_kind of a dataset); None if it does not.
    nonstandard_type : str or None
        The type its header gives an axis that FITS does not define (e.g.
        'WAVELENGTH', as CALIFA writes the spectral axis of its cubes; see
        nonstandard_spectral_axis), if it does: a dataset of it gives what the axis
        is (its spectral type), and so does a description of it; until
        then, its coordinates cannot be read (one of its problems).
    """
    index: int
    name: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    unit: str | None
    beam: Beam | None
    coords: Coords | None
    rms: float | None
    ambiguous_velocity: bool
    problems: tuple[str, ...]
    holds: str | None = None
    nonstandard_type: str | None = None


def describe_fits(
        filename: str, velocity: str | None = None, rest: Any = None,
        spectral_type: str | None = None
) -> list[HDUDescription]:
    """
    Describe the HDUs of a FITS file that hold an array (e.g. data, their
    errors and their mask), from their headers alone: their arrays are
    not read.

    Parameters
    ----------
    filename : str
        The name of the file.
    velocity : str, optional
        The convention of the velocities of spectral axes that the headers
        leave ambiguous (see read_data): 'radio' or 'optical'.
    rest : str or Quantity, optional
        The rest of their spectral axes (e.g. '6562.8 Angstrom'), which
        turns axes of wavelengths or frequencies into velocities, as a
        dataset with that rest reads them (see read_coords).
    spectral_type : str, optional
        What an axis whose header gives it a type FITS does not define
        is (see HDUDescription.nonstandard_type, and with_spectral_type).

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
    # (the file is read once: its headers are described as they are read,
    # each of its HDUs then in memory; a compressed file is read from its
    # start each time it is opened)
    with astropy.io.fits.open(filename) as hdus:
        headers = [
            (index, hdu.name, hdu.header.copy())
            for index, hdu in enumerate(hdus)
            if hdu.is_image and hdu.header.get('NAXIS', 0) > 0]
        beams = 'BEAMS' in hdus
    return [
        _describe(filename, *header, velocity, rest, spectral_type, beams)
        for header in headers]


def _describe(
        filename: str,
        index: int,
        name: str,
        header: astropy.io.fits.Header,
        velocity: str | None,
        rest: Any,
        spectral_type: str | None,
        beams: bool
) -> HDUDescription:
    """
    Return the description of an HDU of a FITS file with an array, from
    its header (and the table of the beams of the file, if it has one:
    beams), as read_coords, read_unit and read_beam read them.
    """
    problems = []
    order = _model_order(header)
    # (world axes beyond those of the array are of length 1, see
    # model_header)
    sizes = header_shape(header)[::-1]
    sizes += (1,) * (len(order) - len(sizes))
    shape = tuple(sizes[axis] for axis in order)
    axes = tuple(
        str(header.get(f'CTYPE{axis + 1}', '')).strip() for axis in order)
    ambiguous = velocity_is_ambiguous(header)
    coords = None
    if not ambiguous or velocity is not None:
        try:
            model, _ = model_header(
                filename, header, header_shape(header), rest, velocity,
                spectral_type)
            coords = coords_from_header(filename, model)
        except ValueError as e:
            problems.append(str(e))
    unit = header_unit(header)
    beam = None
    try:
        options = header_beam(header, filename if beams else None)
        beam = None if options is None else Beam(**options)
    except ValueError as e:
        problems.append(str(e))
    if unit is not None:
        try:
            per_beam = unitutils.spatial_unit(
                unitutils.parse_unit(unit)) == astropy.units.beam
        except ValueError as e:
            problems.append(f"{e}: give the unit (option unit of the data)")
        else:
            if per_beam and beam is None:
                problems.append(
                    f"the unit, {unit}, is per beam, but the header has no "
                    f"beam (BMAJ and BMIN): give it (option beam of the "
                    f"data)")
    rms = next(
        (float(header[key]) for key in _RMS_KEYS
         if isinstance(header.get(key), (int, float))
         and not isinstance(header.get(key), bool)), None)
    holds = None
    if str(header.get('HDUCLAS2', '')).strip().upper() == 'ERROR':
        holds = _ERROR_CLASSES.get(
            str(header.get('HDUCLAS3', '')).strip().upper())
    nonstandard = nonstandard_spectral_axis(header, header_shape(header))
    return HDUDescription(
        index=index, name=name, shape=shape, axes=axes, unit=unit,
        beam=beam, coords=coords, rms=rms,
        ambiguous_velocity=ambiguous,
        problems=tuple(problems), holds=holds,
        nonstandard_type=None if nonstandard is None else nonstandard[1])


def _model_order(header: astropy.io.fits.Header) -> tuple[int, ...]:
    """
    Return the axes of the header in the order of the model (see
    model_axis_order), or in their own if its world coordinates
    are invalid.
    """
    shape = header_shape(header)
    try:
        # (the fixes of astropy to the header are for the coords)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', astropy.wcs.FITSFixedWarning)
            wcs = astropy.wcs.WCS(
                centre_missing_crpix(header, shape))
    except Exception:
        # (any error of astropy: the problem is reported with the coords)
        return tuple(range(len(shape)))
    return model_axis_order(wcs)
