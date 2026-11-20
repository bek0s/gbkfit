"""
Descriptions of FITS files of data from their headers alone (see
describe_fits): what each HDU holds, and what a dataset of it refuses
or needs given, so that the user can say what each holds before making
a dataset of it.
"""

import dataclasses
from typing import Any

import astropy.io.fits
import astropy.units

import gbkfit.units
from gbkfit.config import ConfigProblem
from gbkfit.grid import Coords
from .beams import Beam, header_beam
from .coords import coords_from_header
from .headers import (
    header_shape, header_unit, ignoring_wcs_fixes, parse_wcs)
from .orientation import model_axis_order, model_header
from .spectral import (
    _missing_unit, _statement_errors, velocity_is_ambiguous)


__all__ = [
    'HDUDescription',
    'describe_fits'
]


# The keywords of headers that state the noise of the values (see
# HDUDescription.rms), in order
_RMS_KEYS = ('RMS', 'NOISE')

# What an HDU of errors holds, as its header states it in the classes of
# HDUs of ESO's science data products (e.g. MUSE, and MaNGA): HDUCLAS2 =
# 'ERROR', and HDUCLAS3 one of these (see HDUDescription.error_kind)
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
        The number of pixels along each axis, in the order of the axes of
        the model (x, y, then the spectral axis; see model_axis_order),
        or in that of the file if its world coordinates are invalid; with
        its axes of length 1, which reading leaves out.
    axes : tuple of str
        The type of each axis (CTYPE, e.g. 'RA---SIN' or 'FREQ'; '' if
        none), in the order of shape.
    axis_units : tuple of str
        The unit of each axis as the header writes it (CUNIT, e.g. 'deg'
        or 'KMS', which is not one; '' if none), in the order of shape.
    unit : str or None
        The unit of its values, as the header writes it (BUNIT).
    beam : Beam or None
        Its beam (see header_beam).
    coords : Coords or None
        The world coordinates of its data in the model (see
        read_data); None if they cannot be read, or if its
        velocities are ambiguous and their convention is not given.
    rms : float or None
        The noise of its values that the header states (keyword RMS or
        NOISE, in the unit of the values; e.g. that of the cubes of
        LVHIS), if any.
    ambiguous_velocity : bool
        Whether the header leaves it open whether the velocities of its
        spectral axis are radio or optical (see velocity_is_ambiguous):
        a dataset of it gives their convention (until then, a problem).
    problems : tuple of ConfigProblem
        What a dataset of it refuses (errors) or reads as FITS says, but
        may not be meant (warnings: e.g. a spectral axis without a unit,
        read in SI units), each at the option of the dataset that fixes it
        (e.g. ['spectral', 'unit']; [] if none), with its message (without
        the name of the file): one at a time of the spectral axis, the
        first that reading it finds; described again with the option, the
        next.
    error_kind : str or None
        What its values are, as errors, if its header states it (HDUCLAS2
        = 'ERROR', and HDUCLAS3 'RMSE', 'MSE' or 'INVMSE'): 'sigma'
        (standard deviations), 'variance' or 'inverse_variance' (the
        error_kind of a dataset); None if it does not.
    """
    index: int
    name: str
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    axis_units: tuple[str, ...]
    unit: str | None
    beam: Beam | None
    coords: Coords | None
    rms: float | None
    ambiguous_velocity: bool
    problems: tuple[ConfigProblem, ...]
    error_kind: str | None = None


def describe_fits(
        filename: str, velocity: str | None = None, rest: Any = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None
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
        leave ambiguous (see model_header): 'radio' or 'optical'.
    rest : str or Quantity, optional
        The rest of their spectral axes (e.g. '6562.8 Angstrom'), which
        turns axes of wavelengths or frequencies into velocities, as a
        dataset with that rest reads them (see read_data).
    spectral_type : str, optional
        What a spectral axis whose type does not say what its values are
        is (see model_header).
    spectral_unit : str, optional
        The unit of a spectral axis whose header gives none or one that
        is not valid (see SpectralAxis, option unit); it is not used for
        others.

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
        _describe(
            filename, *header, velocity, rest, spectral_type, spectral_unit,
            beams)
        for header in headers]


def _describe(
        filename: str,
        index: int,
        name: str,
        header: astropy.io.fits.Header,
        velocity: str | None,
        rest: Any,
        spectral_type: str | None,
        spectral_unit: str | None,
        beams: bool
) -> HDUDescription:
    """
    Return the description of an HDU of a FITS file with an array, from
    its header (and the table of the beams of the file, if it has one:
    beams): its world coordinates as read_data reads them, and its unit
    and beam as header_unit and header_beam do.
    """
    problems: list[ConfigProblem] = []

    def problem(error: ValueError) -> None:
        # (at the option of the error, if any; of this file, unnamed)
        message = str(error).removeprefix(f"{filename}: ")
        path = list(getattr(error, 'config_path', []))
        problems.append(ConfigProblem('error', path, message))

    order = _model_order(filename, header)
    # (world axes beyond those of the array are of length 1, see
    # model_header)
    sizes = header_shape(header)[::-1]
    sizes += (1,) * (len(order) - len(sizes))
    shape = tuple(sizes[axis] for axis in order)
    axes = tuple(
        str(header.get(f'CTYPE{axis + 1}', '')).strip() for axis in order)
    axis_units = tuple(
        str(header.get(f'CUNIT{axis + 1}', '')).strip() for axis in order)
    ambiguous = velocity_is_ambiguous(header)
    coords = None
    # (what the spectral axis needs stated, together; once it is, what
    # reading it finds)
    stated = _statement_errors(
        filename, header, header_shape(header), spectral_type, velocity,
        spectral_unit)
    for error in stated:
        problem(error)
    if not stated:
        try:
            model, _ = model_header(
                filename, header, header_shape(header), rest, velocity,
                spectral_type, spectral_unit)
            coords = coords_from_header(filename, model)
        except ValueError as e:
            problem(e)
    if spectral_unit is None:
        try:
            missing = _missing_unit(
                filename, header, header_shape(header), spectral_type)
        except ValueError:
            # (of a type not given: a problem of the coords)
            missing = None
        if missing is not None:
            problems.append(ConfigProblem(
                'warning', ['spectral', 'unit'], f"CUNIT{missing[0]} missing"))
    unit = header_unit(header)
    beam = None
    try:
        beam = header_beam(header, filename if beams else None)
    except ValueError as e:
        problems.append(ConfigProblem(
            'error', ['beam'], str(e).removeprefix(f"{filename}: ")))
    if unit is not None:
        try:
            per_beam = gbkfit.units.spatial_unit(
                gbkfit.units.parse_unit(unit)) == astropy.units.beam
        except ValueError:
            problems.append(ConfigProblem(
                'error', ['unit'], f"BUNIT {unit!r} invalid"))
        else:
            if per_beam and beam is None:
                problems.append(ConfigProblem(
                    'error', ['beam'], f"BMAJ, BMIN missing (BUNIT {unit})"))
    rms = next(
        (float(header[key]) for key in _RMS_KEYS
         if isinstance(header.get(key), (int, float))
         and not isinstance(header.get(key), bool)), None)
    error_kind = None
    if str(header.get('HDUCLAS2', '')).strip().upper() == 'ERROR':
        error_kind = _ERROR_CLASSES.get(
            str(header.get('HDUCLAS3', '')).strip().upper())
    return HDUDescription(
        index=index, name=name, shape=shape, axes=axes,
        axis_units=axis_units, unit=unit,
        beam=beam, coords=coords, rms=rms,
        ambiguous_velocity=ambiguous,
        problems=tuple(problems), error_kind=error_kind)


def _model_order(
        filename: str, header: astropy.io.fits.Header
) -> tuple[int, ...]:
    """
    Return the axes of the header in the order of the model (see
    model_axis_order), or in their own if its world coordinates
    are invalid (a problem reported with the coords).
    """
    try:
        # (the fixes of astropy to the header are reported with the coords)
        with ignoring_wcs_fixes():
            return model_axis_order(parse_wcs(filename, header))
    except ValueError:
        return tuple(range(len(header_shape(header))))
