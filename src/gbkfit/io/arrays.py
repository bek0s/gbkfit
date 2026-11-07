"""
Reading data from files with their errors and mask, in the orientation of
the model: the one reader of the datasets (see dataset.Data.from_files)
and of prep (see prep.FitsArrays).

The files are of any format (see formats: FITS, or ASDF), all read alike
once a file states what it says of its array as a FITS header.
"""

import dataclasses
from typing import Annotated, Any, Literal, TypeAlias

import astropy.io.fits
import astropy.units as u
import astropy.wcs
import numpy as np

from gbkfit.config import Choices, DataFile
from gbkfit.grid import Rest
from gbkfit.utils import unitutils
from .formats import read_image, split_file
from .headers import (
    has_wcs, header_shape, header_unit, parse_wcs, with_shape, with_wcs)
from .orientation import Orientation, model_header
from .reading import header_beam


__all__ = [
    'ERROR_KINDS',
    'Arrays',
    'ErrorKind',
    'read_arrays',
    'read_sky_map'
]


# What an error file holds: standard deviations, variances or inverse
# variances, and the power of the unit of the data that each is in
ERROR_KINDS = ('sigma', 'variance', 'inverse_variance')
ErrorKind: TypeAlias = Annotated[
    Literal[ERROR_KINDS], Choices(
        sigma='Standard deviations', variance='Variances',
        inverse_variance='Inverse variances')]
_ERROR_POWERS = dict(sigma=1, variance=2, inverse_variance=-2)

# How far (in pixels) the world coordinates of an error or mask file may
# put the pixels from those of the data
_GRID_TOLERANCE = 1e-3


@dataclasses.dataclass(frozen=True)
class Arrays:
    """
    Data read from files in the orientation of the model, with their
    errors and mask (see read_arrays).

    Attributes
    ----------
    data : np.ndarray
        The values; NaN where invalid.
    error : np.ndarray or None
        The errors: standard deviations, in the unit of the values; NaN
        where invalid.
    mask : np.ndarray
        The mask: 1 where valid, 0 elsewhere.
    header : astropy.io.fits.Header
        The header of the values, in the orientation of the model (see
        model_header).
    unit : astropy.units.UnitBase or None
        The unit of the values, if known.
    beam : dict or None
        The beam of the values, if they have one (see header_beam): of a
        cube of CASA, that of each of its channels, in the order of the
        channels of the values.
    footprint : np.ndarray or None
        Where the instrument looked, if given (see Data), on the pixels of
        the values.
    orientation : Orientation or None
        How the file of the values was put in the orientation of the
        model, if they were read from one (see read_sky_map).
    """
    data: np.ndarray
    error: np.ndarray | None
    mask: np.ndarray
    header: astropy.io.fits.Header
    unit: u.UnitBase | None
    beam: dict[str, Any] | None = None
    footprint: np.ndarray | None = None
    orientation: Orientation | None = None


def read_arrays(
        data: DataFile,
        error: DataFile | None = None,
        mask: DataFile | None = None,
        *,
        error_kind: ErrorKind = 'sigma',
        unit: str | u.UnitBase | None = None,
        rest: Rest = None,
        velocity: str | None = None,
        footprint: DataFile | None = None,
        spectral_type: str | None = None,
        medium: str | None = None,
        ndim: int | None = None
) -> Arrays:
    """
    Read data, and their errors, mask and footprint, from files in the
    orientation of the model.

    A file is a FITS file and its HDU, or, by default, its first HDU with
    data (see read_hdu); or an ASDF file and the path of its array in its
    tree (e.g. 'roman/data'), whose unit is that of a Quantity, and world
    coordinates those of the WCS of the file, if it has one (see
    formats.asdf). Each is put in the orientation of the model by its own
    world coordinates (see model_header). The errors, the mask and the
    footprint follow the axes of the file of the data, whose shape they
    must have, if they or the data have no world coordinates; else theirs
    must put their pixels on those of the data.

    The errors are standard deviations, variances or inverse variances
    (error_kind), and become standard deviations in the unit of the data.
    If the errors and the data have units (BUNIT, or unit for the data),
    those of the errors are converted to it (e.g. mJy/beam to Jy/beam),
    and must be those of their kind (e.g. the square of the unit of the
    data for variances).

    Values that are not finite, errors that are not finite or are 0
    (inverse variances of 0: no data), and values that the mask marks
    with 0 are invalid: NaN, with the mask 0.

    The beam of the values is that of the header of their file, or of
    each channel of a cube of CASA (see header_beam), in the order of the
    channels of the values (reversed with the spectral axis, e.g. of
    decreasing frequency).

    Parameters
    ----------
    data : DataFile
        The file of the values.
    error, mask : DataFile, optional
        The files of the errors and of the mask.
    error_kind : str, optional
        What the errors are: 'sigma' (standard deviations), 'variance'
        or 'inverse_variance'.
    unit : str or UnitBase, optional
        The unit of the values; by default, that of their header (BUNIT),
        if it is a unit.
    rest : str or Quantity, optional
        The rest of the spectral axis, which turns one of wavelengths or
        frequencies into velocities (see model_header).
    velocity : str, optional
        The convention of the velocities of a spectral axis that the
        headers leave ambiguous: 'radio' or 'optical'.
    footprint : DataFile, optional
        The file of where the instrument looked (see Data).
    spectral_type : str, optional
        The type of an axis whose headers give one that FITS does not
        define (see model_header).
    medium : str, optional
        The medium of the wavelengths of an axis of optical velocities and
        of the rest, if the headers do not state it (see model_header).
    ndim : int, optional
        The number of axes of the values, if known, which axes of length
        1 make up (e.g. 2 of spectra of one region; see model_header).

    Returns
    -------
    Arrays
        The arrays.

    Raises
    ------
    ValueError
        If a file cannot be read, or has world coordinates the model
        cannot take (see model_header); error_kind is not one of
        ERROR_KINDS, the errors are negative, or their unit is not that of
        their kind; the errors or the mask are not on the pixels of the
        data; or the beams of a cube are not one for each channel.
    """
    if error_kind not in ERROR_KINDS:
        raise ValueError(
            f"the kind of the errors must be one of {list(ERROR_KINDS)}; "
            f"it is {error_kind!r}")
    read = dict(
        rest=rest, velocity=velocity, spectral_type=spectral_type,
        medium=medium, ndim=ndim)
    values, header, orientation, beam = _read(data, **read)
    given = unit is not None
    if isinstance(unit, str):
        unit = unitutils.parse_unit(unit)
    if unit is None:
        unit = _parsed_unit(split_file(data)[0], header)
    errors = None
    if error is not None:
        errors, error_header = _read_companion(
            error, header, orientation, **read)
        errors = _standard_deviations(
            split_file(error)[0], errors, error_kind,
            _parsed_unit(split_file(error)[0], error_header, given), unit)
    masks = None
    if mask is not None:
        masks, _ = _read_companion(mask, header, orientation, **read)
    valid = np.isfinite(values)
    if errors is not None:
        valid &= np.isfinite(errors) & (errors > 0)
    if masks is not None:
        valid &= np.isfinite(masks) & (masks != 0)
    footprints = None
    if footprint is not None:
        footprints, _ = _read_companion(
            footprint, header, orientation, **read)
    return Arrays(
        np.where(valid, values, np.nan),
        None if errors is None else np.where(valid, errors, np.nan),
        valid.astype(float), header, unit, beam, footprints, orientation)


def read_sky_map(
        file: DataFile,
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation,
        plane: int | None = None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Read an image of the sky of data read from a file (e.g. the bin of
    each of their pixels) on the pixels of the data, as their errors and
    mask are read (see read_arrays): by its own world coordinates, which
    must put its pixels on those of the celestial axes of the data; or,
    if it or the data have none, in the axes of the file of the data,
    whose sky it must have the shape of.

    Parameters
    ----------
    file : DataFile
        The file of the image.
    data_header : astropy.io.fits.Header
        The header of the data, in the orientation of the model (see
        Arrays).
    data_orientation : Orientation
        How the file of the data was put in the orientation of the model
        (see Arrays).
    plane : int, optional
        The plane to read of a file of images of the sky, along its last
        axis (FITS order; e.g. the BINID of the MAPS of MaNGA's DAP).

    Returns
    -------
    np.ndarray
        The image, in the orientation of the model.
    astropy.io.fits.Header
        Its header.

    Raises
    ------
    ValueError
        If the data have no sky of two axes, the image is not on the
        pixels of their sky, or the file has no such plane.
    """
    orientation = data_orientation.sky()
    shape = header_shape(data_header)[-2:]
    header = with_shape(data_header, shape)
    if has_wcs(data_header):
        wcs = parse_wcs(split_file(file)[0], data_header)
        if wcs.wcs.spec in (0, 1):
            raise ValueError(
                f"{split_file(file)[0]}: the data have no sky of two axes "
                f"to put an image of the sky on")
        header = with_wcs(header, wcs.sub(2))
    return _read_companion(
        file, header, orientation, rest=None, velocity=None,
        spectral_type=None, medium=None, ndim=None, plane=plane)


def _read(
        file: DataFile,
        rest: Rest,
        velocity: str | None,
        spectral_type: str | None,
        medium: str | None,
        ndim: int | None
) -> tuple[
        np.ndarray, astropy.io.fits.Header, Orientation,
        dict[str, Any] | None]:
    """
    Return the values of a file in the orientation of the model, their
    header in it, how they were put in it (see model_header), and their
    beam (see _beam).
    """
    filename, key = split_file(file)
    values, header = read_image(filename, key)
    model, orientation = model_header(
        filename, header, values.shape, rest, velocity, spectral_type,
        medium, ndim)
    return (_floats(orientation.apply(values)), model, orientation,
            _beam(filename, header, model, orientation))


def _beam(
        filename: str,
        header: astropy.io.fits.Header,
        model: astropy.io.fits.Header,
        orientation: Orientation
) -> dict[str, Any] | None:
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
    if beam is None or 'spectral' not in beam:
        return beam
    # A cube with a table BEAMS has a spectral axis (see casa_beam_table).
    # The axes that the orientation reverses are those of the model, where
    # the spectral axis can be at another index than in the file (e.g. of
    # RA, Dec, Stokes and frequency, once the Stokes axis of 1 is dropped).
    axis = astropy.wcs.WCS(model).wcs.spec
    if axis < 0:
        # (a spectral axis of one channel, left out: the beam of it)
        if len(beam['bmaj']) != 1:
            raise ValueError(
                f"{filename}: the cube has 1 channel, but its table BEAMS "
                f"has beams for {len(beam['bmaj'])}")
        return {key: float(beam[key][0]) for key in ('bmaj', 'bmin', 'bpa')}
    nchannels = int(model[f'NAXIS{axis + 1}'])
    if len(beam['bmaj']) != nchannels:
        raise ValueError(
            f"{filename}: the cube has {nchannels} channels, but its table "
            f"BEAMS has beams for {len(beam['bmaj'])}")
    order = slice(None, None, -1) if axis in orientation.flips \
        else slice(None)
    return {key: value[order] for key, value in beam.items()}


def _read_companion(
        file: DataFile,
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation,
        rest: Rest,
        velocity: str | None,
        spectral_type: str | None,
        medium: str | None,
        ndim: int | None,
        plane: int | None = None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the values of a file (of a plane of it, if given; see
    read_sky_map) on the pixels of data (their errors, mask or footprint,
    see read_arrays, or an image of their sky, see read_sky_map) in the
    orientation of the model, and its header.
    """
    filename, key = split_file(file)
    values, header = read_image(filename, key)
    if plane is not None:
        values, header = _plane(filename, values, header, plane)
    if has_wcs(header) and has_wcs(data_header):
        header, orientation = model_header(
            filename, header, values.shape, rest, velocity, spectral_type,
            medium, ndim)
        values = orientation.apply(values)
        _check_same_pixels(filename, header, values.shape, data_header)
        return _floats(values), header
    shape = data_orientation.file_shape
    if values.shape != shape:
        raise ValueError(
            f"{filename}: without world coordinates (or with data without "
            f"them), its values must be of the shape of the data, {shape}, "
            f"in the axes of their file; they are of {values.shape}")
    return _floats(data_orientation.apply(values)), header


def _plane(
        source: str,
        values: np.ndarray,
        header: astropy.io.fits.Header,
        plane: int
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return a plane of an array of three axes, along its last (FITS
    order), and its header without that axis.
    """
    if values.ndim != 3 or not 0 <= plane < values.shape[0]:
        raise ValueError(
            f"{source}: plane {plane} is not one of its array of shape "
            f"{values.shape} (of planes along its first axis, numpy order)")
    header = with_shape(header, values.shape[1:])
    if has_wcs(header):
        header = with_wcs(header, parse_wcs(source, header).sub(2))
    return values[plane], header


def _check_same_pixels(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        data_header: astropy.io.fits.Header
) -> None:
    """
    Raise ValueError unless the values of a header in the orientation of
    the model (of the given shape) are on the pixels of the data of
    another: of their shape, and their world coordinates at the corners.
    """
    data_shape = header_shape(data_header)
    if shape != data_shape:
        raise ValueError(
            f"{source}: its values must be of the shape of the data, "
            f"{data_shape} (in the orientation of the model); they are of "
            f"{shape}")
    wcs, data_wcs = astropy.wcs.WCS(header), astropy.wcs.WCS(data_header)
    if list(wcs.wcs.ctype) != list(data_wcs.wcs.ctype):
        raise ValueError(
            f"{source}: its world coordinates are of the axes "
            f"{list(wcs.wcs.ctype)}, those of the data of "
            f"{list(data_wcs.wcs.ctype)}")
    corners = np.array(np.meshgrid(
        *[[0, size - 1] for size in shape[::-1]])).reshape(len(shape), -1)
    world = data_wcs.pixel_to_world_values(*corners)
    pixels = np.array(wcs.world_to_pixel_values(*world)).reshape(corners.shape)
    if not np.allclose(pixels, corners, rtol=0, atol=_GRID_TOLERANCE):
        raise ValueError(
            f"{source}: its world coordinates do not put its pixels on "
            f"those of the data")


def _standard_deviations(
        source: str,
        errors: np.ndarray,
        kind: str,
        unit: u.UnitBase | None,
        data_unit: u.UnitBase | None
) -> np.ndarray:
    """
    Return errors of a kind (see read_arrays) as standard deviations in
    the unit of the data; a unit of the errors that is not that of their
    kind is an error, which names the kind it is of, if any.
    """
    power = _ERROR_POWERS[kind]
    if unit is not None and data_unit is not None:
        expected = data_unit ** power
        if not unit.is_equivalent(expected):
            kinds = [
                name for name, p in _ERROR_POWERS.items()
                if unit.is_equivalent(data_unit ** p)]
            hint = f"; it is that of {kinds[0]} (error_kind: {kinds[0]})" \
                if kinds else ""
            raise ValueError(
                f"{source}: the unit of the errors "
                f"({unitutils.format_unit(unit)}) is not that of {kind} of "
                f"the data ({unitutils.format_unit(expected)}){hint}")
        errors = errors * unit.to(expected)
    with np.errstate(invalid='ignore'):
        negative = errors < 0
    if np.any(negative):
        raise ValueError(
            f"{source}: the errors cannot be negative; "
            f"{np.count_nonzero(negative)} are, down to {np.nanmin(errors):g}")
    if kind == 'variance':
        return np.sqrt(errors)
    if kind == 'inverse_variance':
        with np.errstate(divide='ignore'):
            return np.where(errors > 0, 1 / np.sqrt(errors), np.nan)
    return errors


def _parsed_unit(
        source: str, header: astropy.io.fits.Header, given: bool = False
) -> u.UnitBase | None:
    """
    Return the unit of the values of a header (BUNIT), if it has one.

    Raise ValueError if it is not a unit (see unitutils.parse_unit),
    unless the unit of the data is given: then return None, and the values
    are taken to be in that unit, as for a header without BUNIT.
    """
    text = header_unit(header)
    if text is None:
        return None
    try:
        return unitutils.parse_unit(text, 'fits')
    except ValueError:
        if given:
            return None
        raise ValueError(
            f"{source}: BUNIT = {text!r} is not a valid unit (see the FITS "
            f"standard, section 4.3); give the unit of the data (option "
            f"unit), or correct BUNIT in a copy of the file") from None


def _floats(values: np.ndarray) -> np.ndarray:
    """Return values as floats (which can be NaN where invalid)."""
    if np.issubdtype(values.dtype, np.floating):
        return values
    return values.astype(float)
