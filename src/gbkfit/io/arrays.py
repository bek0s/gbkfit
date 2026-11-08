"""
Arrays: reading them from files, in the orientation of the model.

- An array with the world coordinates of the model (see read_data).
- Data with their errors and mask (see read_arrays): the one reader of
  the datasets (see dataset.Data.from_files) and of prep (see
  prep.FitsArrays); and images of the sky on the pixels of such data,
  e.g. bins (see read_sky_map).
- An array in the orientation of the model as far as its world
  coordinates allow, which need not be ones the model represents, to
  look at it (see read_oriented).

The files are of any format (see formats: FITS, or ASDF), all read alike
once a file states what it says of its array as a FITS header.
"""

import dataclasses
from collections.abc import Sequence
from typing import Annotated, Literal, TypeAlias

import astropy.io.fits
import astropy.wcs
import numpy as np

import gbkfit.units
from gbkfit.config import Choices, DataFile, split_data_file
from gbkfit.grid import Coords, Rest
from .beams import Beam, _oriented_beam
from .coords import coords_from_header
from .formats import read_image
from .headers import (
    _with_shape, _with_wcs, has_wcs, header_shape, header_unit,
    ignoring_wcs_fixes, parse_wcs)
from .orientation import (
    Orientation, _oriented_header, _squeezed_shape, model_header)


__all__ = [
    'Arrays',
    'ErrorKind',
    'MaskKind',
    'read_arrays',
    'read_data',
    'read_oriented',
    'read_sky_map'
]


# What an error file holds: standard deviations, variances or inverse
# variances, and the power of the unit of the data that each is in
_ERROR_KINDS = ('sigma', 'variance', 'inverse_variance')
ErrorKind: TypeAlias = Annotated[
    Literal[_ERROR_KINDS], Choices(
        sigma='Standard deviations', variance='Variances',
        inverse_variance='Inverse variances')]
_ERROR_POWERS = dict(sigma=1, variance=2, inverse_variance=-2)

# What a mask file marks with values other than 0: the valid values (e.g.
# the masks of SoFiA), or the invalid ones (e.g. a mask of bad pixels, or
# of bits of flags, as CALIFA's BADPIX or the masks of MaNGA's DAP)
_MASK_KINDS = ('valid', 'invalid')
MaskKind: TypeAlias = Annotated[
    Literal[_MASK_KINDS], Choices(
        valid='Nonzero where valid',
        invalid='Nonzero where invalid (e.g. flags)')]

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
    unit : AnyUnit or None
        The unit of the values, if known.
    beam : Beam or None
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
    unit: gbkfit.units.AnyUnit | None = None
    beam: Beam | None = None
    footprint: np.ndarray | None = None
    orientation: Orientation | None = None


def read_data(
        file: DataFile,
        *,
        rpix: float | Sequence[float | None] | None = None,
        rval: float | Sequence[float | None] | None = None,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> tuple[np.ndarray, Coords]:
    """
    Read the data of a file and its world coordinates.

    The data are in the orientation of the model (see model_header),
    without their axes of length 1 (e.g. an axis of Stokes, or the
    spectral axis of a moment map), and their world coordinates in its
    units (see coords_from_header).

    Parameters
    ----------
    file : DataFile
        The file of the data, and where they are in it (see read_image).
    rpix, rval : optional
        World coordinates that replace those of the header (see
        coords_from_header), for the axes in the orientation of the model.
    rest : str or Quantity, optional
        The rest of the spectral axis, which turns one of wavelengths or
        frequencies into velocities (see model_header).
    velocity : str, optional
        The convention of the velocities of a spectral axis that the
        header leaves ambiguous: 'radio' or 'optical' (see model_header).
    spectral_type : str, optional
        The type of an axis whose header gives one that FITS does not
        define (see model_header).
    medium : str, optional
        The medium of the wavelengths of the spectral axis and of the
        rest, if the header does not state it (see model_header and
        coords_from_header).
    spectral_axis : int, optional
        The index (FITS order, from 0) that the spectral axis of the data
        (if any) must have in the orientation of the model: the data have
        an axis of their own before it for each, of length 1 too (e.g. the
        region of spectra of one region; see model_header).

    Returns
    -------
    np.ndarray
        The data.
    Coords
        Their world coordinates.

    Raises
    ------
    ValueError
        If the file has no such data (see read_image), or their header
        has invalid world coordinates or ones the model cannot represent
        (see model_header and coords_from_header).
    """
    filename = split_data_file(file)[0]
    data, header = read_image(file)
    header, orientation = model_header(
        filename, header, data.shape, rest=rest, velocity=velocity,
        spectral_type=spectral_type, medium=medium,
        spectral_axis=spectral_axis)
    coords = coords_from_header(
        filename, header, rpix=rpix, rval=rval, spectral_axis=spectral_axis,
        medium=medium, rest=rest)
    return orientation.apply(data), coords


def read_oriented(
        file: DataFile
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
    file : DataFile
        The file of the array, and where it is in it (see read_image).

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
        If the file has no such array (see read_image).
    """
    filename = split_data_file(file)[0]
    data, header = read_image(file)
    try:
        # (the fixes of astropy to the header are not of the orientation)
        with ignoring_wcs_fixes():
            header, orientation = _oriented_header(
                filename, header, data.shape)
            wcs = astropy.wcs.WCS(header) if has_wcs(header) else None
    except ValueError:
        # (world coordinates that astropy cannot read, or that cannot be
        # put in the orientation of the model, leave the array as it is;
        # the errors of astropy's WCS are ValueErrors)
        return data.reshape(_squeezed_shape(data.shape)), None
    return orientation.apply(data), wcs


def read_arrays(
        data: DataFile,
        error: DataFile | None = None,
        mask: DataFile | None = None,
        *,
        footprint: DataFile | None = None,
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid',
        unit: str | None = None,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> Arrays:
    """
    Read data, and their errors, mask and footprint, from files in the
    orientation of the model.

    Each file is read by read_image (see DataFile) and put in the
    orientation of the model by its own world coordinates (see
    model_header). The errors, the mask and the
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
    (inverse variances of 0: no data), and values that the mask marks as
    invalid (with 0, or of mask_kind 'invalid', with other values) are
    invalid: NaN, with the mask 0.

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
    footprint : DataFile, optional
        The file of where the instrument looked (see Data).
    error_kind : str, optional
        What the errors are: 'sigma' (standard deviations), 'variance'
        or 'inverse_variance'.
    mask_kind : str, optional
        What the mask marks with values other than 0: 'valid' (by
        default) or 'invalid' values (see _MASK_KINDS).
    unit : str, optional
        The unit of the values, as gbkfit.units.parse_unit reads it; by
        default, that of their header (BUNIT), if it is a unit.
    rest, velocity, spectral_type, medium, spectral_axis : optional
        As read_data, for the headers of all the files (the spectral axis
        is not checked: see coords_from_header).

    Returns
    -------
    Arrays
        The arrays.

    Raises
    ------
    ValueError
        If a file cannot be read, or has world coordinates the model
        cannot take (see model_header); error_kind or mask_kind is not
        one of its kinds; the errors are negative, or their unit is not
        that of their kind; the errors or the mask are not on the pixels
        of the data; or the beams of a cube are not one for each channel.
    """
    if error_kind not in _ERROR_KINDS:
        raise ValueError(
            f"the kind of the errors must be one of {list(_ERROR_KINDS)}; "
            f"it is {error_kind!r}")
    if mask_kind not in _MASK_KINDS:
        raise ValueError(
            f"the kind of the mask must be one of {list(_MASK_KINDS)}; it "
            f"is {mask_kind!r}")
    read = dict(
        rest=rest, velocity=velocity, spectral_type=spectral_type,
        medium=medium, spectral_axis=spectral_axis)
    values, header, orientation, beam = _read(data, **read)
    given = unit is not None
    if given:
        unit = gbkfit.units.parse_unit(unit)
    else:
        unit = _parsed_unit(split_data_file(data)[0], header)
    errors = None
    if error is not None:
        error_name = split_data_file(error)[0]
        errors, error_header = _read_companion(
            error, header, orientation, **read)
        errors = _standard_deviations(
            error_name, errors, error_kind,
            _parsed_unit(error_name, error_header, given), unit)
    masks = None
    if mask is not None:
        masks, _ = _read_companion(mask, header, orientation, **read)
    valid = np.isfinite(values)
    if errors is not None:
        valid &= np.isfinite(errors) & (errors > 0)
    if masks is not None:
        marked = masks != 0
        valid &= np.isfinite(masks) & (
            marked if mask_kind == 'valid' else ~marked)
    footprints = None
    if footprint is not None:
        footprints, _ = _read_companion(
            footprint, header, orientation, **read)
    return Arrays(
        np.where(valid, values, np.nan),
        None if errors is None else np.where(valid, errors, np.nan),
        valid.astype(float), header, unit, beam, footprints, orientation)


def read_sky_map(
        file: DataFile, data: Arrays, plane: int | None = None
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
    data : Arrays
        The data, as read_arrays reads them (on the pixels of their file).
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
        If the data are not on the pixels of their file (e.g. cropped),
        have no sky of two axes, the image is not on the pixels of their
        sky, or the file has no such plane.
    """
    if data.orientation is None:
        raise ValueError(
            "an image of the sky is read on the pixels of the file of the "
            "data, before they are cropped or padded")
    orientation = data.orientation.sky()
    shape = header_shape(data.header)[-2:]
    header = _with_shape(data.header, shape)
    if has_wcs(data.header):
        filename = split_data_file(file)[0]
        wcs = parse_wcs(filename, data.header)
        if wcs.wcs.spec in (0, 1):
            raise ValueError(
                f"{filename}: the data have no sky of two axes to put an "
                f"image of the sky on")
        header = _with_wcs(header, wcs.sub(2))
    return _read_companion(
        file, header, orientation, rest=None, velocity=None,
        spectral_type=None, medium=None, spectral_axis=None, plane=plane)


def _read(
        file: DataFile,
        rest: Rest,
        velocity: str | None,
        spectral_type: str | None,
        medium: str | None,
        spectral_axis: int | None
) -> tuple[np.ndarray, astropy.io.fits.Header, Orientation, Beam | None]:
    """
    Return the values of a file in the orientation of the model, their
    header in it, how they were put in it (see model_header), and their
    beam (see _oriented_beam).
    """
    filename = split_data_file(file)[0]
    values, header = read_image(file)
    model, orientation = model_header(
        filename, header, values.shape, rest=rest, velocity=velocity,
        spectral_type=spectral_type, medium=medium,
        spectral_axis=spectral_axis)
    return (_floats(orientation.apply(values)), model, orientation,
            _oriented_beam(filename, header, model, orientation))


def _read_companion(
        file: DataFile,
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation,
        rest: Rest,
        velocity: str | None,
        spectral_type: str | None,
        medium: str | None,
        spectral_axis: int | None,
        plane: int | None = None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the values of a file (of a plane of it, if given; see
    read_sky_map) on the pixels of data (their errors, mask or footprint,
    see read_arrays, or an image of their sky, see read_sky_map) in the
    orientation of the model, and its header.
    """
    filename = split_data_file(file)[0]
    values, header = read_image(file)
    if plane is not None:
        values, header = _plane(filename, values, header, plane)
    if has_wcs(header) and has_wcs(data_header):
        header, orientation = model_header(
            filename, header, values.shape, rest=rest, velocity=velocity,
            spectral_type=spectral_type, medium=medium,
            spectral_axis=spectral_axis)
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
    header = _with_shape(header, values.shape[1:])
    if has_wcs(header):
        header = _with_wcs(header, parse_wcs(source, header).sub(2))
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
        unit: gbkfit.units.AnyUnit | None,
        data_unit: gbkfit.units.AnyUnit | None
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
            hint = (
                f"; it is that of {kinds[0]} (error_kind: {kinds[0]})"
                if kinds else "")
            raise ValueError(
                f"{source}: the unit of the errors "
                f"({gbkfit.units.format_unit(unit)}) is not that of {kind} of "
                f"the data ({gbkfit.units.format_unit(expected)}){hint}")
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
) -> gbkfit.units.AnyUnit | None:
    """
    Return the unit of the values of a header (BUNIT), if it has one.

    Raise ValueError if it is not a unit (see
    gbkfit.units.parse_unit), unless the unit of the data is given:
    then return None, and the values are taken to be in that unit, as for
    a header without BUNIT.
    """
    text = header_unit(header)
    if text is None:
        return None
    try:
        return gbkfit.units.parse_unit(text)
    except ValueError as e:
        if given:
            return None
        raise ValueError(
            f"{source}: BUNIT: {e}; give the unit of the data (option "
            f"unit), or correct BUNIT in a copy of the file") from None


def _floats(values: np.ndarray) -> np.ndarray:
    """Return values as floats (which can be NaN where invalid)."""
    if np.issubdtype(values.dtype, np.floating):
        return values
    return values.astype(float)
