"""
Arrays: reading them from files, in the orientation of the model.

- An array with the world coordinates of the model (see read_data).
- Data with their errors and mask (see read_arrays): the one reader of
  the datasets (see dataset.Data.from_files) and of prep (see
  prep.FitsArrays); and images of the sky on the pixels of such data,
  e.g. bins (see read_sky_map). Data are opened first, from the headers
  of their files alone (see open_arrays: every check of what the headers
  say), and read after, which a dataset defers until its values are
  first used.
- An array in the orientation of the model as far as its world
  coordinates allow, which need not be ones the model represents, to
  look at it (see read_oriented).

The files are of any format (see formats: FITS, or ASDF), all read alike
once a file states what it says of its array as a FITS header.
"""

import dataclasses
import os
from collections.abc import Collection, Mapping, Sequence
from typing import Annotated, Any, Literal, NamedTuple, TypeAlias

import astropy.io.fits
import astropy.wcs
import numpy as np

import gbkfit.units
from gbkfit.config import (
    Choices, DataFile, InputError, config_path, raise_all, shape_text,
    split_data_file)
from gbkfit.grid import Coords, Rest
from .beams import Beam, _oriented_beam
from .coords import coords_from_header
from .formats import read_header, read_image
from .headers import (
    _with_shape, _with_wcs, data_unit, has_wcs, header_shape,
    ignoring_wcs_fixes, parse_wcs)
from .orientation import (
    Orientation, _oriented_header, _squeezed_shape, model_header,
    model_shape)
from .spectral import _stated_header, _warn_of_missing_unit
from .statements import statement_errors


__all__ = [
    'Arrays',
    'ErrorKind',
    'MaskKind',
    'OpenedArray',
    'OpenedArrays',
    'OpenedFootprint',
    'check_companion',
    'check_footprint',
    'open_arrays',
    'open_data',
    'open_footprint',
    'read_arrays',
    'read_data',
    'read_footprint',
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
    orientation: Orientation | None = None


class _Stamp(NamedTuple):
    """
    A file as it was opened: its size and the time it last changed, which
    its reading checks (see OpenedArray.read).
    """
    path: str
    size: int
    mtime: int

    @classmethod
    def of(cls, path: str) -> '_Stamp':
        """Return the stamp of a file as it is now."""
        status = os.stat(path)
        return cls(path, status.st_size, status.st_mtime_ns)

    def check(self) -> None:
        """Raise InputError if the file changed since it was stamped."""
        if _Stamp.of(self.path) != self:
            raise InputError(
                f"{self.path}: the file changed after it was opened; load "
                f"it again")


@dataclasses.dataclass(frozen=True)
class OpenedArray:
    """
    The array of a file, opened from its header alone, as it is put in the
    orientation of the model (see open_arrays), before it is read.

    Attributes
    ----------
    file : DataFile
        The file of the array.
    header : astropy.io.fits.Header
        Its header in the orientation of the model (of an array without
        world coordinates put on the pixels of data, as it is in its file).
    orientation : Orientation
        How the array of the file is put in that orientation.
    plane : int or None
        The plane read of a file of images of the sky, if any (see
        read_sky_map).
    """
    file: DataFile
    header: astropy.io.fits.Header
    orientation: Orientation
    plane: int | None
    stamp: _Stamp = dataclasses.field(repr=False)

    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array as it is read (numpy order)."""
        return header_shape(self.header)

    def read(self, floats: bool = True) -> np.ndarray:
        """
        Read the array in the orientation of the model: as floats (which
        can be NaN where invalid), or of the type of its file.

        Raises
        ------
        ValueError
            If the file changed since it was opened, or cannot be read.
        """
        self.stamp.check()
        values, _ = read_image(self.file)
        if self.plane is not None:
            values = values[self.plane]
        values = self.orientation.apply(values)
        return _floats(values) if floats else values


@dataclasses.dataclass(frozen=True)
class OpenedFootprint(OpenedArray):
    """
    A footprint opened from the headers of its file and of its data (see
    open_footprint), read as 1 inside (where finite and not 0) and 0
    outside.
    """

    def read(self, floats: bool = True) -> np.ndarray:
        values = super().read()
        return (np.isfinite(values) & (values != 0)).astype(float)


@dataclasses.dataclass(frozen=True)
class OpenedArrays:
    """
    Data, and their errors and mask, opened from the headers of their files
    alone (see open_arrays), before their values are read (see read).

    Attributes
    ----------
    data : OpenedArray
        The values.
    error, mask : OpenedArray or None
        The errors and the mask, if given.
    unit : AnyUnit or None
        The unit of the values, if known.
    beam : Beam or None
        The beam of the values, if they have one (see Arrays).
    error_kind, mask_kind : str
        What the errors are, and what the mask marks (see read_arrays).
    """
    data: OpenedArray
    error: OpenedArray | None
    mask: OpenedArray | None
    unit: gbkfit.units.AnyUnit | None
    beam: Beam | None
    error_kind: ErrorKind
    mask_kind: MaskKind
    # (the factor of the errors to the unit of their kind of that of the
    # values, see _error_scale)
    error_scale: float = dataclasses.field(repr=False)

    def header(self) -> astropy.io.fits.Header:
        """
        Return the header of the values, in the orientation of the model
        (see model_header).
        """
        return self.data.header

    def shape(self) -> tuple[int, ...]:
        """Return the shape of the values as they are read (numpy order)."""
        return self.data.shape()

    def read(self) -> Arrays:
        """
        Read the values, their errors and mask (see read_arrays).

        Raises
        ------
        ValueError
            If a file changed since it was opened or cannot be read, or the
            errors are negative.
        """
        values = self.data.read()
        errors = None
        if self.error is not None:
            errors = self.error.read()
            if self.error_scale != 1:
                errors = errors * self.error_scale
            errors = _standard_deviations(
                split_data_file(self.error.file)[0], errors, self.error_kind)
        masks = None if self.mask is None else self.mask.read()
        valid = np.isfinite(values)
        if errors is not None:
            valid &= np.isfinite(errors) & (errors > 0)
        if masks is not None:
            marked = masks != 0
            valid &= np.isfinite(masks) & (
                marked if self.mask_kind == 'valid' else ~marked)
        return Arrays(
            np.where(valid, values, np.nan),
            None if errors is None else np.where(valid, errors, np.nan),
            valid.astype(float), self.data.header, self.unit, self.beam,
            self.data.orientation)


def read_data(
        file: DataFile,
        *,
        rpix: float | Sequence[float | None] | None = None,
        rval: float | Sequence[float | None] | None = None,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
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
        The type of a spectral axis whose header gives one that FITS does
        not define, or none (see model_header).
    spectral_unit : str, optional
        The unit of a spectral axis whose header gives none or one that
        is not valid (see SpectralAxis, option unit); it is not used for
        others.
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
    opened, coords = open_data(
        file, rpix=rpix, rval=rval, rest=rest, velocity=velocity,
        spectral_type=spectral_type, spectral_unit=spectral_unit,
        medium=medium, spectral_axis=spectral_axis)
    return opened.read(floats=False), coords


def open_data(
        file: DataFile,
        *,
        rpix: float | Sequence[float | None] | None = None,
        rval: float | Sequence[float | None] | None = None,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> tuple[OpenedArray, Coords]:
    """
    Open the data of a file from its header alone, as read_data reads
    them (the same options), with their world coordinates, to read them
    after (see OpenedArray.read).

    Returns
    -------
    OpenedArray
        The data, opened.
    Coords
        Their world coordinates.

    Raises
    ------
    ValueError
        If the file has no such data, or their header has invalid world
        coordinates or ones the model cannot represent (see read_data).
    """
    filename = split_data_file(file)[0]
    stamp = _Stamp.of(filename)
    header = read_header(file)
    shape = header_shape(header)
    model, orientation = model_header(
        filename, header, shape, rest=rest, velocity=velocity,
        spectral_type=spectral_type, spectral_unit=spectral_unit,
        medium=medium,
        spectral_axis=spectral_axis)
    _warn_of_missing_unit(
        filename, header, shape, spectral_type, spectral_unit)
    coords = coords_from_header(
        filename, model, rpix=rpix, rval=rval, spectral_axis=spectral_axis,
        medium=medium, rest=rest)
    return OpenedArray(file, model, orientation, None, stamp), coords


def read_oriented(
        file: DataFile,
        *,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        medium: str | None = None,
        unit: str | None = None
) -> tuple[np.ndarray, astropy.wcs.WCS | None]:
    """
    Read the array of a file in the orientation of the model (see
    model_header), without its axes of length 1, with its world
    coordinates in that orientation, to look at it: its spectral axis is
    read as read_data reads it, of what is given of it, but unlike
    read_data, its coordinates need not be ones the model can represent
    (e.g. of wavelengths without a rest, or of velocities whose convention
    the header does not state and that is not given).

    Parameters
    ----------
    file : DataFile
        The file of the array, and where it is in it (see read_image).
    rest, velocity, spectral_type, spectral_unit, medium : optional
        As read_data: each, if given, is how the spectral axis is read (a
        rest, its velocities; see model_header); an array without a
        spectral axis is read as it is.
    unit : str, optional
        The unit of the values, as read_arrays takes it, which replaces
        that of the header: it must be a unit (see data_unit); the values
        are as they are.

    Returns
    -------
    np.ndarray
        The array; as it is in the file (without its axes of length 1) if
        its header has no world coordinates that astropy reads and no rest
        is given.
    astropy.wcs.WCS or None
        Its world coordinates, if any.

    Raises
    ------
    ValueError
        If the file has no such array (see read_image), or what is given
        of its spectral axis or of its unit is not valid or does not apply
        to it (see model_header and data_unit).
    """
    filename = split_data_file(file)[0]
    data, header = read_image(file)
    if unit is not None:
        data_unit(filename, header, unit)
    header = _stated_header(
        filename, header, data.shape, spectral_type, velocity,
        spectral_unit, unstated='keep')
    try:
        # (the fixes of astropy to the header are not of the orientation)
        with ignoring_wcs_fixes():
            header, orientation = _oriented_header(
                filename, header, data.shape, rest, medium)
            wcs = astropy.wcs.WCS(header) if has_wcs(header) else None
    except ValueError:
        # (world coordinates that astropy cannot read, or that cannot be
        # put in the orientation of the model, leave the array as it is;
        # the errors of astropy's WCS are ValueErrors) but of a rest, the
        # velocities asked for cannot be read
        if rest is not None:
            raise
        return data.reshape(_squeezed_shape(data.shape)), None
    return orientation.apply(data), wcs


def read_arrays(
        data: DataFile,
        error: DataFile | None = None,
        mask: DataFile | None = None,
        *,
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid',
        unit: str | None = None,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None,
        stated: Collection[str] = ()
) -> Arrays:
    """
    Read data, and their errors and mask, from files in the orientation of
    the model.

    Each file is read by read_image (see DataFile) and put in the
    orientation of the model by its own world coordinates (see
    model_header). The errors and the mask follow the axes of the file of
    the data, whose shape they must have, if they or the data have no
    world coordinates; else theirs must put their pixels on those of the
    data.

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
    error_kind : str, optional
        What the errors are: 'sigma' (standard deviations), 'variance'
        or 'inverse_variance'.
    mask_kind : str, optional
        What the mask marks with values other than 0: 'valid' (by
        default) or 'invalid' values (see _MASK_KINDS).
    unit : str, optional
        The unit of the values, as gbkfit.units.parse_unit reads it; by
        default, that of their header (BUNIT), if it is a unit.
    rest, velocity, spectral_type, spectral_unit, medium, spectral_axis :
            optional
        As read_data, for the headers of all the files (the spectral axis
        is not checked: see coords_from_header).
    stated : Collection of str, optional
        The options of the grid of a dataset that it states (e.g. 'step'
        or 'spectral.rval'), which the header of the values need not have
        (see reference_errors).

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
        Of the header of the values, the errors of what it needs stated
        are raised together (see statement_errors; an ExceptionGroup of
        several).
    """
    return open_arrays(
        data, error, mask, error_kind=error_kind, mask_kind=mask_kind,
        unit=unit, rest=rest, velocity=velocity, spectral_type=spectral_type,
        spectral_unit=spectral_unit, medium=medium,
        spectral_axis=spectral_axis, stated=stated).read()


def open_arrays(
        data: DataFile,
        error: DataFile | None = None,
        mask: DataFile | None = None,
        *,
        error_kind: ErrorKind = 'sigma',
        mask_kind: MaskKind = 'valid',
        unit: str | None = None,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None,
        stated: Collection[str] = ()
) -> OpenedArrays:
    """
    Open data, and their errors and mask, from the headers of their files
    alone, as read_arrays reads them (the same options): every check of
    what the headers say is made here, and the values are read after (see
    OpenedArrays.read), but for the checks of the values themselves (e.g.
    errors that are negative).

    Returns
    -------
    OpenedArrays
        The arrays, opened.

    Raises
    ------
    ValueError
        As read_arrays, of what the headers say.
    """
    if error_kind not in _ERROR_KINDS:
        raise InputError(
            f"the kind of the errors must be one of {list(_ERROR_KINDS)}; "
            f"it is {error_kind!r}")
    if mask_kind not in _MASK_KINDS:
        raise InputError(
            f"the kind of the mask must be one of {list(_MASK_KINDS)}; it "
            f"is {mask_kind!r}")
    read = dict(
        rest=rest, velocity=velocity, spectral_type=spectral_type,
        spectral_unit=spectral_unit,
        medium=medium, spectral_axis=spectral_axis)
    values, beam = _open(data, **read, unit=unit, stated=stated)
    given = unit is not None
    unit = data_unit(split_data_file(data)[0], values.header, unit)
    errors, scale = None, 1.0
    if error is not None:
        errors = _open_companion(
            error, values.header, values.orientation, name='errors',
            path=['error'], **read)
        error_name = split_data_file(error)[0]
        scale = _error_scale(
            error_name, error_kind,
            _parsed_unit(error_name, read_header(error), given), unit)
    masks = None
    if mask is not None:
        masks = _open_companion(
            mask, values.header, values.orientation, name='mask',
            path=['mask'], **read)
    return OpenedArrays(
        values, errors, masks, unit, beam, error_kind, mask_kind, scale)


def read_footprint(
        file: DataFile,
        data: DataFile,
        *,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> np.ndarray:
    """
    Read where the instrument looked (a footprint: 1 inside, 0 outside)
    on the pixels of data read from a file, in the orientation of the
    model, as their errors and mask are read (see read_arrays): inside
    where the file is finite and not 0 (e.g. the number of fibres on each
    pixel, as CALIFA gives it), outside elsewhere; of the shape of the
    data, or, of a cube (two axes on the sky and a spectral axis), of its
    sky alone (an image of its sky, see read_sky_map), which is then that
    of every channel.

    Parameters
    ----------
    file : DataFile
        The file of the footprint.
    data : DataFile
        The file of the data.
    rest, velocity, spectral_type, spectral_unit, medium, spectral_axis :
            optional
        How the data are read (see read_arrays).

    Returns
    -------
    np.ndarray
        The footprint, of the shape of the data or of their sky.

    Raises
    ------
    ValueError
        If a file cannot be read, or the footprint is not on the pixels
        of the data or of their sky.
    """
    return open_footprint(
        file, data, rest=rest, velocity=velocity,
        spectral_type=spectral_type, spectral_unit=spectral_unit,
        medium=medium, spectral_axis=spectral_axis).read()


def open_footprint(
        file: DataFile,
        data: DataFile,
        *,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> OpenedFootprint:
    """
    Open a footprint from the headers of its file and of the file of its
    data alone, as read_footprint reads it (the same options), to read it
    after (see OpenedFootprint).

    Raises
    ------
    ValueError
        If a header cannot be read, or the footprint is not on the pixels
        of the data or of their sky.
    """
    read = dict(
        rest=rest, velocity=velocity, spectral_type=spectral_type,
        spectral_unit=spectral_unit, medium=medium,
        spectral_axis=spectral_axis)
    header, orientation = _data_model(data, read)
    with config_path('footprint'):
        target, target_orientation, target_read = _footprint_target(
            split_data_file(file)[0], header_shape(read_header(file)),
            header, orientation, read)
    return _open_companion(
        file, target, target_orientation, name='footprint',
        path=['footprint'], opened=OpenedFootprint, **target_read)


def check_companion(
        file: DataFile,
        data: DataFile,
        *,
        name: str,
        path: Sequence[str | int] = (),
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> None:
    """
    Check, from their headers alone, that the array of a file is on the
    pixels of data read from a file, as their errors and mask are read
    (see read_arrays): of their shape, from the types of the axes alone,
    and, once the header of the data can be read in the orientation of the
    model (see model_header; e.g. it states the convention of its
    velocities), of their world coordinates.

    Parameters
    ----------
    file : DataFile
        The file of the array.
    data : DataFile
        The file of the data.
    name : str
        What the array is to the data (e.g. 'errors'), in the message of
        an error.
    path : Sequence of str or int, optional
        The path of the option of its file, of the errors of it (see
        config_path).
    rest, velocity, spectral_type, spectral_unit, medium, spectral_axis :
            optional
        How the data are read (see read_arrays).

    Raises
    ------
    OSError
        If a file cannot be read.
    ValueError
        If the world coordinates of the data are invalid (see
        model_shape), or the array is not on their pixels.
    """
    read = dict(
        rest=rest, velocity=velocity, spectral_type=spectral_type,
        spectral_unit=spectral_unit, medium=medium,
        spectral_axis=spectral_axis)
    model, target = _data_target(data, read)
    filename = split_data_file(file)[0]
    with config_path(*path):
        file_header = read_header(file)
        shape = header_shape(file_header)
        if model is None:
            _check_shape(
                filename, file_header, shape, target, name=name,
                spectral_axis=spectral_axis)
        else:
            _companion_orientation(
                filename, file_header, shape, *model, name=name, **read)


def check_footprint(
        file: DataFile,
        data: DataFile,
        *,
        rest: Rest = None,
        velocity: str | None = None,
        spectral_type: str | None = None,
        spectral_unit: str | None = None,
        medium: str | None = None,
        spectral_axis: int | None = None
) -> None:
    """
    Check, from their headers alone, that a footprint is on the pixels of
    data read from a file, or of their sky, as it is read (see
    read_footprint): as check_companion checks an array.

    Parameters
    ----------
    file : DataFile
        The file of the footprint.
    data : DataFile
        The file of the data.
    rest, velocity, spectral_type, spectral_unit, medium, spectral_axis :
            optional
        How the data are read (see read_arrays).

    Raises
    ------
    OSError
        If a file cannot be read.
    ValueError
        If the world coordinates of the data are invalid (see
        model_shape), or the footprint is not on their pixels or on those
        of their sky (an error of the option footprint, see config_path).
    """
    read = dict(
        rest=rest, velocity=velocity, spectral_type=spectral_type,
        spectral_unit=spectral_unit, medium=medium,
        spectral_axis=spectral_axis)
    model, target = _data_target(data, read)
    filename = split_data_file(file)[0]
    with config_path('footprint'):
        file_header = read_header(file)
        shape = header_shape(file_header)
        if model is None:
            sky = _footprint_on_sky(shape, target.shape)
            _check_shape(
                filename, file_header, shape,
                target.sky() if sky else target, name='footprint',
                spectral_axis=None if sky else spectral_axis)
        else:
            on, on_orientation, on_read = _footprint_target(
                filename, shape, *model, read)
            _companion_orientation(
                filename, file_header, shape, on, on_orientation,
                name='footprint', **on_read)


def _data_model(
        data: DataFile, read: Mapping[str, Any]
) -> tuple[astropy.io.fits.Header, Orientation]:
    """
    Return the header of data read from a file in the orientation of the
    model, and how they are put in it (see model_header), from their
    header alone (read: how they are read, see read_arrays).
    """
    header = read_header(data)
    return model_header(
        split_data_file(data)[0], header, header_shape(header), **read)


class _Target(NamedTuple):
    """
    What an array is put on, the pixels of data or of their sky, as its
    shape is checked (see _check_shape): whether the data have world
    coordinates, and the shape (numpy order) of the target in the axes of
    the file of the data (of an array of no world coordinates, which is
    put on them as it is) and in the orientation of the model.
    """
    wcs: bool
    file_shape: tuple[int, ...]
    shape: tuple[int, ...]

    def sky(self) -> '_Target':
        """
        Return the sky of data of three axes: the first two of the model
        (see Orientation.sky).
        """
        return _Target(self.wcs, self.shape[-2:], self.shape[-2:])


def _data_target(
        data: DataFile, read: Mapping[str, Any]
) -> tuple[tuple[astropy.io.fits.Header, Orientation] | None, _Target]:
    """
    Return the header of data read from a file in the orientation of the
    model and how they are put in it (see model_header), or none while
    their header cannot be read so (e.g. it does not state the convention
    of its velocities: a problem of the data themselves, see
    describe_fits); and what an array put on their pixels is checked
    against, from the types of their axes alone (see _check_shape; read:
    how they are read, see read_arrays).

    Raises
    ------
    ValueError
        If the world coordinates of the data are invalid (see
        model_shape).
    """
    filename = split_data_file(data)[0]
    header = read_header(data)
    shape = header_shape(header)
    target = _Target(
        has_wcs(header), shape,
        model_shape(filename, header, shape, read['spectral_axis']))
    try:
        return model_header(filename, header, shape, **read), target
    except ValueError:
        return None, target


def _check_shape(
        filename: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        target: _Target,
        *,
        name: str,
        spectral_axis: int | None
) -> None:
    """
    Raise InputError unless an array of a header (of a shape, numpy order,
    in its file) has the shape of what it is put on, from the types of
    their axes alone: in the orientation of the model, of an array and
    data of world coordinates (see model_shape), else in the axes of
    their files (name: what it is to the data, e.g. 'errors').
    """
    if has_wcs(header) and target.wcs:
        found = model_shape(filename, header, shape, spectral_axis)
        wanted = target.shape
    else:
        found, wanted = tuple(shape), target.file_shape
    if found != wanted:
        raise InputError(_mismatch(name, found, wanted))


def _footprint_target(
        filename: str,
        shape: tuple[int, ...],
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation,
        read: Mapping[str, Any]
) -> tuple[astropy.io.fits.Header, Orientation, Mapping[str, Any]]:
    """
    Return what a footprint of a shape (numpy order, of its file) is put
    on, of data of a header and an orientation in the model (see
    read_footprint): the pixels of the data, or, of a cube, those of its
    sky (see DatasetGrid), as a header and an orientation, and how it is
    read; or raise InputError if it is of neither.
    """
    if _footprint_on_sky(shape, header_shape(data_header)):
        return *_sky_target(filename, data_header, data_orientation), \
            _NO_READING
    return data_header, data_orientation, read


def _footprint_on_sky(
        shape: tuple[int, ...], data_shape: tuple[int, ...]
) -> bool:
    """
    Return whether a footprint of a shape (numpy order, of its file) is
    put on the sky of data of a shape in the model (numpy order), not on
    their pixels: of a cube, a footprint of two axes is; raise InputError
    if it is of neither.
    """
    shape = _squeezed_shape(shape)
    if len(shape) == len(data_shape):
        return False
    if len(data_shape) == 3 and len(shape) == 2:
        return True
    raise InputError(
        _mismatch('footprint', shape, data_shape)
        + (f" or {shape_text(data_shape[1:])}"
           if len(data_shape) == 3 else ""))


def _mismatch(
        name: str, shape: tuple[int, ...], data_shape: tuple[int, ...]
) -> str:
    """
    Return the message of an array (e.g. 'errors') whose shape (numpy
    order) is not that of the data.
    """
    return (
        f"{name} {shape_text(shape)} · must match the data, "
        f"{shape_text(data_shape)}")


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
        raise InputError(
            "an image of the sky is read on the pixels of the file of the "
            "data, before they are cropped or padded")
    return _read_sky_image(file, data.header, data.orientation, plane)


def _read_sky_image(
        file: DataFile,
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation,
        plane: int | None = None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return an image of the sky of data (see read_sky_map) on the pixels of
    their sky, of their header and orientation in the model, and its
    header.
    """
    header, orientation = _sky_target(
        split_data_file(file)[0], data_header, data_orientation)
    return _read_companion(
        file, header, orientation, name='image', path=(), plane=plane,
        **_NO_READING)


def _sky_target(
        filename: str,
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation
) -> tuple[astropy.io.fits.Header, Orientation]:
    """
    Return the header and the orientation in the model of the sky of data
    of a header and an orientation in it, that an image of their sky (of a
    file, see read_sky_map) is put on.
    """
    orientation = data_orientation.sky()
    header = _with_shape(data_header, header_shape(data_header)[-2:])
    if has_wcs(data_header):
        wcs = parse_wcs(filename, data_header)
        if wcs.wcs.spec in (0, 1):
            raise InputError(
                f"{filename}: the data have no sky of two axes to put an "
                f"image of the sky on")
        header = _with_wcs(header, wcs.sub(2))
    return header, orientation


# How an image of the sky is read (see _sky_target): of no spectral axis
_NO_READING = dict(
    rest=None, velocity=None, spectral_type=None, spectral_unit=None,
    medium=None, spectral_axis=None)


def _open(
        file: DataFile,
        rest: Rest,
        velocity: str | None,
        spectral_type: str | None,
        spectral_unit: str | None,
        medium: str | None,
        spectral_axis: int | None,
        unit: str | None,
        stated: Collection[str]
) -> tuple[OpenedArray, Beam | None]:
    """
    Return the values of a file opened in the orientation of the model, of
    its header alone (see model_header), and their beam (see
    _oriented_beam). The errors of what their header needs stated, of the
    options given, are raised together, first (see statement_errors).
    """
    filename = split_data_file(file)[0]
    stamp = _Stamp.of(filename)
    header = read_header(file)
    shape = header_shape(header)
    raise_all(statement_errors(
        filename, header, shape, rest=rest, velocity=velocity,
        spectral_type=spectral_type, spectral_unit=spectral_unit,
        unit=unit, stated=stated), filename)
    model, orientation = model_header(
        filename, header, shape, rest=rest, velocity=velocity,
        spectral_type=spectral_type, spectral_unit=spectral_unit,
        medium=medium,
        spectral_axis=spectral_axis)
    _warn_of_missing_unit(
        filename, header, shape, spectral_type, spectral_unit)
    return (OpenedArray(file, model, orientation, None, stamp),
            _oriented_beam(filename, header, model, orientation))


def _read_companion(
        file: DataFile,
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation,
        *,
        name: str,
        path: Sequence[str | int],
        rest: Rest,
        velocity: str | None,
        spectral_type: str | None,
        spectral_unit: str | None,
        medium: str | None,
        spectral_axis: int | None,
        plane: int | None = None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the values of a file (of a plane of it, if given; see
    read_sky_map) on the pixels of data in the orientation of the model,
    and its header (see _open_companion).
    """
    opened = _open_companion(
        file, data_header, data_orientation, name=name, path=path, rest=rest,
        velocity=velocity, spectral_type=spectral_type,
        spectral_unit=spectral_unit, medium=medium,
        spectral_axis=spectral_axis, plane=plane)
    return opened.read(), opened.header


def _open_companion(
        file: DataFile,
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation,
        *,
        name: str,
        path: Sequence[str | int],
        rest: Rest,
        velocity: str | None,
        spectral_type: str | None,
        spectral_unit: str | None,
        medium: str | None,
        spectral_axis: int | None,
        plane: int | None = None,
        opened: type[OpenedArray] = OpenedArray
) -> OpenedArray:
    """
    Return the array of a file (of a plane of it, if given; see
    read_sky_map) opened on the pixels of data (their errors or mask, see
    read_arrays; their footprint, see read_footprint; or an image of their
    sky, see read_sky_map) in the orientation of the model, from the
    headers alone (see _companion_orientation; name: what it is to the
    data, in the message of an error; path: of the option of its file, of
    its errors, see config_path; opened: what it is opened as).
    """
    filename = split_data_file(file)[0]
    with config_path(*path):
        stamp = _Stamp.of(filename)
        header = read_header(file)
        if plane is not None:
            header = _plane(filename, header, plane)
        header, orientation = _companion_orientation(
            filename, header, header_shape(header), data_header,
            data_orientation, name=name, rest=rest, velocity=velocity,
            spectral_type=spectral_type, spectral_unit=spectral_unit,
            medium=medium, spectral_axis=spectral_axis)
    return opened(file, header, orientation, plane, stamp)


def _companion_orientation(
        filename: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        data_header: astropy.io.fits.Header,
        data_orientation: Orientation,
        *,
        name: str,
        **read: Any
) -> tuple[astropy.io.fits.Header, Orientation]:
    """
    Return the header of an array of a file (of a shape, numpy order)
    that is put on the pixels of data of a header and an orientation in
    the model, and how it is put on them, from the headers alone: by its
    own world coordinates, which must put its pixels on theirs, or, if it
    or the data have none, in the axes of the file of the data. Raise
    InputError if it is not on them: first if it is not of their shape
    (see _check_shape, as data whose header cannot yet be read in the
    orientation of the model are checked), then if its world coordinates
    are not on theirs (name: what it is to the data, e.g. 'errors'; read:
    how it is read, see read_arrays).
    """
    target = _Target(
        has_wcs(data_header), data_orientation.file_shape,
        header_shape(data_header))
    _check_shape(
        filename, header, shape, target, name=name,
        spectral_axis=read.get('spectral_axis'))
    if has_wcs(header) and has_wcs(data_header):
        model, orientation = model_header(filename, header, shape, **read)
        _check_same_pixels(model, data_header, name=name)
        return model, orientation
    return header, data_orientation


def _plane(
        source: str, header: astropy.io.fits.Header, plane: int
) -> astropy.io.fits.Header:
    """
    Return the header of a plane of an array of three axes of a header,
    along its last (FITS order), without that axis; raise InputError if it
    has no such plane.
    """
    shape = header_shape(header)
    if len(shape) != 3 or not 0 <= plane < shape[0]:
        raise InputError(
            f"{source}: plane {plane} is not one of its array of shape "
            f"{shape} (of planes along its first axis, numpy order)")
    header = _with_shape(header, shape[1:])
    if has_wcs(header):
        header = _with_wcs(header, parse_wcs(source, header).sub(2))
    return header


def _check_same_pixels(
        header: astropy.io.fits.Header,
        data_header: astropy.io.fits.Header,
        *,
        name: str
) -> None:
    """
    Raise InputError unless an array of a header in the orientation of
    the model is on the pixels of the data of another: of their shape,
    and their world coordinates at the corners (name: what it is to the
    data, e.g. 'errors').
    """
    shape, data_shape = header_shape(header), header_shape(data_header)
    if shape != data_shape:
        raise InputError(_mismatch(name, shape, data_shape))
    wcs, data_wcs = astropy.wcs.WCS(header), astropy.wcs.WCS(data_header)
    if list(wcs.wcs.ctype) != list(data_wcs.wcs.ctype):
        raise InputError(
            f"{name} on the axes {', '.join(wcs.wcs.ctype)} · the data on "
            f"{', '.join(data_wcs.wcs.ctype)}")
    corners = np.array(np.meshgrid(
        *[[0, size - 1] for size in shape[::-1]])).reshape(len(shape), -1)
    world = data_wcs.pixel_to_world_values(*corners)
    pixels = np.array(wcs.world_to_pixel_values(*world)).reshape(corners.shape)
    if not np.allclose(pixels, corners, rtol=0, atol=_GRID_TOLERANCE):
        raise InputError(
            f"{name} off the pixels of the data (by their world "
            f"coordinates)")


def _error_scale(
        source: str,
        kind: str,
        unit: gbkfit.units.AnyUnit | None,
        data_unit: gbkfit.units.AnyUnit | None
) -> float:
    """
    Return the factor of errors of a kind (see read_arrays) in a unit to
    the unit of their kind of that of the data (1 if either is not known);
    a unit of the errors that is not that of their kind is an error, which
    names the kind it is of, if any.
    """
    if unit is None or data_unit is None:
        return 1.0
    power = _ERROR_POWERS[kind]
    expected = data_unit ** power
    if not unit.is_equivalent(expected):
        kinds = [
            name for name, p in _ERROR_POWERS.items()
            if unit.is_equivalent(data_unit ** p)]
        hint = (
            f"; it is that of {kinds[0]} (error_kind: {kinds[0]})"
            if kinds else "")
        raise InputError(
            f"{source}: the unit of the errors "
            f"({gbkfit.units.format_unit(unit)}) is not that of {kind} of "
            f"the data ({gbkfit.units.format_unit(expected)}){hint}")
    return float(unit.to(expected))


def _standard_deviations(
        source: str, errors: np.ndarray, kind: str) -> np.ndarray:
    """
    Return errors of a kind (see read_arrays), in the unit of their kind of
    that of the data, as standard deviations; negative errors are an
    error.
    """
    with np.errstate(invalid='ignore'):
        negative = errors < 0
    if np.any(negative):
        raise InputError(
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
    Return the unit of the values of a header (BUNIT), if it has one (see
    data_unit), or None if it is not a unit and the unit of the data is
    given: the values are then taken to be in that unit, as for a header
    without BUNIT.
    """
    try:
        return data_unit(source, header)
    except InputError:
        if given:
            return None
        raise


def _floats(values: np.ndarray) -> np.ndarray:
    """Return values as floats (which can be NaN where invalid)."""
    if np.issubdtype(values.dtype, np.floating):
        return values
    return values.astype(float)
