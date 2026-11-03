"""
Reading data from files with their errors and mask, in the orientation of
the model: the one reader of the datasets (see dataset.Data.from_files)
and of prep (see prep.FitsArrays).

The files are FITS files, or ASDF files, whose world coordinates (a gwcs)
become those of a FITS header (see fitsutils.linear_header): past that,
both are read alike.
"""

import dataclasses
import typing

import asdf
import astropy.io.fits
import astropy.units
import astropy.wcs
import gwcs
import numpy as np

from gbkfit.utils import fitsutils, parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from gbkfit.utils.typeutils import DataFile, is_asdf_file


__all__ = [
    'ERROR_KINDS',
    'Arrays',
    'ErrorKind',
    'read_arrays'
]


# What an error file holds: standard deviations, variances or inverse
# variances, and the power of the unit of the data that each is in
ErrorKind: typing.TypeAlias = typing.Literal[
    'sigma', 'variance', 'inverse_variance']
ERROR_KINDS = typing.get_args(ErrorKind)
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
        fitsutils.model_header).
    unit : astropy.units.UnitBase or None
        The unit of the values, if known.
    beam : dict or None
        The beam of the values, if they have one (see
        fitsutils.header_beam): of a cube of CASA, that of each of its
        channels, in the order of the channels of the values.
    footprint : np.ndarray or None
        Where the instrument looked, if given (see Data): as the file
        has it.
    """
    data: np.ndarray
    error: np.ndarray | None
    mask: np.ndarray
    header: astropy.io.fits.Header
    unit: astropy.units.UnitBase | None
    beam: dict[str, typing.Any] | None = None
    footprint: np.ndarray | None = None


def read_arrays(
        data: DataFile,
        error: DataFile | None = None,
        mask: DataFile | None = None,
        *,
        error_kind: ErrorKind = 'sigma',
        unit: str | astropy.units.UnitBase | None = None,
        rest: str | astropy.units.Quantity | None = None,
        velocity: str | None = None,
        footprint: DataFile | None = None
) -> Arrays:
    """
    Read data, and their errors, mask and footprint, from files in the
    orientation of the model.

    A file is a FITS file and its HDU, or, by default, its first HDU with
    data (see fitsutils.read_hdu); or an ASDF file and the path of its
    array in its tree (e.g. 'roman/data'), whose unit is that of a
    Quantity, and world coordinates those of the WCS of the file, if it
    has one (see _read_asdf). Each is put in the orientation of the
    model by its own world coordinates (see fitsutils.model_header). The
    errors, the mask and the footprint follow the axes of the file of the
    data, whose shape they must have, if they or the data have no world
    coordinates; else theirs must put their pixels on those of the data.

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
    each channel of a cube of CASA (see fitsutils.header_beam), in the
    order of the channels of the values (reversed with the spectral axis,
    e.g. of decreasing frequency).

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
        frequencies into velocities (see fitsutils.model_header).
    velocity : str, optional
        The convention of the velocities of a spectral axis that the
        headers leave ambiguous: 'radio' or 'optical'.
    footprint : DataFile, optional
        The file of where the instrument looked (see Data).

    Returns
    -------
    Arrays
        The arrays.

    Raises
    ------
    ConfigError
        If a file cannot be read, or has world coordinates the model
        cannot take (see fitsutils.model_header); error_kind is not one of
        ERROR_KINDS, the errors are negative, or their unit is not that of
        their kind; the errors or the mask are not on the pixels of the
        data; or the beams of a cube are not one for each channel.
    """
    if error_kind not in ERROR_KINDS:
        raise ConfigError(
            f"the kind of the errors must be one of {list(ERROR_KINDS)}; "
            f"it is {error_kind!r}")
    values, header, orientation, beam = _read(data, rest, velocity)
    if isinstance(unit, str):
        unit = unitutils.parse_unit(unit)
    unit = unit or _header_unit(_filename(data), header)
    errors = None
    if error is not None:
        errors, error_header = _read_companion(
            error, header, orientation, rest, velocity)
        errors = _standard_deviations(
            _filename(error), errors, error_kind,
            _header_unit(_filename(error), error_header), unit)
    masks = None
    if mask is not None:
        masks, _ = _read_companion(mask, header, orientation, rest, velocity)
    valid = np.isfinite(values)
    if errors is not None:
        valid &= np.isfinite(errors) & (errors > 0)
    if masks is not None:
        valid &= np.isfinite(masks) & (masks != 0)
    footprints = None
    if footprint is not None:
        footprints, _ = _read_companion(
            footprint, header, orientation, rest, velocity)
    return Arrays(
        np.where(valid, values, np.nan),
        None if errors is None else np.where(valid, errors, np.nan),
        valid.astype(float), header, unit, beam, footprints)


def _filename(file: DataFile) -> str:
    """Return the filename of a file."""
    return file if isinstance(file, str) else file[0]


def _read(
        file: DataFile,
        rest: typing.Any,
        velocity: str | None
) -> tuple[
        np.ndarray, astropy.io.fits.Header, fitsutils.Orientation,
        dict[str, typing.Any] | None]:
    """
    Return the values of a file in the orientation of the model, their
    header in it, how they were put in it (see fitsutils.model_header),
    and their beam (see _beam).
    """
    filename = _filename(file)
    values, header = _read_file(file)
    model, orientation = fitsutils.model_header(
        filename, header, values.shape, rest, velocity)
    return (_floats(orientation.apply(values)), model, orientation,
            _beam(filename, header, model, orientation))


def _beam(
        filename: str,
        header: astropy.io.fits.Header,
        model: astropy.io.fits.Header,
        orientation: fitsutils.Orientation
) -> dict[str, typing.Any] | None:
    """
    Return the beam of the values of a file of the given header (see
    fitsutils.header_beam; the header of an ASDF file has none), the
    beams of the channels of a cube of CASA in the order of its channels
    in the orientation of the model (of the header model).

    Raises
    ------
    ConfigError
        If the beams of a cube are not one for each channel.
    """
    beam = fitsutils.header_beam(
        header, None if is_asdf_file(filename) else filename)
    if beam is None or 'spectral' not in beam:
        return beam
    # A cube with a table BEAMS has a spectral axis (see casa_beam_table).
    # The axes that the orientation reverses are those of the model, where
    # the spectral axis can be at another index than in the file (e.g. of
    # RA, Dec, Stokes and frequency, once the Stokes axis of 1 is dropped).
    axis = astropy.wcs.WCS(model).wcs.spec
    nchannels = int(model[f'NAXIS{axis + 1}'])
    if len(beam['bmaj']) != nchannels:
        raise ConfigError(
            f"{filename}: the cube has {nchannels} channels, but its table "
            f"BEAMS has beams for {len(beam['bmaj'])}")
    order = slice(None, None, -1) if axis in orientation.flips \
        else slice(None)
    return {key: value[order] for key, value in beam.items()}


def _read_file(file: DataFile) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the array of a file and its header: of a FITS file, or a FITS
    header of an ASDF file (see _read_asdf).
    """
    filename, key = (file, None) if isinstance(file, str) else file
    if is_asdf_file(filename):
        return _read_asdf(filename, key)
    return fitsutils.read_hdu(filename, key)


def _read_asdf(
        filename: str, path: str | None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the array of an ASDF file at a path of its tree (e.g.
    'roman/data'), and a FITS header of its unit (BUNIT, of a Quantity)
    and of the world coordinates of the only WCS of the file (a gwcs), if
    it has one (see fitsutils.linear_header).

    Raises
    ------
    ConfigError
        If the path is not given or not in the tree, or the file has more
        than one WCS.
    """
    if path is None:
        raise ConfigError(
            f"{filename}: give the path of the array in the tree of the "
            f"file (path, e.g. roman/data)")
    with asdf.open(filename) as file:
        node = file.tree
        for key in str(path).split('/'):
            try:
                node = node[key]
            except (KeyError, IndexError, TypeError) as e:
                raise ConfigError(
                    f"{filename}: its tree has no {path!r}") from e
        unit = getattr(node, 'unit', None)
        values = np.array(getattr(node, 'value', node))
        wcss = list(_instances(file.tree, gwcs.WCS))
    header = astropy.io.fits.Header()
    if unit is not None:
        header['BUNIT'] = unitutils.format_unit(unit)
    if len(wcss) > 1:
        raise ConfigError(
            f"{filename}: its tree has {len(wcss)} WCSs; files of one are "
            f"read")
    if wcss:
        header.update(fitsutils.linear_header(filename, wcss[0], values.shape))
    return values, header


def _instances(node: typing.Any, cls: type) -> typing.Iterator[typing.Any]:
    """Return the objects of a class in a tree of dicts and lists."""
    if isinstance(node, cls):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _instances(value, cls)
    elif isinstance(node, list | tuple):
        for value in node:
            yield from _instances(value, cls)


def _read_companion(
        file: DataFile,
        data_header: astropy.io.fits.Header,
        data_orientation: fitsutils.Orientation,
        rest: typing.Any,
        velocity: str | None
) -> tuple[np.ndarray, astropy.io.fits.Header]:
    """
    Return the values of the errors or the mask of data in the orientation
    of the model, and their header (see read_arrays).
    """
    filename = _filename(file)
    values, header = _read_file(file)
    if fitsutils.has_wcs(header) and fitsutils.has_wcs(data_header):
        header, orientation = fitsutils.model_header(
            filename, header, values.shape, rest, velocity)
        values = orientation.apply(values)
        _check_same_pixels(filename, header, values.shape, data_header)
        return _floats(values), header
    shape = data_orientation.file_shape
    if values.shape != shape:
        raise ConfigError(
            f"{filename}: without world coordinates (or with data without "
            f"them), its values must be of the shape of the data, {shape}, "
            f"in the axes of their file; they are of {values.shape}")
    return _floats(data_orientation.apply(values)), header


def _check_same_pixels(
        source: str,
        header: astropy.io.fits.Header,
        shape: tuple[int, ...],
        data_header: astropy.io.fits.Header
) -> None:
    """
    Raise ConfigError unless the values of a header in the orientation of
    the model (of the given shape) are on the pixels of the data of
    another: of their shape, and their world coordinates at the corners.
    """
    data_shape = fitsutils.header_shape(data_header)
    if shape != data_shape:
        raise ConfigError(
            f"{source}: its values must be of the shape of the data, "
            f"{data_shape} (in the orientation of the model); they are of "
            f"{shape}")
    wcs, data_wcs = astropy.wcs.WCS(header), astropy.wcs.WCS(data_header)
    if list(wcs.wcs.ctype) != list(data_wcs.wcs.ctype):
        raise ConfigError(
            f"{source}: its world coordinates are of the axes "
            f"{list(wcs.wcs.ctype)}, those of the data of "
            f"{list(data_wcs.wcs.ctype)}")
    corners = np.array(np.meshgrid(
        *[[0, size - 1] for size in shape[::-1]])).reshape(len(shape), -1)
    world = data_wcs.pixel_to_world_values(*corners)
    pixels = np.array(wcs.world_to_pixel_values(*world)).reshape(corners.shape)
    if not np.allclose(pixels, corners, rtol=0, atol=_GRID_TOLERANCE):
        raise ConfigError(
            f"{source}: its world coordinates do not put its pixels on "
            f"those of the data")


def _standard_deviations(
        source: str,
        errors: np.ndarray,
        kind: str,
        unit: astropy.units.UnitBase | None,
        data_unit: astropy.units.UnitBase | None
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
            raise ConfigError(
                f"{source}: the unit of the errors "
                f"({unitutils.format_unit(unit)}) is not that of {kind} of "
                f"the data ({unitutils.format_unit(expected)}){hint}")
        errors = errors * unit.to(expected)
    with np.errstate(invalid='ignore'):
        negative = errors < 0
    if np.any(negative):
        raise ConfigError(
            f"{source}: the errors cannot be negative; "
            f"{np.count_nonzero(negative)} are, down to {np.nanmin(errors):g}")
    if kind == 'variance':
        return np.sqrt(errors)
    if kind == 'inverse_variance':
        with np.errstate(divide='ignore'):
            return np.where(errors > 0, 1 / np.sqrt(errors), np.nan)
    return errors


def _header_unit(
        source: str, header: astropy.io.fits.Header
) -> astropy.units.UnitBase | None:
    """
    Return the unit of the values of a header (BUNIT), if it has one that
    is a unit (see unitutils.parse_unit); one that is not is left out,
    with a warning (see parseutils.warn).
    """
    text = str(header.get('BUNIT', '')).strip()
    if not text:
        return None
    try:
        return unitutils.parse_unit(text)
    except ConfigError as e:
        parseutils.warn(f"{source}: BUNIT is left out: {e}")
        return None


def _floats(values: np.ndarray) -> np.ndarray:
    """Return values as floats (which can be NaN where invalid)."""
    if np.issubdtype(values.dtype, np.floating):
        return values
    return values.astype(float)
