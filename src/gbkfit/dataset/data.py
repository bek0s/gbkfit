
"""
Data items: measured values, with their mask and error.
"""

import os.path
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, TypeAlias

import astropy.units
import numpy as np

from gbkfit.utils import fitsutils, gridutils, parseutils, unitutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Beam',
    'Data',
    'FitsFile',
    'dump_data',
    'fits_file'
]


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
        """Return its solid angle (arcsec^2): pi / (4 ln 2) bmaj bmin."""
        return np.pi / (4 * np.log(2)) * np.multiply(self.bmaj, self.bmin)


# A FITS file: its filename, or its filename and the HDU to read (by
# default the first)
FitsFile: TypeAlias = str | tuple[str, int | str]


def _read(
        file: FitsFile,
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None,
        rest: str | astropy.units.Quantity | None = None,
        spectral_axis: int | None = None,
        medium: str | None = None
) -> tuple[np.ndarray, gridutils.Coords]:
    """Read a FITS file and its world coordinates (see fitsutils.read_data)."""
    filename, hdu = (file, 0) if isinstance(file, str) else file
    return fitsutils.read_data(
        filename, hdu, rpix, rval, rest, spectral_axis, medium)


def fits_file(x: str | Mapping[str, Any], prefix: str = '') -> FitsFile:
    """
    Return the FITS file of a file option.

    Parameters
    ----------
    x : str or Mapping
        A filename, or a dict with the filename ('file') and, optionally,
        the HDU ('hdu', e.g. 'SCI').
    prefix : str, optional
        Prepended to the filename.

    Returns
    -------
    tuple of str and (int or str)
        The filename and the HDU.

    Raises
    ------
    ConfigError
        If the option is invalid.
    """
    file, hdu = parseutils.parse_file(x)
    return prefix + file, hdu


def _as_float(x: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """
    Return a copy of an array of floats, in native byte order (the copy
    leaves the caller's array unchanged): of the given dtype, or else
    floats keep their type and other types become float64.
    """
    x = np.asarray(x)
    if dtype is None:
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) \
            else np.dtype(np.float64)
    return np.array(x, dtype=np.dtype(dtype).newbyteorder('='))


class Data:
    """
    Measured values with their mask and error: arrays of one shape, as
    copies (float arrays keep their type, and the others become float64;
    the mask and the error take the type of the values), and their unit
    and beam, if known.

    Where the values were measured (e.g. on the pixels of a grid) is
    described by their dataset. The values that are masked (mask 0), not
    finite, or have an error that is not finite or is 0 (e.g. blank
    pixels of an error map) are NaN, with mask 0.

    Parameters
    ----------
    data : ndarray
        The values.
    mask : ndarray, optional
        The mask: 0 for the values to leave out; by default, none.
    error : ndarray, optional
        The error of each value (its standard deviation).
    unit : str or astropy.units.UnitBase, optional
        The unit of the values and the errors (see unitutils.parse_unit).
    beam : Beam or Mapping, optional
        Their beam, for units per beam (e.g. Jy/beam): a Beam, or its
        options bmaj, bmin and bpa.

    Raises
    ------
    ConfigError
        If the mask is not finite, an error is negative, the arrays are
        not of one shape, the unit is not a unit, or the widths of the
        beam are not positive.
    """

    def __init__(
            self,
            data: np.ndarray,
            mask: np.ndarray | None = None,
            error: np.ndarray | None = None,
            unit: str | astropy.units.UnitBase | None = None,
            beam: Beam | Mapping[str, Any] | None = None
    ):
        data = _as_float(data)
        mask = np.ones_like(data) if mask is None \
            else _as_float(mask, data.dtype)
        if error is not None:
            error = _as_float(error, data.dtype)
        if np.any(~np.isfinite(mask)):
            raise ConfigError("the mask must be finite")
        if data.shape != mask.shape:
            raise ConfigError(
                f"the data and the mask must have one shape; their shapes "
                f"are {data.shape} and {mask.shape}")
        if error is not None and data.shape != error.shape:
            raise ConfigError(
                f"the data and the error must have one shape; their shapes "
                f"are {data.shape} and {error.shape}")
        if error is not None and np.any(error < 0):
            raise ConfigError(
                f"errors cannot be negative; {np.count_nonzero(error < 0)} "
                f"are, down to {np.nanmin(error):g}")
        # The total mask: the values that are finite, not masked, and
        # with a finite, positive error
        total_mask = np.isfinite(data) & (mask != 0)
        if error is not None:
            total_mask &= np.isfinite(error) & (error > 0)
        data[~total_mask] = np.nan
        if error is not None:
            error[~total_mask] = np.nan
        self._data = data
        self._mask = total_mask.astype(data.dtype)
        self._error = error
        self._unit = unitutils.parse_unit(unit) if isinstance(unit, str) \
            else unit
        self._beam = _make_beam(beam)

    @classmethod
    def from_files(
            cls,
            data: FitsFile,
            mask: FitsFile | None = None,
            error: FitsFile | float | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rest: str | astropy.units.Quantity | None = None,
            spectral_axis: int | None = None,
            medium: str | None = None,
            unit: str | None = None,
            beam: Mapping[str, Any] | None = None
    ) -> tuple['Data', gridutils.Coords]:
        """
        Read a data item from FITS files, with the unit and the beam of the
        header of the file of its values unless given (see
        fitsutils.read_unit and read_beam; a unit of the header that is
        not a unit is left out).

        Parameters
        ----------
        data : str or tuple
            The file of the values: a filename, or a filename and the HDU.
        mask : str or tuple, optional
            The file of the mask.
        error : str or tuple or float, optional
            The file of the errors, or one error for all the values.
        rpix, rval, rest, spectral_axis, medium : optional
            Passed to fitsutils.read_data for the file of the values.
        unit : str, optional
            The unit of the values (see Data).
        beam : Mapping, optional
            The options of their beam (see Beam).

        Returns
        -------
        tuple of Data and gridutils.Coords
            The data item, and the world coordinates of the file of its
            values.

        Raises
        ------
        ConfigError
            If the error is not positive, or the arrays are invalid (see
            Data).
        """
        values, coords = _read(data, rpix, rval, rest, spectral_axis, medium)
        masks = None if mask is None else _read(mask)[0]
        if error is None:
            errors = None
        elif isinstance(error, bool):
            raise ConfigError(
                "the error must be a number or a file; it is a bool")
        elif isinstance(error, (int, float)):
            # (an error that is not positive would mask every value)
            if not error > 0:
                raise ConfigError(f"the error must be positive; it is {error}")
            errors = np.full(values.shape, error, dtype=float)
        else:
            errors = _read(error)[0]
        filename, hdu = (data, 0) if isinstance(data, str) else data
        if unit is None:
            unit = _header_unit(filename, hdu)
        if beam is None:
            beam = fitsutils.read_beam(filename, hdu)
        return cls(values, masks, errors, unit, beam), coords

    def ndim(self) -> int:
        """Return the number of axes of the arrays."""
        return self._data.ndim

    def size(self) -> int:
        """Return the number of values."""
        return self._data.size

    def shape(self) -> tuple[int, ...]:
        """Return the shape of the arrays (numpy order)."""
        return self._data.shape

    def data(self) -> np.ndarray:
        """Return the values (NaN where left out)."""
        return self._data

    def mask(self) -> np.ndarray:
        """Return the mask: 1 for the values used, 0 for the others."""
        return self._mask

    def error(self) -> np.ndarray | None:
        """Return the errors, if any (NaN where left out)."""
        return self._error

    def dtype(self) -> np.dtype:
        """Return the dtype of the arrays."""
        return self._data.dtype

    def unit(self) -> astropy.units.UnitBase | None:
        """Return the unit of the values, if known."""
        return self._unit

    def beam(self) -> Beam | None:
        """Return the beam of the values, if known."""
        return self._beam

    def with_data(self, data: np.ndarray) -> 'Data':
        """
        Return a copy with other values (e.g. those of a bootstrap), of its
        dtype, and its mask, error, unit and beam: the values it masks
        stay masked.

        Raises
        ------
        ConfigError
            If the values are not of its shape.
        """
        return Data(
            np.asarray(data, self.dtype()), self._mask, self._error,
            self._unit, self._beam)


def _header_unit(
        filename: str, hdu: int | str
) -> astropy.units.UnitBase | None:
    """
    Return the unit of the header of a FITS file (BUNIT), if it has one
    that is a unit (see unitutils.parse_unit).
    """
    text = fitsutils.read_unit(filename, hdu)
    if text is None:
        return None
    try:
        return unitutils.parse_unit(text)
    except ConfigError:
        return None


def _make_beam(beam: Beam | Mapping[str, Any] | None) -> Beam | None:
    """
    Return a beam from itself or its options. Raise ConfigError if its
    widths are not positive.
    """
    if beam is None:
        return None
    if not isinstance(beam, Beam):
        beam = Beam(**parseutils.parse_options(
            beam, required={'bmaj', 'bmin'}, optional={'bpa', 'spectral'}))
    if not (np.all(np.asarray(beam.bmaj) > 0)
            and np.all(np.asarray(beam.bmin) > 0)):
        raise ConfigError(
            f"the widths of a beam must be positive; they are {beam.bmaj} "
            f"and {beam.bmin}")
    return beam


def dump_data(
        data: Data,
        filenames: Mapping[str, str],
        write: Callable[[str, np.ndarray], None],
        dump_path: bool = True
) -> dict[str, Any]:
    """
    Write the arrays of a data item to files, and return its
    configuration: the file of each array, and its unit and beam (one
    beam, not those of the channels of a cube; see Data.from_files).

    Parameters
    ----------
    data : Data
        The data item.
    filenames : Mapping
        The file of each array to write ('data', 'mask', 'error').
    write : Callable
        Writes an array to a file, as write(filename, array) (e.g. with
        the world coordinates of the dataset).
    dump_path : bool, optional
        Whether the configuration has the paths of the files, or only
        their names.

    Returns
    -------
    dict
        The configuration.
    """
    arrays = dict(data=data.data(), mask=data.mask(), error=data.error())
    info = {}
    for key, filename in filenames.items():
        if arrays[key] is not None:
            write(filename, arrays[key])
            info[key] = filename if dump_path else os.path.basename(filename)
    if data.unit() is not None:
        info['unit'] = unitutils.format_unit(data.unit())
    beam = data.beam()
    if beam is not None and beam.spectral is None:
        info['beam'] = dict(
            bmaj=float(beam.bmaj), bmin=float(beam.bmin), bpa=float(beam.bpa))
    elif beam is not None:
        parseutils.warn(
            "the beams of the channels of a data item are not dumped; give "
            "its beam in its configuration (beam: {bmaj, bmin, bpa})")
    return info
