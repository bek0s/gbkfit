
"""
Data items: measured values, with their mask and error.
"""

import os.path
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import astropy.units
import numpy as np

from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Data',
    'FitsFile',
    'dump_data',
    'fits_file'
]


# A FITS file: its filename, or its filename and the HDU to read (by
# default the first)
FitsFile: TypeAlias = str | tuple[str, int | str]


def _read(
        file: FitsFile,
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None,
        rest: str | astropy.units.Quantity | None = None,
        spectral_axis: int | None = None
) -> tuple[np.ndarray, gridutils.Coords]:
    """Read a FITS file and its world coordinates (see fitsutils.read_data)."""
    filename, hdu = (file, 0) if isinstance(file, str) else file
    return fitsutils.read_data(filename, hdu, rpix, rval, rest, spectral_axis)


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


def _as_float32(x: np.ndarray) -> np.ndarray:
    """
    Return a float32 copy of an array, in native byte order (the drivers
    support float32 only, and the copy leaves the caller's array
    unchanged).
    """
    return np.array(x, dtype=np.float32)


class Data:
    """
    Measured values with their mask and error: arrays of one shape, as
    float32 copies.

    Where the values were measured (e.g. on the pixels of a grid) is
    described by their dataset. The values that are masked (mask 0), not
    finite, or have an error that is not finite and positive are NaN, with
    mask 0.

    Parameters
    ----------
    data : ndarray
        The values.
    mask : ndarray, optional
        The mask: 0 for the values to leave out; by default, none.
    error : ndarray, optional
        The error of each value (its standard deviation).

    Raises
    ------
    ConfigError
        If the mask is not finite, or the arrays are not of one shape.
    """

    def __init__(
            self,
            data: np.ndarray,
            mask: np.ndarray | None = None,
            error: np.ndarray | None = None
    ):
        data = _as_float32(data)
        mask = np.ones_like(data) if mask is None else _as_float32(mask)
        if error is not None:
            error = _as_float32(error)
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
        # The total mask: the values that are finite, not masked, and
        # with a finite, positive error
        total_mask = np.isfinite(data) & (mask != 0)
        if error is not None:
            total_mask &= np.isfinite(error) & (error > 0)
        data[~total_mask] = np.nan
        if error is not None:
            error[~total_mask] = np.nan
        self._data = data
        self._mask = total_mask.astype(np.float32)
        self._error = error

    @classmethod
    def from_files(
            cls,
            data: FitsFile,
            mask: FitsFile | None = None,
            error: FitsFile | float | None = None,
            rpix: float | Sequence[float] | None = None,
            rval: float | Sequence[float] | None = None,
            rest: str | astropy.units.Quantity | None = None,
            spectral_axis: int | None = None
    ) -> tuple['Data', gridutils.Coords]:
        """
        Read a data item from FITS files.

        Parameters
        ----------
        data : str or tuple
            The file of the values: a filename, or a filename and the HDU.
        mask : str or tuple, optional
            The file of the mask.
        error : str or tuple or float, optional
            The file of the errors, or one error for all the values.
        rpix, rval, rest, spectral_axis : optional
            Passed to fitsutils.read_data for the file of the values.

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
        values, coords = _read(data, rpix, rval, rest, spectral_axis)
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
        return cls(values, masks, errors), coords

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
        """Return the dtype of the arrays (float32)."""
        return self._data.dtype


def dump_data(
        data: Data,
        filenames: Mapping[str, str],
        write: Callable[[str, np.ndarray], None],
        dump_path: bool = True
) -> dict[str, Any]:
    """
    Write the arrays of a data item to files, and return its
    configuration: the file of each array (see Data.from_files).

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
    return info
