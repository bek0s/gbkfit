
"""
Data items: measured values, with their mask and error.
"""

import os.path
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Data',
    'dump_data',
    'load_data'
]


def _read_file(
        x: str | Mapping[str, Any],
        prefix: str,
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None,
        rest: str | astropy.units.Quantity | None = None,
        spectral_axis: int | None = None
) -> tuple[np.ndarray, gridutils.Coords]:
    """
    Read the data of a file option (a filename, or a dict with the
    filename and the HDU) and its world coordinates (see
    fitsutils.read_data).
    """
    file, hdu = parseutils.parse_file(x)
    return fitsutils.read_data(
        prefix + file, hdu, rpix, rval, rest, spectral_axis)


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


def load_data(
        info: dict[str, Any],
        prefix: str = '',
        rpix: float | Sequence[float] | None = None,
        rval: float | Sequence[float] | None = None,
        rest: str | astropy.units.Quantity | None = None,
        spectral_axis: int | None = None
) -> tuple[Data, gridutils.Coords]:
    """
    Load a data item from the configuration of its files.

    Parameters
    ----------
    info : dict
        The file of the data ('data'), and optionally that of the mask
        ('mask'), and that of the error or one error for all the values
        ('error'). A file is a filename, or a dict with the filename
        ('file') and the HDU ('hdu').
    prefix : str, optional
        Prepended to the filenames.
    rpix, rval, rest, spectral_axis : optional
        Passed to fitsutils.read_data for the data file.

    Returns
    -------
    tuple of Data and gridutils.Coords
        The data item, and the world coordinates of its data file.

    Raises
    ------
    ConfigError
        If the configuration is invalid, or the files cannot be read.
    """
    if not isinstance(info, Mapping):
        raise ConfigError(
            f"a data item has the file of its data ('data'), and optionally "
            f"those of its mask ('mask') and error ('error'); it is {info!r}")
    parseutils.parse_options(
        info, required={'data'}, optional={'mask', 'error'})
    data_d, coords = parseutils.load_option(
        lambda x: _read_file(x, prefix, rpix, rval, rest, spectral_axis),
        info, 'data', required=True)
    data_m = None
    data_e = None
    if (mask := info.get('mask')) is not None:
        with parseutils.config_path('mask'):
            data_m = _read_file(mask, prefix)[0]
    if (error := info.get('error')) is not None:
        if isinstance(error, bool):
            raise ConfigError(
                "the error must be a number or a file; it is a bool")
        if isinstance(error, (int, float)):
            # (an error that is not positive would mask every value)
            if not error > 0:
                raise ConfigError(f"the error must be positive; it is {error}")
            data_e = np.full(np.shape(data_d), error, dtype=float)
        else:
            with parseutils.config_path('error'):
                data_e = _read_file(error, prefix)[0]
    return Data(data_d, data_m, data_e), coords


def dump_data(
        data: Data,
        filenames: Mapping[str, str],
        write: Callable[[str, np.ndarray], None],
        dump_path: bool = True
) -> dict[str, Any]:
    """
    Write the arrays of a data item to files, and return its
    configuration (see load_data).

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
