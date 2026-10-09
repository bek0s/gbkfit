
import os.path
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'Data',
    'dump_data',
    'load_data'
]


def _read_file(
        x, prefix, rpix=None, rval=None, rest=None, spectral_axis=None):
    """
    The data of a file and its world coordinates (see fitsutils.Coords).
    x is a filename, or a dict with the filename ('file') and the HDU to
    read ('hdu', e.g. 'SCI'; by default the first).
    """
    if isinstance(x, str):
        x = dict(file=x)
    options = parseutils.parse_options(
        x, 'data file', required={'file'}, optional={'hdu'})
    return fitsutils.read_data(
        prefix + options['file'], options.get('hdu', 0), rpix, rval, rest,
        spectral_axis)


def _as_float32(x):
    """
    A float32 copy of an array, in native byte order. The drivers support
    float32 only, and the copy leaves the caller's array unchanged.
    """
    return np.array(x, dtype=np.float32)


class Data:
    """
    Measured values with their mask and error: arrays of one shape, as
    float32. Where the values were measured (e.g. on the pixels of a grid)
    is described by their dataset.
    """

    def __init__(
            self,
            data: np.ndarray,
            mask: np.ndarray | None = None,
            error: np.ndarray | None = None
    ):
        """
        The values that are masked (mask 0), not finite, or have an error
        that is not finite and positive are NaN, with mask 0. Without a
        mask, every value is measured.
        """
        data = _as_float32(data)
        mask = np.ones_like(data) if mask is None else _as_float32(mask)
        if error is not None:
            error = _as_float32(error)
        # Ensure mask contains only finite values
        if np.any(~np.isfinite(mask)):
            raise RuntimeError("mask contains non-finite values")
        # Validate shapes
        if data.shape != mask.shape:
            raise RuntimeError(
                f"data and mask have incompatible shapes "
                f"({data.shape} != {mask.shape})")
        if error is not None and data.shape != error.shape:
            raise RuntimeError(
                f"data and error have incompatible shapes "
                f"({data.shape} != {error.shape})")
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
        return self._data.ndim

    def npix(self) -> int:
        return self._data.size

    def shape(self) -> tuple[int, ...]:
        """The shape of the arrays (numpy order)."""
        return self._data.shape

    def data(self) -> np.ndarray:
        return self._data

    def mask(self) -> np.ndarray:
        return self._mask

    def error(self) -> np.ndarray | None:
        return self._error

    def dtype(self) -> np.dtype:
        return self._data.dtype


def load_data(
        info: dict[str, Any],
        prefix: str = '',
        rpix: Any = None,
        rval: Any = None,
        rest: Any = None,
        spectral_axis: int | None = None
) -> tuple[Data, fitsutils.Coords]:
    """
    A data item from files, and the world coordinates of its data file
    (see fitsutils.read_data, which also explains rpix, rval, rest and
    spectral_axis). info has
    the data file and, optionally, the mask file and the error file or a
    scalar error. A file is a filename, or a dict with the filename
    ('file') and the HDU ('hdu'). prefix is prepended to the filenames.
    """
    parseutils.parse_options(
        info, 'data', required={'data'}, optional={'mask', 'error'})
    data_d, coords = parseutils.load_option(
        lambda x: _read_file(x, prefix, rpix, rval, rest, spectral_axis),
        info, 'data', True, False)
    data_m = None
    data_e = None
    if (mask := info.get('mask')) is not None:
        with parseutils.config_path('mask'):
            data_m = _read_file(mask, prefix)[0]
    if (error := info.get('error')) is not None:
        if isinstance(error, (int, float)):
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
    Write the arrays of a data item, and return its info (see load_data).
    filenames has a filename for each array to write ('data', 'mask',
    'error'), and write(filename, array) writes an array (e.g. with the
    world coordinates of its dataset). Without dump_path, the info has the
    filenames without their directories.
    """
    arrays = dict(data=data.data(), mask=data.mask(), error=data.error())
    info = {}
    for key, filename in filenames.items():
        if arrays[key] is not None:
            write(filename, arrays[key])
            info[key] = filename if dump_path else os.path.basename(filename)
    return info
