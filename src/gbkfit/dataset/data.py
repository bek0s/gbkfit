
import os.path
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'Data',
    'data_parser'
]


def _make_filename(filename, dump_path):
    return filename if dump_path else os.path.basename(filename)


def _read_file(x, prefix, rpix=None, rval=None):
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
        prefix + options['file'], options.get('hdu', 0), rpix, rval)


def _as_float32(x):
    """
    A float32 copy of an array, in native byte order. The drivers support
    float32 only, and the copy leaves the caller's array unchanged.
    """
    return np.array(x, dtype=np.float32)


class Data(parseutils.BasicSerializable):
    """
    An array of data, with its mask and error, and the world coordinates
    of its pixels in the units of the model (see fitsutils.Coords). The
    spatial axes are measured from the reference pixel; the spectral
    axis (spectral_axis, if any) from its world value there.
    """

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            step: Real = None,
            rpix: Real = None,
            rval: Real = None,
            rota: Real = None,
            prefix: str = '',
            spectral_axis: int | None = None
    ):
        """
        Load data from files. step, rpix, rval and rota are the defaults
        of the options of the same name (e.g. those of the dataset), and
        spectral_axis the index of the spectral axis of the dataset.
        """
        desc = parseutils.make_basic_desc(cls, 'data')
        # Local information has higher priority than global. Without
        # either, the header has it.
        step = info.get('step', step)
        rpix = info.get('rpix', rpix)
        rval = info.get('rval', rval)
        rota = info.get('rota', rota)
        data_d, coords = parseutils.load_option(
            lambda x: _read_file(x, prefix, rpix, rval),
            info, 'data', True, False)
        data_m = None
        data_e = None
        if (mask := info.get('mask')) is not None:
            data_m = _read_file(mask, prefix)[0]
        if (error := info.get('error')) is not None:
            if isinstance(error, (int, float)):
                data_e = np.full(np.shape(data_d), error, dtype=float)
            else:
                data_e = _read_file(error, prefix)[0]
        info.update(dict(
            data=data_d,
            mask=data_m,
            error=data_e,
            step=coords.step if step is None else step,
            rpix=coords.rpix,
            rval=coords.rval,
            rota=coords.rota if rota is None else rota))
        opts = parseutils.parse_options_for_callable(
            info, desc, cls.__init__, fun_ignore_args=['spectral_axis'])
        return cls(**opts, spectral_axis=spectral_axis)

    def dump(
            self,
            filename_d: str,
            filename_m: str | None = None,
            filename_e: str | None = None,
            dump_wcs: bool = True,
            dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        info = dict()
        coords = fitsutils.Coords(
            self.step(), self.rpix(), self.rval(), self.rota())
        # Dump the world coordinates as options too (if requested)
        if dump_wcs:
            info.update(coords._asdict())
        files = dict(
            data=(filename_d, self.data()),
            mask=(filename_m, self.mask()),
            error=(filename_e, self.error()))
        for key, (filename, data) in files.items():
            if filename and data is not None:
                info[key] = filename = _make_filename(filename, dump_path)
                fitsutils.write_data(
                    filename, data, coords, self._spectral_axis, overwrite)
        return info

    def __init__(
            self,
            data: np.ndarray,
            mask: np.ndarray | None = None,
            error: np.ndarray | None = None,
            step: Real | Sequence[Real] | None = None,
            rpix: Real | Sequence[Real] | None = None,
            rval: Real | Sequence[Real] | None = None,
            rota: Real | None = None,
            spectral_axis: int | None = None
    ):
        """
        spectral_axis is the index of the spectral axis (in FITS order,
        e.g. 2 for a spectral cube), or None if there is none.
        """
        if spectral_axis is not None and not 0 <= spectral_axis < data.ndim:
            raise RuntimeError(
                f"spectral axis {spectral_axis} of data with {data.ndim} "
                f"axes")
        # If mask was not provided, use a default mask.
        if mask is None:
            mask = np.ones_like(data)
        if step is None:
            step = (1,) * data.ndim
        # By default, origin is at the center of the dataset.
        if rpix is None:
            rpix = tuple((np.asarray(data.shape[::-1]) / 2 - 0.5).tolist())
        if rval is None:
            rval = (0,) * data.ndim
        if rota is None:
            rota = 0
        if isinstance(step, Real):
            step = (step,) * data.ndim
        if isinstance(rpix, Real):
            rpix = (rpix,) * data.ndim
        if isinstance(rval, Real):
            rval = (rval,) * data.ndim
        data = _as_float32(data)
        mask = _as_float32(mask)
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
        if data.ndim != len(step):
            raise RuntimeError(
                f"data dimensionality and step length are incompatible "
                f"({data.ndim} != {len(step)})")
        if data.ndim != len(rpix):
            raise RuntimeError(
                f"data dimensionality and rpix length are incompatible "
                f"({data.ndim} != {len(rpix)})")
        if data.ndim != len(rval):
            raise RuntimeError(
                f"data dimensionality and rval length are incompatible "
                f"({data.ndim} != {len(rval)})")
        # The total mask: the pixels with a finite value, not masked, and
        # with a finite, positive error
        total_mask = np.isfinite(data) & (mask != 0)
        if error is not None:
            total_mask &= np.isfinite(error) & (error > 0)
        data[~total_mask] = np.nan
        if error is not None:
            error[~total_mask] = np.nan
        mask = total_mask.astype(np.float32)
        # The world coordinates of the first pixel: the spatial axes are
        # measured from the reference pixel, and the spectral axis from
        # its world value there
        zero = [
            (rval[axis] if axis == spectral_axis else 0)
            - rpix[axis] * step[axis]
            for axis in range(data.ndim)]
        self._data = data
        self._mask = mask
        self._error = error
        self._step = tuple(step)
        self._zero = tuple(zero)
        self._rpix = tuple(rpix)
        self._rval = tuple(rval)
        self._rota = rota
        self._spectral_axis = spectral_axis

    def ndim(self):
        return self._data.ndim

    def npix(self):
        return self._data.size

    def size(self):
        return self._data.shape[::-1]

    def step(self):
        return self._step

    def zero(self):
        return self._zero

    def rpix(self):
        return self._rpix

    def rval(self):
        return self._rval

    def rota(self):
        return self._rota

    def spectral_axis(self):
        return self._spectral_axis

    def data(self):
        return self._data

    def mask(self):
        return self._mask

    def error(self):
        return self._error

    def dtype(self):
        return self._data.dtype


data_parser = parseutils.BasicParser(Data)
