
import logging
import os.path
from collections.abc import Sequence
from numbers import Real
from typing import Any

import astropy.wcs
import numpy as np

from gbkfit.utils import fitsutils, miscutils, parseutils


__all__ = [
    'Data',
    'data_parser'
]


_log = logging.getLogger(__name__)


def _make_filename(filename, dump_path):
    return filename if dump_path else os.path.basename(filename)


def _ensure_floating_or_float32(x, label):
    result = x
    if not np.issubdtype(x.dtype, np.floating):
        _log.warning(
            f"{label} array is not of floating type; will convert to float32")
        result = x.astype(np.float32)
    return result


class Data(parseutils.BasicSerializable):

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            step: Real = None,
            rpix: Real = None,
            rval: Real = None,
            rota: Real = None,
            prefix: str = ''
    ):
        desc = parseutils.make_basic_desc(cls, 'data')
        data_d, wcs_d = parseutils.load_option(
            lambda x: fitsutils.load_fits(prefix + x),
            info, 'data', True, False)
        data_m = None
        data_e = None
        if (mask := info.get('mask')) is not None:
            data_m = fitsutils.load_fits(prefix + mask)[0]
        if (error := info.get('error')) is not None:
            if isinstance(error, (int, float)):
                data_e = np.full_like(data_d, error)
            elif isinstance(error, str):
                data_e = fitsutils.load_fits(prefix + error)[0]
        # Local information has higher priority than global
        step = info.get('step', step)
        rpix = info.get('rpix', rpix)
        rval = info.get('rval', rval)
        rota = info.get('rota', rota)
        # If no information is provided, use fits header
        if step is None:
            step = wcs_d.wcs.cdelt.tolist()  # noqa
        if rpix is None:
            rpix = wcs_d.wcs.crpix.tolist()  # noqa
        if rval is None:
            rval = wcs_d.wcs.crval.tolist()  # noqa
        # todo: deal with rotation (PC Matrix and CROTA (deprecated))
        # Build class arguments dict
        info.update(dict(
            data=data_d,
            mask=data_m,
            error=data_e,
            step=step,
            rpix=rpix,
            rval=rval,
            rota=rota))
        # Parse options and create object
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

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
        # Some shortcuts
        dat = self.data()
        msk = self.mask()
        err = self.error()
        step = self.step()
        rpix = self.rpix()
        rval = self.rval()
        rota = self.rota()
        # Create WCS object
        wcs = astropy.wcs.WCS(naxis=dat.ndim, relax=False)
        wcs.wcs.cdelt = step    # noqa
        wcs.wcs.crpix = rpix    # noqa
        wcs.wcs.crval = rval    # noqa
        wcs.wcs.pc = np.identity(dat.ndim)  # noqa
        wcs.wcs.pc[0][0] = +np.cos(rota)    # noqa
        wcs.wcs.pc[0][1] = -np.sin(rota)    # noqa
        wcs.wcs.pc[1][0] = +np.sin(rota)    # noqa
        wcs.wcs.pc[1][1] = +np.cos(rota)    # noqa
        # Dump WCS as meta-data (if requested)
        if dump_wcs:
            info.update(dict(step=step, rpix=rpix, rval=rval, rota=rota))
        # Dump data
        info['data'] = filename_d = _make_filename(filename_d, dump_path)
        fitsutils.dump_fits(filename_d, dat, wcs, overwrite)
        if filename_m and msk is not None:
            info['mask'] = filename_m = _make_filename(filename_m, dump_path)
            fitsutils.dump_fits(filename_m, msk, wcs, overwrite)
        if filename_e and err is not None:
            info['error'] = filename_e = _make_filename(filename_e, dump_path)
            fitsutils.dump_fits(filename_e, err, wcs, overwrite)
        return info

    def __init__(
            self,
            data: np.ndarray,
            mask: np.ndarray | None = None,
            error: np.ndarray | None = None,
            step: Real | Sequence[Real] | None = None,
            rpix: Real | Sequence[Real] | None = None,
            rval: Real | Sequence[Real] | None = None,
            rota: Real | None = None
    ):
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
        # Convert to native byte order and ensure fp format.
        data = miscutils.to_native_byteorder(data)
        data = _ensure_floating_or_float32(data, 'data')
        mask = miscutils.to_native_byteorder(mask)
        mask = _ensure_floating_or_float32(mask, 'mask')
        if error is not None:
            error = miscutils.to_native_byteorder(error)
            error = _ensure_floating_or_float32(error, 'error')
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
        # Create and apply the "total mask"
        total_mask = np.isfinite(data) & (mask != 0)
        if error is not None:
            total_mask &= np.isfinite(error)
        # Apply total mask to data
        data[~total_mask] = np.nan
        if error is not None:
            error[~total_mask] = np.nan
        # Convert mask to the same dtype as data
        mask[:] = total_mask.astype(data.dtype)
        # Ensure mask and total_mask are identical
        if not np.array_equal(mask, total_mask):
            raise RuntimeError("impossible")
        del total_mask
        # Calculate the world coordinates at the very first pixel
        zero = (np.array(rval) - np.array(rpix) * np.array(step)).tolist()
        # Keep copies of the supplied data
        dtype = data.dtype
        self._data = data.astype(dtype)
        self._mask = mask.astype(dtype)
        self._error = error.astype(dtype) if error is not None else None
        self._step = tuple(step)
        self._zero = tuple(zero)
        self._rpix = tuple(rpix)
        self._rval = tuple(rval)
        self._rota = rota

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

    def data(self):
        return self._data

    def mask(self):
        return self._mask

    def error(self):
        return self._error

    def dtype(self):
        return self._data.dtype


data_parser = parseutils.BasicParser(Data)
