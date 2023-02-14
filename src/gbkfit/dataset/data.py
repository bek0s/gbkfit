
import logging
import os.path
from collections.abc import Sequence

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


class Data:

    @classmethod
    def load(cls, info, step=None, rpix=None, rval=None, rota=None, prefix=''):
        desc = parseutils.make_basic_desc(cls, 'data')
        data_d, wcs_d = fitsutils.load_fits(prefix + info['data'])
        data_m = None
        data_e = None
        if mask := info.get('mask', None):
            data_m = fitsutils.load_fits(prefix + mask)[0]
        if error := info.get('error', None):
            if isinstance(error, (int, float)):
                data_m = np.full_like(data_d, error)
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
            self, filename_d, filename_m=None, filename_e=None,
            dump_wcs=True, dump_path=True, overwrite=False):
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
            step: Sequence[int | float] | None = None,
            rpix: Sequence[int | float] | None = None,
            rval: Sequence[int | float] | None = None,
            rota: int | float | None = 0
    ):
        # if mask is None:
        #     mask = np.ones_like(data)
        # if error is None:
        #     error = np.ones_like(data)
        if step is None:
            step = (1,) * data.ndim
        if rpix is None:
            rpix = tuple((np.array(data.shape[::-1]) / 2 - 0.5).tolist())
        if rval is None:
            rval = (0,) * data.ndim
        if rota is None:
            rota = 0
        if isinstance(step, (int, float)):
            step = (step,) * data.ndim
        if isinstance(rpix, (int, float)):
            rpix = (rpix,) * data.ndim
        if isinstance(rval, (int, float)):
            rval = (rval,) * data.ndim
        # Convert to native byte order and ensure fp format.
        # If not in fp format, converting to fp32 should be enough
        data = miscutils.to_native_byteorder(data)
        data = _ensure_floating_or_float32(data, 'data')
        if mask is not None:
            mask = miscutils.to_native_byteorder(mask)
            mask = _ensure_floating_or_float32(mask, 'mask')
        if error is not None:
            error = miscutils.to_native_byteorder(error)
            error = _ensure_floating_or_float32(error, 'error')
        # Make sure the supplied arguments are compatible
        if mask is not None and data.shape != mask.shape:
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
        # Create and apply the "total mask" which takes into account
        # the supplied mask as well as all the nan values in the data.
        total_mask = np.ones_like(data, dtype=int)
        total_mask *= np.isfinite(data)
        if mask is not None:
            total_mask *= np.isfinite(mask)
            total_mask *= mask != 0
        if error is not None:
            total_mask *= np.isfinite(error)
        # Apply the total mask to all available data
        data[total_mask == 0] = np.nan
        if mask is not None:
            mask[total_mask == 0] = 0
            mask[total_mask != 0] = 1
        if error is not None:
            error[total_mask == 0] = np.nan
        # If no mask was supplied but the global mask contains zeros,
        # use the global mask as a mask
        if mask is None and np.any(total_mask == 0):
            mask = total_mask
        # Calculate the world coordinates at the very first pixel
        zero = (np.array(rval) - np.array(rpix) * np.array(step)).tolist()
        # Keep copies of the supplied data
        dtype = data.dtype
        self._data = data.copy()
        self._mask = mask.copy().astype(dtype) if mask is not None else None
        self._error = error.copy().attype(dtype) if error is not None else None
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
