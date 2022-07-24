
import logging
import os.path

import astropy.wcs
import numpy as np

from gbkfit.utils import fitsutils, iterutils, miscutils, parseutils


__all__ = ['Data', 'data_parser']


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
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        data_d, wcs_d = fitsutils.load_fits(prefix + opts['data'])
        data_m = None
        data_e = None
        if 'mask' in opts:
            data_m = fitsutils.load_fits(prefix + opts['mask'])[0]
        if 'error' in opts:
            data_e = fitsutils.load_fits(prefix + opts['error'])[0]
        # Local information has higher priority than global
        step = opts.get('step', step)
        rpix = opts.get('rpix', rpix)
        rval = opts.get('rval', rval)
        # If no information is provided, use fits header
        if step is None:
            step = wcs_d.wcs.cdelt  # noqa
        if rpix is None:
            rpix = wcs_d.wcs.crpix  # noqa
        if rval is None:
            rval = wcs_d.wcs.crval  # noqa
        # todo: deal with rotation (PC Matrix and CROTA (deprecated))
        # Build class arguments dict
        opts.update(dict(
            data=data_d,
            mask=data_m,
            error=data_e,
            step=step,
            rpix=rpix,
            rval=rval,
            rota=rota))
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
            self, data, mask=None, error=None,
            step=None, rpix=None, rval=None, rota=0):
        step = iterutils.tuplify(step, False)
        rpix = iterutils.tuplify(rpix, False)
        rval = iterutils.tuplify(rval, False)
        if mask is None:
            mask = np.ones_like(data)
        if error is None:
            error = np.ones_like(data)
        if step is None:
            step = (1,) * data.ndim
        if rpix is None:
            rpix = tuple((np.array(data.shape[::-1]) / 2 - 0.5).tolist())
        if rval is None:
            rval = (0,) * data.ndim
        data = miscutils.to_native_byteorder(data)
        mask = miscutils.to_native_byteorder(mask)
        error = miscutils.to_native_byteorder(error)
        # If data is not floating point, we need to convert it.
        # float32 should be more than enough.
        data = _ensure_floating_or_float32(data, 'data')
        mask = _ensure_floating_or_float32(mask, 'mask')
        error = _ensure_floating_or_float32(error, 'error')
        # Make sure the supplied arguments are compatible
        if data.shape != mask.shape:
            raise RuntimeError(
                f"data and mask have incompatible shapes "
                f"({data.shape} != {mask.shape})")
        if data.shape != error.shape:
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
                f"({data.dim} != {len(rval)})")
        # Create and apply the "total mask" which takes into account
        # the supplied mask as well as all the nan values in the data.
        total_mask = np.ones_like(data)
        total_mask *= np.isfinite(data)
        total_mask *= np.isfinite(mask)
        total_mask *= np.isfinite(error)
        total_mask *= mask != 0
        data[total_mask == 0] = np.nan
        mask[total_mask == 0] = 0
        mask[total_mask != 0] = 1
        error[total_mask == 0] = np.nan
        # Calculate the world coordinates at pixel (0, 0)
        zero = (np.array(rval) - np.array(rpix) * np.array(step)).tolist()
        # Keep copies of the supplied data
        self._data = data.copy()
        self._mask = mask.copy().astype(data.dtype)
        self._error = error.copy().astype(data.dtype)
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
