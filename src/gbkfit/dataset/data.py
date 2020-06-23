
import sys

import astropy.io.fits as fits
import numpy as np

from gbkfit.utils import parseutils


class Data:

    @classmethod
    def load(cls, info, *args, **kwargs):
        cls_args = parseutils.parse_class_args(cls, info)
    #   byteorder = dict(little='<', big='>')[sys.byteorder]
        data_d = fits.getdata(cls_args['data']).astype(np.float32)
        data_m = fits.getdata(cls_args['mask']).astype(np.float32) \
            if 'mask' in info else None
        data_e = fits.getdata(cls_args['error']).astype(np.float32) \
            if 'error' in info else None
        step = None
        cval = None
        cls_args.update(
            data=data_d,
            mask=data_m,
            error=data_e,
            step=kwargs.get('step', step),
            cval=kwargs.get('cval', cval))
        return cls(**cls_args)

    def dump(self, *args, **kwargs):
        prefix = kwargs.get('prefix', '')
        info = dict(
            data=f'{prefix}d.fits',
            mask=f'{prefix}m.fits',
            error=f'{prefix}e.fits',
            step=self.step(),
            cval=self.cval())
        fits.writeto(info['data'], self.data(), overwrite=True)
        fits.writeto(info['mask'], self.mask(), overwrite=True)
        fits.writeto(info['error'], self.error(), overwrite=True)
        return info

    def __init__(self, data, mask=None, error=None, step=None, cval=None):
        if mask is None:
            mask = np.ones_like(data)
        if error is None:
            error = np.ones_like(data)
        if step is None:
            step = [1] * data.ndim
        if cval is None:
            cval = [0] * data.ndim
        if data.shape != mask.shape:
            raise RuntimeError(
                f"data and mask have incompatible shapes "
                f"({data.shape} != {mask.shape})")
        if data.shape != error.shape:
            raise RuntimeError(
                f"data and error have incompatible shapes "
                f"({data.shape} != {mask.shape})")
        if data.ndim != len(step):
            raise RuntimeError(
                f"data dimensionality and step length are incompatible "
                f"({data.dim} != {len(step)})")
        if data.ndim != len(cval):
            raise RuntimeError(
                f"data dimensionality and cval length are incompatible "
                f"({data.dim} != {len(cval)})")
        finite_mask = np.ones_like(data)
        finite_mask *= np.isfinite(data)
        finite_mask *= np.isfinite(mask)
        finite_mask *= np.isfinite(error)
        data[finite_mask == 0] = np.nan
        mask[finite_mask == 0] = 0
        mask[finite_mask != 0] = 1
        error[finite_mask == 0] = np.nan
        self._data = data.copy()
        self._mask = mask.copy().astype(data.dtype)
        self._error = error.copy().astype(data.dtype)
        self._step = tuple(step)
        self._cval = tuple(cval)

    def ndim(self):
        return self._data.ndim

    def npix(self):
        return self._data.size

    def size(self):
        return self._data.shape[::-1]

    def step(self):
        return self._step

    def cval(self):
        return self._cval

    def data(self):
        return self._data

    def mask(self):
        return self._mask

    def error(self):
        return self._error

    def dtype(self):
        return self._data.dtype


parser = parseutils.SimpleParser(Data)
