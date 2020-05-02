
import astropy.io.fits as fits
import numpy as np

from gbkfit.utils import parseutils


class Data:

    @classmethod
    def load(cls, info, **kwargs):
        args = parseutils.parse_class_args(Data, info)
        data_d_data = fits.getdata(info['data'])
        data_m_data = None
        if 'mask' in info:
            data_m_data = fits.getdata(info['mask'])
        data_e_data = None
        if 'error' in info:
            data_e_data = fits.getdata(info['error'])
        args.update(
            data=data_d_data,
            mask=data_m_data,
            error=data_e_data,
            step=kwargs.get('step'),
            cval=kwargs.get('cval'))
        return cls(**args)

    def dump(self, **kwargs):
        prefix = kwargs.get('prefix', '')
        info = dict(
            data=f'{prefix}_d.fits',
            mask=f'{prefix}_m.fits',
            error=f'{prefix}_e.fits',
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
        if data.shape != mask.shape or data.shape != error.shape:
            raise RuntimeError(
                f"data (shape={data.shape}), mask (shape={mask.shape}), and "
                f"error (shape={error.shape}) arrays must have the same shape")
        if data.ndim != len(step) or data.ndim != len(cval):
            raise RuntimeError(
                f"step (length={len(step)}) and cval (length={len(cval)}) "
                f"must have a length equal to the dimensionality of the "
                f"data (ndim={data.ndim})")
        finite_mask = np.ones_like(data)
        finite_mask *= np.isfinite(data)
        finite_mask *= np.isfinite(mask)
        finite_mask *= np.isfinite(error)
        data[finite_mask == 0] = np.nan
        mask[finite_mask == 0] = 0
        mask[finite_mask != 0] = 1
        error[finite_mask == 0] = np.nan
        # TODO: fix this
        data = data.astype(np.float32)
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


parser = parseutils.SimpleParser(Data)
