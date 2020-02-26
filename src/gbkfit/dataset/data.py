
import astropy.io.fits as fits
import numpy as np


def make_mask(data_list):
    if any([data.shape != data_list[0].shape for data in data_list]):
        raise RuntimeError('')
    mask = np.ones_like(data_list[0])
    for data in data_list:
        mask *= np.isfinite(data)
    return mask


class Data:

    @classmethod
    def load(cls, info):
        data_d_hdu = fits.open(info['data'])[0]
        data_d_data = data_d_hdu.data
        data_m_data = None
        if 'mask' in info:
            data_m_hdu = fits.open(info['mask'])[0]
            data_m_data = data_m_hdu.data
        data_e_data = None
        if 'error' in info:
            data_e_hdu = fits.open(info['error'])[0]
            data_e_data = data_e_hdu.data
        step = None
        cval = None
        dtype = info.get('dtype')
        return cls(data_d_data, data_m_data, data_e_data, step, cval, dtype)

    def dump(self, prefix=''):
        info = {
            'data': prefix + 'd.fits',
            'mask': prefix + 'm.fits',
            'error': prefix + 'e.fits',
            'step': self.step(),
            'cval': self.cval()}
        fits.writeto(info['data'], self.data(), overwrite=True)
        fits.writeto(info['mask'], self.mask(), overwrite=True)
        fits.writeto(info['error'], self.error(), overwrite=True)
        return info

    def __init__(
            self, data, mask=None, error=None,
            step=None, cval=None, dtype=None):
        if mask is None:
            mask = np.ones_like(data)
        if error is None:
            error = np.ones_like(data)
        if step is None:
            step = [1] * data.ndim
        if cval is None:
            cval = [0] * data.ndim
        if dtype is None:
            dtype = data.dtype
        if mask.shape != data.shape or error.shape != data.shape:
            raise RuntimeError(
                f"data (shape={data.shape}), mask (shape={mask.shape}), and "
                f"error (shape={error.shape}) must have the same shape.")
        if len(step) != data.ndim or len(cval) != data.ndim:
            raise RuntimeError(
                f"step (length={len(step)}) and cval (length={len(cval)}) "
                f"must have a length equal to the dimensionality of the "
                f"data (ndim={data.ndim}).")
        mmask = make_mask([data, error, mask])
        data[mmask == 0] = np.nan
        mask[mmask == 0] = 0
        error[mmask == 0] = np.nan
        mask[mmask != 0] = 1
        """
        import astropy.io.fits as fits
        fits.writeto(f'mmask_d.fits', data, overwrite=True)
        fits.writeto(f'mmask_m.fits', mask, overwrite=True)
        fits.writeto(f'mmask_e.fits', error, overwrite=True)
        """
        self._data = data.copy().astype(dtype)
        self._mask = mask.copy().astype(dtype)
        self._error = error.copy().astype(dtype)
        self._step = tuple(step)
        self._cval = tuple(cval)

    def npix(self):
        return self._data.size

    def ndim(self):
        return self._data.ndim

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