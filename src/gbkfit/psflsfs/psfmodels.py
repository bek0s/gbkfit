
import astropy.io.fits as fits  # TODO: Remove dependency.
import numpy as np
import scipy.ndimage

import gbkfit.math
import gbkfit.psflsf
from gbkfit.utils import iterutils


def _create_grid_2d(size, step, offset, ratio, posa):
    center_x = size[0] // 2 + offset[0]
    center_y = size[1] // 2 + offset[1]
    x = (np.array(range(size[0])) - center_x) * step[0]
    y = (np.array(range(size[1])) - center_y) * step[1]
    x = x[None, :]
    y = y[:, None]
    x, y = gbkfit.math.transform_lh_rotate_z(x, y, np.radians(posa))
    return np.sqrt(x * x + y * y / (ratio * ratio))


class PSFGauss(gbkfit.psflsf.PSF):

    _CUTOFF_COEFF = 6

    @staticmethod
    def type():
        return 'gauss'

    @classmethod
    def load(cls, info):
        sigma = info['sigma']
        ratio = info.get('ratio')
        posa = info.get('posa')
        return cls(sigma, ratio, posa)

    def dump(self, **kwargs):
        return {
            'type': self.type(),
            'sigma': self._sigma,
            'ratio': self._ratio,
            'posa': self._posa}

    def __init__(self, sigma, ratio=None, posa=None):
        if ratio is None:
            ratio = 1
        if posa is None:
            posa = 0
        self._sigma = sigma
        self._ratio = ratio
        self._posa = posa

    def _size_impl(self, step):
        return (2 * self._CUTOFF_COEFF * self._sigma / step[0],
                2 * self._CUTOFF_COEFF * self._sigma / step[1])

    def _asarray_impl(self, step, size, offset):
        r = _create_grid_2d(size, step, offset, self._ratio, self._posa)
        indices = r > self._CUTOFF_COEFF * self._sigma
        data = gbkfit.math.gauss_1d_fun(r, 1, 0, self._sigma)
        data[indices] = 0
        return data / np.sum(data)


class PSFGGauss(gbkfit.psflsf.PSF):

    _CUTOFF_COEFF = 10

    @staticmethod
    def type():
        return 'ggauss'

    @classmethod
    def load(cls, info):
        alpha = info['alpha']
        beta = info['beta']
        ratio = info.get('ratio')
        posa = info.get('posa')
        return cls(alpha, beta, ratio, posa)

    def dump(self, **kwargs):
        return {
            'type': self.type(),
            'alpha': self._alpha,
            'beta': self._beta,
            'ratio': self._ratio,
            'posa': self._posa}

    def __init__(self, alpha, beta, ratio=None, posa=None):
        if ratio is None:
            ratio = 1
        if posa is None:
            posa = 0
        self._alpha = alpha
        self._beta = beta
        self._ratio = ratio
        self._posa = posa

    def _size_impl(self, step):
        return (2 * self._CUTOFF_COEFF * self._alpha / step[0],
                2 * self._CUTOFF_COEFF * self._alpha / step[1])

    def _asarray_impl(self, step, size, offset):
        r = _create_grid_2d(size, step, offset, self._ratio, self._posa)
        indices = r > self._CUTOFF_COEFF * self._alpha
        data = gbkfit.math.ggauss_1d_fun(r, 1, 0, self._alpha, self._beta)
        data[indices] = 0
        return data / np.sum(data)


class PSFLorentz(gbkfit.psflsf.PSF):

    _CUTOFF_COEFF = 10

    @staticmethod
    def type():
        return 'lorentz'

    @classmethod
    def load(cls, info):
        gamma = info['gamma']
        ratio = info.get('ratio')
        posa = info.get('posa')
        return cls(gamma, ratio, posa)

    def dump(self, **kwargs):
        return {
            'type': self.type(),
            'gamma': self._gamma,
            'ratio': self._ratio,
            'posa': self._posa}

    def __init__(self, gamma, ratio=None, posa=None):
        if ratio is None:
            ratio = 1
        if posa is None:
            posa = 0
        self._gamma = gamma
        self._ratio = ratio
        self._posa = posa

    def _size_impl(self, step):
        return (2 * self._CUTOFF_COEFF * self._gamma / step[0],
                2 * self._CUTOFF_COEFF * self._gamma / step[1])

    def _asarray_impl(self, step, size, offset):
        r = _create_grid_2d(size, step, offset, self._ratio, self._posa)
        indices = r > self._CUTOFF_COEFF * self._gamma
        data = gbkfit.math.lorentz_1d_fun(r, 1, 0, self._gamma)
        data[indices] = 0
        return data / np.sum(data)


class PSFMoffat(gbkfit.psflsf.PSF):

    _CUTOFF_COEFF = 10

    @staticmethod
    def type():
        return 'moffat'

    @classmethod
    def load(cls, info):
        alpha = info['alpha']
        beta = info['beta']
        ratio = info.get('ratio')
        posa = info.get('posa')
        return cls(alpha, beta, ratio, posa)

    def dump(self, **kwargs):
        return {
            'type': self.type(),
            'alpha': self._alpha,
            'beta': self._beta,
            'ratio': self._ratio,
            'posa': self._posa}

    def __init__(self, alpha, beta, ratio=None, posa=None):
        if ratio is None:
            ratio = 1
        if posa is None:
            posa = 0
        self._alpha = alpha
        self._beta = beta
        self._ratio = ratio
        self._posa = posa

    def _size_impl(self, step):
        return (2 * self._CUTOFF_COEFF * self._alpha / step[0],
                2 * self._CUTOFF_COEFF * self._alpha / step[1])

    def _asarray_impl(self, step, size, offset):
        r = _create_grid_2d(size, step, offset, self._ratio, self._posa)
        indices = r > self._CUTOFF_COEFF * self._alpha
        data = gbkfit.math.moffat_1d_fun(r, 1, 0, self._alpha, self._beta)
        data[indices] = 0
        return data / np.sum(data)


class PSFImage(gbkfit.psflsf.PSF):

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info):
        file = info['file']
        step = info['step']
        data = fits.getdata(file)
        return cls(data, step)

    def dump(self, **kwargs):
        file = kwargs['file']
        fits.writeto(file, self._data)
        return {
            'file': file,
            'step': self._step}

    def __init__(self, data, step):
        step = iterutils.tuplify(step)
        if len(step) == 1:
            step = step + (step[0],)
        self._data = data.copy()
        self._step = step

    def _size_impl(self, step):
        return (
            (self._step[0] / step[0]) * self._data.shape[1],
            (self._step[1] / step[1]) * self._data.shape[0])

    def _asarray_impl(self, step, size, offset):
        scale_x = step[0] / self._step[0]
        scale_y = step[1] / self._step[1]
        old_center_x = self._data.shape[0] / 2 + 0.5
        old_center_y = self._data.shape[1] / 2 + 0.5
        new_center_x = size[0] / 2 + 0.5 + offset[0]
        new_center_y = size[1] / 2 + 0.5 + offset[1]
        x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
        nx = (x - new_center_x) * scale_x + old_center_x
        ny = (y - new_center_y) * scale_y + old_center_y
        data = scipy.ndimage.map_coordinates(self._data, [ny, nx], order=5)
        return data / np.sum(data)
