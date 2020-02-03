
import astropy.io.fits as fits  # TODO: Remove dependency.
import numpy as np
import scipy.ndimage

import gbkfit.math
import gbkfit.psflsf


def _create_grid_1d(size, step, offset):
    center = size // 2 + offset
    vz = (np.array(range(size)) - center) * step
    return vz


class LSFGauss(gbkfit.psflsf.LSF):

    _CUTOFF_COEFF = 10

    @staticmethod
    def type():
        return 'gauss'

    @classmethod
    def load(cls, info):
        sigma = info['sigma']
        return cls(sigma)

    def dump(self, **kwargs):
        return {
            'type': self.type(),
            'sigma': self._sigma}

    def __init__(self, sigma):
        self._sigma = sigma

    def _size_impl(self, step):
        return 2 * self._CUTOFF_COEFF * self._sigma / step

    def _asarray_impl(self, step, size, offset):
        vz = _create_grid_1d(size, step, offset)
        data = gbkfit.math.gauss_1d_fun(vz, 1, 0, self._sigma)
        return data / np.sum(data)


class LSFGGauss(gbkfit.psflsf.LSF):

    _CUTOFF_COEFF = 10

    @staticmethod
    def type():
        return 'ggauss'

    @classmethod
    def load(cls, info):
        alpha = info['alpha']
        beta = info['beta']
        return cls(alpha, beta)

    def dump(self, **kwargs):
        return {
            'type': self.type(),
            'alpha': self._alpha,
            'beta': self._beta}

    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    def _size_impl(self, step):
        return 2 * self._CUTOFF_COEFF * self._alpha / step

    def _asarray_impl(self, step, size, offset):
        vz = _create_grid_1d(size, step, offset)
        data = gbkfit.math.ggauss_1d_fun(vz, 1, 0, self._alpha, self._beta)
        return data / np.sum(data)


class LSFLorentz(gbkfit.psflsf.LSF):

    _CUTOFF_COEFF = 10

    @staticmethod
    def type():
        return 'lorentz'

    @classmethod
    def load(cls, info):
        gamma = info['gamma']
        return cls(gamma)

    def dump(self, **kwargs):
        return {
            'type': self.type(),
            'gamma': self._gamma}

    def __init__(self, gamma):
        self._gamma = gamma

    def _size_impl(self, step):
        return 2 * self._CUTOFF_COEFF * self._gamma / step

    def _asarray_impl(self, step, size, offset):
        vz = _create_grid_1d(size, step, offset)
        data = gbkfit.math.lorentz_1d_fun(vz, 1, 0, self._gamma)
        return data / np.sum(data)


class LSFMoffat(gbkfit.psflsf.LSF):

    _CUTOFF_COEFF = 10

    @staticmethod
    def type():
        return 'moffat'

    @classmethod
    def load(cls, info):
        alpha = info['alpha']
        beta = info['beta']
        return cls(alpha, beta)

    def dump(self, **kwargs):
        return {
            'type': self.type(),
            'alpha': self._alpha,
            'beta': self._beta}

    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    def _size_impl(self, step):
        return 2 * self._CUTOFF_COEFF * self._alpha / step

    def _asarray_impl(self, step, size, offset):
        vz = _create_grid_1d(size, step, offset)
        data = gbkfit.math.moffat_1d_fun(vz, 1, 0, self._alpha, self._beta)
        return data / np.sum(data)


class LSFImage(gbkfit.psflsf.LSF):

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
        self._data = np.copy(data)
        self._step = step

    def _size_impl(self, step):
        return (self._step[1] / step[1]) * self._data.shape[0]

    def _asarray_impl(self, step, size, offset):
        scale = step / self._step
        old_center = self._data.shape[0] / 2 + 0.5
        new_center = size / 2 + 0.5 + offset
        x = np.arange(size)
        nx = (x - new_center) * scale + old_center
        data = scipy.ndimage.map_coordinates(self._data, [nx], order=5)
        return data / np.sum(data)
