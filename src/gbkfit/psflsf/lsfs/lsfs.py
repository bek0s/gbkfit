
import astropy.io.fits as fits  # TODO: Remove dependency somehow
import numpy as np
import scipy.ndimage

import gbkfit.math
from gbkfit.psflsf import LSF
from gbkfit.utils import miscutils, parseutils


def _create_grid_1d(size, step, offset):
    center = size // 2 + offset
    return (np.array(range(size)) - center) * step


def _load_lsf_common(cls, info):
    desc = parseutils.make_typed_desc(cls, 'LSF')
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


class LSFGauss(LSF):

    _CUTOFF_COEFF = 8

    @staticmethod
    def type():
        return 'gauss'

    @classmethod
    def load(cls, info):
        opts = _load_lsf_common(cls, info)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            sigma=self._sigma)

    def __init__(self, sigma):
        self._sigma = sigma

    def _size_impl(self, step):
        return 2 * self._CUTOFF_COEFF * self._sigma / step

    def _asarray_impl(self, step, size, offset):
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.gauss_1d_fun(z, 1, 0, self._sigma)
        data[z > self._CUTOFF_COEFF * self._sigma] = 0
        return data / np.sum(data)


class LSFGGauss(LSF):

    _CUTOFF_COEFF = 8

    @staticmethod
    def type():
        return 'ggauss'

    @classmethod
    def load(cls, info):
        opts = _load_lsf_common(cls, info)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta)

    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    def _size_impl(self, step):
        return 2 * self._CUTOFF_COEFF * self._alpha / step

    def _asarray_impl(self, step, size, offset):
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.ggauss_1d_fun(z, 1, 0, self._alpha, self._beta)
        data[z > self._CUTOFF_COEFF * self._alpha] = 0
        return data / np.sum(data)


class LSFLorentz(LSF):

    _CUTOFF_COEFF = 8

    @staticmethod
    def type():
        return 'lorentz'

    @classmethod
    def load(cls, info):
        opts = _load_lsf_common(cls, info)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            gamma=self._gamma)

    def __init__(self, gamma):
        self._gamma = gamma

    def _size_impl(self, step):
        return 2 * self._CUTOFF_COEFF * self._gamma / step

    def _asarray_impl(self, step, size, offset):
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.lorentz_1d_fun(z, 1, 0, self._gamma)
        data[z > self._CUTOFF_COEFF * self._gamma] = 0
        return data / np.sum(data)


class LSFMoffat(LSF):

    _CUTOFF_COEFF = 8

    @staticmethod
    def type():
        return 'moffat'

    @classmethod
    def load(cls, info):
        opts = _load_lsf_common(cls, info)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta)

    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    def _size_impl(self, step):
        return 2 * self._CUTOFF_COEFF * self._alpha / step

    def _asarray_impl(self, step, size, offset):
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.moffat_1d_fun(z, 1, 0, self._alpha, self._beta)
        data[z > self._CUTOFF_COEFF * self._alpha] = 0
        return data / np.sum(data)


class LSFImage(LSF):

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info):
        opts = _load_lsf_common(cls, info)
        opts.update(dict(
            data=miscutils.to_native_byteorder(fits.getdata(opts['data']))))
        return cls(**opts)

    def dump(self, file=None):
        if not file:
            file = 'lsf.fits'
        info = dict(
            data=file,
            step=self._step)
        fits.writeto(file, self._data, overwrite=True)
        return info

    def __init__(self, data, step):
        self._data = np.copy(data)
        self._step = step

    def _size_impl(self, step):
        return (self._step / step) * self._data.shape[0]

    def _asarray_impl(self, step, size, offset):
        scale = step / self._step
        old_center = self._data.shape[0] / 2 + 0.5
        new_center = size / 2 + 0.5 + offset
        x = np.arange(size)
        nx = (x - new_center) * scale + old_center
        data = scipy.ndimage.map_coordinates(self._data, [nx], order=5)
        return data / np.sum(data)
