
import astropy.wcs
import numpy as np
import scipy.ndimage

import gbkfit.dataset
import gbkfit.math
from gbkfit.psflsf.core import LSF
from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'LSFPoint',
    'LSFGauss',
    'LSFGGauss',
    'LSFLorentz',
    'LSFMoffat',
    'LSFImage'
]


def _create_grid_1d(size, step, offset):
    center = size // 2 + offset
    return (np.array(range(size)) - center) * step


def _load_lsf_common(cls, info):
    desc = parseutils.make_typed_desc(cls, 'LSF')
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


class LSFPoint(LSF):

    @staticmethod
    def type():
        return 'point'

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return dict(type=self.type())

    def _size_impl(self, step):
        return 1

    def _asarray_impl(self, step, size, offset):
        data = np.zeros(size)
        data[size // 2 + offset] = 1
        return data


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
        try:
            data, wcs = fitsutils.load_fits(opts['data'])
        except Exception as e:
            raise RuntimeError(
                f"could not load LSF image with filename '{opts['data']}'; "
                f"see preceding exception for additional information") from e
        opts.update(dict(data=data, step=opts.get('step', wcs.wcs.cdelt[0])))
        return cls(**opts)

    def dump(self, filename='lsf.fits', overwrite=False):
        info = dict(type=self.type(), data=filename, step=self._step)
        wcs = astropy.wcs.WCS(naxis=1, relax=False)
        wcs.wcs.cdelt = self._step  # noqa
        fitsutils.dump_fits(filename, self._data, wcs, overwrite)
        return info

    def __init__(self, data, step=1):
        if not np.all(np.isfinite(data)):
            raise RuntimeError(
                "non-finite pixels found in the supplied LSF image")
        self._data = data
        self._step = step

    def _size_impl(self, step):
        return (self._step / step) * self._data.shape[0]

    def _asarray_impl(self, step, size, offset):
        scale = step / self._step
        old_center = self._data.shape[0] / 2 - 0.5
        new_center = size / 2 - 0.5 + offset
        x = np.arange(size)
        nx = (x - new_center) * scale + old_center
        data = scipy.ndimage.map_coordinates(self._data, [nx], order=5)  # noqa
        return data / np.sum(data)
