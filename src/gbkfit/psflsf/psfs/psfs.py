
import astropy.wcs
import numpy as np
import scipy.ndimage

import gbkfit.math
from gbkfit.psflsf.core import PSF
from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'PSFPoint',
    'PSFGauss',
    'PSFGGauss',
    'PSFLorentz',
    'PSFMoffat',
    'PSFImage'
]


def _create_grid_2d(size, step, offset, ratio, posa):
    center_x = size[0] // 2 + offset[0]
    center_y = size[1] // 2 + offset[1]
    x = (np.array(range(size[0])) - center_x) * step[0]
    y = (np.array(range(size[1])) - center_y) * step[1]
    x = x[None, :]
    y = y[:, None]
    x, y = gbkfit.math.transform_lh_rotate_z(x, y, np.radians(posa))
    return np.sqrt(x * x + y * y / (ratio * ratio))


def _load_psf_common(cls, info):
    desc = parseutils.make_typed_desc(cls, 'PSF')
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


class PSFPoint(PSF):

    @staticmethod
    def type():
        return 'point'

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return dict(type=self.type())

    def _size_impl(self, step):
        return 1, 1

    def _asarray_impl(self, step, size, offset):
        data = np.zeros(size)
        data[size[0] // 2 + offset[0], size[1] // 2 + offset[1]] = 1
        return data


class PSFGauss(PSF):

    _CUTOFF_COEFF = 8

    @staticmethod
    def type():
        return 'gauss'

    @classmethod
    def load(cls, info):
        opts = _load_psf_common(cls, info)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            sigma=self._sigma,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(self, sigma, ratio=1, posa=0):
        self._sigma = sigma
        self._ratio = ratio
        self._posa = posa

    def _size_impl(self, step):
        return (2 * self._CUTOFF_COEFF * self._sigma / step[0],
                2 * self._CUTOFF_COEFF * self._sigma / step[1])

    def _asarray_impl(self, step, size, offset):
        r = _create_grid_2d(size, step, offset, self._ratio, self._posa)
        data = gbkfit.math.gauss_1d_fun(r, 1, 0, self._sigma)
        data[r > self._CUTOFF_COEFF * self._sigma] = 0
        return data / np.sum(data)


class PSFGGauss(PSF):

    _CUTOFF_COEFF = 8

    @staticmethod
    def type():
        return 'ggauss'

    @classmethod
    def load(cls, info):
        opts = _load_psf_common(cls, info)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(self, alpha, beta, ratio=1, posa=0):
        self._alpha = alpha
        self._beta = beta
        self._ratio = ratio
        self._posa = posa

    def _size_impl(self, step):
        return (2 * self._CUTOFF_COEFF * self._alpha / step[0],
                2 * self._CUTOFF_COEFF * self._alpha / step[1])

    def _asarray_impl(self, step, size, offset):
        r = _create_grid_2d(size, step, offset, self._ratio, self._posa)
        data = gbkfit.math.ggauss_1d_fun(r, 1, 0, self._alpha, self._beta)
        data[r > self._CUTOFF_COEFF * self._alpha] = 0
        return data / np.sum(data)


class PSFLorentz(PSF):

    _CUTOFF_COEFF = 8

    @staticmethod
    def type():
        return 'lorentz'

    @classmethod
    def load(cls, info):
        opts = _load_psf_common(cls, info)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            gamma=self._gamma,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(self, gamma, ratio=1, posa=0):
        self._gamma = gamma
        self._ratio = ratio
        self._posa = posa

    def _size_impl(self, step):
        return (2 * self._CUTOFF_COEFF * self._gamma / step[0],
                2 * self._CUTOFF_COEFF * self._gamma / step[1])

    def _asarray_impl(self, step, size, offset):
        r = _create_grid_2d(size, step, offset, self._ratio, self._posa)
        data = gbkfit.math.lorentz_1d_fun(r, 1, 0, self._gamma)
        data[r > self._CUTOFF_COEFF * self._gamma] = 0
        return data / np.sum(data)


class PSFMoffat(PSF):

    _CUTOFF_COEFF = 8

    @staticmethod
    def type():
        return 'moffat'

    @classmethod
    def load(cls, info):
        opts = _load_psf_common(cls, info)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(self, alpha, beta, ratio=1, posa=0):
        self._alpha = alpha
        self._beta = beta
        self._ratio = ratio
        self._posa = posa

    def _size_impl(self, step):
        return (2 * self._CUTOFF_COEFF * self._alpha / step[0],
                2 * self._CUTOFF_COEFF * self._alpha / step[1])

    def _asarray_impl(self, step, size, offset):
        r = _create_grid_2d(size, step, offset, self._ratio, self._posa)
        data = gbkfit.math.moffat_1d_fun(r, 1, 0, self._alpha, self._beta)
        data[r > self._CUTOFF_COEFF * self._alpha] = 0
        return data / np.sum(data)


class PSFImage(PSF):

    @staticmethod
    def type():
        return 'image'

    @classmethod
    def load(cls, info):
        opts = _load_psf_common(cls, info)
        try:
            data, wcs = fitsutils.load_fits(opts['data'])
        except Exception as e:
            raise RuntimeError(
                f"could not load PSF image with filename '{opts['data']}'; "
                f"see preceding exception for additional information") from e
        opts.update(dict(data=data, step=opts.get('step', wcs.wcs.cdelt)))
        return cls(**opts)

    def dump(self, filename='psf.fits', overwrite=False):
        info = dict(type=self.type(), data=filename, step=self._step)
        wcs = astropy.wcs.WCS(naxis=2, relax=False)
        wcs.wcs.cdelt = self._step  # noqa
        fitsutils.dump_fits(filename, self._data, wcs, overwrite)
        return info

    def __init__(self, data, step=(1, 1)):
        if not np.all(np.isfinite(data)):
            raise RuntimeError(
                "non-finite pixels found in the supplied PSF image")
        self._data = data
        self._step = step

    def _size_impl(self, step):
        return (
            (self._step[0] / step[0]) * self._data.shape[1],
            (self._step[1] / step[1]) * self._data.shape[0])

    def _asarray_impl(self, step, size, offset):
        scale_x = step[0] / self._step[0]
        scale_y = step[1] / self._step[1]
        old_center_x = self._data.shape[1] / 2 - 0.5
        old_center_y = self._data.shape[0] / 2 - 0.5
        new_center_x = size[0] / 2 - 0.5 + offset[0]
        new_center_y = size[1] / 2 - 0.5 + offset[1]
        x, y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
        nx = (x - new_center_x) * scale_x + old_center_x
        ny = (y - new_center_y) * scale_y + old_center_y
        data = scipy.ndimage.map_coordinates(self._data, [ny, nx], order=5)
        return data / np.sum(data)
