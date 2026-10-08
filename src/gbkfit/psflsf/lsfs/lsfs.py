
from typing import Any

import astropy.wcs
import numpy as np
import scipy.ndimage

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


def _create_grid_1d(size: int, step: float, offset: int) -> np.ndarray:
    center = size // 2 + offset
    return (np.array(range(size)) - center) * step


def _load_lsf_common(cls, info: dict[str, Any]):
    desc = parseutils.make_typed_desc(cls, 'LSF')
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


class LSFPoint(LSF):
    """
    A point-like Line Spread Function (LSF).

    Represents an LSF where all the energy is concentrated at a single
    point. The output array contains a single nonzero value (1) at the
    center.
    """

    @staticmethod
    def type() -> str:
        return 'point'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFPoint':
        return cls()

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def _size_impl(self, step: float) -> float:
        return 1

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        data = np.zeros(size)
        data[size // 2 + offset] = 1
        return data


class LSFGauss(LSF):
    """
    A Gaussian Line Spread Function (LSF).
    """

    _CUTOFF_FACTOR = 8

    @staticmethod
    def type() -> str:
        return 'gauss'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFGauss':
        opts = _load_lsf_common(cls, info)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            sigma=self._sigma)

    def __init__(self, sigma: float):
        self._sigma = sigma

    def _size_impl(self, step: float) -> float:
        return 2 * self._CUTOFF_FACTOR * self._sigma / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.gauss_1d_fun(z, 1, 0, self._sigma)
        data[z > self._CUTOFF_FACTOR * self._sigma] = 0
        return data / np.sum(data)


class LSFGGauss(LSF):
    """
    A Generalized Gaussian Line Spread Function (LSF).
    """

    _CUTOFF_FACTOR = 8

    @staticmethod
    def type() -> str:
        return 'ggauss'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFGGauss':
        opts = _load_lsf_common(cls, info)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta)

    def __init__(self, alpha: float, beta: float):
        self._alpha = alpha
        self._beta = beta

    def _size_impl(self, step: float) -> float:
        return 2 * self._CUTOFF_FACTOR * self._alpha / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.ggauss_1d_fun(z, 1, 0, self._alpha, self._beta)
        data[z > self._CUTOFF_FACTOR * self._alpha] = 0
        return data / np.sum(data)


class LSFLorentz(LSF):
    """
    A Lorentzian Line Spread Function (LSF).
    """

    _CUTOFF_FACTOR = 8

    @staticmethod
    def type() -> str:
        return 'lorentz'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFLorentz':
        opts = _load_lsf_common(cls, info)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            gamma=self._gamma)

    def __init__(self, gamma: float):
        self._gamma = gamma

    def _size_impl(self, step: float) -> float:
        return 2 * self._CUTOFF_FACTOR * self._gamma / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.lorentz_1d_fun(z, 1, 0, self._gamma)
        data[z > self._CUTOFF_FACTOR * self._gamma] = 0
        return data / np.sum(data)


class LSFMoffat(LSF):
    """
    A Moffat Line Spread Function (LSF).
    """

    _CUTOFF_FACTOR = 8

    @staticmethod
    def type() -> str:
        return 'moffat'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFMoffat':
        opts = _load_lsf_common(cls, info)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta)

    def __init__(self, alpha: float, beta: float):
        self._alpha = alpha
        self._beta = beta

    def _size_impl(self, step: float) -> float:
        return 2 * self._CUTOFF_FACTOR * self._alpha / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.moffat_1d_fun(z, 1, 0, self._alpha, self._beta)
        data[z > self._CUTOFF_FACTOR * self._alpha] = 0
        return data / np.sum(data)


class LSFImage(LSF):
    """
    An LSF defined by an image, loaded from a FITS file.

    The image is resampled based on the provided step size.
    """

    @staticmethod
    def type() -> str:
        return 'image'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFImage':
        opts = _load_lsf_common(cls, info)
        try:
            data, wcs = fitsutils.load_fits(opts['data'])
        except Exception as e:
            raise RuntimeError(
                f"could not load LSF image with filename '{opts['data']}'; "
                f"see preceding exception for additional information") from e
        opts.update(dict(data=data, step=opts.get('step', wcs.wcs.cdelt[0])))  # noqa
        return cls(**opts)

    def dump(
            self, filename='lsf.fits', overwrite: bool = False
    ) -> dict[str, Any]:
        info = dict(type=self.type(), data=filename, step=self._step)
        wcs = astropy.wcs.WCS(naxis=1, relax=False)
        wcs.wcs.cdelt = [self._step]  # noqa
        fitsutils.dump_fits(filename, self._data, wcs, overwrite)
        return info

    def __init__(self, data: np.ndarray, step: float = 1.0):
        data = np.squeeze(data)  # Remove singleton dimensions
        if data.ndim != 1:
            raise RuntimeError(
                f"expected a 1D LSF image, but got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise RuntimeError(
                "non-finite pixels found in the supplied LSF image")
        self._data = data
        self._step = step

    def _size_impl(self, step: float) -> float:
        """Computes the LSF size based on the step ratio."""
        return (self._step / step) * self._data.shape[0]

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        """
        Resamples the stored LSF image to match the desired step and
        size. Uses spline interpolation (order=5).
        """
        scale = step / self._step
        old_center = self._data.shape[0] / 2 - 0.5
        new_center = size / 2 - 0.5 + offset
        x = np.arange(size)
        nx = (x - new_center) * scale + old_center
        data = scipy.ndimage.map_coordinates(self._data, [nx], order=5)  # noqa
        return data / np.sum(data)
