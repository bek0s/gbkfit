
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.special

import gbkfit.math
from gbkfit.psflsf.core import (
    MIN_EXTENT, PSF, WING_FLUX, check_ratio, check_scale)
from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'PSFPoint',
    'PSFGauss',
    'PSFGGauss',
    'PSFMoffat',
    'PSFImage'
]


def _create_grid_2d(
        size: tuple[int, int],
        step: tuple[float, float],
        offset: tuple[int, int],
        ratio: float,
        posa: float
) -> np.ndarray:
    center_x = size[0] // 2 + offset[0]
    center_y = size[1] // 2 + offset[1]
    x = (np.array(range(size[0])) - center_x) * step[0]
    y = (np.array(range(size[1])) - center_y) * step[1]
    x = x[None, :]
    y = y[:, None]
    x, y = gbkfit.math.transform_lh_rotate_z(x, y, np.radians(posa))
    return np.sqrt(x * x + y * y / (ratio * ratio))


def _load_psf_common(cls, info: dict[str, Any]):
    desc = parseutils.make_typed_desc(cls, 'PSF')
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


class PSFPoint(PSF):
    """
    A point-like Point Spread Function (PSF).

    Represents a PSF where all the energy is concentrated at a single
    point. The output array contains a single nonzero value (1) at the
    center.
    """

    @staticmethod
    def type() -> str:
        return 'point'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFPoint':
        return cls()

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return 1, 1

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        # Like all images, the array has shape (y, x)
        data = np.zeros(size[::-1])
        data[size[1] // 2 + offset[1], size[0] // 2 + offset[0]] = 1
        return data


class PSFGauss(PSF):
    """
    A Gaussian Point Spread Function (PSF).
    """


    @staticmethod
    def type() -> str:
        return 'gauss'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFGauss':
        opts = _load_psf_common(cls, info)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            sigma=self._sigma,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(self, sigma: float, ratio: float = 1.0, posa: float = 0.0):
        check_scale('sigma', sigma)
        check_ratio(ratio)
        self._sigma = sigma
        self._ratio = ratio
        self._posa = posa

    def _extent(self):
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        wings of a 2D Gaussian beyond r hold exp(-r^2 / 2 sigma^2).
        """
        return max(
            MIN_EXTENT * self._sigma,
            self._sigma * np.sqrt(-2 * np.log(WING_FLUX)))

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return (2 * self._extent() / step[0],
                2 * self._extent() / step[1])

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        r = _create_grid_2d(
            size, step, offset, self._ratio, self._posa - rota)
        data = gbkfit.math.gauss_1d_fun(r, 1, 0, self._sigma)
        data[r > self._extent()] = 0
        return data / np.sum(data)


class PSFGGauss(PSF):
    """
    A Generalized Gaussian Point Spread Function (PSF).
    """


    @staticmethod
    def type() -> str:
        return 'ggauss'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs):
        opts = _load_psf_common(cls, info)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(
            self,
            alpha: float,
            beta: float,
            ratio: float = 1.0,
            posa: float = 0.0
    ):
        check_scale('alpha', alpha)
        check_scale('beta', beta)
        check_ratio(ratio)
        self._alpha = alpha
        self._beta = beta
        self._ratio = ratio
        self._posa = posa

    def _extent(self):
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        flux of a 2D ggauss within r is P(2 / beta, (r / alpha)^beta),
        P the regularized lower incomplete gamma function.
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * scipy.special.gammaincinv(
                2 / self._beta, 1 - WING_FLUX) ** (1 / self._beta))

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return (2 * self._extent() / step[0],
                2 * self._extent() / step[1])

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        r = _create_grid_2d(
            size, step, offset, self._ratio, self._posa - rota)
        data = gbkfit.math.ggauss_1d_fun(r, 1, 0, self._alpha, self._beta)
        data[r > self._extent()] = 0
        return data / np.sum(data)


class PSFMoffat(PSF):
    """
   A Moffat Point Spread Function (PSF).
   """


    @staticmethod
    def type():
        return 'moffat'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFMoffat':
        opts = _load_psf_common(cls, info)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            alpha=self._alpha,
            beta=self._beta,
            ratio=self._ratio,
            posa=self._posa)

    def __init__(
            self,
            alpha: float,
            beta: float,
            ratio: float = 1.0,
            posa: float = 0.0
    ):
        check_scale('alpha', alpha)
        if not beta > 1:
            raise RuntimeError(
                "a Moffat PSF has finite flux only for beta > 1; "
                f"beta is {beta}")
        check_ratio(ratio)
        self._alpha = alpha
        self._beta = beta
        self._ratio = ratio
        self._posa = posa

    def _extent(self):
        """
        The radius out to which it is drawn (see MIN_EXTENT). The
        wings of a 2D Moffat beyond r hold (1 + (r / alpha)^2)^(1 - beta).
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * np.sqrt(
                WING_FLUX ** (1 / (1 - self._beta)) - 1))

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return (2 * self._extent() / step[0],
                2 * self._extent() / step[1])

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        r = _create_grid_2d(
            size, step, offset, self._ratio, self._posa - rota)
        data = gbkfit.math.moffat_1d_fun(r, 1, 0, self._alpha, self._beta)
        data[r > self._extent()] = 0
        return data / np.sum(data)


class PSFImage(PSF):
    """
    An PSF defined by an image, loaded from a FITS file.

    The image is resampled based on the provided step size. It is in the
    frame of the pixels of the data, so it is not rotated with the grid.
    """

    @staticmethod
    def type() -> str:
        return 'image'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'PSFImage':
        # Read the image, and its pixel scale in arcsec
        data, coords = parseutils.load_option(
            fitsutils.read_data, info, 'data', True, False)
        info.update(data=data, step=info.get('step', coords.step))
        return cls(**_load_psf_common(cls, info))

    def dump(self, filename='psf.fits', overwrite=False):
        info = dict(type=self.type(), data=filename, step=self._step)
        rpix = tuple(np.array(self._data.shape[::-1]) / 2 - 0.5)
        coords = fitsutils.Coords(tuple(self._step), rpix, (0.0, 0.0), 0.0)
        fitsutils.write_data(filename, self._data, coords, None, overwrite)
        return info

    def __init__(
            self, data: np.ndarray, step: tuple[float, float] = (1.0, 1.0)
    ):
        data = np.squeeze(data)  # Remove singleton dimensions
        if data.ndim != 2:
            raise RuntimeError(
                f"expected a 2D PSF image, but got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise RuntimeError(
                "non-finite pixels found in the supplied PSF image")
        self._data = data
        self._step = step

    def _size_impl(self, step: tuple[float, float]) -> tuple[float, float]:
        return (
            (self._step[0] / step[0]) * self._data.shape[1],
            (self._step[1] / step[1]) * self._data.shape[0])

    def _asarray_impl(
            self,
            step: tuple[float, float],
            size: tuple[int, int],
            offset: tuple[int, int],
            rota: float
    ) -> np.ndarray:
        scale_x = step[0] / self._step[0]
        scale_y = step[1] / self._step[1]
        # The centre of the image, and the centre of the PSF in the array,
        # where the analytic PSFs put it
        old_center_x = self._data.shape[1] / 2 - 0.5
        old_center_y = self._data.shape[0] / 2 - 0.5
        new_center_x = size[0] // 2 + offset[0]
        new_center_y = size[1] // 2 + offset[1]
        # Like all images, the arrays have shape (y, x)
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        nx = (x - new_center_x) * scale_x + old_center_x
        ny = (y - new_center_y) * scale_y + old_center_y
        data = scipy.ndimage.map_coordinates(self._data, [ny, nx], order=5)  # noqa
        return data / np.sum(data)
