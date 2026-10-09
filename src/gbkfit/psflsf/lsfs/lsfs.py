
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.special

import gbkfit.math
from gbkfit.psflsf.core import (
    LSF, MIN_EXTENT, WING_FLUX, check_scale, lsf_parser)
from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'LSFPoint',
    'LSFGauss',
    'LSFGGauss',
    'LSFLorentz',
    'LSFMoffat',
    'LSFImage',
    'LSFSum'
]


def _create_grid_1d(size: int, step: float, offset: int) -> np.ndarray:
    center = size // 2 + offset
    return (np.array(range(size)) - center) * step


def _sum_weights(weights, n, desc):
    """
    The weights of the terms of a sum (one for each of n terms), each
    positive, normalised to sum to 1.
    """
    weights = np.asarray(weights, dtype=float)
    if n < 1:
        raise RuntimeError(f"{desc} needs at least one term")
    if weights.shape != (n,) or not np.all(weights > 0):
        raise RuntimeError(
            f"{desc} needs a positive weight for each of its {n} terms; "
            f"its weights are {weights.tolist()}")
    return weights / weights.sum()


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
        check_scale('sigma', sigma)
        self._sigma = sigma

    def _extent(self):
        """
        The half-width out to which it is drawn (see MIN_EXTENT). The
        flux of a Gaussian within |z| < w is erf(w / sqrt(2) sigma).
        """
        return max(
            MIN_EXTENT * self._sigma,
            self._sigma * np.sqrt(2) * scipy.special.erfinv(1 - WING_FLUX))

    def _size_impl(self, step: float) -> float:
        return 2 * self._extent() / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.gauss_1d_fun(z, 1, 0, self._sigma)
        data[np.abs(z) > self._extent()] = 0
        return data / np.sum(data)


class LSFGGauss(LSF):
    """
    A Generalized Gaussian Line Spread Function (LSF).
    """


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
        check_scale('alpha', alpha)
        check_scale('beta', beta)
        self._alpha = alpha
        self._beta = beta

    def _extent(self):
        """
        The half-width out to which it is drawn (see MIN_EXTENT). The
        flux of a ggauss within |z| < w is P(1 / beta, (w / alpha)^beta),
        P the regularized lower incomplete gamma function.
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * scipy.special.gammaincinv(
                1 / self._beta, 1 - WING_FLUX) ** (1 / self._beta))

    def _size_impl(self, step: float) -> float:
        return 2 * self._extent() / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.ggauss_1d_fun(z, 1, 0, self._alpha, self._beta)
        data[np.abs(z) > self._extent()] = 0
        return data / np.sum(data)


class LSFLorentz(LSF):
    """
    A Lorentzian Line Spread Function (LSF).
    """


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
        check_scale('gamma', gamma)
        self._gamma = gamma

    def _extent(self):
        """
        The half-width out to which it is drawn (see MIN_EXTENT). The
        flux of a Lorentzian within |z| < w is 2 / pi arctan(w / gamma).
        """
        return max(
            MIN_EXTENT * self._gamma,
            self._gamma * np.tan(np.pi / 2 * (1 - WING_FLUX)))

    def _size_impl(self, step: float) -> float:
        return 2 * self._extent() / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.lorentz_1d_fun(z, 1, 0, self._gamma)
        data[np.abs(z) > self._extent()] = 0
        return data / np.sum(data)


class LSFMoffat(LSF):
    """
    A Moffat Line Spread Function (LSF).
    """


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
        check_scale('alpha', alpha)
        if not beta > 0.5:
            raise RuntimeError(
                "a Moffat LSF has finite flux only for beta > 0.5; "
                f"beta is {beta}")
        self._alpha = alpha
        self._beta = beta

    def _extent(self):
        """
        The half-width out to which it is drawn (see MIN_EXTENT). The
        flux of a Moffat within |z| < w is I_u(1/2, beta - 1/2), I the
        regularized incomplete beta function and u = w^2 / (alpha^2 + w^2).
        """
        return max(
            MIN_EXTENT * self._alpha,
            self._alpha * np.sqrt(1 / (1 - scipy.special.betaincinv(
                0.5, self._beta - 0.5, 1 - WING_FLUX)) - 1))

    def _size_impl(self, step: float) -> float:
        return 2 * self._extent() / step

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
        z = _create_grid_1d(size, step, offset)
        data = gbkfit.math.moffat_1d_fun(z, 1, 0, self._alpha, self._beta)
        data[np.abs(z) > self._extent()] = 0
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
        # Read the image, and its channel width in km/s
        data, coords = parseutils.load_option(
            fitsutils.read_data, info, 'data', True, False)
        info.update(data=data, step=info.get('step', coords.step[0]))
        return cls(**_load_lsf_common(cls, info))

    def dump(
            self, filename='lsf.fits', overwrite: bool = False
    ) -> dict[str, Any]:
        info = dict(type=self.type(), data=filename, step=self._step)
        rpix = self._data.size / 2 - 0.5
        coords = fitsutils.Coords((self._step,), (rpix,), (0.0,), 0.0)
        fitsutils.write_data(filename, self._data, coords, 0, overwrite)
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
        # The centre of the image, and the centre of the LSF in the array,
        # where the analytic LSFs put it
        old_center = self._data.shape[0] / 2 - 0.5
        new_center = size // 2 + offset
        x = np.arange(size)
        nx = (x - new_center) * scale + old_center
        data = scipy.ndimage.map_coordinates(self._data, [nx], order=5)  # noqa
        return data / np.sum(data)


class LSFSum(LSF):
    """
    A sum of LSFs (e.g. a double Gaussian), each with its fraction of the
    light: the weights, normalised to sum to 1.
    """

    @staticmethod
    def type() -> str:
        return 'sum'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFSum':
        parseutils.load_option_and_update_info(
            lsf_parser, info, 'lsfs', required=True)
        return cls(**_load_lsf_common(cls, info))

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(),
            lsfs=lsf_parser.dump(list(self._lsfs)),
            weights=self._weights.tolist())

    def __init__(self, lsfs: Sequence[LSF], weights: Sequence[float]):
        self._lsfs = tuple(lsfs)
        self._weights = _sum_weights(weights, len(self._lsfs), "a sum of LSFs")

    def _size_impl(self, step: float) -> float:
        return max(lsf._size_impl(step) for lsf in self._lsfs)

    def _asarray_impl(self, step: float, size: int, offset: int) -> np.ndarray:
        # Each term holds its fraction of the light on the same grid
        return sum(
            weight * lsf.asarray(step, size, offset)
            for weight, lsf in zip(self._weights, self._lsfs))
