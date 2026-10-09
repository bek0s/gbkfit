
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.special

import gbkfit.math
from gbkfit.psflsf.core import (
    LSF, MIN_EXTENT, WING_FLUX, check_scale, embed, lsf_parser)
from gbkfit.utils import fitsutils, parseutils


__all__ = [
    'LSFPoint',
    'LSFGauss',
    'LSFGGauss',
    'LSFLorentz',
    'LSFMoffat',
    'LSFImage',
    'LSFSum',
    'LSFConvolution',
    'LSFHanning'
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

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
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

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
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

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
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

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
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

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
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
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}lsf.fits'
        rpix = self._data.size / 2 - 0.5
        coords = fitsutils.Coords((self._step,), (rpix,), (0.0,), 0.0)
        fitsutils.write_data(filename, self._data, coords, 0, overwrite)
        return dict(
            type=self.type(),
            data=filename if dump_path else os.path.basename(filename),
            step=self._step)

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
        data = scipy.ndimage.map_coordinates(
            self._data, [nx], order=5, mode='grid-constant')
        return data / np.sum(data)


def _dump_terms(
        lsfs: Sequence[LSF], prefix: str, dump_path: bool, overwrite: bool
) -> list[dict[str, Any]]:
    """
    The options of the LSFs of a sum or a convolution, the files of each
    named with its own prefix (term0_, term1_, ...).
    """
    return [
        lsf_parser.dump(
            lsf, prefix=f'{prefix}term{i}_', dump_path=dump_path,
            overwrite=overwrite)
        for i, lsf in enumerate(lsfs)]


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

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            lsfs=_dump_terms(self._lsfs, prefix, dump_path, overwrite),
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


class LSFConvolution(LSF):
    """
    The convolution of LSFs: e.g. the LSF of an instrument and the
    Hanning smoothing of the channels of the data (see LSFHanning).
    """

    @staticmethod
    def type() -> str:
        return 'convolution'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFConvolution':
        parseutils.load_option_and_update_info(
            lsf_parser, info, 'lsfs', required=True)
        return cls(**_load_lsf_common(cls, info))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            lsfs=_dump_terms(self._lsfs, prefix, dump_path, overwrite))

    def __init__(self, lsfs: Sequence[LSF]):
        if not lsfs:
            raise RuntimeError("a convolution of LSFs needs at least one LSF")
        self._lsfs = tuple(lsfs)

    def _size_impl(self, step: float) -> float:
        # The sum of the (odd) sizes of the terms, less one for each
        # convolution
        sizes = [lsf.size(step) for lsf in self._lsfs]
        return sum(sizes) - len(sizes) + 1

    def _asarray_impl(self, step: float, size: int, offset: int) -> np.ndarray:
        data = np.ones(1)
        for lsf in self._lsfs:
            data = np.convolve(data, lsf.asarray(step))
        return embed(data / data.sum(), (size,), (offset,))


class LSFHanning(LSF):
    """
    The response of the Hanning smoothing of channels of the given width
    (km/s): 1/4, 1/2 and 1/4 of the light of each channel goes to the
    channel before, the channel itself and the channel after. The width
    must be a multiple of the step it is drawn with (e.g. the channels of
    the data, with any oversampling).
    """

    @staticmethod
    def type() -> str:
        return 'hanning'

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'LSFHanning':
        return cls(**_load_lsf_common(cls, info))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(type=self.type(), width=self._width)

    def __init__(self, width: float):
        check_scale('width', width)
        self._width = width

    def _steps(self, step: float) -> int:
        """The number of steps in a channel width."""
        steps = self._width / step
        if abs(steps - round(steps)) > 1e-6 * steps:
            raise RuntimeError(
                f"the channel width of the Hanning smoothing ({self._width}) "
                f"must be a multiple of the step it is drawn with ({step})")
        return round(steps)

    def _size_impl(self, step: float) -> float:
        return 2 * self._steps(step) + 1

    def _asarray_impl(self, step: float, size: int, offset: int) -> np.ndarray:
        steps = self._steps(step)
        data = np.zeros(2 * steps + 1)
        data[[0, steps, 2 * steps]] = 0.25, 0.5, 0.25
        return embed(data, (size,), (offset,))
