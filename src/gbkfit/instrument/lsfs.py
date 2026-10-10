"""
Line spread functions (LSFs): how the instrument spreads the light of a
single velocity along the spectral axis.

Widths are in km/s.
"""

import abc
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.ndimage
import scipy.special

import gbkfit.math
from gbkfit.utils import fitsutils, gridutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail
from ._detail import MIN_EXTENT, WING_FLUX, check_scale


__all__ = [
    'LSF',
    'LSFConvolution',
    'LSFGauss',
    'LSFGGauss',
    'LSFHanning',
    'LSFImage',
    'LSFLorentz',
    'LSFMoffat',
    'LSFPoint',
    'LSFSum',
    'lsf_parser'
]


class LSF(parseutils.TypedSerializable, abc.ABC):
    """
    A line spread function, drawn on arrays of channels.
    """

    @abc.abstractmethod
    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        """
        Dump the LSF to its configuration.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of its data (e.g. an
            image), if any.
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.
        """

    def size(self, step: float) -> int:
        """
        Return the size of the array that holds the LSF.

        Parameters
        ----------
        step : float
            The width of the channels (km/s).

        Returns
        -------
        int
            The number of channels, odd.

        Raises
        ------
        RuntimeError
            If the wings of the LSF are too heavy for a finite array.
        """
        size = self._size_impl(step)
        _detail.check_finite_size(self, size)
        return int(gbkfit.math.roundu_odd(size))

    def asarray(
            self,
            step: float,
            size: int | None = None,
            offset: int = 0
    ) -> np.ndarray:
        """
        Return the LSF on an array of channels, normalised to sum to 1.

        Parameters
        ----------
        step : float
            The width of the channels (km/s).
        size : int, optional
            The number of channels; by default, that of size.
        offset : int, optional
            The offset of the centre of the LSF from the channel
            size // 2. size + offset must be odd.

        Returns
        -------
        ndarray
            The LSF.

        Raises
        ------
        RuntimeError
            If size + offset is even.
        """
        if size is None:
            size = self.size(step)
        if gbkfit.math.is_even(size + offset):
            raise RuntimeError(
                f"invalid LSF size: (size + offset) = "
                f"({size} + {offset} = {size + offset}), "
                f"but it must be odd")
        return self._asarray_impl(step, size, offset)

    @abc.abstractmethod
    def _size_impl(self, step: float) -> float:
        """The size of the array that holds the LSF, before rounding."""

    @abc.abstractmethod
    def _asarray_impl(
            self,
            step: float,
            size: int,
            offset: int
    ) -> np.ndarray:
        """The LSF on an array (see asarray)."""


def _create_grid_1d(size: int, step: float, offset: int) -> np.ndarray:
    """The velocities of the channels of an array from its centre."""
    center = size // 2 + offset
    return (np.array(range(size)) - center) * step


class LSFPoint(LSF):
    """
    A point: all the light stays in its channel.
    """

    @staticmethod
    def type() -> str:
        return 'point'

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
    A Gaussian, exp(-v^2 / 2 sigma^2).

    Parameters
    ----------
    sigma : float
        The standard deviation (km/s).
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

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
    A generalised Gaussian, exp(-(|v| / alpha)^beta).

    Parameters
    ----------
    alpha : float
        The scale (km/s).
    beta : float
        The shape, > 0: 2 is a Gaussian, and 1 an exponential.
    """

    @staticmethod
    def type() -> str:
        return 'ggauss'

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
    A Lorentzian, gamma^2 / (v^2 + gamma^2).

    Parameters
    ----------
    gamma : float
        The scale (km/s): the half width at half maximum.
    """

    @staticmethod
    def type() -> str:
        return 'lorentz'

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
    A Moffat, (1 + (v / alpha)^2)^(-beta).

    Parameters
    ----------
    alpha : float
        The scale of the core (km/s).
    beta : float
        The slope of the wings, > 0.5 (for a finite flux).
    """

    @staticmethod
    def type() -> str:
        return 'moffat'

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
            raise ConfigError(
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
    An LSF given as a 1D image, centred on the centre of the image, and
    resampled to the channels it is drawn on (with splines of order 5).

    Its configuration has its file (a filename, or a dict with the
    filename and the HDU), and the width of its channels comes from the
    header unless step is given.

    Parameters
    ----------
    data : ndarray
        The image, finite.
    step : float, optional
        The width of the channels of the image (km/s).
    """

    @staticmethod
    def type() -> str:
        return 'image'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'LSFImage':
        data, coords = parseutils.load_option(
            _detail.read_image, info, 'file', required=True)
        info = dict(info)
        del info['file']
        info.update(data=data, step=info.get('step', coords.step[0]))
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        filename = f'{prefix}lsf.fits'
        rpix = self._data.size / 2 - 0.5
        coords = gridutils.Coords((self._step,), (rpix,), (0.0,), 0.0)
        fitsutils.write_data(filename, self._data, coords, 0, overwrite)
        return dict(
            type=self.type(),
            file=filename if dump_path else os.path.basename(filename),
            step=self._step)

    def __init__(self, data: np.ndarray, step: float = 1.0):
        data = np.squeeze(data)  # Remove singleton dimensions
        if data.ndim != 1:
            raise ConfigError(
                f"expected a 1D LSF image, but got shape {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ConfigError(
                "non-finite pixels found in the supplied LSF image")
        self._data = data
        self._step = step

    def _size_impl(self, step: float) -> float:
        return (self._step / step) * self._data.shape[0]

    def _asarray_impl(
            self, step: float, size: int, offset: int
    ) -> np.ndarray:
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


class LSFSum(LSF):
    """
    A sum of LSFs (e.g. a double Gaussian), each with its fraction of the
    light.

    Parameters
    ----------
    lsfs : Sequence of LSF
        The LSFs.
    weights : Sequence of float
        The weight of each LSF, positive; they are normalised to sum to 1.
    """

    @staticmethod
    def type() -> str:
        return 'sum'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'LSFSum':
        parseutils.load_option_and_update_info(
            lsf_parser, info, 'lsfs', required=True)
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            lsfs=_detail.dump_terms(
                lsf_parser, self._lsfs, prefix, dump_path, overwrite),
            weights=self._weights.tolist())

    def __init__(self, lsfs: Sequence[LSF], weights: Sequence[float]):
        self._lsfs = tuple(lsfs)
        self._weights = _detail.sum_weights(
            weights, len(self._lsfs), "a sum of LSFs")

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

    Parameters
    ----------
    lsfs : Sequence of LSF
        The LSFs, at least one.
    """

    @staticmethod
    def type() -> str:
        return 'convolution'

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'LSFConvolution':
        parseutils.load_option_and_update_info(
            lsf_parser, info, 'lsfs', required=True)
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        return dict(
            type=self.type(),
            lsfs=_detail.dump_terms(
                lsf_parser, self._lsfs, prefix, dump_path, overwrite))

    def __init__(self, lsfs: Sequence[LSF]):
        if not lsfs:
            raise ConfigError("a convolution of LSFs needs at least one LSF")
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
        return _detail.embed(data / data.sum(), (size,), (offset,))


class LSFHanning(LSF):
    """
    The response of the Hanning smoothing of channels: 1/4, 1/2 and 1/4 of
    the light of each channel go to the channel before, the channel itself
    and the channel after.

    Parameters
    ----------
    width : float
        The width of the smoothed channels (km/s); a multiple of the
        width of the channels it is drawn on (e.g. those of the data,
        with any oversampling).
    """

    @staticmethod
    def type() -> str:
        return 'hanning'

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
        return _detail.embed(data, (size,), (offset,))


lsf_parser = parseutils.TypedParser(LSF, [
    LSFPoint,
    LSFGauss,
    LSFGGauss,
    LSFLorentz,
    LSFMoffat,
    LSFImage,
    LSFSum,
    LSFConvolution,
    LSFHanning])
