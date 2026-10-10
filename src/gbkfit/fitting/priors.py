"""
The priors of the free parameters of a fit: one probability distribution
for each free element, independent of the others.
"""

import abc
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.special
import scipy.stats

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Prior',
    'PriorGauss',
    'PriorGaussTrunc',
    'PriorLogUniform',
    'PriorUniform',
    'prior_parser'
]


def _check_range(minimum: float, maximum: float) -> None:
    """Check that a range is finite and not empty."""
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        raise ConfigError(
            f"the minimum and maximum must be finite; they are {minimum} "
            f"and {maximum}")
    if minimum >= maximum:
        raise ConfigError(
            f"the minimum must be less than the maximum; they are "
            f"{minimum} and {maximum}")


def _check_std(std: float) -> None:
    """Check that a standard deviation is finite and positive."""
    if not (math.isfinite(std) and std > 0):
        raise ConfigError(
            f"the standard deviation must be finite and positive; it is "
            f"{std}")


class Prior(parseutils.TypedSerializable, abc.ABC):
    """
    The prior of a free element: a probability distribution of its value.
    Priors are immutable, and take arrays of values.

    The support (minimum, maximum) is where the density is positive; the
    finite differences of the uncertainty stay inside it. The nested
    samplers draw from the unit cube through ppf, and the MCMC samplers
    add log_pdf to the log-likelihood.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'Prior':
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__,
            rename_params=dict(minimum='min', maximum='max')))

    @abc.abstractmethod
    def minimum(self) -> float:
        """Return the lower end of the support (may be -inf)."""

    @abc.abstractmethod
    def maximum(self) -> float:
        """Return the upper end of the support (may be inf)."""

    @abc.abstractmethod
    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Return the log of the probability density.

        Parameters
        ----------
        x : ArrayLike
            The values.

        Returns
        -------
        np.ndarray
            The log-density of each value (-inf outside the support).
        """

    @abc.abstractmethod
    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        """
        Return the percent point function: the inverse of the cumulative
        distribution function.

        Parameters
        ----------
        u : ArrayLike
            The probabilities, in [0, 1].

        Returns
        -------
        np.ndarray
            The value of each probability.
        """

    def sample(
            self,
            rng: np.random.Generator,
            size: int | tuple[int, ...] | None = None
    ) -> np.ndarray:
        """
        Draw random values.

        Parameters
        ----------
        rng : np.random.Generator
            The random number generator.
        size : int or tuple of int, optional
            The shape of the values (by default, one value).

        Returns
        -------
        np.ndarray
            The values.
        """
        return self.ppf(rng.uniform(size=size))

    def width(self) -> float:
        """
        Return a typical width of the distribution: the distance between
        its 16th and 84th percentiles (two standard deviations, for a
        Gaussian).
        """
        return float(self.ppf(0.84) - self.ppf(0.16))


class PriorUniform(Prior):
    """
    A uniform prior.

    Parameters
    ----------
    minimum, maximum : float
        The range (config: min, max).

    Raises
    ------
    ConfigError
        If the range is not finite or is empty.
    """

    @staticmethod
    def type() -> str:
        return 'uniform'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), min=self._minimum, max=self._maximum)

    def __init__(self, minimum: float, maximum: float):
        _check_range(minimum, maximum)
        self._minimum = float(minimum)
        self._maximum = float(maximum)

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum

    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        inside = (x >= self._minimum) & (x <= self._maximum)
        return np.where(
            inside, -math.log(self._maximum - self._minimum), -np.inf)

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self._minimum + u * (self._maximum - self._minimum)


class PriorLogUniform(Prior):
    """
    A log-uniform prior: uniform in the logarithm of the value (for a
    positive scale whose order of magnitude is unknown).

    Parameters
    ----------
    minimum, maximum : float
        The range, positive (config: min, max).

    Raises
    ------
    ConfigError
        If the range is not finite and positive, or is empty.
    """

    @staticmethod
    def type() -> str:
        return 'log_uniform'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), min=self._minimum, max=self._maximum)

    def __init__(self, minimum: float, maximum: float):
        _check_range(minimum, maximum)
        if minimum <= 0:
            raise ConfigError(
                f"the minimum of a log-uniform prior must be positive; it "
                f"is {minimum}")
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._log_ratio = math.log(self._maximum / self._minimum)

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum

    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        inside = (x >= self._minimum) & (x <= self._maximum)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(
                inside, -np.log(x) - math.log(self._log_ratio), -np.inf)

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self._minimum * np.exp(u * self._log_ratio)


class PriorGauss(Prior):
    """
    A Gaussian (normal) prior.

    Parameters
    ----------
    mean : float
        The mean.
    std : float
        The standard deviation.

    Raises
    ------
    ConfigError
        If the mean is not finite, or the standard deviation is not
        finite and positive.
    """

    @staticmethod
    def type() -> str:
        return 'gauss'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), mean=self._mean, std=self._std)

    def __init__(self, mean: float, std: float):
        if not math.isfinite(mean):
            raise ConfigError(f"the mean must be finite; it is {mean}")
        _check_std(std)
        self._mean = float(mean)
        self._std = float(std)

    def mean(self) -> float:
        """Return the mean."""
        return self._mean

    def std(self) -> float:
        """Return the standard deviation."""
        return self._std

    def minimum(self) -> float:
        return -math.inf

    def maximum(self) -> float:
        return math.inf

    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self._mean) / self._std
        return -0.5 * z * z - math.log(self._std * math.sqrt(2 * math.pi))

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        return self._mean + self._std * scipy.special.ndtri(
            np.asarray(u, dtype=float))


class PriorGaussTrunc(Prior):
    """
    A Gaussian prior truncated to a range (and normalised in it).

    Parameters
    ----------
    mean : float
        The mean of the Gaussian before the truncation.
    std : float
        The standard deviation of the Gaussian before the truncation.
    minimum, maximum : float
        The range (config: min, max).

    Raises
    ------
    ConfigError
        If the mean or the range is not finite, the range is empty, or
        the standard deviation is not finite and positive.
    """

    @staticmethod
    def type() -> str:
        return 'gauss_trunc'

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), mean=self._mean, std=self._std,
            min=self._minimum, max=self._maximum)

    def __init__(
            self,
            mean: float,
            std: float,
            minimum: float,
            maximum: float
    ):
        if not math.isfinite(mean):
            raise ConfigError(f"the mean must be finite; it is {mean}")
        _check_std(std)
        _check_range(minimum, maximum)
        self._mean = float(mean)
        self._std = float(std)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        # (scipy's truncnorm keeps the tails far from the mean accurate)
        self._dist = scipy.stats.truncnorm(
            (minimum - mean) / std, (maximum - mean) / std, mean, std)

    def mean(self) -> float:
        """Return the mean of the Gaussian before the truncation."""
        return self._mean

    def std(self) -> float:
        """
        Return the standard deviation of the Gaussian before the
        truncation.
        """
        return self._std

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum

    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray:
        return np.asarray(self._dist.logpdf(np.asarray(x, dtype=float)))

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        return np.asarray(self._dist.ppf(np.asarray(u, dtype=float)))


prior_parser = parseutils.TypedParser(Prior)
prior_parser.register(PriorUniform)
prior_parser.register(PriorLogUniform)
prior_parser.register(PriorGauss)
prior_parser.register(PriorGaussTrunc)
