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

from gbkfit.utils import funcutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Prior',
    'PriorGauss',
    'PriorGaussTrunc',
    'PriorHalfCauchy',
    'PriorHalfGauss',
    'PriorLogUniform',
    'PriorSine',
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


def _check_scale(scale: float) -> None:
    """Check that a scale is finite and positive."""
    if not (math.isfinite(scale) and scale > 0):
        raise ConfigError(
            f"the scale must be finite and positive; it is {scale}")


class Prior(parseutils.TypedSerializable, abc.ABC):
    """
    The prior of a free element: a probability distribution of its value.
    Priors are immutable, and take arrays of values.

    The support (minimum, maximum) is where the density is positive; the
    finite differences of the uncertainty stay inside it. The nested
    samplers draw from the unit cube through ppf, the MCMC samplers add
    log_pdf to the log-likelihood, and the optimizers add chi2 to the
    chi-square (least squares: residual to the residuals; see
    FitProblem.cost).
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'Prior':
        # (config: min and max, of the priors that have a range)
        names = funcutils.parameter_names(cls.__init__)
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__, rename_params={
                name: name[:3] for name in ('minimum', 'maximum')
                if name in names.required + names.optional}))

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

    @abc.abstractmethod
    def chi2(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Return the chi-square of the prior: -2 log_pdf(x) plus a constant
        that makes it 0 at the most probable value (or at the mean of a
        truncated Gaussian); inf outside the support.
        """

    def has_residual(self) -> bool:
        """
        Return whether the prior has a residual for least squares (see
        residual).
        """
        return False

    def residual(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Return the residual of the prior for least squares: a smooth
        function r of the value with r^2 = chi2(x).

        Raises
        ------
        TypeError
            If the prior has none (see has_residual).
        """
        raise TypeError(
            f"a prior of type '{self.type()}' has no residual for least "
            f"squares")

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

    def chi2(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        inside = (x >= self._minimum) & (x <= self._maximum)
        return np.where(inside, 0.0, np.inf)

    def has_residual(self) -> bool:
        return True

    def residual(self, x: npt.ArrayLike) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=float))

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

    def chi2(self, x: npt.ArrayLike) -> np.ndarray:
        # (no residual: its square root has an infinite slope at minimum)
        x = np.asarray(x, dtype=float)
        inside = (x >= self._minimum) & (x <= self._maximum)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(inside, 2 * np.log(x / self._minimum), np.inf)

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self._minimum * np.exp(u * self._log_ratio)


class PriorSine(Prior):
    """
    A sine prior on an angle in degrees: p(x) proportional to sin x, the
    distribution of the inclination of a disk oriented at random (seen
    edge-on more often than face-on).

    Its chi-square is -2 ln sin x, 0 at 90 degrees also when the range
    ends before (as a truncated Gaussian's is 0 at its mean), so that the
    residual, sign(x - 90) sqrt(chi2), is smooth everywhere.

    Parameters
    ----------
    minimum, maximum : float, optional
        The range, in degrees, within [0, 180] (config: min, max; by
        default [0, 90]).

    Raises
    ------
    ConfigError
        If the range is not within [0, 180], or is empty.
    """

    @staticmethod
    def type() -> str:
        return 'sine'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), min=self._minimum, max=self._maximum)

    def __init__(self, minimum: float = 0.0, maximum: float = 90.0):
        _check_range(minimum, maximum)
        if not (0 <= minimum and maximum <= 180):
            raise ConfigError(
                f"the range of a sine prior must be within [0, 180] "
                f"degrees; it is [{minimum}, {maximum}]")
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        # (the cosines of the ends, whose difference normalises it)
        self._cos_min = math.cos(math.radians(self._minimum))
        self._cos_max = math.cos(math.radians(self._maximum))

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum

    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        inside = (x >= self._minimum) & (x <= self._maximum)
        # (a density per degree)
        norm = math.log(math.radians(1) / (self._cos_min - self._cos_max))
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(
                inside, np.log(np.sin(np.radians(x))) + norm, -np.inf)

    def chi2(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        inside = (x >= self._minimum) & (x <= self._maximum)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(
                inside, -2 * np.log(np.sin(np.radians(x))), np.inf)

    def has_residual(self) -> bool:
        return True

    def residual(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.sign(x - 90) * np.sqrt(self.chi2(x))

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return np.degrees(np.arccos(
            self._cos_min - u * (self._cos_min - self._cos_max)))


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

    def chi2(self, x: npt.ArrayLike) -> np.ndarray:
        return self.residual(x) ** 2

    def has_residual(self) -> bool:
        return True

    def residual(self, x: npt.ArrayLike) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self._mean) / self._std

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

    def chi2(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        inside = (x >= self._minimum) & (x <= self._maximum)
        return np.where(inside, self.residual(x) ** 2, np.inf)

    def has_residual(self) -> bool:
        return True

    def residual(self, x: npt.ArrayLike) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self._mean) / self._std

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        return np.asarray(self._dist.ppf(np.asarray(u, dtype=float)))


class PriorHalfGauss(Prior):
    """
    A half-Gaussian prior on a value of at least 0: a Gaussian of mean 0
    and standard deviation s, folded onto x >= 0 (e.g. for a jitter that
    is likely below 2 s, as 95% of the prior is). Its chi-square is (x /
    s)^2, and its residual x / s.

    Parameters
    ----------
    std : float
        The standard deviation s of the Gaussian.

    Raises
    ------
    ConfigError
        If the standard deviation is not finite and positive.
    """

    @staticmethod
    def type() -> str:
        return 'half_gauss'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), std=self._std)

    def __init__(self, std: float):
        _check_std(std)
        self._std = float(std)

    def std(self) -> float:
        """Return the standard deviation of the Gaussian."""
        return self._std

    def minimum(self) -> float:
        return 0.0

    def maximum(self) -> float:
        return math.inf

    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        norm = math.log(math.sqrt(2 / math.pi) / self._std)
        return np.where(x >= 0, norm - 0.5 * (x / self._std) ** 2, -np.inf)

    def chi2(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, self.residual(x) ** 2, np.inf)

    def has_residual(self) -> bool:
        return True

    def residual(self, x: npt.ArrayLike) -> np.ndarray:
        return np.asarray(x, dtype=float) / self._std

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self._std * scipy.special.ndtri((1 + u) / 2)


class PriorHalfCauchy(Prior):
    """
    A half-Cauchy prior on a value of at least 0: p(x) proportional to 1 /
    (1 + (x / s)^2), x >= 0, of scale s. About flat below s, with a heavy
    tail, it lets the data prefer values well above s: a weakly
    informative prior for scale parameters, e.g. a jitter (Gelman 2006,
    Bayesian Analysis 1, 515).

    Its chi-square is 2 log(1 + (x / s)^2), and its residual sign(x)
    sqrt(2 log(1 + (x / s)^2)), which has the slope sqrt(2) / s at 0.

    Parameters
    ----------
    scale : float
        The scale s, the median.

    Raises
    ------
    ConfigError
        If the scale is not finite and positive.
    """

    @staticmethod
    def type() -> str:
        return 'half_cauchy'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type(), scale=self._scale)

    def __init__(self, scale: float):
        _check_scale(scale)
        self._scale = float(scale)

    def scale(self) -> float:
        """Return the scale."""
        return self._scale

    def minimum(self) -> float:
        return 0.0

    def maximum(self) -> float:
        return math.inf

    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        norm = math.log(2 / (math.pi * self._scale))
        return np.where(
            x >= 0, norm - np.log1p((x / self._scale) ** 2), -np.inf)

    def chi2(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where(
            x >= 0, 2 * np.log1p((x / self._scale) ** 2), np.inf)

    def has_residual(self) -> bool:
        return True

    def residual(self, x: npt.ArrayLike) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.sign(x) * np.sqrt(2 * np.log1p((x / self._scale) ** 2))

    def ppf(self, u: npt.ArrayLike) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self._scale * np.tan(np.pi / 2 * u)


prior_parser = parseutils.TypedParser(Prior)
prior_parser.register(PriorUniform)
prior_parser.register(PriorLogUniform)
prior_parser.register(PriorSine)
prior_parser.register(PriorGauss)
prior_parser.register(PriorGaussTrunc)
prior_parser.register(PriorHalfGauss)
prior_parser.register(PriorHalfCauchy)
