"""
The solves of parameters: how the values of solved elements are worked
out at each evaluation, rather than fitted.

An element of a parameter that the model is linear in (e.g. a scale of
the flux of an observation) can be solved: the objective compares the
model with the data, and the solve of the element gives its best value
from the sums of that comparison, under its prior (see Solve).
"""

import abc
import math
from typing import NamedTuple

import numpy as np
import scipy.special

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Solution',
    'Solve',
    'SolveGaussian',
    'solve_parser'
]


class Solution(NamedTuple):
    """
    What a solve gives at an evaluation (see Solve): the solved values,
    the residuals of their prior (one for each value, flat), which count
    in the chi-square, and the term of the log-likelihood beyond -chi2 / 2
    (of the values integrated out under their prior).
    """
    values: np.ndarray
    residuals: np.ndarray
    log_term: float


class Solve(parseutils.TypedSerializable, abc.ABC):
    """
    How the values of elements of parameters that the model is linear in
    (e.g. a scale of the flux of the model) are solved at each evaluation
    of an objective, rather than fitted: each is the best value for the
    data under its prior, from the sums of the comparison of the model
    with the data (see Comparison.scale_sums), with exact formulas.
    Optimizers take the best values, with the residuals of their prior;
    samplers take the log-likelihood with the values integrated out under
    their prior. Each solved value counts as a parameter in the degrees of
    freedom, and its prior as a residual: they cancel (see
    Objective.nnuisance).
    """

    @abc.abstractmethod
    def solve(self, a: np.ndarray, b: np.ndarray) -> Solution:
        """
        Return the solved values of the sums of their values: A = sum w
        m^2 and B = sum w m d over the values that each scales (m the
        model, d the data and w the weight of the value in the
        chi-square), arrays of one shape, one of each per solved value.
        """
        pass

    @abc.abstractmethod
    def bounds(self) -> tuple[float, float]:
        """Return the smallest and the largest values it can give."""
        pass

    def default(self) -> float:
        """
        Return its value without data (e.g. of an evaluation of the model
        alone): that of sums of 0.
        """
        return float(self.solve(np.zeros(1), np.zeros(1)).values[0])


class SolveGaussian(Solve):
    """
    Solved values whose prior is a Gaussian of a mean (mu) and a standard
    deviation (s), cut below a minimum (c), if any: each value x times the
    model, of the sums A and B of its values, adds to the chi-square
    A x^2 - 2 B x (and a term without x) and the chi-square of its prior,
    (x - mu)^2 / s^2. With A' = A + 1/s^2 and B' = B + mu/s^2:

    - the best value is x = max(B' / A', c), and the residual of its prior
      (x - mu) / s;
    - the log of the integral over x >= c of exp(-chi2(x) / 2) times the
      prior is -chi2 / 2 at the best value + log(Phi(z) / (s sqrt(A')
      Phi((mu - c) / s))) + (z^2 / 2 if z < 0), with z = (B' - c A') /
      sqrt(A') and Phi the normal cdf (without a minimum, the terms of Phi
      are 0).

    A value whose values do not count in the chi-square (A = 0) is the
    mean, without a residual or a term.

    Parameters
    ----------
    mean : float
        The mean of the prior.
    sigma : float
        The standard deviation of the prior, finite and positive.
    minimum : float, optional
        The smallest value (e.g. 0 for a scale of the flux); none by
        default.

    Raises
    ------
    ConfigError
        If the mean or the minimum is not finite, or sigma is not finite
        and positive.
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def dump(self) -> dict[str, str | float]:
        minimum = {} if self._minimum is None else dict(minimum=self._minimum)
        return dict(type=self.type(), mean=self._mean, sigma=self._sigma) \
            | minimum

    def __init__(
            self,
            mean: float,
            sigma: float,
            minimum: float | None = None
    ):
        if not math.isfinite(mean):
            raise ConfigError(f"the mean must be finite; it is {mean}")
        if not (math.isfinite(sigma) and sigma > 0):
            raise ConfigError(
                f"sigma must be finite and positive; it is {sigma}")
        if minimum is not None and not math.isfinite(minimum):
            raise ConfigError(f"the minimum must be finite; it is {minimum}")
        self._mean = float(mean)
        self._sigma = float(sigma)
        self._minimum = None if minimum is None else float(minimum)

    def bounds(self) -> tuple[float, float]:
        return -math.inf if self._minimum is None else self._minimum, math.inf

    def solve(self, a: np.ndarray, b: np.ndarray) -> Solution:
        mean, sigma, minimum = self._mean, self._sigma, self._minimum
        precision = 1 / sigma ** 2
        a, b = a + precision, b + mean * precision
        values = b / a
        if minimum is not None:
            values = np.maximum(values, minimum)
        terms = -np.log(sigma * np.sqrt(a))
        if minimum is not None:
            z = (b - minimum * a) / np.sqrt(a)
            terms = np.where(z < 0, 0.5 * z ** 2, 0.0) \
                + scipy.special.log_ndtr(z) + terms \
                - scipy.special.log_ndtr((mean - minimum) / sigma)
        return Solution(
            values, ((values - mean) / sigma).ravel(), float(np.sum(terms)))


solve_parser = parseutils.TypedParser(Solve, [SolveGaussian])
