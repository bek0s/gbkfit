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
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.special

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'LIKELIHOODS',
    'Solution',
    'Solve',
    'SolveGaussian',
    'check_together',
    'solve_parser',
    'solve_together',
    'solve_together_rows'
]


# What the log-likelihood takes of solved values (see Solve): their
# integral under their prior, or their best values (the profile)
LIKELIHOODS = ('integrated', 'profiled')

# The most Newton steps of solve_together_rows
_ROWS_MAX_STEPS = 100


class Solution(NamedTuple):
    """
    What a solve gives at an evaluation (see Solve): the solved values,
    the residuals of their prior (one for each value, flat), which count
    in the chi-square, the term of the log-likelihood beyond -chi2 / 2
    (of the values integrated out under their prior; 0 when profiled),
    and the effective number of the values in the degrees of freedom.
    """
    values: np.ndarray
    residuals: np.ndarray
    log_term: float
    m_eff: float


class Solve(parseutils.TypedSerializable, abc.ABC):
    """
    How the values of elements of parameters that the model is linear in
    (e.g. a scale of the flux of the model) are solved at each evaluation
    of an objective, rather than fitted: the best values for the data
    under their prior, from the sums of the comparison of the model with
    the data (see Comparison.scale_sums), each alone (solve) or together,
    where their parts of the model fall on the same data (solve_coupled).
    Optimizers take the best values, with the residuals of their prior;
    the log-likelihood takes the values integrated out under their prior
    ('integrated'), or at their best values ('profiled': the profile
    likelihood, e.g. where no exact integral exists).

    Each solved value counts in the degrees of freedom by how much the
    data decide it, its effective number: the trace of the hat matrix of
    the values (how much each data value pulls its own fitted value), 1
    for a value that the data decide, 0 for one that its prior decides, or
    that is at its minimum. The residuals of its prior are not of the
    data: they count in the chi-square, but not in the degrees of freedom
    (see Objective.nnuisance).
    """

    @abc.abstractmethod
    def likelihood(self) -> str:
        """
        Return what the log-likelihood takes of the solved values (see
        LIKELIHOODS).
        """
        pass

    @abc.abstractmethod
    def solve(self, a: np.ndarray, b: np.ndarray) -> Solution:
        """
        Return the solved values of the sums of their values, each value
        alone: A = sum w m^2 and B = sum w m d over the values that each
        scales (m the model, d the data and w the weight of the value in
        the chi-square), arrays of one shape, one of each per solved
        value.
        """
        pass

    @abc.abstractmethod
    def check_coupled(self, size: int) -> None:
        """
        Raise ConfigError unless it can solve size values together (see
        solve_coupled).
        """
        pass

    @abc.abstractmethod
    def solve_coupled(self, normal: np.ndarray, rhs: np.ndarray) -> Solution:
        """
        Return the solved values of the sums of their values, together:
        for the model m_0 + sum_k u_k m_k of the values u_k, the matrix
        N_kl = sum w m_k m_l (K x K) and the vector b_k = sum w m_k (d -
        m_0) over the data values.
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
      (x - mu) / s; its effective number A / A' (0 at the minimum);
    - the log of the integral over x >= c of exp(-chi2(x) / 2) times the
      prior is -chi2 / 2 at the best value + log(Phi(z) / (s sqrt(A')
      Phi((mu - c) / s))) + (z^2 / 2 if z < 0), with z = (B' - c A') /
      sqrt(A') and Phi the normal cdf (without a minimum, the terms of Phi
      are 0).

    A value whose values do not count in the chi-square (A = 0) is the
    mean, without a residual or a term.

    Values solved together (see solve_coupled), of the sums N and b, with
    N' = N + I/s^2 and b' = b + mu/s^2: without a minimum, the best values
    solve N' x = b', the integral is -chi2 / 2 at them - log det(N') / 2 -
    K log(s), and their effective number is the trace of N'^-1 N; with a
    minimum, the best values are those of the least squares of N' and b'
    above it (bounded variables), their effective number that trace over
    those above it, and the integral has no exact formula (in more than
    one value): 'profiled' only.

    Parameters
    ----------
    mean : float
        The mean of the prior.
    sigma : float
        The standard deviation of the prior, finite and positive.
    minimum : float, optional
        The smallest value (e.g. 0 for a scale of the flux); none by
        default.
    likelihood : str, optional
        What the log-likelihood takes of the values: 'integrated' (the
        default) or 'profiled' (see Solve).

    Raises
    ------
    ConfigError
        If the mean or the minimum is not finite, sigma is not finite and
        positive, or likelihood is not one of LIKELIHOODS.
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def dump(self) -> dict[str, str | float]:
        info: dict[str, str | float] = dict(
            type=self.type(), mean=self._mean, sigma=self._sigma)
        if self._minimum is not None:
            info.update(minimum=self._minimum)
        if self._likelihood != LIKELIHOODS[0]:
            info.update(likelihood=self._likelihood)
        return info

    def __init__(
            self,
            mean: float,
            sigma: float,
            minimum: float | None = None,
            likelihood: str = LIKELIHOODS[0]
    ):
        if not math.isfinite(mean):
            raise ConfigError(f"the mean must be finite; it is {mean}")
        if not (math.isfinite(sigma) and sigma > 0):
            raise ConfigError(
                f"sigma must be finite and positive; it is {sigma}")
        if minimum is not None and not math.isfinite(minimum):
            raise ConfigError(f"the minimum must be finite; it is {minimum}")
        if likelihood not in LIKELIHOODS:
            raise ConfigError(
                f"likelihood must be one of {list(LIKELIHOODS)}; it is "
                f"{likelihood!r}")
        self._mean = float(mean)
        self._sigma = float(sigma)
        self._minimum = None if minimum is None else float(minimum)
        self._likelihood = likelihood

    def likelihood(self) -> str:
        return self._likelihood

    def bounds(self) -> tuple[float, float]:
        return -math.inf if self._minimum is None else self._minimum, math.inf

    def solve(self, a: np.ndarray, b: np.ndarray) -> Solution:
        mean, sigma, minimum = self._mean, self._sigma, self._minimum
        precision = 1 / sigma ** 2
        a_prior, b_prior = a + precision, b + mean * precision
        values = b_prior / a_prior
        # (the effective number of each value: the share of a_prior of its
        # data, a / a_prior; none at its minimum, which its data do not
        # set)
        free = np.ones(values.shape, dtype=bool)
        if minimum is not None:
            free = values > minimum
            values = np.maximum(values, minimum)
        m_eff = float(np.sum(np.where(free, a / a_prior, 0.0)))
        log_term = 0.0
        if self._likelihood == 'integrated':
            terms = -np.log(sigma * np.sqrt(a_prior))
            if minimum is not None:
                z = (b_prior - minimum * a_prior) / np.sqrt(a_prior)
                terms = np.where(z < 0, 0.5 * z ** 2, 0.0) \
                    + scipy.special.log_ndtr(z) + terms \
                    - scipy.special.log_ndtr((mean - minimum) / sigma)
            log_term = float(np.sum(terms))
        return Solution(
            values, ((values - mean) / sigma).ravel(), log_term, m_eff)

    def mean(self) -> float:
        """Return the mean of the prior."""
        return self._mean

    def sigma(self) -> float:
        """Return the standard deviation of the prior."""
        return self._sigma

    def minimum(self) -> float | None:
        """Return the smallest value, if any."""
        return self._minimum

    def check_coupled(self, size: int) -> None:
        check_together([self], [size])

    def solve_coupled(self, normal: np.ndarray, rhs: np.ndarray) -> Solution:
        return solve_together([self], [len(rhs)], normal, rhs)


def check_together(solves: Sequence[Solve], sizes: Sequence[int]) -> None:
    """
    Raise ConfigError unless values of the given solves, the given number
    of each, can be solved together (see solve_together): of Gaussian
    priors and one likelihood, and 'profiled' if more than one value has
    a minimum (their integral has no exact formula).
    """
    if not all(isinstance(solve, SolveGaussian) for solve in solves):
        raise ConfigError(
            "values solved together take Gaussian priors (solves of type "
            "'gaussian')")
    likelihoods = {solve.likelihood() for solve in solves}
    if len(likelihoods) > 1:
        raise ConfigError(
            f"values solved together take one likelihood; they have "
            f"{sorted(likelihoods)}")
    if sum(sizes) > 1 and likelihoods == {'integrated'} and any(
            solve.minimum() is not None for solve in solves):
        raise ConfigError(
            f"{sum(sizes)} values solved together above a minimum have no "
            f"exact integral for the log-likelihood: give their solves the "
            f"likelihood 'profiled' (the log-likelihood at their best "
            f"values), or no minimum")


def solve_together(
        solves: Sequence[Solve],
        sizes: Sequence[int],
        normal: np.ndarray,
        rhs: np.ndarray
) -> Solution:
    """
    Return the values of the given solves solved together, the given
    number of each in their order, of the sums of their values (see
    Solve.solve_coupled), each of the Gaussian prior of its solve (mean
    mu, standard deviation s, minimum c): with N' = N + diag(1/s^2) and
    b' = b + mu/s^2, without a minimum, the best values solve N' x = b',
    the integral is -chi2 / 2 at them - log det(N') / 2 - sum log(s), and
    their effective number is the trace of N'^-1 N; with a minimum, the
    best values are those of the least squares of N' and b' above it
    (bounded variables), their effective number that trace over those
    above it, and they are profiled (see check_together). One value alone
    takes the formulas of SolveGaussian.

    Raises
    ------
    ConfigError
        If they cannot be solved together (see check_together).
    """
    check_together(solves, sizes)
    if len(rhs) == 1:
        return solves[0].solve(normal[0], rhs)
    mean, sigma, lower = _priors(solves, sizes)
    precision = 1 / sigma ** 2
    normal_prior = normal + np.diag(precision)
    rhs_prior = rhs + mean * precision
    log_term = 0.0
    if np.all(np.isinf(lower)):
        factor = scipy.linalg.cho_factor(normal_prior)
        values = scipy.linalg.cho_solve(factor, rhs_prior)
        free = np.ones(len(rhs), dtype=bool)
        if solves[0].likelihood() == 'integrated':
            log_term = -float(np.sum(np.log(np.diag(factor[0])))) \
                - float(np.sum(np.log(sigma)))
    else:
        # (the least squares of R x = t, R^T R = N' and R^T t = b', whose
        # chi-square is that of the values but a constant)
        upper = scipy.linalg.cholesky(normal_prior)
        target = scipy.linalg.solve_triangular(upper, rhs_prior, trans='T')
        values = scipy.optimize.lsq_linear(
            upper, target, bounds=(lower, np.inf), method='bvls').x
        free = values > lower
    # (the effective number of the values: the trace of N'^-1 N over those
    # above their minimum, which their data set)
    normal_free = normal[np.ix_(free, free)]
    m_eff = float(np.trace(np.linalg.solve(
        normal_free + np.diag(precision[free]), normal_free))) \
        if free.any() else 0.0
    return Solution(values, (values - mean) / sigma, log_term, m_eff)


def solve_together_rows(
        solves: Sequence[Solve],
        sizes: Sequence[int],
        rows: np.ndarray,
        target: np.ndarray,
        ndata: int | None = None
) -> Solution:
    """
    Return the values of the given solves solved together, as
    solve_together, of the rows of their chi-square rather than its sums:
    of the residuals A u - t of the values u (e.g. over the errors of the
    data), N = A^T A and b = A^T t. For many values and fewer rows (e.g.
    the weights of the orbits of Schwarzschild models), as N is never
    made.

    They are solved on the dual, in one variable per row (the semismooth
    Newton method): y = t - A u, u = max(c, mu + s^2 A^T y), whose
    Newton matrix is H = I + A_F S_F^2 A_F^T over the values F above their
    minimum (the dual is quadratic where F holds, so a full step that
    keeps it reaches the solution), from the solution without the minimum
    (from the means of the priors, the minimum cuts the steps short, and
    from the solution of similar rows it takes more of them). Then the
    effective number of the values is n - tr(H^-1), and the integral of
    the likelihood (without a minimum) -log det(H) / 2: those of
    solve_together, by the identities of Woodbury and Sylvester.

    Parameters
    ----------
    solves : sequence of Solve
        The solves.
    sizes : sequence of int
        The number of values of each, in their order.
    rows : np.ndarray
        The matrix A, of shape (n, K) for K values.
    target : np.ndarray
        The vector t, of shape (n,).
    ndata : int, optional
        The number of the rows of the data, the first ones; the others
        are of the prior (e.g. constraints): their residuals count in the
        chi-square, but not in the effective number (that of the rows of
        the data, sum of 1 - (H^-1)_ii over them). By default, all.

    Returns
    -------
    Solution
        The solution.

    Raises
    ------
    ConfigError
        If they cannot be solved together (see check_together).
    RuntimeError
        If the Newton steps do not reach the solution.
    """
    check_together(solves, sizes)
    mean, sigma, lower = _priors(solves, sizes)
    variance = sigma ** 2

    def dual(y: np.ndarray) -> tuple[float, np.ndarray]:
        # (the dual, as a minimum, and the values of y: ||y||^2 / 2 + sum
        # G_k((A^T y)_k) - t^T y, G_k(s) = max over u >= c_k of u s - (u -
        # mu_k)^2 / (2 s_k^2), convex, whose derivative is u_k(s))
        slopes = rows.T @ y
        values = np.maximum(lower, mean + variance * slopes)
        conjugates = values * slopes - (values - mean) ** 2 / (2 * variance)
        return float(y @ y / 2 + np.sum(conjugates) - target @ y), values

    # (from the solution without the minimum, all values free)
    identity = np.eye(len(target))
    free = np.ones(len(mean), dtype=bool)
    gram = (rows * variance) @ rows.T
    factor = scipy.linalg.cho_factor(identity + gram)
    y = scipy.linalg.cho_solve(factor, target - rows @ mean)
    value, values = dual(y)
    for _ in range(_ROWS_MAX_STEPS):
        # (the Newton matrix, of the values entering and leaving F, or
        # anew when most change)
        changed = free != (values > lower)
        free = values > lower
        if 2 * changed.sum() > free.sum():
            gram = (rows[:, free] * variance[free]) @ rows[:, free].T
        else:
            sign = np.where(free[changed], 1.0, -1.0)
            gram += (rows[:, changed] * (sign * variance[changed])) \
                @ rows[:, changed].T
        if changed.any():
            factor = scipy.linalg.cho_factor(identity + gram)
        # (at the solution, the gradient of the dual is 0, to rounding)
        gradient = y + rows @ values - target
        if np.linalg.norm(gradient) <= 1e-12 * (1 + np.linalg.norm(target)):
            break
        step = -scipy.linalg.cho_solve(factor, gradient)
        # (backtracking, over the kinks of max)
        slope = float(step @ gradient)
        length = 1.0
        while True:
            trial, trial_values = dual(y + length * step)
            if trial <= value + 1e-4 * length * slope or length < 1e-10:
                break
            length /= 2
        y, value, values = y + length * step, trial, trial_values
        if length == 1 and np.array_equal(values > lower, free):
            break
    else:
        raise RuntimeError(
            f"the values solved on rows did not converge in "
            f"{_ROWS_MAX_STEPS} Newton steps")
    log_term = 0.0
    if solves[0].likelihood() == 'integrated':
        log_term = -float(np.sum(np.log(np.diag(factor[0]))))
    # (the diagonal of H^-1, of the inverse of its Cholesky factor: H =
    # U^T U, or L L^T)
    inverse = scipy.linalg.solve_triangular(
        factor[0], identity, lower=factor[1])
    diagonal = np.sum(inverse ** 2, axis=0 if factor[1] else 1)
    counted = len(y) if ndata is None else ndata
    m_eff = counted - float(np.sum(diagonal[:counted]))
    return Solution(values, (values - mean) / sigma, log_term, m_eff)


def _priors(
        solves: Sequence[Solve], sizes: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the mean, the standard deviation and the minimum (-inf without
    one) of the Gaussian prior of each value of the given solves, the given
    number of each, in their order.
    """
    mean, sigma = (
        np.repeat([getter(solve) for solve in solves], sizes)
        for getter in (SolveGaussian.mean, SolveGaussian.sigma))
    lower = np.repeat([
        -np.inf if solve.minimum() is None else solve.minimum()
        for solve in solves], sizes)
    return mean, sigma, lower


solve_parser = parseutils.TypedParser(Solve, [SolveGaussian])
