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
import typing
from collections.abc import Callable, Sequence
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

# The most Newton steps of solve_together_rows, and the relative change
# of its dual below which it is rounding (of sums of many terms)
_ROWS_MAX_STEPS = 100
_ROWS_ROUNDING = 1e-14
# The tolerance of ln(sigma) chosen by AICp (see SolveGaussian)
_AICP_TOLERANCE = 1e-3


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

    Solves of the same options are equal (e.g. those of copies of the
    parameters; see ParamSpace.with_values).
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) \
            and typing.cast(Solve, other).dump() == self.dump()

    def __hash__(self) -> int:
        return hash((type(self), repr(self.dump())))

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

    The standard deviation can instead be chosen at each evaluation, in a
    range (sigma_aicp), as the one of the least AICp = chi2 + 2 m_eff of
    the data (Lipka & Thomas 2021; chi2 of the data alone, not of the
    prior): the regularisation of the values that the data favour, e.g.
    of the weights of orbits at each mass. It is found by a bounded
    minimisation in ln(s), which changes smoothly with the data; the
    log-likelihood takes the values 'profiled' (the integral of a prior
    chosen by the data is not that of a prior).

    Parameters
    ----------
    mean : float
        The mean of the prior.
    sigma : float, optional
        The standard deviation of the prior, finite and positive; or else
        sigma_aicp.
    minimum : float, optional
        The smallest value (e.g. 0 for a scale of the flux); none by
        default.
    likelihood : str, optional
        What the log-likelihood takes of the values: 'integrated' (the
        default) or 'profiled' (see Solve).
    sigma_aicp : sequence of float, optional
        The range of the standard deviation chosen by AICp (two finite
        positive numbers, increasing), in place of sigma.

    Raises
    ------
    ConfigError
        If the mean or the minimum is not finite, not one of sigma and
        sigma_aicp is given, sigma is not finite and positive, sigma_aicp
        not two of them increasing, likelihood is not one of LIKELIHOODS,
        or it is 'integrated' with sigma_aicp.
    """

    @staticmethod
    def type() -> str:
        return 'gaussian'

    def dump(self) -> dict[str, str | float | list[float]]:
        info: dict[str, str | float | list[float]] = dict(
            type=self.type(), mean=self._mean)
        if self._sigma_aicp is None:
            info.update(sigma=self._sigma)
        else:
            info.update(sigma_aicp=list(self._sigma_aicp))
        if self._minimum is not None:
            info.update(minimum=self._minimum)
        if self._likelihood != LIKELIHOODS[0]:
            info.update(likelihood=self._likelihood)
        return info

    def __init__(
            self,
            mean: float,
            sigma: float | None = None,
            minimum: float | None = None,
            likelihood: str = LIKELIHOODS[0],
            sigma_aicp: Sequence[float] | None = None
    ):
        if not math.isfinite(mean):
            raise ConfigError(f"the mean must be finite; it is {mean}")
        if (sigma is None) == (sigma_aicp is None):
            raise ConfigError(
                "a Gaussian solve has sigma, or sigma_aicp (a range of sigma "
                "chosen by AICp), not both")
        if sigma is not None and not (math.isfinite(sigma) and sigma > 0):
            raise ConfigError(
                f"sigma must be finite and positive; it is {sigma}")
        if sigma_aicp is not None:
            if not (len(sigma_aicp) == 2
                    and all(math.isfinite(s) and s > 0 for s in sigma_aicp)
                    and sigma_aicp[0] < sigma_aicp[1]):
                raise ConfigError(
                    f"sigma_aicp must be two finite positive numbers, "
                    f"increasing; it is {sigma_aicp!r}")
            if likelihood != 'profiled':
                raise ConfigError(
                    "values of a prior of sigma chosen by AICp take the "
                    "likelihood 'profiled'")
        if minimum is not None and not math.isfinite(minimum):
            raise ConfigError(f"the minimum must be finite; it is {minimum}")
        if likelihood not in LIKELIHOODS:
            raise ConfigError(
                f"likelihood must be one of {list(LIKELIHOODS)}; it is "
                f"{likelihood!r}")
        self._mean = float(mean)
        self._sigma = None if sigma is None else float(sigma)
        self._sigma_aicp = None if sigma_aicp is None \
            else tuple(float(s) for s in sigma_aicp)
        self._minimum = None if minimum is None else float(minimum)
        self._likelihood = likelihood

    def likelihood(self) -> str:
        return self._likelihood

    def bounds(self) -> tuple[float, float]:
        return -math.inf if self._minimum is None else self._minimum, math.inf

    def default(self) -> float:
        # (the mean, at least the minimum, at any sigma: no need to choose
        # one by AICp)
        return self._mean if self._minimum is None \
            else max(self._mean, self._minimum)

    def solve(self, a: np.ndarray, b: np.ndarray) -> Solution:
        if self._sigma_aicp is not None:
            return _chosen(
                lambda sigma: self._solve_with(a, b, sigma), self._sigma_aicp)
        return self._solve_with(a, b, self._sigma)[0]

    def _solve_with(
            self, a: np.ndarray, b: np.ndarray, sigma: float
    ) -> tuple[Solution, float]:
        """
        Return the solution of the sums at the standard deviation sigma,
        and the chi-square of the data (but a term without the values).
        """
        mean, minimum = self._mean, self._minimum
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
        chi2 = float(np.sum(a * values ** 2 - 2 * b * values))
        return Solution(
            values, ((values - mean) / sigma).ravel(), log_term, m_eff), chi2

    def mean(self) -> float:
        """Return the mean of the prior."""
        return self._mean

    def sigma(self) -> float | None:
        """
        Return the standard deviation of the prior; None if it is chosen
        (see sigma_aicp).
        """
        return self._sigma

    def sigma_aicp(self) -> tuple[float, float] | None:
        """
        Return the range of the standard deviation chosen by AICp, if it
        is chosen.
        """
        return self._sigma_aicp

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
    if len({solve.sigma_aicp() for solve in solves}) > 1:
        raise ConfigError(
            "values solved together take one sigma chosen by AICp: their "
            "solves have the same sigma_aicp, or none")


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
    return _chosen_together(
        solves, lambda sigma: _together_with(
            solves, sizes, normal, rhs, sigma))


def _together_with(
        solves: Sequence[Solve],
        sizes: Sequence[int],
        normal: np.ndarray,
        rhs: np.ndarray,
        sigma: float | None
) -> tuple[Solution, float]:
    """
    Return the values of solve_together at the standard deviation sigma
    of all priors (theirs if None), and the chi-square of the data (but a
    term without the values).
    """
    mean, sigma, lower = _priors(solves, sizes, sigma)
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
    chi2 = float(values @ normal @ values - 2 * rhs @ values)
    return Solution(
        values, (values - mean) / sigma, log_term, m_eff), chi2


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
    from the solution of similar rows it takes more of them); at the
    standard deviations that AICp tries (see SolveGaussian), each from the
    solution at the last one (see _RowsSolver). Then the
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
    return _chosen_together(
        solves, _RowsSolver(solves, sizes, rows, target, ndata).solve)


class _RowsSolver:
    """
    The values of solve_together_rows at a standard deviation of the
    priors: theirs, once, or one of all of them after another (see
    _chosen_together). Each of the latter starts from the solution of the
    last one: its dual y, its values above their minimum F (most stay so),
    and their matrix A_F B_F A_F^T (B the variances of the priors at a
    standard deviation of 1), which the standard deviation only scales.
    The matrix of new values F comes from that of the last ones, or from
    that of all values (kept) without the others, or anew: whichever
    takes the fewest columns (e.g. thousands change between distant
    standard deviations, but few values are at their minimum).
    """

    def __init__(
            self,
            solves: Sequence[Solve],
            sizes: Sequence[int],
            rows: np.ndarray,
            target: np.ndarray,
            ndata: int | None
    ):
        self._solves = solves
        self._sizes = sizes
        self._rows = rows
        self._target = target
        self._ndata = ndata
        # The solution of the last standard deviation of all priors: its
        # dual, its free values and their matrix (None before one)
        self._last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # The matrix of all values, and its variances B (None before one)
        self._all: tuple[np.ndarray, np.ndarray] | None = None

    def solve(self, sigma: float | None) -> tuple[Solution, float]:
        """
        Return the values at the standard deviation sigma of all priors
        (theirs if None), and the chi-square of the rows of the data.
        """
        rows, target = self._rows, self._target
        mean, sigmas, lower = _priors(self._solves, self._sizes, sigma)
        # (the variances: scale times those of a scale of 1, base)
        # (exactly 1 with one standard deviation of all priors, so that
        # the matrix of all values is made once for all of them)
        scale = 1.0 if sigma is None else sigma ** 2
        base = sigmas ** 2 if sigma is None else np.ones(len(sigmas))
        variance = scale * base

        def dual(y: np.ndarray) -> tuple[float, np.ndarray]:
            # (the dual, as a minimum, and the values of y: ||y||^2 / 2 +
            # sum G_k((A^T y)_k) - t^T y, G_k(s) = max over u >= c_k of u s
            # - (u - mu_k)^2 / (2 s_k^2), convex, whose derivative is
            # u_k(s))
            slopes = rows.T @ y
            values = np.maximum(lower, mean + variance * slopes)
            conjugates = \
                values * slopes - (values - mean) ** 2 / (2 * variance)
            return float(y @ y / 2 + np.sum(conjugates) - target @ y), values

        identity = np.eye(len(target))
        factor = None
        if sigma is not None and self._last is not None:
            y, free, gram = self._last
        else:
            # (from the solution without the minimum, all values free)
            free = np.ones(len(mean), dtype=bool)
            gram = self._all_matrix(base).copy()
            factor = scipy.linalg.cho_factor(identity + scale * gram)
            y = scipy.linalg.cho_solve(factor, target - rows @ mean)
        value, values = dual(y)
        for _ in range(_ROWS_MAX_STEPS):
            # (the Newton matrix of the values F above their minimum)
            changed = free != (values > lower)
            free = values > lower
            if changed.any():
                gram = self._matrix(gram, free, changed, base)
                factor = None
            if factor is None:
                factor = scipy.linalg.cho_factor(identity + scale * gram)
            # (at the solution, the gradient of the dual is 0, to rounding)
            gradient = y + rows @ values - target
            if np.linalg.norm(gradient) \
                    <= 1e-12 * (1 + np.linalg.norm(target)):
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
            # (the solution to rounding: a full step that keeps F, or no
            # step that lowers the dual beyond rounding, e.g. of a value
            # at its minimum, in F and out of it from step to step)
            stalled = not value - trial > _ROWS_ROUNDING * abs(value)
            if not stalled:
                y, value, values = y + length * step, trial, trial_values
            if stalled or length == 1 \
                    and np.array_equal(values > lower, free):
                break
        else:
            raise RuntimeError(
                f"the values solved on rows did not converge in "
                f"{_ROWS_MAX_STEPS} Newton steps")
        if sigma is not None:
            self._last = y, free, gram
        log_term = 0.0
        if self._solves[0].likelihood() == 'integrated':
            log_term = -float(np.sum(np.log(np.diag(factor[0]))))
        # (the diagonal of H^-1, of the inverse of its Cholesky factor: H =
        # U^T U, or L L^T)
        triangle, lower = factor
        trtri, = scipy.linalg.get_lapack_funcs(('trtri',), (triangle,))
        inverse, info = trtri(triangle, lower=lower)
        if info != 0:
            raise RuntimeError(
                f"the Newton matrix of the values solved on rows is "
                f"singular (trtri: {info})")
        inverse = np.tril(inverse) if lower else np.triu(inverse)
        diagonal = np.sum(inverse ** 2, axis=0 if lower else 1)
        counted = len(y) if self._ndata is None else self._ndata
        m_eff = counted - float(np.sum(diagonal[:counted]))
        residuals = rows[:counted] @ values - target[:counted]
        return Solution(
            values, (values - mean) / sigmas, log_term, m_eff), \
            float(residuals @ residuals)

    def _all_matrix(self, base: np.ndarray) -> np.ndarray:
        """
        Return the matrix A B A^T of all values, of the variances B (made
        once for them).
        """
        if self._all is None or not np.array_equal(self._all[1], base):
            self._all = (self._rows * base) @ self._rows.T, base
        return self._all[0]

    def _matrix(
            self,
            gram: np.ndarray,
            free: np.ndarray,
            changed: np.ndarray,
            base: np.ndarray
    ) -> np.ndarray:
        """
        Return the matrix A_F B A_F^T of the free values F, of the
        variances B: from that of the values before the changed ones
        changed, gram (updated in place), from that of all values without
        the others, or anew, whichever takes the fewest columns.
        """
        rows = self._rows
        nchanged, nfree = int(changed.sum()), int(free.sum())
        nfixed = len(free) - nfree
        if nchanged <= min(nfree, nfixed):
            sign = np.where(free[changed], 1.0, -1.0)
            gram += (rows[:, changed] * (sign * base[changed])) \
                @ rows[:, changed].T
            return gram
        if nfixed <= nfree:
            fixed = ~free
            return self._all_matrix(base) \
                - (rows[:, fixed] * base[fixed]) @ rows[:, fixed].T
        return (rows[:, free] * base[free]) @ rows[:, free].T


def _chosen_together(
        solves: Sequence[Solve],
        solve_with: Callable[[float | None], tuple[Solution, float]]
) -> Solution:
    """
    Return the solution of values solved together, of the function of
    their solution at a standard deviation of their priors (and the
    chi-square of their data): at theirs, or chosen by AICp in their range
    (see SolveGaussian).
    """
    widths = solves[0].sigma_aicp()
    if widths is None:
        return solve_with(None)[0]
    return _chosen(solve_with, widths)


def _chosen(
        solve_with: Callable[[float], tuple[Solution, float]],
        widths: tuple[float, float]
) -> Solution:
    """
    Return the solution of the standard deviation of a prior in a range
    of the least AICp = chi2 + 2 m_eff of the data (see SolveGaussian), of
    the function of the solution at a standard deviation and the
    chi-square of the data: by a bounded minimisation in its ln.
    """
    best: list[tuple[float, Solution]] = []

    def aicp(ln_sigma: float) -> float:
        solution, chi2 = solve_with(math.exp(ln_sigma))
        value = chi2 + 2 * solution.m_eff
        if not best or value < best[0][0]:
            best[:] = [(value, solution)]
        return value
    scipy.optimize.minimize_scalar(
        aicp, bounds=np.log(widths), method='bounded',
        options=dict(xatol=_AICP_TOLERANCE))
    return best[0][1]


def _priors(
        solves: Sequence[Solve],
        sizes: Sequence[int],
        sigma: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the mean, the standard deviation (that given, if any) and the
    minimum (-inf without one) of the Gaussian prior of each value of the
    given solves, the given number of each, in their order.
    """
    mean = np.repeat([solve.mean() for solve in solves], sizes)
    sigma = np.repeat([solve.sigma() for solve in solves], sizes) \
        if sigma is None else np.full(len(mean), sigma)
    lower = np.repeat([
        -np.inf if solve.minimum() is None else solve.minimum()
        for solve in solves], sizes)
    return mean, sigma, lower


solve_parser = parseutils.TypedParser(Solve, [SolveGaussian])
