"""
Helpers shared by the fitters.
"""

import math
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from gbkfit.fitting.result import Evidence
from gbkfit.fitting.uncertainty import finite_difference_steps, jacobian
from gbkfit.utils import threadutils


def relative_step(
        precision: float,
        scheme: Literal['forward', 'central']
) -> float:
    """
    Return the best relative step of finite differences for models of a
    precision (see FitProblem.precision): its square root for forward
    differences, and its cube root for central ones.
    """
    return precision ** (1 / 3 if scheme == 'central' else 1 / 2)


def corrected_evidence(
        evidence: Evidence,
        normalisation: Evidence
) -> Evidence:
    """
    Return the evidence of a nested sampler corrected for the priors on
    several parameters (see JointPrior).

    A nested sampler draws its points through the priors of the elements
    one at a time (FitProblem.prior_transform), which cannot hold a prior
    on several elements; so that goes into its likelihood instead
    (FitProblem.nested_log_likelihood_batch), as BayesianBBarolo adds its
    curvature penalty. The posterior is exact, but the evidence is then
    that of the prior pi(x) e^(-chi2/2) unnormalised: C times the
    evidence, with C = int pi(x) e^(-chi2/2) dx, which depends on the
    priors alone, not on the data (a strong smoothness prior over 30
    rings can make it 1e-20). C is the evidence of a run of the same
    sampler on the priors alone (FitProblem.joint_prior_log_density_batch;
    no models, so cheap): log Z = log Z' - log C, with the errors added in
    quadrature.

    Parameters
    ----------
    evidence : Evidence
        The evidence of the run on the likelihood with these priors.
    normalisation : Evidence
        The evidence of the run on these priors alone (C).
    """
    errors = (evidence.log_z_error, normalisation.log_z_error)
    return Evidence(
        log_z=evidence.log_z - normalisation.log_z,
        log_z_error=None if None in errors else math.hypot(*errors))


def inside_bounds(
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        steps: np.ndarray
) -> np.ndarray:
    """
    Return values moved off their bounds by a step (to the middle, if the
    bounds are narrower than two steps). Least-squares methods need this:
    the map of bounded values of MINPACK (lmfit's leastsq, scipy's 'lm')
    is flat at the bounds, and TRF's first steps from a bound are tiny,
    so a value that starts on one does not move.
    """
    inside = np.clip(x, lower + steps, upper - steps)
    narrow = upper - lower < 2 * steps
    inside[narrow] = ((lower + upper) / 2)[narrow]
    return inside


class LeastSquaresFunctions:
    """
    Residuals and their Jacobian by finite differences, for least-squares
    libraries that take a Jacobian function.

    Their own finite differences do not suit GBKFIT's models: scipy's
    steps are relative to |x|, and at x = 0 (a common start, e.g. of
    xpos or vsys) fall back to a step for float64 (1.5e-8), below the
    precision of float32 models, so the Jacobian column is 0 and the
    value never moves. These steps are of the scale of each value (see
    finite_difference_steps).

    The last residuals are kept, so a forward-difference Jacobian at the
    point evaluated last costs one evaluation per variable. The steps of a
    Jacobian are evaluated in one batch (see FitProblem.residuals_batch).

    Parameters
    ----------
    func_batch : Callable
        The residuals at several values of the variables (one per row), a
        new array for each.
    lower, upper : np.ndarray
        The bounds of the variables.
    scales : np.ndarray
        The typical scales of the variables (see FitParam.scale).
    scheme : {'forward', 'central'}
        The finite differences.
    relative_step : float
        The relative step (see relative_step).
    """

    def __init__(
            self,
            func_batch: Callable[[np.ndarray], Sequence[np.ndarray]],
            lower: np.ndarray,
            upper: np.ndarray,
            scales: np.ndarray,
            scheme: Literal['forward', 'central'],
            relative_step: float
    ):
        self._func_batch = func_batch
        self._lower = lower
        self._upper = upper
        self._scales = scales
        self._scheme: Literal['forward', 'central'] = scheme
        self._relative_step = relative_step
        self._last_x: np.ndarray | None = None
        self._last_residuals: np.ndarray | None = None
        self._last_jacobian_x: np.ndarray | None = None
        self._last_jacobian: np.ndarray | None = None
        self._best_x: np.ndarray | None = None
        self._best_cost = np.inf

    def best(self) -> np.ndarray | None:
        """
        Return the point of the lowest sum of squared residuals that the
        library asked for, if any.
        """
        return None if self._best_x is None else self._best_x.copy()

    def residuals(self, x: np.ndarray) -> np.ndarray:
        """Return the residuals at x (a new array)."""
        residuals = self._func_batch(np.atleast_2d(x))[0]
        self._last_x = np.array(x, dtype=float)
        self._last_residuals = residuals
        # (see FitProblem.residuals)
        with threadutils.single_threaded_blas():
            cost = float(residuals @ residuals)
        if cost < self._best_cost:
            self._best_cost = cost
            self._best_x = self._last_x.copy()
        return residuals

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Return the Jacobian of the residuals at x (a new array; that of
        the last point is kept, since some libraries ask for it twice).
        """
        x = np.array(x, dtype=float)
        if not np.array_equal(x, self._last_jacobian_x):
            f0 = self._last_residuals \
                if np.array_equal(x, self._last_x) else None
            steps = finite_difference_steps(
                x, self._scales, self._relative_step)
            self._last_jacobian = jacobian(
                self._func_batch, x, steps, self._lower, self._upper,
                self._scheme, f0)
            self._last_jacobian_x = x
        return self._last_jacobian.copy()
