"""
Helpers shared by the fitters.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np

from gbkfit.fitting.uncertainty import finite_difference_steps, jacobian


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


def inside_bounds(
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        steps: np.ndarray
) -> np.ndarray:
    """
    Return values moved off their bounds by a step (to the middle, if the
    bounds are narrower than two steps). Methods that map bounded values
    to unbounded ones (MINPACK in lmfit and scipy) need this: the map is
    flat at the bounds, so a value that starts on one never moves.
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
    point evaluated last costs one evaluation per variable.

    Parameters
    ----------
    func : Callable
        The residuals, of the variables (a new array each call).
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
            func: Callable[[np.ndarray], np.ndarray],
            lower: np.ndarray,
            upper: np.ndarray,
            scales: np.ndarray,
            scheme: Literal['forward', 'central'],
            relative_step: float
    ):
        self._func = func
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
        residuals = self._func(x)
        self._last_x = np.array(x, dtype=float)
        self._last_residuals = residuals
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
                self._func, x, steps, self._lower, self._upper,
                self._scheme, f0)
            self._last_jacobian_x = x
        return self._last_jacobian.copy()
