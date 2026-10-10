"""
The uncertainty of a solution of any fitter: the covariance from the
Jacobian of the residuals, and the Gaussian (Laplace) approximation of
the posterior from it.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import Covariance, Posterior
from gbkfit.params import ParamEvaluationError
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'UncertaintyOptions',
    'covariance',
    'finite_difference_steps',
    'jacobian',
    'laplace_posterior'
]


class UncertaintyOptions(parseutils.Serializable):
    """
    How fitters work out the uncertainty of their solutions.

    Parameters
    ----------
    covariance : bool, optional
        Whether to compute the covariance at each solution (2 model
        evaluations per free element).
    scale : bool, optional
        Whether to scale the covariance by the reduced chi-square (when
        the errors of the data are not trusted).
    laplace_samples : int, optional
        The number of samples of the Gaussian (Laplace) approximation of
        the posterior at each solution (0 for none).
    rcond : float, optional
        Singular values of the Jacobian below rcond times the largest
        one count as 0 (unconstrained directions).

    Raises
    ------
    ConfigError
        If a number is negative.
    """

    def dump(self) -> dict[str, Any]:
        return dict(
            covariance=self._covariance, scale=self._scale,
            laplace_samples=self._laplace_samples, rcond=self._rcond)

    def __init__(
            self,
            covariance: bool = True,
            scale: bool = False,
            laplace_samples: int = 1000,
            rcond: float = 1e-10
    ):
        if laplace_samples < 0 or rcond < 0:
            raise ConfigError(
                "options 'laplace_samples' and 'rcond' cannot be negative")
        self._covariance = covariance
        self._scale = scale
        self._laplace_samples = laplace_samples
        self._rcond = rcond

    def covariance(self) -> bool:
        """Return whether to compute the covariance."""
        return self._covariance

    def scale(self) -> bool:
        """Return whether to scale the covariance by the reduced chi2."""
        return self._scale

    def laplace_samples(self) -> int:
        """Return the number of samples of the Laplace posterior."""
        return self._laplace_samples

    def rcond(self) -> float:
        """Return the cutoff of the singular values of the Jacobian."""
        return self._rcond


def finite_difference_steps(
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        relative: float
) -> np.ndarray:
    """
    Return the steps of finite differences for each free element: a
    relative step times the larger of |x| and 1% of the width of the
    bounds (or 1, if unbounded), so that values at or near 0 get a step
    of their scale.

    The best relative steps are the square root of the precision of the
    models for forward differences, and its cube root for central ones
    (see FitProblem.precision).

    Parameters
    ----------
    x : np.ndarray
        The free values.
    lower, upper : np.ndarray
        The bounds (may be infinite).
    relative : float
        The relative step.

    Returns
    -------
    np.ndarray
        The steps.
    """
    width = upper - lower
    scale = np.where(np.isfinite(width), 0.01 * width, 1.0)
    return relative * np.maximum(np.abs(x), scale)


def jacobian(
        func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        steps: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        scheme: Literal['central', 'forward'] = 'central',
        f0: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the Jacobian of a function by finite differences: central or
    forward, and one-sided where a step would cross a bound.

    Parameters
    ----------
    func : Callable
        The function, of a vector, giving a vector.
    x : np.ndarray
        Where.
    steps : np.ndarray
        The step of each variable.
    lower, upper : np.ndarray
        The bounds of the variables.
    scheme : {'central', 'forward'}, optional
        The differences (central: 2 evaluations per variable; forward:
        1, plus func(x) if f0 is not given).
    f0 : np.ndarray, optional
        func(x), if known.

    Returns
    -------
    np.ndarray
        The Jacobian: one row per value of the function, one column per
        variable (a new array).
    """
    columns = []
    for i, h in enumerate(steps):
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        can_up = up[i] <= upper[i]
        if scheme == 'central' and can_up and down[i] >= lower[i]:
            columns.append((func(up) - func(down)) / (2 * h))
            continue
        if f0 is None:
            f0 = func(x)
        if can_up:
            columns.append((func(up) - f0) / h)
        else:
            columns.append((f0 - func(down)) / h)
    return np.stack(columns, axis=1)


def covariance(
        problem: FitProblem,
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        options: UncertaintyOptions,
        reduced_chi2: float
) -> tuple[Covariance, int]:
    """
    Return the covariance of a solution, from the Jacobian of the
    weighted residuals: (J^T J)^-1, by an SVD pseudo-inverse; and the
    number of model evaluations it took.

    The errors are propagated to all elements through the Jacobian of
    the parameter space (no models).

    Parameters
    ----------
    problem : FitProblem
        The problem (its counter of evaluations is not used).
    x : np.ndarray
        The free values of the solution.
    lower, upper : np.ndarray
        The bounds of the free values.
    options : UncertaintyOptions
        The options.
    reduced_chi2 : float
        The reduced chi-square of the solution (to scale the covariance).

    Returns
    -------
    Covariance
        The covariance.
    int
        The number of model evaluations.
    """
    objective = problem.objective()
    params = problem.params()
    nevals = 0

    def residuals(point: np.ndarray) -> np.ndarray:
        nonlocal nevals
        nevals += 1
        return objective.residuals(params.evaluate(point))

    steps = finite_difference_steps(
        x, lower, upper, problem.precision() ** (1 / 3))
    jac = jacobian(residuals, x, steps, lower, upper)
    _, s, vt = np.linalg.svd(jac, full_matrices=False)
    keep = s > options.rcond() * s[0]
    vt_kept = vt[keep] / s[keep, None]
    matrix = vt_kept.T @ vt_kept
    if options.scale():
        matrix *= reduced_chi2
    std = np.sqrt(np.diag(matrix))
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = matrix / np.outer(std, std)
    names = problem.names()
    unconstrained = tuple(
        name for name, weight in zip(names, np.abs(vt[~keep]).max(
            axis=0, initial=0)) if weight > 0.1)
    # The errors of all elements: J_e cov J_e^T, with J_e the Jacobian of
    # the element values by the free values
    element_names = list(params.element_values(params.evaluate(x)))

    def element_values(point: np.ndarray) -> np.ndarray:
        return np.array(list(params.element_values(
            params.evaluate(point)).values()))

    jac_elements = jacobian(element_values, x, steps, lower, upper)
    element_var = np.einsum(
        'ij,jk,ik->i', jac_elements, matrix, jac_elements)
    return Covariance(
        names=names,
        matrix=matrix,
        std=dict(zip(names, std.tolist())),
        correlation=correlation,
        element_std=dict(zip(
            element_names, np.sqrt(np.maximum(element_var, 0)).tolist())),
        rank=int(keep.sum()),
        unconstrained=unconstrained,
        scaled=options.scale()), nevals


def laplace_posterior(
        problem: FitProblem,
        x: np.ndarray,
        cov: Covariance,
        nsamples: int,
        rng: np.random.Generator
) -> Posterior:
    """
    Return samples of the Gaussian (Laplace) approximation of the
    posterior at a solution: a normal distribution with its covariance,
    mapped to all elements. Samples where the parameters cannot be
    evaluated (e.g. outside their valid range) are left out.

    Parameters
    ----------
    problem : FitProblem
        The problem.
    x : np.ndarray
        The free values of the solution.
    cov : Covariance
        Its covariance.
    nsamples : int
        The number of samples to draw.
    rng : np.random.Generator
        The random number generator.

    Returns
    -------
    Posterior
        The samples (approximate).
    """
    params = problem.params()
    free = rng.multivariate_normal(
        x, cov.matrix, size=nsamples, method='eigh')
    rows = []
    for point in free:
        try:
            rows.append(list(params.element_values(
                params.evaluate(point)).values()))
        except ParamEvaluationError:
            pass
    names = tuple(params.element_values(params.evaluate(x)))
    return Posterior(
        names=names, samples=np.array(rows).reshape(-1, len(names)),
        weights=None, approximate=True)
