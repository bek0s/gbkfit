"""
The uncertainty of a solution of any fitter: the covariance from the
Jacobian of the residuals, and the Gaussian (Laplace) approximation of
the posterior from it.
"""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import Covariance, Posterior
from gbkfit.objective import Objective
from gbkfit.params import ParamEvaluationError, ParamSpace
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'UncertaintyOptions',
    'covariance',
    'finite_difference_steps',
    'jacobian',
    'laplace_posterior',
    'sample_values'
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
        the errors of the data are not trusted): the chi-square over the
        degrees of freedom, of the independent residuals (see
        FitSolution.dof), so that correlated noise of its stated errors
        leaves the covariance as it is.
    laplace_samples : int, optional
        The number of samples of the Gaussian (Laplace) approximation of
        the posterior at each solution (0 for none).
    rcond : float, optional
        Singular values of the Jacobian (with normalised columns) below
        rcond times the largest one count as 0: the data do not constrain
        those directions, which are listed instead of given errors of
        numerical noise. By default, 10 times the precision of the
        models to the power 2/3, the relative error of the central
        differences (2.4e-4 for float32 models, 5e-10 for float64).

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
            rcond: float | None = None
    ):
        if laplace_samples < 0 or (rcond is not None and rcond < 0):
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

    def rcond(self) -> float | None:
        """
        Return the cutoff of the singular values of the Jacobian (None
        for the default, from the precision of the models).
        """
        return self._rcond


def finite_difference_steps(
        x: np.ndarray,
        scales: np.ndarray,
        relative: float
) -> np.ndarray:
    """
    Return the steps of finite differences for each free element: a
    relative step times the larger of |x| and the typical scale of the
    value (see FitParam.scale), so that values at or near 0 get a step
    of their scale.

    The best relative steps are the square root of the precision of the
    models for forward differences, and its cube root for central ones
    (see FitProblem.precision).

    Parameters
    ----------
    x : np.ndarray
        The free values.
    scales : np.ndarray
        The typical scale of each.
    relative : float
        The relative step.

    Returns
    -------
    np.ndarray
        The steps.
    """
    return relative * np.maximum(np.abs(x), scales)


def jacobian(
        func_batch: Callable[[np.ndarray], Sequence[np.ndarray]],
        x: np.ndarray,
        steps: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        scheme: Literal['central', 'forward'] = 'central',
        f0: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the Jacobian of a function by finite differences: central or
    forward, and one-sided where a step would cross a bound. The function
    is evaluated at all the steps at once (in one batch).

    Parameters
    ----------
    func_batch : Callable
        The function at several points (one per row), giving a vector for
        each.
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
    # Each column is (f(a) - f(b)) / d: the two points a and b of its
    # difference (None for x itself, whose value is f0), and d
    differences = []
    for i, h in enumerate(steps):
        room_up = upper[i] - x[i]
        room_down = x[i] - lower[i]
        if h > room_up and h > room_down:
            # (the bounds are narrower than the step: step to the bound
            # with more room)
            h = max(room_up, room_down)
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        can_up = h <= room_up
        if scheme == 'central' and can_up and h <= room_down:
            differences.append((up, down, 2 * h))
        elif can_up:
            differences.append((up, None, h))
        else:
            differences.append((None, down, h))
    points = [p for a, b, _ in differences for p in (a, b) if p is not None]
    needs_f0 = f0 is None \
        and any(a is None or b is None for a, b, _ in differences)
    values = iter(func_batch(np.array(([x] if needs_f0 else []) + points)))
    if needs_f0:
        f0 = next(values)
    columns = []
    for a, b, d in differences:
        fa = f0 if a is None else next(values)
        fb = f0 if b is None else next(values)
        columns.append((fa - fb) / d)
    return np.stack(columns, axis=1)


def sample_values(
        params: ParamSpace,
        x: np.ndarray,
        derived: Sequence[str] = ()
) -> np.ndarray:
    """
    Return the values of a sample of the posterior at the given free
    values: those of every element (see ParamSpace.element_values), then
    those of the given elements of the derived values (see
    ParamSpace.derived_elements), NaN where they cannot be evaluated.

    Raises
    ------
    ParamEvaluationError
        If the parameters cannot be evaluated there.
    """
    values = params.evaluate(x)
    elements = list(params.element_values(values).values())
    if not derived:
        return np.array(elements)
    try:
        derived_values = params.derived_elements(values)
    except ParamEvaluationError:
        derived_values = {}
    return np.array(
        elements + [derived_values.get(name, np.nan) for name in derived])


def covariance(
        problem: FitProblem,
        x: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        scales: np.ndarray,
        options: UncertaintyOptions,
        reduced_chi2: float,
        derived: Sequence[str] = ()
) -> tuple[Covariance, int]:
    """
    Return the covariance of a solution, from the Jacobian of the
    weighted residuals and of those of the priors that have them (see
    FitProblem.prior_residuals): (J^T J)^-1, by an SVD pseudo-inverse of
    J with its columns normalised (so that the cutoff of the singular
    values does not depend on the units); and the number of model
    evaluations it took.

    The errors are propagated to all elements, and to the given elements
    of the derived values, through their Jacobian (the delta method; no
    models but those of the derived values). Elements at a bound are
    listed: the covariance ignores the bound. If options.scale() is set, the
    covariance is scaled by the reduced chi-square, when that is defined
    (more residuals than free elements).

    Parameters
    ----------
    problem : FitProblem
        The problem (its counter of evaluations is not used).
    x : np.ndarray
        The free values of the solution.
    lower, upper : np.ndarray
        The bounds of the free values.
    scales : np.ndarray
        The typical scales of the free values (see FitParam.scale).
    options : UncertaintyOptions
        The options.
    reduced_chi2 : float
        The reduced chi-square of the solution (to scale the covariance).
    derived : Sequence of str, optional
        The elements of the derived values (see FitSolution.derived).

    Returns
    -------
    Covariance
        The covariance.
    int
        The number of model evaluations.

    Raises
    ------
    ParamEvaluationError
        If the parameters cannot be evaluated at a step (e.g. out of
        their valid range).
    """
    params = problem.params()
    nevals = 0

    # (with those of the priors: the covariance of the posterior; on the
    # pool of the problem, if any)
    def residuals(points: np.ndarray) -> list[np.ndarray]:
        nonlocal nevals
        nevals += len(points)
        evaluated = problem.map_objective(
            Objective.residuals, [params.evaluate(p) for p in points])
        return [
            np.concatenate([r, problem.prior_residuals(p)])
            for p, r in zip(points, evaluated)]

    steps = finite_difference_steps(
        x, scales, problem.precision() ** (1 / 3))
    jac = jacobian(residuals, x, steps, lower, upper)
    norms = np.linalg.norm(jac, axis=0)
    norms[norms == 0] = 1
    _, s, vt = np.linalg.svd(jac / norms, full_matrices=False)
    rcond = options.rcond()
    if rcond is None:
        rcond = 10 * problem.precision() ** (2 / 3)
    keep = s > rcond * s[0]
    vt_kept = vt[keep] / s[keep, None]
    matrix = (vt_kept.T @ vt_kept) / np.outer(norms, norms)
    scaled = options.scale() and np.isfinite(reduced_chi2)
    if scaled:
        matrix *= reduced_chi2
    std = np.sqrt(np.diag(matrix))
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = matrix / np.outer(std, std)
    names = problem.names()
    unconstrained = tuple(
        name for name, weight in zip(names, np.abs(vt[~keep]).max(
            axis=0, initial=0)) if weight > 0.1)
    # The errors of all elements and of the derived values: J_e cov J_e^T,
    # with J_e the Jacobian of their values by the free values
    element_names = list(params.element_values(params.evaluate(x)))
    jac_elements = jacobian(
        lambda points: [sample_values(params, p, derived) for p in points],
        x, steps, lower, upper)
    element_std = np.sqrt(np.maximum(np.einsum(
        'ij,jk,ik->i', jac_elements, matrix, jac_elements), 0)).tolist()
    nelements = len(element_names)
    return Covariance(
        names=names,
        matrix=matrix,
        std=dict(zip(names, std.tolist())),
        correlation=correlation,
        element_std=dict(zip(element_names, element_std[:nelements])),
        derived_std=dict(zip(derived, element_std[nelements:])),
        rank=int(keep.sum()),
        unconstrained=unconstrained,
        at_bounds=tuple(
            name for name, xi, lo, hi, h in zip(names, x, lower, upper, steps)
            if xi - lo < h or hi - xi < h),
        scaled=bool(scaled)), nevals


def laplace_posterior(
        problem: FitProblem,
        x: np.ndarray,
        cov: Covariance,
        nsamples: int,
        rng: np.random.Generator,
        derived: Sequence[str] = ()
) -> Posterior:
    """
    Return samples of the Gaussian (Laplace) approximation of the
    posterior at a solution: a normal distribution with its covariance,
    mapped to all elements and the given elements of the derived values
    (see sample_values). Samples where the parameters cannot be
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
    derived : Sequence of str, optional
        The elements of the derived values (see FitSolution.derived).

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
            rows.append(sample_values(params, point, derived))
        except ParamEvaluationError:
            pass
    names = tuple(params.kinds()) + tuple(derived)
    return Posterior(
        names=names, samples=np.array(rows).reshape(-1, len(names)),
        weights=None, approximate=True)
