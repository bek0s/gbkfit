"""
The result of a fit: what was run, the best fit and its uncertainty, the
other solutions, the posterior and the evidence (when the fitter gives
them), and the cost.
"""

import csv
import dataclasses
import json
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from gbkfit.params import ParamValues
from gbkfit.utils import numutils


__all__ = [
    'Covariance',
    'Evidence',
    'FitResult',
    'FitSolution',
    'Posterior',
    'weighted_quantiles'
]


def weighted_quantiles(
        samples: np.ndarray,
        q: npt.ArrayLike,
        weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Return quantiles of weighted samples, along the first axis.

    Parameters
    ----------
    samples : np.ndarray
        The samples, one per row.
    q : ArrayLike
        The quantiles, in [0, 1].
    weights : np.ndarray, optional
        The weight of each sample (by default, equal).

    Returns
    -------
    np.ndarray
        The quantiles of each column: shape (len(q), ncolumns); nan if
        there are no samples.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    samples = np.asarray(samples, dtype=float).reshape(len(samples), -1)
    if len(samples) == 0:
        return np.full((len(q), samples.shape[1]), np.nan)
    if weights is None:
        return np.quantile(samples, q, axis=0)
    result = np.empty((len(q), samples.shape[1]))
    for j in range(samples.shape[1]):
        order = np.argsort(samples[:, j])
        cdf = np.cumsum(weights[order])
        cdf /= cdf[-1]
        result[:, j] = np.interp(q, cdf, samples[order, j])
    return result


@dataclasses.dataclass(frozen=True)
class Covariance:
    """
    The covariance of the free elements at a solution, from the Jacobian
    of the residuals: (J^T J)^-1, by an SVD pseudo-inverse.

    Attributes
    ----------
    names : tuple of str
        The free elements, in order.
    matrix : np.ndarray
        The covariance matrix.
    std : dict
        The standard error of each free element.
    correlation : np.ndarray
        The correlation matrix.
    element_std : dict
        The standard error of every element (propagated to the tied
        elements; 0 for the fixed ones).
    rank : int
        The rank of the Jacobian. Below the number of free elements, some
        directions are not constrained, and the errors are left out of
        them: see unconstrained.
    unconstrained : tuple of str
        The free elements in the directions that the data do not
        constrain.
    at_bounds : tuple of str
        The free elements at a bound (within a finite-difference step):
        their errors assume the bound is not there.
    scaled : bool
        Whether the matrix is scaled by the reduced chi-square.
    """
    names: tuple[str, ...]
    matrix: np.ndarray
    std: dict[str, float]
    correlation: np.ndarray
    element_std: dict[str, float]
    rank: int
    unconstrained: tuple[str, ...]
    at_bounds: tuple[str, ...]
    scaled: bool


@dataclasses.dataclass(frozen=True)
class Posterior:
    """
    Samples of the posterior of all the elements of the parameters.

    Attributes
    ----------
    names : tuple of str
        The elements (all of them: free, tied and fixed), in order.
    samples : np.ndarray
        The samples, one per row.
    weights : np.ndarray or None
        The weight of each sample (nested samplers), or None for equal
        weights.
    approximate : bool
        Whether it is the Gaussian (Laplace) approximation at a solution,
        from its covariance.
    """
    names: tuple[str, ...]
    samples: np.ndarray
    weights: np.ndarray | None
    approximate: bool

    def quantiles(self, q: Sequence[float]) -> dict[str, np.ndarray]:
        """
        Return quantiles of each element.

        Parameters
        ----------
        q : Sequence of float
            The quantiles, in [0, 1].

        Returns
        -------
        dict
            The quantiles of each element.
        """
        values = weighted_quantiles(self.samples, q, self.weights)
        return {name: values[:, i] for i, name in enumerate(self.names)}

    def summary(self) -> dict[str, tuple[float, float, float]]:
        """
        Return the 16th, 50th and 84th percentiles of each element (the
        median and a 68% interval).
        """
        return {
            name: (float(v[0]), float(v[1]), float(v[2]))
            for name, v in self.quantiles([0.16, 0.5, 0.84]).items()}


@dataclasses.dataclass(frozen=True)
class Evidence:
    """
    The Bayesian evidence (marginal likelihood) of nested samplers.

    Attributes
    ----------
    log_z : float
        Its logarithm (of the likelihood without its normalisation; see
        Objective.log_likelihood).
    log_z_error : float or None
        The error of log_z, if the sampler gives one.
    """
    log_z: float
    log_z_error: float | None


@dataclasses.dataclass(frozen=True)
class FitSolution:
    """
    A solution of a fit: the parameters, the goodness of fit and the
    uncertainty there.

    Attributes
    ----------
    free : dict
        The value of each free element.
    values : ParamValues
        The values of all parameters (to evaluate the models).
    elements : dict
        The value of every element.
    chi2 : float
        The chi-square.
    nresiduals : int
        The number of residuals that count (not masked).
    dof : int
        The degrees of freedom: nresiduals less the free elements and the
        nuisance parameters that the objective solves (see
        Objective.nnuisance).
    reduced_chi2 : float
        chi2 / dof (nan if dof <= 0).
    log_likelihood : float
        The log-likelihood (-chi2 / 2, unless the objective has nuisance
        parameters; see Objective.log_likelihood).
    log_prior : float or None
        The log of the prior density, if the fit has priors.
    covariance : Covariance or None
        The covariance, or None if disabled or if the parameters cannot
        be evaluated at a step of its Jacobian (see the diagnostics
        'uncertainty_errors' of FitResult).
    laplace : Posterior or None
        The Gaussian approximation of the posterior from the covariance
        (approximate), or None if disabled or there is no covariance.
    """
    free: dict[str, float]
    values: ParamValues
    elements: dict[str, float]
    chi2: float
    nresiduals: int
    dof: int
    reduced_chi2: float
    log_likelihood: float
    log_prior: float | None
    covariance: Covariance | None
    laplace: Posterior | None

    @property
    def nfitted(self) -> int:
        """
        The number of fitted quantities: the free elements and the nuisance
        parameters (nresiduals - dof).
        """
        return self.nresiduals - self.dof

    @property
    def aic(self) -> float:
        """
        The Akaike information criterion, 2 k - 2 ln L, of the k fitted
        quantities (see nfitted) and the log-likelihood L: the lower, the
        better a model, of several fitted to the same data.
        """
        return 2 * self.nfitted - 2 * self.log_likelihood

    @property
    def bic(self) -> float:
        """
        The Bayesian information criterion, k ln n - 2 ln L, of the k fitted
        quantities (see nfitted), the n residuals and the log-likelihood L.
        Like the likelihood, it takes the values as independent (of
        correlated values, e.g. the pixels within a beam, n is more than
        the independent ones, and k ln n penalises too much).
        """
        return self.nfitted * float(np.log(self.nresiduals)) \
            - 2 * self.log_likelihood


@dataclasses.dataclass(frozen=True)
class FitResult:
    """
    The result of a fit.

    Attributes
    ----------
    fitter : dict
        The configuration of the fitter (with its seed).
    params : dict or None
        The configuration of the parameters (None if it cannot be dumped,
        e.g. a transforms function not defined in a file).
    versions : dict
        The versions of gbkfit, numpy, scipy and the libraries of the
        fitter.
    solutions : tuple of FitSolution
        The solutions, best first.
    posterior : Posterior or None
        The posterior samples of a sampler.
    evidence : Evidence or None
        The evidence of a nested sampler.
    nevals : int
        The number of model evaluations of the fitter.
    time : float
        The time of the fit, in seconds.
    success : bool
        Whether the fitter reports success.
    message : str
        The message of the fitter.
    diagnostics : dict
        Numbers that tell how well the fitter worked (e.g. the acceptance
        fraction of an MCMC sampler).
    raw : object
        The result object of the fitter's library, for experts (not
        dumped).
    """
    fitter: dict[str, Any]
    params: dict[str, Any] | None
    versions: dict[str, str]
    solutions: tuple[FitSolution, ...]
    posterior: Posterior | None
    evidence: Evidence | None
    nevals: int
    time: float
    success: bool
    message: str
    diagnostics: dict[str, Any]
    raw: object = dataclasses.field(default=None, repr=False)

    def best(self) -> FitSolution:
        """Return the best solution."""
        return self.solutions[0]

    def summary(self) -> str:
        """Return a summary for humans: the best fit and its errors."""
        best = self.best()
        lines = [
            f"fitter: {self.fitter.get('type')}; success: {self.success} "
            f"({self.message})",
            f"model evaluations: {self.nevals}; time: {self.time:.1f} s",
            f"chi2: {best.chi2:.6g}; dof: {best.dof}; reduced chi2: "
            f"{best.reduced_chi2:.6g}"]
        if self.evidence is not None:
            lines.append(f"log Z: {self.evidence.log_z:.6g}")
        posterior = self.posterior or best.laplace
        quantiles = posterior.summary() if posterior else {}
        std = best.covariance.std if best.covariance else {}
        label = 'posterior' if self.posterior else 'laplace'
        lines.append(
            f"{'name':<16} {'best':>12} {'std':>12} "
            f"{label + ' 16/50/84':>38}")
        for name, value in best.free.items():
            q = quantiles.get(name)
            q_text = '' if q is None else (
                f"{q[0]:>12.6g} {q[1]:>12.6g} {q[2]:>12.6g}")
            lines.append(
                f"{name:<16} {value:>12.6g} {std.get(name, math.nan):>12.6g} "
                f"{q_text}")
        if best.covariance and best.covariance.unconstrained:
            lines.append(
                f"unconstrained directions involve: "
                f"{list(best.covariance.unconstrained)}")
        if best.covariance and best.covariance.at_bounds:
            lines.append(
                f"at a bound: {list(best.covariance.at_bounds)}")
        return '\n'.join(lines)

    def dump(self, output_dir: str) -> None:
        """
        Write the result to a directory: a summary (result.json), and the
        arrays as CSV files with the names of the elements as headers:
        the covariance matrix and the Laplace samples of each solution
        (solution_<i>_covariance.csv, solution_<i>_laplace.csv) and the
        posterior samples, with their weights if any (posterior.csv).

        Parameters
        ----------
        output_dir : str
            The directory (made if missing).
        """
        os.makedirs(output_dir, exist_ok=True)
        summary = dict(
            fitter=self.fitter, params=self.params, versions=self.versions,
            nevals=self.nevals, time=self.time, success=self.success,
            message=self.message, diagnostics=self.diagnostics,
            evidence=None if self.evidence is None
            else dataclasses.asdict(self.evidence),
            posterior=None if self.posterior is None
            else self.posterior.summary(),
            solutions=[_solution_summary(s) for s in self.solutions])
        with open(os.path.join(output_dir, 'result.json'), 'w') as f:
            json.dump(numutils.nativify(summary), f, indent=2)

        def path(name: str) -> str:
            return os.path.join(output_dir, name)

        for i, solution in enumerate(self.solutions):
            if (cov := solution.covariance) is not None:
                # (one row per free element, named in its first column)
                _write_csv(
                    path(f'solution_{i}_covariance.csv'),
                    ['name', *cov.names],
                    [[name, *row] for name, row in zip(cov.names, cov.matrix)])
            if (laplace := solution.laplace) is not None:
                _write_csv(
                    path(f'solution_{i}_laplace.csv'), list(laplace.names),
                    laplace.samples)
        if (posterior := self.posterior) is not None:
            header = list(posterior.names)
            rows = posterior.samples
            if posterior.weights is not None:
                header.append('weight')
                rows = np.column_stack((rows, posterior.weights))
            _write_csv(path('posterior.csv'), header, rows)


def _write_csv(
        filename: str,
        header: Sequence[str],
        rows: Iterable[Sequence[str | float]]
) -> None:
    """Write rows of values to a CSV file, numbers to 12 digits."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                value if isinstance(value, str) else f'{value:.12g}'
                for value in row])


def _solution_summary(solution: FitSolution) -> dict[str, Any]:
    """Return the summary of a solution (without its arrays)."""
    info: dict[str, Any] = dict(
        free=solution.free, elements=solution.elements, chi2=solution.chi2,
        nresiduals=solution.nresiduals, dof=solution.dof,
        reduced_chi2=solution.reduced_chi2,
        log_likelihood=solution.log_likelihood,
        log_prior=solution.log_prior, aic=solution.aic, bic=solution.bic)
    if (cov := solution.covariance) is not None:
        info['covariance'] = dict(
            names=list(cov.names), std=cov.std, element_std=cov.element_std,
            rank=cov.rank, unconstrained=list(cov.unconstrained),
            at_bounds=list(cov.at_bounds), scaled=cov.scaled)
    if solution.laplace is not None:
        info['laplace'] = solution.laplace.summary()
    return info
