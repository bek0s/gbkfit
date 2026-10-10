"""
The result of a fit: what was run, the best fit and its uncertainty, the
other solutions, the posterior and the evidence (when the fitter gives
them), and the cost.
"""

import dataclasses
import json
import math
import os
from collections.abc import Sequence
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
        The quantiles of each column: shape (len(q), ncolumns).
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    samples = np.asarray(samples, dtype=float).reshape(len(samples), -1)
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
        The degrees of freedom: nresiduals less the free elements.
    reduced_chi2 : float
        chi2 / dof.
    log_likelihood : float
        -chi2 / 2.
    log_prior : float or None
        The log of the prior density, if the fit has priors.
    covariance : Covariance or None
        The covariance, unless disabled.
    laplace : Posterior or None
        The Gaussian approximation of the posterior from the covariance
        (approximate), unless disabled.
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
        The versions of gbkfit and of the libraries of the fitter.
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
        return '\n'.join(lines)

    def dump(self, output_dir: str) -> None:
        """
        Write the result to a directory: a summary (result.json) and the
        arrays (result.npz: the covariance matrices, and the posterior
        samples).

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
        arrays: dict[str, np.ndarray] = {}
        for i, solution in enumerate(self.solutions):
            if solution.covariance is not None:
                arrays[f'solution_{i}_covariance'] = \
                    solution.covariance.matrix
            if solution.laplace is not None:
                arrays[f'solution_{i}_laplace_samples'] = \
                    solution.laplace.samples
        if self.posterior is not None:
            arrays['posterior_names'] = np.array(self.posterior.names)
            arrays['posterior_samples'] = self.posterior.samples
            if self.posterior.weights is not None:
                arrays['posterior_weights'] = self.posterior.weights
        np.savez(os.path.join(output_dir, 'result.npz'), **arrays)


def _solution_summary(solution: FitSolution) -> dict[str, Any]:
    """Return the summary of a solution (without its arrays)."""
    info: dict[str, Any] = dict(
        free=solution.free, elements=solution.elements, chi2=solution.chi2,
        nresiduals=solution.nresiduals, dof=solution.dof,
        reduced_chi2=solution.reduced_chi2,
        log_likelihood=solution.log_likelihood,
        log_prior=solution.log_prior)
    if (cov := solution.covariance) is not None:
        info['covariance'] = dict(
            names=list(cov.names), std=cov.std, element_std=cov.element_std,
            rank=cov.rank, unconstrained=list(cov.unconstrained),
            scaled=cov.scaled)
    if solution.laplace is not None:
        info['laplace'] = solution.laplace.summary()
    return info
