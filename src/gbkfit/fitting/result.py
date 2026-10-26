"""
The result of a fit: what was run, the best fit and its uncertainty, the
other solutions, the posterior and the evidence (when the fitter gives
them), and the cost.
"""

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from astropy.table import Table

from gbkfit.outputs import RunWriter
from gbkfit.params import ParamValues


__all__ = [
    'Bootstrap',
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
    Return quantiles of weighted samples, along the first axis, of the
    values that are not NaN in each column (e.g. a row of a table that a
    sample does not use).

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
        The quantiles of each column: shape (len(q), ncolumns); nan for a
        column without values.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    samples = np.asarray(samples, dtype=float).reshape(len(samples), -1)
    if weights is None:
        weights = np.ones(len(samples))
    result = np.full((len(q), samples.shape[1]), np.nan)
    for j in range(samples.shape[1]):
        kept = ~np.isnan(samples[:, j])
        if not np.any(kept):
            continue
        if np.all(weights[kept] == weights[kept][0]):
            result[:, j] = np.quantile(samples[kept, j], q)
            continue
        order = np.argsort(samples[kept, j])
        cdf = np.cumsum(weights[kept][order])
        cdf /= cdf[-1]
        result[:, j] = np.interp(q, cdf, samples[kept, j][order])
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
        The covariance matrix, in the directions that the data constrain
        (see rank).
    std : dict
        The standard error of each free element: infinite for those in
        the directions that the data do not constrain (see unconstrained).
    correlation : np.ndarray
        The correlation matrix (NaN for the free elements of infinite
        errors).
    element_std : dict
        The standard error of every element (propagated to the tied
        elements; 0 for the fixed ones, infinite for those that change in
        the directions that the data do not constrain).
    derived_std : dict
        The standard error of each element of the derived values (see
        FitSolution.derived), propagated as those of the tied elements
        (the delta method; infinite as theirs); NaN where they cannot be
        evaluated at a step.
    rank : int
        The rank of the Jacobian. Below the number of free elements, some
        directions are not constrained (see unconstrained).
    unconstrained : tuple of str
        The free elements in the directions that the data do not
        constrain (more than a tenth of a unit vector of such a
        direction, in the free values scaled by the Jacobian): their
        errors are infinite.
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
    derived_std: dict[str, float]
    rank: int
    unconstrained: tuple[str, ...]
    at_bounds: tuple[str, ...]
    scaled: bool

    def table(self) -> Table:
        """
        Return the matrix as a table: a row of each free element, named in
        the column 'name', and a column of each.
        """
        return Table(
            [list(self.names), *self.matrix.T], names=['name', *self.names])


@dataclasses.dataclass(frozen=True)
class Posterior:
    """
    Samples of the posterior of all the elements of the parameters, and
    of the derived values.

    Attributes
    ----------
    names : tuple of str
        The elements (all of them: free, tied and fixed), in order, then
        those of the derived values (see FitSolution.derived), which are
        NaN in the samples where they cannot be evaluated.
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

    def table(self) -> Table:
        """
        Return the samples as a table: a column of each element, and of
        the weights, if any ('weight').
        """
        table = _samples_table(self.names, self.samples)
        if self.weights is not None:
            table['weight'] = self.weights
        return table


@dataclasses.dataclass(frozen=True)
class Bootstrap:
    """
    The bootstrap of a solution (see bootstrap): the best fits of refits
    of data made of its model plus its residuals redrawn, which scatter
    as the best fit does over data with other noise of the same kind.

    Attributes
    ----------
    names : tuple of str
        The elements (all of them: free, tied and fixed), in order, then
        those of the derived values (see FitSolution.derived), which are
        NaN in the samples where they cannot be evaluated.
    samples : np.ndarray
        The best fit of each refit that succeeded, one per row.
    failures : tuple of str
        Why each of the other refits failed.
    options : dict
        The configuration of the bootstrap (see BootstrapOptions).
    """
    names: tuple[str, ...]
    samples: np.ndarray
    failures: tuple[str, ...]
    options: dict[str, Any]

    def std(self) -> dict[str, float]:
        """
        Return the standard deviation of each element over the refits
        (NaN with fewer than two).
        """
        std = np.std(self.samples, axis=0, ddof=1) \
            if len(self.samples) >= 2 else np.full(len(self.names), np.nan)
        return dict(zip(self.names, std.tolist()))

    def summary(self) -> dict[str, tuple[float, float, float]]:
        """
        Return the 16th, 50th and 84th percentiles of each element over the
        refits (the median and a 68% interval).
        """
        values = weighted_quantiles(self.samples, [0.16, 0.5, 0.84])
        return {
            name: (float(v[0]), float(v[1]), float(v[2]))
            for name, v in zip(self.names, values.T)}

    def info(self) -> dict[str, Any]:
        """
        Return its summary, for result.json: the number of refits that
        succeeded and failed, why those failed, and the standard deviation
        and the percentiles of each element (see std and summary).
        """
        return dict(
            nsamples=len(self.samples), nfailed=len(self.failures),
            failures=list(self.failures), std=self.std(),
            quantiles=self.summary(), options=self.options)

    def table(self) -> Table:
        """Return the samples as a table: a column of each element."""
        return _samples_table(self.names, self.samples)


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
    derived : dict
        The value of each element of each derived value (see
        ParamSpace.derived_elements); none if they cannot be evaluated
        there (see the diagnostics 'derived_errors' of FitResult).
    chi2 : float
        The chi-square.
    nresiduals : int
        The number of residuals that count (not masked).
    nfitted : float
        The number of fitted quantities: the free elements and the
        effective number of the nuisance parameters that the objective
        solves (see Objective.nnuisance).
    dof : float
        The degrees of freedom: the number of independent residuals (see
        Objective.nindependent; nresiduals, but fewer for correlated
        noise) less nfitted.
    reduced_chi2 : float
        chi2 / dof (nan if dof <= 0).
    log_likelihood : float
        The log-likelihood (-chi2 / 2, unless the objective has nuisance
        parameters; see Objective.log_likelihood).
    log_prior : float or None
        The log of the prior density, if the fit has priors.
    aicp : float
        The AICp (Lipka & Thomas 2021): the chi-square of the data (see
        Objective.aicp) plus twice the number of fitted quantities (see
        nfitted): the lower, the better a model, of several fitted to the
        same data, whose solved values are regularised by their priors
        (e.g. Schwarzschild models). Without solved values, it is the
        chi-square plus twice the number of the free elements.
    covariance : Covariance or None
        The covariance, or None if disabled, if the parameters cannot be
        evaluated at a step of its Jacobian, or if the fit was asked to
        stop (see the diagnostics 'uncertainty_errors' of FitResult).
    laplace : Posterior or None
        The Gaussian approximation of the posterior from the covariance
        (approximate), or None if disabled, there is no covariance, or it
        has directions that the data do not constrain (see the
        diagnostics 'uncertainty_errors' of FitResult).
    """
    free: dict[str, float]
    values: ParamValues
    elements: dict[str, float]
    derived: dict[str, float]
    chi2: float
    nresiduals: int
    nfitted: float
    dof: float
    reduced_chi2: float
    log_likelihood: float
    log_prior: float | None
    aicp: float
    covariance: Covariance | None
    laplace: Posterior | None

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
    bootstrap : Bootstrap or None
        The bootstrap of the best solution, if run (see bootstrap).
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
    bootstrap: Bootstrap | None = None
    raw: object = dataclasses.field(default=None, repr=False)

    def best(self) -> FitSolution:
        """Return the best solution."""
        return self.solutions[0]

    def summary(self) -> str:
        """
        Return a summary for humans: the best fit and its errors, of the
        free elements and the derived values (with those of the bootstrap,
        if run).
        """
        best = self.best()
        lines = [
            f"fitter: {self.fitter.get('type')}; success: {self.success} "
            f"({self.message})",
            f"model evaluations: {self.nevals}; time: {self.time:.1f} s",
            f"chi2: {best.chi2:.6g}; dof: {best.dof:.6g}; reduced chi2: "
            f"{best.reduced_chi2:.6g}; AICp: {best.aicp:.6g}"]
        if self.evidence is not None:
            lines.append(f"log Z: {self.evidence.log_z:.6g}")
        posterior = self.posterior or best.laplace
        columns = [posterior.summary() if posterior else {}]
        labels = ['posterior' if self.posterior else 'laplace']
        if self.bootstrap is not None:
            columns.append(self.bootstrap.summary())
            labels.append('bootstrap')
        std = best.covariance.element_std | best.covariance.derived_std \
            if best.covariance else {}
        lines.append(
            f"{'name':<16} {'best':>12} {'std':>12} "
            + ' '.join(f"{label + ' 16/50/84':>38}" for label in labels))
        for name, value in (_free_elements(best) | best.derived).items():
            q_texts = [
                f"{q[0]:>12.6g} {q[1]:>12.6g} {q[2]:>12.6g}"
                if (q := column.get(name)) is not None else ' ' * 38
                for column in columns]
            lines.append((
                f"{name:<16} {value:>12.6g} {std.get(name, math.nan):>12.6g} "
                + ' '.join(q_texts)).rstrip())
        if best.covariance and best.covariance.unconstrained:
            lines.append(
                f"unconstrained directions involve: "
                f"{list(best.covariance.unconstrained)}")
        if best.covariance and best.covariance.at_bounds:
            lines.append(
                f"at a bound: {list(best.covariance.at_bounds)}")
        return '\n'.join(lines)

    def info(self) -> dict[str, Any]:
        """
        Return its summary (without the arrays): what was run, the
        solutions, the summaries of the posterior and of the bootstrap,
        and the evidence.
        """
        return dict(
            fitter=self.fitter, params=self.params, versions=self.versions,
            nevals=self.nevals, time=self.time, success=self.success,
            message=self.message, diagnostics=self.diagnostics,
            evidence=None if self.evidence is None
            else dataclasses.asdict(self.evidence),
            posterior=None if self.posterior is None
            else self.posterior.summary(),
            solutions=[_solution_summary(s) for s in self.solutions],
            bootstrap=None if self.bootstrap is None
            else self.bootstrap.info())

    def write(self, run: RunWriter, folder: str = 'fit') -> None:
        """
        Write the result as products of a run (see gbkfit.outputs), in a
        folder: its summary (result.json, kind 'fit_result'; see info),
        the covariance matrix and the Laplace samples of each solution i
        (solutions/<i>/covariance.ecsv and laplace.ecsv, kinds
        'covariance' and 'laplace', id 'solution'), the posterior samples,
        with their weights if any (posterior.ecsv, 'posterior'), and the
        samples of the bootstrap, if run (bootstrap.ecsv, 'bootstrap').
        The tables have a column of each element (see Posterior.table,
        Covariance.table and Bootstrap.table).

        Parameters
        ----------
        run : RunWriter
            The run.
        folder : str, optional
            The folder of the products in the run.
        """
        run.write(f'{folder}/result.json', 'fit_result', self.info())
        for i, solution in enumerate(self.solutions):
            if solution.covariance is not None:
                run.write(
                    f'{folder}/solutions/{i}/covariance.ecsv', 'covariance',
                    solution.covariance.table(), solution=str(i))
            if solution.laplace is not None:
                run.write(
                    f'{folder}/solutions/{i}/laplace.ecsv', 'laplace',
                    solution.laplace.table(), solution=str(i))
        if self.posterior is not None:
            run.write(
                f'{folder}/posterior.ecsv', 'posterior',
                self.posterior.table())
        if self.bootstrap is not None:
            run.write(
                f'{folder}/bootstrap.ecsv', 'bootstrap',
                self.bootstrap.table())


def _samples_table(names: Sequence[str], samples: np.ndarray) -> Table:
    """Return samples (one per row) as a table, a column of each name."""
    return Table(list(np.asarray(samples).T), names=list(names))


def _free_elements(solution: FitSolution) -> dict[str, float]:
    """
    Return the values of the free elements of a solution, decoded (those
    of parameters with modes are coded in its free values), as the
    posterior, the Laplace samples and the bootstrap give them.
    """
    return {name: solution.elements[name] for name in solution.free}


def _solution_summary(solution: FitSolution) -> dict[str, Any]:
    """Return the summary of a solution (without its arrays)."""
    info: dict[str, Any] = dict(
        free=solution.free, elements=solution.elements,
        derived=solution.derived, chi2=solution.chi2,
        nresiduals=solution.nresiduals, nfitted=solution.nfitted,
        dof=solution.dof,
        reduced_chi2=solution.reduced_chi2,
        log_likelihood=solution.log_likelihood,
        log_prior=solution.log_prior, aic=solution.aic, bic=solution.bic,
        aicp=solution.aicp)
    if (cov := solution.covariance) is not None:
        info['covariance'] = dict(
            names=list(cov.names), std=cov.std, element_std=cov.element_std,
            derived_std=cov.derived_std, rank=cov.rank,
            unconstrained=list(cov.unconstrained),
            at_bounds=list(cov.at_bounds), scaled=cov.scaled)
    if solution.laplace is not None:
        info['laplace'] = solution.laplace.summary()
    return info
