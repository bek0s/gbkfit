"""
The base class of the fitters, and what their runs give.
"""

import abc
import copy
import dataclasses
import importlib.metadata
import time
from collections.abc import Sequence
from typing import Any, Self

import numpy as np

from gbkfit.fitting.control import FitControl, FitStopped
from gbkfit.fitting.evaluation_log import EvaluationLog
from gbkfit.fitting.fitparams import FitParam, FitParamClass
from gbkfit.fitting.pool import EvaluationPool
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.progress import ProgressObserver
from gbkfit.fitting.result import (
    Evidence, FitResult, FitSolution, Posterior)
from gbkfit.fitting.uncertainty import (
    UncertaintyOptions, covariance, laplace_posterior, sample_values)
from gbkfit.objective import Objective
from gbkfit.params import ParamEvaluationError, ParamSpace
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitRun',
    'Fitter',
    'fitter_parser'
]


@dataclasses.dataclass
class FitRun:
    """
    What the run of a fitter gives (see Fitter._run).

    Attributes
    ----------
    solutions : list of np.ndarray
        The free values of each solution, best first.
    success : bool
        Whether the fitter reports success.
    message : str
        The message of the fitter.
    samples : np.ndarray, optional
        The free values of the posterior samples (samplers), one per row.
    unused : np.ndarray, optional
        Which free elements each sample does not use (e.g. the rows of a
        table that DNest4 leaves out; samples by free elements, bool):
        NaN in the posterior. The samples hold the values with which the
        model was evaluated there.
    weights : np.ndarray, optional
        The weight of each sample (nested samplers).
    evidence : Evidence, optional
        The evidence (nested samplers).
    diagnostics : dict, optional
        Numbers that tell how well the fitter worked.
    raw : object, optional
        The result object of the fitter's library.
    """
    solutions: list[np.ndarray]
    success: bool
    message: str
    samples: np.ndarray | None = None
    unused: np.ndarray | None = None
    weights: np.ndarray | None = None
    evidence: Evidence | None = None
    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)
    raw: object = None


class Fitter(parseutils.TypedSerializable, abc.ABC):
    """
    A fitter: it finds the free values of a parameter space that fit the
    data of an objective, and works out their uncertainty.

    Each fitter reads the free property of each element (see
    ParamSpace.free_properties) into its class of FitParam (see
    fit_param_class): e.g. initial values and bounds for optimizers, and
    priors for samplers. After its run, the fitter evaluates each
    solution, and computes its covariance and Laplace posterior (see
    UncertaintyOptions).

    Parameters
    ----------
    seed : int
        The seed of the random numbers (of the fitter, if random, and of
        the Laplace samples).
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the solutions.
    """

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'Fitter':
        info = dict(info)
        if 'uncertainty' in info:
            info['uncertainty'] = parseutils.load_option(
                UncertaintyOptions.load, info, 'uncertainty')
        return cls(**parseutils.parse_options_for_callable(info, cls.__init__))

    def __init__(
            self,
            seed: int,
            uncertainty: UncertaintyOptions | None
    ):
        self._seed = seed
        self._uncertainty = uncertainty or UncertaintyOptions()

    def seed(self) -> int:
        """Return the seed of the random numbers."""
        return self._seed

    def uncertainty(self) -> UncertaintyOptions:
        """Return the options of the uncertainty."""
        return self._uncertainty

    def without_uncertainty(self) -> Self:
        """
        Return a copy of it that works out no uncertainty of its solutions
        (e.g. the refits of a bootstrap, whose spread is their
        uncertainty; see bootstrap).
        """
        fitter = copy.copy(self)
        fitter._uncertainty = UncertaintyOptions(
            covariance=False, laplace_samples=0)
        return fitter

    def dump_common(self) -> dict[str, Any]:
        """Return the options of all fitters, for dump."""
        return dict(seed=self._seed, uncertainty=self._uncertainty.dump())

    @staticmethod
    @abc.abstractmethod
    def fit_param_class() -> FitParamClass:
        """Return the class of the options of the free elements."""

    @abc.abstractmethod
    def libraries(self) -> tuple[str, ...]:
        """Return the libraries (distribution names) the fitter uses."""

    def load_fit_params(self, params: ParamSpace) -> dict[str, FitParam]:
        """
        Load the options of the free elements from their free properties.

        Parameters
        ----------
        params : ParamSpace
            The parameters.

        Returns
        -------
        dict
            The options of each free element, in order.

        Raises
        ------
        ConfigError
            If the properties of an element are invalid for the fitter.
        """
        cls = self.fit_param_class()
        fit_params = {}
        for name, info in params.free_properties().items():
            with parseutils.config_path(name):
                fit_params[name] = cls.load(info)
        return fit_params

    def fit(
            self,
            objective: Objective,
            params: ParamSpace,
            observers: Sequence[ProgressObserver] = (),
            control: FitControl | None = None,
            log: EvaluationLog | None = None,
            pool: EvaluationPool | None = None
    ) -> FitResult:
        """
        Fit the data of an objective.

        A fit asked to stop (see FitControl) ends with what the fitter has:
        the best point so far (of an MCMC sampler, also its chain so far),
        and success false, without the covariance of its solutions (nor
        those of a fit asked to stop as it computes them). A fit with a
        log of its evaluations can resume from it (see EvaluationLog).

        Parameters
        ----------
        objective : Objective
            The objective.
        params : ParamSpace
            The parameters of its models.
        observers : Sequence of ProgressObserver, optional
            Who receives the progress of the fit as it runs (see
            FitProgress).
        control : FitControl, optional
            The control of the fit, to stop it.
        log : EvaluationLog, optional
            The log of the evaluations: a new one, or that of a fit to
            resume (with the same objective, parameters and fitter).
        pool : EvaluationPool, optional
            Copies of the objective (e.g. one per GPU) that evaluate the
            batches of points of the fitter at once (e.g. the members of
            a generation of CMA-ES, the walkers of emcee, or the steps of
            a Jacobian); the result is that of the fit without it, to
            the rounding of the devices.

        Returns
        -------
        FitResult
            The result.

        Raises
        ------
        ConfigError
            If the free properties or the options of the fitter do not
            suit the problem (e.g. too few free elements or walkers), or
            the fitter may try values of free elements outside the valid
            ranges of their parameters (see ParamDesc).
        ValueError
            If no parameter is free, the parameter space lacks parameters
            of the objective, or the pool is of another objective.
        ParamEvaluationError
            If the parameters cannot be evaluated where a least-squares
            fitter evaluates them (e.g. at the start).
        RuntimeError
            If the fitter finds no point with a finite chi-square (or the
            fit was stopped before).
        """
        fit_params = self.load_fit_params(params)
        if not fit_params:
            raise ValueError("no parameter is free")
        _check_valid_ranges(fit_params, params.free_valid_ranges())
        problem = FitProblem(
            objective, params, [p.prior() for p in fit_params.values()],
            observers, control=control, log=log, pool=pool)
        try:
            return self._fit(problem, params, fit_params)
        finally:
            problem.close()

    def _fit(
            self,
            problem: FitProblem,
            params: ParamSpace,
            fit_params: dict[str, FitParam]
    ) -> FitResult:
        """Run the fitter on a problem, and make its result (see fit)."""
        rng = np.random.default_rng(self._seed)
        start = time.perf_counter()
        try:
            run = self._run(problem, list(fit_params.values()), rng)
        except FitStopped:
            run = _stopped_run(problem)
        elapsed = time.perf_counter() - start
        bounds = _Bounds(
            np.array([p.minimum() for p in fit_params.values()]),
            np.array([p.maximum() for p in fit_params.values()]),
            np.array([p.scale() for p in fit_params.values()]))
        diagnostics = dict(run.diagnostics)
        problem.set_stage('uncertainty')
        solutions = []
        uncertainty_nevals = 0
        problems = []
        for x in run.solutions:
            solution, nevals, solution_problems = self._solution(
                problem, x, bounds, rng)
            solutions.append(solution)
            uncertainty_nevals += nevals
            problems += solution_problems
        diagnostics['uncertainty_nevals'] = uncertainty_nevals
        diagnostics['nan_nevals'] = problem.nan_nevals()
        for key, message in problems:
            diagnostics.setdefault(key, []).append(message)
        posterior = None
        if run.samples is not None:
            posterior = _posterior(
                problem, run.samples, run.weights,
                tuple(solutions[0].derived), run.unused)
        problem.finish()
        return FitResult(
            fitter=self.dump(),
            params=_dump_params(params),
            versions=_versions(self.libraries()),
            solutions=tuple(solutions),
            posterior=posterior,
            evidence=run.evidence,
            nevals=problem.nevals(),
            time=elapsed,
            success=run.success,
            message=run.message,
            diagnostics=diagnostics,
            raw=run.raw)

    @abc.abstractmethod
    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        """
        Run the fitter.

        Parameters
        ----------
        problem : FitProblem
            The problem.
        fit_params : list of FitParam
            The options of each free element, in order (of the class of
            fit_param_class).
        rng : np.random.Generator
            The random number generator, from the seed (fitters whose
            library takes a seed draw it from rng).

        Returns
        -------
        FitRun
            What the run gives.
        """

    def _solution(
            self,
            problem: FitProblem,
            x: np.ndarray,
            bounds: '_Bounds',
            rng: np.random.Generator
    ) -> tuple[FitSolution, int, list[tuple[str, str]]]:
        """
        Evaluate a solution and its uncertainty; return it, the number of
        model evaluations it took, and its problems, as the keys of the
        diagnostics of the result and the messages: why its uncertainty
        is missing ('uncertainty_errors': the parameters cannot be
        evaluated at a step of its Jacobian, or the fit was asked to
        stop), or its derived values ('derived_errors': an expression
        fails there).
        """
        objective = problem.objective()
        params = problem.params()
        x = np.asarray(x, dtype=float)
        values = params.evaluate(x)
        problems = []
        try:
            derived = params.derived_elements(values)
        except ParamEvaluationError as e:
            derived = {}
            problems.append(('derived_errors', f"no derived values: {e}"))
        chi2, log_likelihood = objective.chi2_and_log_likelihood(values)
        nresiduals = objective.nresiduals()
        nevals = 1
        nfitted = problem.ndim() + objective.nnuisance()
        dof = objective.nindependent() - nfitted
        # (the noise residuals of a fitted noise are not of the data: a
        # reduced chi-square with them would scale the covariance wrongly)
        reduced_chi2 = (chi2 - objective.noise_chi2()) / dof if dof > 0 \
            else np.nan
        log_prior = None if problem.priors() is None \
            else problem.log_prior(x)
        cov = laplace = None
        options = self._uncertainty
        if options.covariance():
            try:
                cov, cov_nevals = covariance(
                    problem, x, bounds.lower, bounds.upper, bounds.scales,
                    options, reduced_chi2, tuple(derived))
                nevals += cov_nevals
            except ParamEvaluationError as e:
                problems.append(('uncertainty_errors', f"no covariance: {e}"))
            except FitStopped:
                problems.append((
                    'uncertainty_errors',
                    "no covariance: the fit was asked to stop"))
        if cov is not None and options.laplace_samples() > 0:
            laplace = laplace_posterior(
                problem, x, cov, options.laplace_samples(), rng,
                tuple(derived))
        return FitSolution(
            free=dict(zip(problem.names(), x.tolist())),
            values=values,
            elements=params.element_values(values),
            derived=derived,
            chi2=chi2,
            nresiduals=nresiduals,
            nfitted=nfitted,
            dof=dof,
            reduced_chi2=reduced_chi2,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            covariance=cov,
            laplace=laplace), nevals, problems


def _stopped_run(problem: FitProblem) -> FitRun:
    """
    Return the run of a fitter that was asked to stop: the best point so
    far.

    Raises
    ------
    RuntimeError
        If it had no point with a finite chi-square.
    """
    best = problem.best()
    if best is None:
        raise RuntimeError(
            "the fit was stopped before any point with a finite chi-square")
    return FitRun(
        solutions=[best[0]], success=False,
        message='stopped before the fitter finished: the best point so far')


def _check_valid_ranges(
        fit_params: dict[str, FitParam],
        ranges: dict[str, tuple[float, float]]
) -> None:
    """
    Raise ConfigError if the fitter may try values of free elements (in
    their bounds, or the support of their priors) outside the valid
    ranges of their parameters (by element name), where the models are
    not defined.
    """
    outside = [
        f"'{name}' [{fit_param.minimum():g}, {fit_param.maximum():g}] "
        f"(valid: [{ranges[name][0]:g}, {ranges[name][1]:g}])"
        for name, fit_param in fit_params.items()
        if name in ranges and not (
            ranges[name][0] <= fit_param.minimum()
            and fit_param.maximum() <= ranges[name][1])]
    if outside:
        raise ConfigError(
            f"the fit may try values of free elements outside the valid "
            f"ranges of their parameters, where the models are not defined: "
            f"{', '.join(outside)}; give them bounds (or priors) within "
            f"those ranges")


@dataclasses.dataclass(frozen=True)
class _Bounds:
    """The bounds and the typical scales of the free values."""
    lower: np.ndarray
    upper: np.ndarray
    scales: np.ndarray


def _posterior(
        problem: FitProblem,
        samples: np.ndarray,
        weights: np.ndarray | None,
        derived: Sequence[str],
        unused: np.ndarray | None = None
) -> Posterior:
    """
    Return the posterior of all elements and of the given elements of the
    derived values (see sample_values), from samples of the free, with
    NaN in the free elements a sample does not use (see FitRun). The
    samples of no weight and those where the parameters cannot be
    evaluated (which have no likelihood) are left out.
    """
    params = problem.params()
    names = tuple(params.kinds()) + tuple(derived)
    columns = np.array([names.index(name) for name in problem.names()])
    rows = []
    kept = []
    for i, x in enumerate(samples):
        if weights is not None and weights[i] == 0:
            continue
        try:
            row = sample_values(params, x, derived)
        except ParamEvaluationError:
            continue
        if unused is not None:
            row[columns[unused[i]]] = np.nan
        rows.append(row)
        kept.append(i)
    return Posterior(
        names=names, samples=np.array(rows).reshape(-1, len(names)),
        weights=None if weights is None else weights[kept],
        approximate=False)


def _dump_params(params: ParamSpace) -> dict[str, Any] | None:
    """Return the configuration of the parameters, if it can be dumped."""
    try:
        return params.dump()
    except ConfigError:
        return None


def _versions(libraries: tuple[str, ...]) -> dict[str, str]:
    """Return the versions of gbkfit, numpy, scipy and the libraries."""
    versions = {}
    for name in ('gbkfit', 'numpy', 'scipy', *libraries):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'unknown'
    return versions


fitter_parser = parseutils.TypedParser(Fitter)
