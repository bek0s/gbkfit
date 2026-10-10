"""
The base class of the fitters, and what their runs give.
"""

import abc
import dataclasses
import importlib.metadata
import time
from typing import Any

import numpy as np

from gbkfit.fitting.fitparams import FitParam, FitParamClass
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import (
    Evidence, FitResult, FitSolution, Posterior)
from gbkfit.fitting.uncertainty import (
    UncertaintyOptions, covariance, laplace_posterior)
from gbkfit.objective import Objective
from gbkfit.params import ParamSpace
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

    def fit(self, objective: Objective, params: ParamSpace) -> FitResult:
        """
        Fit the data of an objective.

        Parameters
        ----------
        objective : Objective
            The objective.
        params : ParamSpace
            The parameters of its models.

        Returns
        -------
        FitResult
            The result.

        Raises
        ------
        ConfigError
            If the free properties are invalid for the fitter.
        """
        fit_params = self.load_fit_params(params)
        priors = [p.prior() for p in fit_params.values()]
        if len({p is None for p in priors}) > 1:
            raise ConfigError("either all free elements have priors, or none")
        problem = FitProblem(
            objective, params, None if priors[0] is None else priors)
        rng = np.random.default_rng(self._seed)
        start = time.perf_counter()
        run = self._run(problem, list(fit_params.values()), rng)
        elapsed = time.perf_counter() - start
        lower = np.array([p.minimum() for p in fit_params.values()])
        upper = np.array([p.maximum() for p in fit_params.values()])
        diagnostics = dict(run.diagnostics)
        solutions = []
        uncertainty_nevals = 0
        for x in run.solutions:
            solution, nevals = self._solution(problem, x, lower, upper, rng)
            solutions.append(solution)
            uncertainty_nevals += nevals
        diagnostics['uncertainty_nevals'] = uncertainty_nevals
        posterior = None
        if run.samples is not None:
            posterior = _posterior(problem, run.samples, run.weights)
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
            lower: np.ndarray,
            upper: np.ndarray,
            rng: np.random.Generator
    ) -> tuple[FitSolution, int]:
        """
        Evaluate a solution and its uncertainty; return it and the number
        of model evaluations it took.
        """
        objective = problem.objective()
        params = problem.params()
        x = np.asarray(x, dtype=float)
        values = params.evaluate(x)
        chi2 = objective.chi2(values)
        nresiduals = objective.nresiduals(values)
        nevals = 2
        dof = nresiduals - problem.ndim()
        reduced_chi2 = chi2 / dof if dof > 0 else np.nan
        log_prior = None if problem.priors() is None \
            else problem.log_prior(x)
        cov = laplace = None
        options = self._uncertainty
        if options.covariance():
            cov, cov_nevals = covariance(
                problem, x, lower, upper, options, reduced_chi2)
            nevals += cov_nevals
            if options.laplace_samples() > 0:
                laplace = laplace_posterior(
                    problem, x, cov, options.laplace_samples(), rng)
        return FitSolution(
            free=dict(zip(problem.names(), x.tolist())),
            values=values,
            elements=params.element_values(values),
            chi2=chi2,
            nresiduals=nresiduals,
            dof=dof,
            reduced_chi2=reduced_chi2,
            log_likelihood=-0.5 * chi2,
            log_prior=log_prior,
            covariance=cov,
            laplace=laplace), nevals


def _posterior(
        problem: FitProblem,
        samples: np.ndarray,
        weights: np.ndarray | None
) -> Posterior:
    """Return the posterior of all elements, from samples of the free."""
    params = problem.params()
    rows = [
        list(params.element_values(params.evaluate(x)).values())
        for x in samples]
    names = tuple(params.element_values(params.evaluate(samples[0])))
    return Posterior(
        names=names, samples=np.array(rows), weights=weights,
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
