"""
The dynesty nested samplers: static and dynamic.
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal

import dynesty
import numpy as np
from annotated_types import Gt

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitters._detail import corrected_evidence
from gbkfit.fitting.fitparams import FitParam, FitParamClass, FitParamPrior
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import Evidence
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import check_options


__all__ = [
    'FitterDynestyDynamic',
    'FitterDynestyStatic'
]


_Sampler = dynesty.NestedSampler | dynesty.DynamicNestedSampler


def _evidence(sampler: _Sampler) -> Evidence:
    """Return the evidence of a run of a dynesty sampler."""
    results = sampler.results
    return Evidence(
        log_z=float(results.logz[-1]),
        log_z_error=float(results.logzerr[-1]))


def _fit_run(
        sampler: _Sampler,
        maxcall: int | None,
        normalisation: _Sampler | None = None
) -> FitRun:
    """
    Return what a run of a dynesty sampler gives, with its evidence
    corrected by a run on the priors on several parameters alone, if any
    (see corrected_evidence).
    """
    results = sampler.results
    ncall = int(np.sum(results.ncall))
    stopped = maxcall is not None and ncall >= maxcall
    log_z = float(results.logz[-1])
    evidence = _evidence(sampler)
    if normalisation is not None:
        evidence = corrected_evidence(evidence, _evidence(normalisation))
    return FitRun(
        solutions=[results.samples[np.argmax(results.logl)]],
        success=not stopped,
        message='stopped by maxcall' if stopped else 'converged',
        samples=results.samples,
        weights=np.exp(results.logwt - log_z),
        evidence=evidence,
        diagnostics=dict(
            ncall=ncall, niter=int(results.niter),
            efficiency=float(results.eff)),
        raw=results)


class FitterDynestyStatic(Fitter):
    """
    Static nested sampling with dynesty (Speagle 2020): a fixed number of
    live points. It gives the posterior (weighted samples) and the
    evidence with its error; its best solution is the sample with the
    highest likelihood.

    The free elements take a prior (or min and max for a uniform prior;
    see FitParamPrior). The likelihood is evaluated one point at a time
    (dynesty batches only through a pool).

    Parameters
    ----------
    nlive : int, optional
        The number of live points.
    bound : {'multi', 'single', 'balls', 'cubes', 'none'}, optional
        The bounds of the live points.
    sample : {'auto', 'unif', 'rwalk', 'slice', 'rslice'}, optional
        How new points are drawn in the bounds ('auto' chooses by the
        number of free elements).
    dlogz : float, optional
        The evidence left in the live points at which the run stops (by
        default, dynesty's).
    maxcall : int, optional
        The most likelihood evaluations (by default, no limit).
    verbose : bool, optional
        Whether dynesty reports its progress.
    seed : int, optional
        The seed of the random numbers.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution.

    Raises
    ------
    ConfigError
        If a number is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'dynesty.static'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamPrior

    def libraries(self) -> tuple[str, ...]:
        return ('dynesty',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), nlive=self._nlive, bound=self._bound,
            sample=self._sample, dlogz=self._dlogz, maxcall=self._maxcall,
            verbose=self._verbose) | self.dump_common()

    def __init__(
            self,
            nlive: Annotated[int, Gt(0)] = 500,
            bound: Literal[
                'multi', 'single', 'balls', 'cubes', 'none'] = 'multi',
            sample: Literal[
                'auto', 'unif', 'rwalk', 'slice', 'rslice'] = 'auto',
            dlogz: Annotated[float, Gt(0)] | None = None,
            maxcall: Annotated[int, Gt(0)] | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        check_options(FitterDynestyStatic.__init__, locals())
        super().__init__(seed, uncertainty)
        self._nlive = nlive
        self._bound = bound
        self._sample = sample
        self._dlogz = dlogz
        self._maxcall = maxcall
        self._verbose = verbose

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        sampler = self._run_sampler(
            problem, problem.nested_log_likelihood, rng)
        normalisation = self._run_sampler(
            problem, problem.joint_prior_log_density, rng) \
            if problem.has_joint_priors() else None
        return _fit_run(sampler, self._maxcall, normalisation)

    def _run_sampler(
            self,
            problem: FitProblem,
            log_likelihood: Callable[[np.ndarray], float],
            rng: np.random.Generator
    ) -> dynesty.NestedSampler:
        """
        Run the sampler on a log-likelihood over the priors of the problem.
        """
        sampler = dynesty.NestedSampler(
            log_likelihood, problem.prior_transform, problem.ndim(),
            nlive=self._nlive, bound=self._bound, sample=self._sample,
            rstate=np.random.default_rng(int(rng.integers(2 ** 31))))
        options = {} if self._dlogz is None else dict(dlogz=self._dlogz)
        sampler.run_nested(
            maxcall=self._maxcall, print_progress=self._verbose, **options)
        return sampler


class FitterDynestyDynamic(Fitter):
    """
    Dynamic nested sampling with dynesty (Speagle 2020): after a first
    run, batches of live points are added where the posterior is, until
    its effective sample size is reached. It gives the posterior
    (weighted samples) and the evidence with its error; its best solution
    is the sample with the highest likelihood.

    The free elements take a prior (or min and max for a uniform prior;
    see FitParamPrior). The likelihood is evaluated one point at a time
    (dynesty batches only through a pool).

    Parameters
    ----------
    nlive : int, optional
        The number of live points of the first run and of each batch.
    bound : {'multi', 'single', 'balls', 'cubes', 'none'}, optional
        The bounds of the live points.
    sample : {'auto', 'unif', 'rwalk', 'slice', 'rslice'}, optional
        How new points are drawn in the bounds ('auto' chooses by the
        number of free elements).
    dlogz_init : float, optional
        The evidence left in the live points at which the first run stops
        (by default, dynesty's).
    n_effective : int, optional
        The effective sample size of the posterior at which sampling
        stops (by default, dynesty's).
    maxcall : int, optional
        The most likelihood evaluations (by default, no limit).
    verbose : bool, optional
        Whether dynesty reports its progress.
    seed : int, optional
        The seed of the random numbers.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution.

    Raises
    ------
    ConfigError
        If a number is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'dynesty.dynamic'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamPrior

    def libraries(self) -> tuple[str, ...]:
        return ('dynesty',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), nlive=self._nlive, bound=self._bound,
            sample=self._sample, dlogz_init=self._dlogz_init,
            n_effective=self._n_effective, maxcall=self._maxcall,
            verbose=self._verbose) | self.dump_common()

    def __init__(
            self,
            nlive: Annotated[int, Gt(0)] = 500,
            bound: Literal[
                'multi', 'single', 'balls', 'cubes', 'none'] = 'multi',
            sample: Literal[
                'auto', 'unif', 'rwalk', 'slice', 'rslice'] = 'auto',
            dlogz_init: Annotated[float, Gt(0)] | None = None,
            n_effective: Annotated[int, Gt(0)] | None = None,
            maxcall: Annotated[int, Gt(0)] | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        check_options(FitterDynestyDynamic.__init__, locals())
        super().__init__(seed, uncertainty)
        self._nlive = nlive
        self._bound = bound
        self._sample = sample
        self._dlogz_init = dlogz_init
        self._n_effective = n_effective
        self._maxcall = maxcall
        self._verbose = verbose

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        sampler = self._run_sampler(
            problem, problem.nested_log_likelihood, rng)
        normalisation = self._run_sampler(
            problem, problem.joint_prior_log_density, rng) \
            if problem.has_joint_priors() else None
        return _fit_run(sampler, self._maxcall, normalisation)

    def _run_sampler(
            self,
            problem: FitProblem,
            log_likelihood: Callable[[np.ndarray], float],
            rng: np.random.Generator
    ) -> dynesty.DynamicNestedSampler:
        """
        Run the sampler on a log-likelihood over the priors of the problem.
        """
        sampler = dynesty.DynamicNestedSampler(
            log_likelihood, problem.prior_transform, problem.ndim(),
            nlive=self._nlive, bound=self._bound, sample=self._sample,
            rstate=np.random.default_rng(int(rng.integers(2 ** 31))))
        options = dict(
            dlogz_init=self._dlogz_init, n_effective=self._n_effective)
        sampler.run_nested(
            maxcall=self._maxcall, print_progress=self._verbose,
            **{k: v for k, v in options.items() if v is not None})
        return sampler
