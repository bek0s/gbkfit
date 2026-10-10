"""
The dynesty nested samplers: static and dynamic.
"""

from typing import Any, Literal

import dynesty
import numpy as np

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitparams import FitParam, FitParamClass, FitParamPrior
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import Evidence
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitterDynestyDynamic',
    'FitterDynestyStatic'
]


def _check_positive(**options: float | None) -> None:
    """Raise ConfigError if an option is given and not positive."""
    for name, value in options.items():
        if value is not None and not value > 0:
            raise ConfigError(
                f"option '{name}' must be positive; it is {value}")


def _fit_run(
        sampler: dynesty.NestedSampler | dynesty.DynamicNestedSampler,
        maxcall: int | None
) -> FitRun:
    """Return what a run of a dynesty sampler gives."""
    results = sampler.results
    ncall = int(np.sum(results.ncall))
    stopped = maxcall is not None and ncall >= maxcall
    log_z = float(results.logz[-1])
    return FitRun(
        solutions=[results.samples[np.argmax(results.logl)]],
        success=not stopped,
        message='stopped by maxcall' if stopped else 'converged',
        samples=results.samples,
        weights=np.exp(results.logwt - log_z),
        evidence=Evidence(
            log_z=log_z, log_z_error=float(results.logzerr[-1])),
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
            nlive: int = 500,
            bound: Literal[
                'multi', 'single', 'balls', 'cubes', 'none'] = 'multi',
            sample: Literal[
                'auto', 'unif', 'rwalk', 'slice', 'rslice'] = 'auto',
            dlogz: float | None = None,
            maxcall: int | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        _check_positive(nlive=nlive, dlogz=dlogz, maxcall=maxcall)
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
        sampler = dynesty.NestedSampler(
            problem.log_likelihood, problem.prior_transform, problem.ndim(),
            nlive=self._nlive, bound=self._bound, sample=self._sample,
            rstate=np.random.default_rng(int(rng.integers(2 ** 31))))
        options = {} if self._dlogz is None else dict(dlogz=self._dlogz)
        sampler.run_nested(
            maxcall=self._maxcall, print_progress=self._verbose, **options)
        return _fit_run(sampler, self._maxcall)


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
            nlive: int = 500,
            bound: Literal[
                'multi', 'single', 'balls', 'cubes', 'none'] = 'multi',
            sample: Literal[
                'auto', 'unif', 'rwalk', 'slice', 'rslice'] = 'auto',
            dlogz_init: float | None = None,
            n_effective: int | None = None,
            maxcall: int | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        _check_positive(
            nlive=nlive, dlogz_init=dlogz_init, n_effective=n_effective,
            maxcall=maxcall)
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
        sampler = dynesty.DynamicNestedSampler(
            problem.log_likelihood, problem.prior_transform, problem.ndim(),
            nlive=self._nlive, bound=self._bound, sample=self._sample,
            rstate=np.random.default_rng(int(rng.integers(2 ** 31))))
        options = dict(
            dlogz_init=self._dlogz_init, n_effective=self._n_effective)
        sampler.run_nested(
            maxcall=self._maxcall, print_progress=self._verbose,
            **{k: v for k, v in options.items() if v is not None})
        return _fit_run(sampler, self._maxcall)
