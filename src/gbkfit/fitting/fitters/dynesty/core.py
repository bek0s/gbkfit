"""
The dynesty nested sampler.
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
    'FitterDynesty'
]


class FitterDynesty(Fitter):
    """
    Nested sampling with dynesty (Speagle 2020): static, or dynamic (which
    adds live points where the posterior is, after a first run). It gives
    the posterior (weighted samples) and the evidence with its error; its
    best solution is the sample with the highest likelihood.

    The free elements take a prior (or min and max for a uniform prior;
    see FitParamPrior). The likelihood is evaluated one point at a time.

    Parameters
    ----------
    dynamic : bool, optional
        Whether to sample dynamically.
    nlive : int, optional
        The number of live points (of the first run, if dynamic).
    bound : {'multi', 'single', 'balls', 'cubes', 'none'}, optional
        The bounds of the live points.
    sample : {'auto', 'unif', 'rwalk', 'slice', 'rslice'}, optional
        How new points are drawn in the bounds ('auto' chooses by the
        number of free elements).
    dlogz : float, optional
        The evidence left in the live points at which a (first) run stops
        (by default, dynesty's).
    n_effective : int, optional
        The effective sample size at which a dynamic run stops (by
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
        If a number is not positive, or n_effective is given for a static
        run.
    """

    @staticmethod
    def type() -> str:
        return 'dynesty'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamPrior

    def libraries(self) -> tuple[str, ...]:
        return ('dynesty',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), dynamic=self._dynamic, nlive=self._nlive,
            bound=self._bound, sample=self._sample, dlogz=self._dlogz,
            n_effective=self._n_effective, maxcall=self._maxcall,
            verbose=self._verbose) | self.dump_common()

    def __init__(
            self,
            dynamic: bool = False,
            nlive: int = 500,
            bound: Literal[
                'multi', 'single', 'balls', 'cubes', 'none'] = 'multi',
            sample: Literal[
                'auto', 'unif', 'rwalk', 'slice', 'rslice'] = 'auto',
            dlogz: float | None = None,
            n_effective: int | None = None,
            maxcall: int | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                nlive=nlive, dlogz=dlogz, n_effective=n_effective,
                maxcall=maxcall).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        if n_effective is not None and not dynamic:
            raise ConfigError("option 'n_effective' needs dynamic: true")
        self._dynamic = dynamic
        self._nlive = nlive
        self._bound = bound
        self._sample = sample
        self._dlogz = dlogz
        self._n_effective = n_effective
        self._maxcall = maxcall
        self._verbose = verbose

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        options: dict[str, Any] = dict(
            nlive=self._nlive, bound=self._bound, sample=self._sample,
            rstate=np.random.default_rng(int(rng.integers(2 ** 31))))
        if self._dynamic:
            sampler = dynesty.DynamicNestedSampler(
                problem.log_likelihood, problem.prior_transform,
                problem.ndim(), **options)
            run_options: dict[str, Any] = dict(
                dlogz_init=self._dlogz, n_effective=self._n_effective)
        else:
            sampler = dynesty.NestedSampler(
                problem.log_likelihood, problem.prior_transform,
                problem.ndim(), **options)
            run_options = dict(dlogz=self._dlogz)
        run_options = {k: v for k, v in run_options.items() if v is not None}
        sampler.run_nested(
            maxcall=self._maxcall, print_progress=self._verbose,
            **run_options)
        results = sampler.results
        ncall = int(np.sum(results.ncall))
        stopped = self._maxcall is not None and ncall >= self._maxcall
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
