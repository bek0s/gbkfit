"""
The UltraNest nested sampler.
"""

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import numpy as np
import ultranest

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitparams import FitParam, FitParamClass, FitParamPrior
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import Evidence
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitterUltraNest'
]


# The log-likelihood UltraNest gets for points where the parameters cannot
# be evaluated: it takes only finite values (UltraNest's documentation
# suggests a very low one), and their posterior weight underflows to 0
_REJECTED = -1e300


@contextlib.contextmanager
def _seeded_global_rng(seed: int) -> Iterator[None]:
    """
    Seed numpy's global random numbers, which UltraNest draws from (it
    takes no generator), and restore their state afterwards.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def _log_level(name: str, level: int) -> Iterator[None]:
    """Set the level of a logger, and restore it afterwards."""
    logger = logging.getLogger(name)
    previous = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)


class FitterUltraNest(Fitter):
    """
    Nested sampling with UltraNest (Buchner 2021): MLFriends regions,
    with as many live points as the posterior needs. It gives the
    posterior (weighted samples) and the evidence with its error; its
    best solution is the sample with the highest likelihood.

    The free elements take a prior (or min and max for a uniform prior;
    see FitParamPrior). The likelihood is evaluated in batches.

    Parameters
    ----------
    min_num_live_points : int, optional
        The fewest live points.
    dlogz : float, optional
        The error of the evidence at which sampling stops.
    frac_remain : float, optional
        The largest fraction of the evidence left in the live points when
        sampling stops.
    min_ess : int, optional
        The fewest effective samples of the posterior.
    max_ncalls : int, optional
        The most likelihood evaluations (by default, no limit).
    verbose : bool, optional
        Whether UltraNest reports its progress.
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
        return 'ultranest'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamPrior

    def libraries(self) -> tuple[str, ...]:
        return ('ultranest',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), min_num_live_points=self._min_num_live_points,
            dlogz=self._dlogz, frac_remain=self._frac_remain,
            min_ess=self._min_ess, max_ncalls=self._max_ncalls,
            verbose=self._verbose) | self.dump_common()

    def __init__(
            self,
            min_num_live_points: int = 400,
            dlogz: float = 0.5,
            frac_remain: float = 0.01,
            min_ess: int = 400,
            max_ncalls: int | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                min_num_live_points=min_num_live_points, dlogz=dlogz,
                frac_remain=frac_remain, min_ess=min_ess,
                max_ncalls=max_ncalls).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        self._min_num_live_points = min_num_live_points
        self._dlogz = dlogz
        self._frac_remain = frac_remain
        self._min_ess = min_ess
        self._max_ncalls = max_ncalls
        self._verbose = verbose

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        def log_likelihood(xs: np.ndarray) -> np.ndarray:
            values = problem.log_likelihood_batch(xs)
            return np.where(np.isfinite(values), values, _REJECTED)

        sampler = ultranest.ReactiveNestedSampler(
            list(problem.names()), log_likelihood, problem.prior_transform,
            vectorized=True, log_dir=None)
        with (_seeded_global_rng(int(rng.integers(2 ** 31))),
              _log_level('ultranest', logging.INFO if self._verbose
                         else logging.WARNING)):
            result = sampler.run(
                min_num_live_points=self._min_num_live_points,
                dlogz=self._dlogz, frac_remain=self._frac_remain,
                min_ess=self._min_ess, max_ncalls=self._max_ncalls,
                show_status=self._verbose, viz_callback=False)
        weighted = result['weighted_samples']
        stopped = self._max_ncalls is not None \
            and result['ncall'] >= self._max_ncalls
        return FitRun(
            solutions=[np.asarray(result['maximum_likelihood']['point'])],
            success=not stopped,
            message='stopped by max_ncalls' if stopped else 'converged',
            samples=np.asarray(weighted['points']),
            weights=np.asarray(weighted['weights']),
            evidence=Evidence(
                log_z=float(result['logz']),
                log_z_error=float(result['logzerr'])),
            diagnostics=dict(
                ncall=int(result['ncall']), niter=int(result['niter']),
                ess=float(result['ess'])),
            raw=result)
