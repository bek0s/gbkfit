"""
The UltraNest nested sampler.
"""

import contextlib
import logging
from collections.abc import Callable, Iterator
from typing import Annotated, Any

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

    requires = {'ultranest': "pip install 'gbkfit[ultranest]'"}

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
            min_num_live_points: Annotated[int, Gt(0)] = 400,
            dlogz: Annotated[float, Gt(0)] = 0.5,
            frac_remain: Annotated[float, Gt(0)] = 0.01,
            min_ess: Annotated[int, Gt(0)] = 400,
            max_ncalls: Annotated[int, Gt(0)] | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        check_options(FitterUltraNest.__init__, locals())
        super().__init__(seed, uncertainty)
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
        result = self._sample(
            problem, problem.nested_log_likelihood_batch, rng)
        evidence = _evidence(result)
        if problem.has_joint_priors():
            evidence = corrected_evidence(evidence, _evidence(self._sample(
                problem, problem.joint_prior_log_density_batch, rng)))
        weighted = result['weighted_samples']
        stopped = self._max_ncalls is not None \
            and result['ncall'] >= self._max_ncalls
        return FitRun(
            solutions=[np.asarray(result['maximum_likelihood']['point'])],
            success=not stopped,
            message='stopped by max_ncalls' if stopped else 'converged',
            samples=np.asarray(weighted['points']),
            weights=np.asarray(weighted['weights']),
            evidence=evidence,
            diagnostics=dict(
                ncall=int(result['ncall']), niter=int(result['niter']),
                ess=float(result['ess'])),
            raw=result)

    def _sample(
            self,
            problem: FitProblem,
            log_likelihood: Callable[[np.ndarray], np.ndarray],
            rng: np.random.Generator
    ) -> dict[str, Any]:
        """
        Run the sampler on a log-likelihood (of points, one per row) over
        the priors of the problem; return its result.
        """
        import ultranest
        def finite(xs: np.ndarray) -> np.ndarray:
            values = log_likelihood(xs)
            return np.where(np.isfinite(values), values, _REJECTED)

        sampler = ultranest.ReactiveNestedSampler(
            list(problem.names()), finite, problem.prior_transform,
            vectorized=True, log_dir=None)
        with (_seeded_global_rng(int(rng.integers(2 ** 31))),
              _log_level('ultranest', logging.INFO if self._verbose
                         else logging.WARNING)):
            return sampler.run(
                min_num_live_points=self._min_num_live_points,
                dlogz=self._dlogz, frac_remain=self._frac_remain,
                min_ess=self._min_ess, max_ncalls=self._max_ncalls,
                show_status=self._verbose, viz_callback=False)


def _evidence(result: dict[str, Any]) -> Evidence:
    """Return the evidence of the result of a run."""
    return Evidence(
        log_z=float(result['logz']), log_z_error=float(result['logzerr']))
