"""
The nautilus nested sampler.
"""

import math
from typing import Any

import nautilus
import numpy as np

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitparams import FitParam, FitParamClass, FitParamPrior
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.result import Evidence
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitterNautilus'
]


class FitterNautilus(Fitter):
    """
    Nested sampling with nautilus (Lange 2023): importance nested
    sampling with neural-network bounds. It gives the posterior (weighted
    samples) and the evidence; its best solution is the sample with the
    highest likelihood. Efficient up to about 50 free elements.

    The free elements take a prior (or min and max for a uniform prior;
    see FitParamPrior). The likelihood is evaluated in batches.

    Parameters
    ----------
    n_live : int, optional
        The number of live points.
    n_networks : int, optional
        The number of neural networks of each bound.
    n_batch : int, optional
        The number of likelihood evaluations per batch (by default,
        nautilus': 100).
    f_live : float, optional
        The largest fraction of the evidence left in the live set when the
        exploration ends.
    n_eff : int, optional
        The effective sample size at which sampling stops.
    n_like_max : int, optional
        The most likelihood evaluations (by default, no limit).
    discard_exploration : bool, optional
        Whether to discard the points of the exploration phase (for an
        unbiased posterior and evidence, at extra cost).
    timeout : float, optional
        The longest time in seconds (by default, no limit).
    verbose : bool, optional
        Whether nautilus reports its progress.
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
        return 'nautilus'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamPrior

    def libraries(self) -> tuple[str, ...]:
        return ('nautilus-sampler',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), n_live=self._n_live,
            n_networks=self._n_networks, n_batch=self._n_batch,
            f_live=self._f_live, n_eff=self._n_eff,
            n_like_max=self._n_like_max,
            discard_exploration=self._discard_exploration,
            timeout=self._timeout, verbose=self._verbose
        ) | self.dump_common()

    def __init__(
            self,
            n_live: int = 2000,
            n_networks: int = 4,
            n_batch: int | None = None,
            f_live: float = 0.01,
            n_eff: int = 10000,
            n_like_max: int | None = None,
            discard_exploration: bool = False,
            timeout: float | None = None,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                n_live=n_live, n_networks=n_networks, n_batch=n_batch,
                f_live=f_live, n_eff=n_eff, n_like_max=n_like_max,
                timeout=timeout).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        self._n_live = n_live
        self._n_networks = n_networks
        self._n_batch = n_batch
        self._f_live = f_live
        self._n_eff = n_eff
        self._n_like_max = n_like_max
        self._discard_exploration = discard_exploration
        self._timeout = timeout
        self._verbose = verbose

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        if problem.ndim() < 2:
            raise ConfigError("nautilus needs at least 2 free elements")
        sampler = nautilus.Sampler(
            problem.prior_transform, problem.log_likelihood_batch,
            n_dim=problem.ndim(), n_live=self._n_live,
            n_networks=self._n_networks, n_batch=self._n_batch,
            vectorized=True, pass_dict=False,
            seed=int(rng.integers(2 ** 31)))
        success = sampler.run(
            f_live=self._f_live, n_eff=self._n_eff,
            n_like_max=self._n_like_max or math.inf,
            discard_exploration=self._discard_exploration,
            timeout=self._timeout or math.inf, verbose=self._verbose)
        points, log_w, log_l = sampler.posterior()
        message = 'converged' if success else \
            'stopped by n_like_max or timeout'
        return FitRun(
            solutions=[points[np.argmax(log_l)]],
            success=bool(success),
            message=message,
            samples=points,
            weights=np.exp(log_w),
            evidence=Evidence(log_z=float(sampler.log_z), log_z_error=None),
            diagnostics=dict(
                n_like=int(sampler.n_like), n_eff=float(sampler.n_eff),
                eta=float(sampler.eta)),
            raw=sampler)
