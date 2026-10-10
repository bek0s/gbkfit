"""
The emcee MCMC sampler.
"""

import math
import typing
from typing import Any, Literal

import emcee
import numpy as np

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitparams import FitParam, FitParamClass, FitParamPrior
from gbkfit.fitting.priors import Prior
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitParamEmcee',
    'FitterEmcee'
]


class FitParamEmcee(FitParamPrior):
    """
    The options of a free element for emcee: its prior, and where the
    walkers start.

    The walkers start uniformly in a box of the given width around the
    value, if a width is given, or else from the prior.

    Parameters
    ----------
    prior : Prior
        The prior (in a config, min and max without a prior give a
        uniform one).
    value : float, optional
        The centre of the box of the walkers.
    width : float, optional
        The width of the box of the walkers (needs a value).

    Raises
    ------
    ConfigError
        If the value is not in the support of the prior, or a width is
        given without a value, or is not positive.
    """

    def dump(self) -> dict[str, Any]:
        info = super().dump()
        if self._width is not None:
            info['width'] = self._width
        return info

    def __init__(
            self,
            prior: Prior,
            value: float | None = None,
            width: float | None = None
    ):
        super().__init__(prior, value)
        if width is not None and value is None:
            raise ConfigError("option 'width' needs option 'value'")
        if width is not None and not (math.isfinite(width) and width > 0):
            raise ConfigError(
                f"option 'width' must be finite and positive; it is {width}")
        self._width = width

    def width(self) -> float | None:
        """Return the width of the box of the walkers, if any."""
        return self._width

    def sample_start(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Return the starting values of the walkers."""
        if self._width is None:
            return self.prior().sample(rng, size)
        value = typing.cast(float, self.value())
        lo = max(value - self._width / 2, self.minimum())
        hi = min(value + self._width / 2, self.maximum())
        return rng.uniform(lo, hi, size)


class FitterEmcee(Fitter):
    """
    MCMC sampling with emcee (Foreman-Mackey et al. 2013): an ensemble of
    walkers, moved by differential evolution (default) or by stretches.
    It gives the posterior (the chain after the burn-in); its best
    solution is the point with the highest likelihood it evaluated.

    The free elements take a prior, and optionally where the walkers
    start (see FitParamEmcee). The walkers are evaluated in batches.

    Parameters
    ----------
    nwalkers : int, optional
        The number of walkers (by default, the larger of 32 and 4 times
        the number of free elements; at least twice that number).
    nsteps : int, optional
        The number of steps of each walker.
    burnin : int, optional
        The steps discarded at the start (by default, half).
    thin : int, optional
        Keep every thin-th step of the chain.
    moves : {'de', 'stretch'}, optional
        The moves: 'de' is differential evolution (ter Braak 2006), and
        'stretch' the affine-invariant stretch move (slower to mix). (Not
        emcee's DESnookerMove: in emcee 3.1.6 its step is not along a unit
        vector, and it samples correlated posteriors too narrowly, by 12%
        on a 10-parameter fit.)
    seed : int, optional
        The seed of the random numbers.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution.

    Raises
    ------
    ConfigError
        If a number is not positive, or the burn-in is not shorter than
        the chain.
    """

    @staticmethod
    def type() -> str:
        return 'emcee'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamEmcee

    def libraries(self) -> tuple[str, ...]:
        return ('emcee',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), nwalkers=self._nwalkers, nsteps=self._nsteps,
            burnin=self._burnin, thin=self._thin, moves=self._moves
        ) | self.dump_common()

    def __init__(
            self,
            nwalkers: int | None = None,
            nsteps: int = 5000,
            burnin: int | None = None,
            thin: int = 1,
            moves: Literal['de', 'stretch'] = 'de',
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                nwalkers=nwalkers, nsteps=nsteps, thin=thin).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        if burnin is not None and not 0 <= burnin < nsteps:
            raise ConfigError(
                f"option 'burnin' must be in [0, nsteps); it is {burnin}")
        self._nwalkers = nwalkers
        self._nsteps = nsteps
        self._burnin = burnin
        self._thin = thin
        self._moves = moves

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        params = typing.cast(list[FitParamEmcee], fit_params)
        ndim = problem.ndim()
        nwalkers = self._nwalkers or max(32, 4 * ndim)
        if nwalkers < 2 * ndim:
            raise ConfigError(
                f"emcee needs at least {2 * ndim} walkers (twice the free "
                f"elements); nwalkers is {nwalkers}")
        burnin = self._nsteps // 2 if self._burnin is None else self._burnin
        moves = emcee.moves.DEMove() if self._moves == 'de' \
            else emcee.moves.StretchMove()
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, problem.log_posterior_batch, moves=moves,
            vectorize=True)
        sampler.random_state = np.random.RandomState(
            int(rng.integers(2 ** 31))).get_state()
        sampler.run_mcmc(
            self._start(problem, params, nwalkers, rng), self._nsteps)
        samples = sampler.get_chain(discard=burnin, thin=self._thin, flat=True)
        tau = sampler.get_autocorr_time(discard=burnin, quiet=True)
        converged = (self._nsteps - burnin) > 50 * np.max(tau)
        best = problem.best()
        if best is None:
            raise RuntimeError("emcee found no point with a finite chi2")
        return FitRun(
            solutions=[best[0]],
            success=bool(converged),
            message='the chain after the burn-in is longer than 50 '
                    'autocorrelation times' if converged else
                    'the chain after the burn-in is shorter than 50 '
                    'autocorrelation times: run longer',
            samples=samples,
            diagnostics=dict(
                acceptance_fraction=float(np.mean(
                    sampler.acceptance_fraction)),
                autocorr_time=dict(zip(problem.names(), tau.tolist())),
                n_eff=float(
                    (self._nsteps - burnin) * nwalkers / np.max(tau))),
            raw=sampler)

    @staticmethod
    def _start(
            problem: FitProblem,
            params: list[FitParamEmcee],
            nwalkers: int,
            rng: np.random.Generator
    ) -> np.ndarray:
        """
        Return the starting points of the walkers, each with a finite
        prior density (redrawn up to 100 times).
        """
        start = np.stack(
            [p.sample_start(rng, nwalkers) for p in params], axis=1)
        for _ in range(100):
            bad = [
                i for i, x in enumerate(start)
                if not math.isfinite(problem.log_prior(x))]
            if not bad:
                return start
            start[bad] = np.stack(
                [p.sample_start(rng, len(bad)) for p in params], axis=1)
        raise ConfigError(
            "could not start the walkers where the prior density is "
            "positive")
