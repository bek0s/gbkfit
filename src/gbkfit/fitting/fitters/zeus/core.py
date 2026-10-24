"""
The zeus ensemble slice sampler.
"""

import contextlib
import random
import typing
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
import zeus

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.control import FitStopped
from gbkfit.fitting.fitparams import FitParam, FitParamClass, FitParamWalkers
from gbkfit.fitting.fitters._detail import start_walkers
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitParamZeus',
    'FitterZeus'
]


class FitParamZeus(FitParamWalkers):
    """
    The options of a free element for zeus: its prior, and where the
    walkers start (see FitParamWalkers).
    """


@contextlib.contextmanager
def _seeded_globals(seed: int) -> Iterator[None]:
    """
    Seed the global random numbers of NumPy and of the module random,
    which zeus draws, and restore their states afterwards.
    """
    numpy_state, random_state = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)


class FitterZeus(Fitter):
    """
    MCMC sampling with zeus (Karamanis & Beutler 2021; Karamanis, Beutler
    & Peacock 2021): an ensemble slice sampler. Each walker moves along a
    direction drawn from the other walkers, by slice sampling along it,
    whose scale zeus tunes as it runs: there is no step size to choose,
    and every step moves, at a few evaluations of each walker. It gives
    the posterior (the chain after the burn-in); its best solution is the
    point with the highest likelihood it evaluated.

    The free elements take a prior, and optionally where the walkers
    start (see FitParamZeus). The walkers are evaluated in batches.

    Parameters
    ----------
    nwalkers : int, optional
        The number of walkers, even (by default, the larger of 32 and 4
        times the number of free elements; at least twice that number).
    nsteps : int, optional
        The number of steps of each walker.
    burnin : int, optional
        The steps discarded at the start (by default, half).
    thin : int, optional
        Keep every thin-th step of the chain.
    moves : {'differential', 'global'}, optional
        The directions of the walkers: 'differential' (the default), the
        differences of pairs of the other walkers; or 'global', for
        posteriors of several modes, the differences between the
        components of a Gaussian mixture of the other walkers, which jump
        between modes (after the burn-in, which takes 'differential', as
        zeus advises).
    light_mode : bool, optional
        Do not widen the slices after zeus has tuned their scale: fewer
        evaluations, for posteriors that are nearly Gaussian.
    seed : int, optional
        The seed of the random numbers (zeus draws the global ones of
        NumPy and of the module random, which it seeds for the run, and
        restores after it).
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution.

    Raises
    ------
    ConfigError
        If nwalkers is not even and positive, nsteps or thin is not
        positive, burnin is not in [0, nsteps), thin keeps no step after
        the burn-in, or the moves are 'global' without a burn-in.
    """

    @staticmethod
    def type() -> str:
        return 'zeus'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamZeus

    def libraries(self) -> tuple[str, ...]:
        return ('zeus',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), nwalkers=self._nwalkers, nsteps=self._nsteps,
            burnin=self._burnin, thin=self._thin, moves=self._moves,
            light_mode=self._light_mode
        ) | self.dump_common()

    def __init__(
            self,
            nwalkers: int | None = None,
            nsteps: int = 2000,
            burnin: int | None = None,
            thin: int = 1,
            moves: Literal['differential', 'global'] = 'differential',
            light_mode: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                nwalkers=nwalkers, nsteps=nsteps, thin=thin).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        if nwalkers is not None and nwalkers % 2:
            raise ConfigError(
                f"option 'nwalkers' must be even (zeus moves two halves of "
                f"the walkers in turn); it is {nwalkers}")
        if burnin is not None and not 0 <= burnin < nsteps:
            raise ConfigError(
                f"option 'burnin' must be in [0, nsteps); it is {burnin}")
        burnin = nsteps // 2 if burnin is None else burnin
        if thin > nsteps - burnin:
            raise ConfigError(
                f"option 'thin' ({thin}) keeps no step of the "
                f"{nsteps - burnin} after the burn-in")
        if moves == 'global' and burnin == 0:
            raise ConfigError(
                "the moves 'global' start after a burn-in: give a positive "
                "burnin")
        self._nwalkers = nwalkers
        self._nsteps = nsteps
        self._burnin = burnin
        self._thin = thin
        self._moves = moves
        self._light_mode = light_mode

    def _stages(self) -> list[tuple[Any, int]]:
        """
        Return the moves of the run and their steps: the differential
        move for all of them, or for the burn-in and then the global move.
        """
        if self._moves == 'differential':
            return [(zeus.moves.DifferentialMove(), self._nsteps)]
        return [(zeus.moves.DifferentialMove(), self._burnin),
                (zeus.moves.GlobalMove(), self._nsteps - self._burnin)]

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        params = typing.cast(list[FitParamZeus], fit_params)
        ndim = problem.ndim()
        nwalkers = self._nwalkers or max(32, 4 * ndim)
        if nwalkers < 2 * ndim:
            raise ConfigError(
                f"zeus needs at least {2 * ndim} walkers (twice the free "
                f"elements); nwalkers is {nwalkers}")
        coords, log_prob = start_walkers(problem, params, nwalkers, rng)
        # Each step of the chain goes to the log of the fit as it runs; a
        # fit asked to stop keeps the chain so far (half of it burn-in, if
        # shorter than the burn-in). The chain is that of the last stage.
        stopped = False
        nsteps = ncall = 0
        with _seeded_globals(int(rng.integers(2 ** 31))):
            for move, steps in self._stages():
                sampler = zeus.EnsembleSampler(
                    nwalkers, ndim, problem.log_posterior_batch, moves=move,
                    vectorize=True, light_mode=self._light_mode,
                    verbose=False)
                try:
                    for x, z, _ in sampler.sample(
                            coords, log_prob0=log_prob, iterations=steps,
                            progress=False):
                        problem.record_chain(x, z)
                except FitStopped:
                    stopped = True
                nsteps += sampler.iteration
                ncall += int(sampler.ncall)
                if stopped:
                    break
                coords = sampler.get_last_sample()
                log_prob = sampler.get_last_log_prob()
        # (a fit stopped within the burn-in keeps half of its chain)
        burnin = self._burnin if self._burnin < nsteps else nsteps // 2
        # (the steps of the burn-in in the chain of the last stage)
        discard = max(0, burnin - (nsteps - sampler.iteration))
        if sampler.iteration - discard < 1:
            raise FitStopped("zeus was stopped before its first steps")
        samples = sampler.get_chain(
            flat=True, discard=discard, thin=self._thin)
        tau = zeus.AutoCorrTime(sampler.get_chain(discard=discard))
        kept = sampler.iteration - discard
        converged = not stopped and kept > 50 * np.max(tau)
        best = problem.best()
        if best is None:
            raise RuntimeError("zeus found no point with a finite chi2")
        if stopped:
            message = f'stopped after {nsteps} of {self._nsteps} steps'
        elif converged:
            message = 'the chain after the burn-in is longer than 50 ' \
                'autocorrelation times'
        else:
            message = 'the chain after the burn-in is shorter than 50 ' \
                'autocorrelation times: run longer'
        return FitRun(
            solutions=[best[0]],
            success=bool(converged),
            message=message,
            samples=samples,
            diagnostics=dict(
                autocorr_time=dict(zip(problem.names(), tau.tolist())),
                n_eff=float(kept * nwalkers / np.max(tau)),
                ncall=ncall,
                nsteps=nsteps),
            raw=sampler)
