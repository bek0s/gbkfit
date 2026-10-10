"""
The CMA-ES fitter (pycma).
"""

import math
import typing
from typing import Any

import cma
import numpy as np

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitparams import (
    FitParam, FitParamBounded, FitParamClass)
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitParamCMAES',
    'FitterCMAES'
]


class FitParamCMAES(FitParamBounded):
    """
    The options of a free element for CMA-ES: its initial value, its
    bounds, and its initial search width.

    Parameters
    ----------
    value : float
        The initial value (the centre of the first search).
    minimum, maximum : float, optional
        The bounds (config: min, max; by default none).
    sigma : float, optional
        The standard deviation of the first search (by default, a quarter
        of the width of the bounds; required without both bounds).

    Raises
    ------
    ConfigError
        If the bounds are empty, the value is not finite or not in them,
        or sigma is not positive or is missing without both bounds.
    """

    def dump(self) -> dict[str, Any]:
        return super().dump() | dict(sigma=self._sigma)

    def __init__(
            self,
            value: float,
            minimum: float = -math.inf,
            maximum: float = math.inf,
            sigma: float | None = None
    ):
        super().__init__(value, minimum, maximum)
        width = self.maximum() - self.minimum()
        if sigma is None and not math.isfinite(width):
            raise ConfigError(
                "option 'sigma' is required without both 'min' and 'max'")
        if sigma is not None and not (math.isfinite(sigma) and sigma > 0):
            raise ConfigError(
                f"option 'sigma' must be finite and positive; it is {sigma}")
        self._sigma = sigma if sigma is not None else width / 4

    def sigma(self) -> float:
        """Return the standard deviation of the first search."""
        return self._sigma


class FitterCMAES(Fitter):
    """
    The covariance matrix adaptation evolution strategy (Hansen 2016),
    with pycma: a global, derivative-free optimizer of the chi-square,
    with optional restarts of growing populations (IPOP or BIPOP).

    The free elements take a value, optional min and max, and sigma (see
    FitParamCMAES). CMA-ES works on the values scaled by their sigma, so
    that each starts with a search of standard deviation 1. Each
    generation is evaluated as a batch; values that the parameters
    cannot take (e.g. outside their valid range) get an infinite
    chi-square.

    Parameters
    ----------
    popsize : int, optional
        The population of the first run (by default, pycma's:
        4 + 3 ln(n)).
    restarts : int, optional
        The number of restarts with larger populations (IPOP; with
        bipop, runs with small populations come on top).
    bipop : bool, optional
        Whether the restarts alternate large and small populations.
    maxfevals : int, optional
        The most model evaluations of all runs (by default, pycma's).
    tolfun : float, optional
        The change of the chi-square below which a run stops.
    tolx : float, optional
        The step (in units of sigma) below which a run stops.
    verbose : bool, optional
        Whether pycma prints its progress.
    seed : int, optional
        The seed of the random numbers.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution.

    Raises
    ------
    ConfigError
        If a number is not positive (restarts: negative).
    """

    @staticmethod
    def type() -> str:
        return 'cmaes'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamCMAES

    def libraries(self) -> tuple[str, ...]:
        return ('cma',)

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), popsize=self._popsize,
            restarts=self._restarts, bipop=self._bipop,
            maxfevals=self._maxfevals, tolfun=self._tolfun,
            tolx=self._tolx, verbose=self._verbose) | self.dump_common()

    def __init__(
            self,
            popsize: int | None = None,
            restarts: int = 0,
            bipop: bool = False,
            maxfevals: int | None = None,
            tolfun: float = 1e-6,
            tolx: float = 1e-6,
            verbose: bool = False,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(
                popsize=popsize, maxfevals=maxfevals, tolfun=tolfun,
                tolx=tolx).items():
            if value is not None and not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        if restarts < 0:
            raise ConfigError(
                f"option 'restarts' cannot be negative; it is {restarts}")
        self._popsize = popsize
        self._restarts = restarts
        self._bipop = bipop
        self._maxfevals = maxfevals
        self._tolfun = tolfun
        self._tolx = tolx
        self._verbose = verbose

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        params = typing.cast(list[FitParamCMAES], fit_params)
        x0 = np.array([p.value() for p in params])
        scale = np.array([p.sigma() for p in params])
        lower = [(p.minimum() - p.value()) / p.sigma() for p in params]
        upper = [(p.maximum() - p.value()) / p.sigma() for p in params]

        def chi2(zs: list[np.ndarray]) -> list[float]:
            return problem.chi2_batch(x0 + scale * np.array(zs)).tolist()

        options: dict[str, Any] = dict(
            bounds=[
                [z if math.isfinite(z) else None for z in lower],
                [z if math.isfinite(z) else None for z in upper]],
            seed=int(rng.integers(1, 2 ** 31)), tolfun=self._tolfun,
            tolx=self._tolx, verbose=1 if self._verbose else -9,
            verb_log=0, verb_disp=100 if self._verbose else 0)
        if self._popsize is not None:
            options['popsize'] = self._popsize
        if self._maxfevals is not None:
            options['maxfevals'] = self._maxfevals
        _, es = cma.fmin2(
            None, np.zeros(len(params)), 1.0, options,
            restarts=self._restarts, bipop=self._bipop,
            parallel_objective=chi2)
        # The best point of all runs
        best = problem.best()
        if best is None:
            raise RuntimeError("CMA-ES found no point with a finite chi2")
        stop = {str(k): v for k, v in es.stop().items()}
        return FitRun(
            solutions=[best[0]],
            success='maxfevals' not in stop,
            message=f"stopped by {stop}",
            diagnostics=dict(
                iterations=int(es.result.iterations), stop=stop),
            raw=es)
