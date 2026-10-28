"""
The CMA-ES fitter (pycma).
"""

import math
import typing
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

import numpy as np
from annotated_types import Ge, Gt
from pydantic import FiniteFloat

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.fitparams import (
    FitParam, FitParamBounded, FitParamClass)
from gbkfit.fitting.problem import MERITS, FitProblem
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.params import FreeElement, Prior
from gbkfit.utils.parseutils import ConfigError, check_options


__all__ = [
    'FitParamCMAES',
    'FitterCMAES'
]


# The stops of a run of CMA-ES that mean it converged (the others: the
# budget of evaluations or iterations, or a run that went wrong)
_CONVERGED = frozenset({'tolfun', 'tolfunhist', 'tolx'})

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
    prior : Prior, optional
        The prior (see FitParamBounded).
    sigma : float, optional
        The standard deviation of the first search (by default, that of
        the fitter, or else a quarter of the width of the bounds; required
        without both bounds).

    Raises
    ------
    ConfigError
        If the bounds are empty, the value is not finite or not in them,
        or sigma is not positive or is missing without both bounds.
    """

    @classmethod
    def of(
            cls, element: FreeElement, defaults: Mapping[str, Any]
    ) -> Self:
        bounded = FitParamBounded.of(element, defaults)
        sigma = element.sigma()
        return cls(
            bounded.value(), bounded.minimum(), bounded.maximum(),
            bounded.prior(),
            defaults.get('sigma') if sigma is None else sigma)

    def __init__(
            self,
            value: float,
            minimum: float = -math.inf,
            maximum: float = math.inf,
            prior: Prior | None = None,
            sigma: float | None = None
    ):
        super().__init__(value, minimum, maximum, prior)
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
    with pycma: a global, derivative-free optimizer of the chi-square (or
    of the AICp; see merit), with optional restarts of growing
    populations (IPOP or BIPOP).

    The free elements take a value, optional min and max, and sigma (see
    FitParamCMAES; the option sigma of the fitter is the default). CMA-ES
    works on the values scaled by their sigma, so that each starts with a
    search of standard deviation 1. Each generation is evaluated as a
    batch; values that the parameters cannot take (e.g. outside their
    valid range) get an infinite chi-square.

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
        The change of the merit below which a run stops (the merit is in
        units of chi-square, whose changes mean the same at any size of
        the data: 1 moves a parameter by one standard deviation). By
        default 0.01, which keeps a run going while it matters (0.1 of a
        standard deviation), and lets it stop above the round-off of a
        merit summed from float32 models (~1e-4 for chi-squares of ~1e4),
        which a smaller one waits for in vain: the run then ends by
        stagnation, without success.
    tolx : float, optional
        The step (in units of sigma) below which a run stops.
    sigma : float, optional
        The sigma of the free elements without their own (see
        FitParamCMAES).
    verbose : bool, optional
        Whether pycma prints its progress.
    merit : {'chi2', 'aicp'}, optional
        What it minimises (see Fitter): the chi-square, or the AICp.
    seed : int, optional
        The seed of the random numbers.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution (by default,
        of merit 'aicp': none).

    Raises
    ------
    ConfigError
        If a number is not positive (restarts: negative), sigma is not
        finite, or the merit and the uncertainty do not go together (see
        Fitter).
    """

    requires = {'cma': "pip install 'gbkfit[cmaes]'"}

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
            tolx=self._tolx, sigma=self._sigma,
            verbose=self._verbose, merit=self.merit()) | self.dump_common()

    def __init__(
            self,
            popsize: Annotated[int, Gt(0)] | None = None,
            restarts: Annotated[int, Ge(0)] = 0,
            bipop: bool = False,
            maxfevals: Annotated[int, Gt(0)] | None = None,
            tolfun: Annotated[float, Gt(0)] = 0.01,
            tolx: Annotated[float, Gt(0)] = 1e-6,
            sigma: Annotated[FiniteFloat, Gt(0)] | None = None,
            verbose: bool = False,
            merit: Literal[MERITS] = 'chi2',
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        check_options(FitterCMAES.__init__, locals())
        super().__init__(seed, uncertainty, merit)
        self._popsize = popsize
        self._restarts = restarts
        self._bipop = bipop
        self._maxfevals = maxfevals
        self._tolfun = tolfun
        self._tolx = tolx
        self._sigma = sigma
        self._verbose = verbose

    def _tuning(self) -> dict[str, Any]:
        return {} if self._sigma is None else dict(sigma=self._sigma)

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        import cma
        params = typing.cast(list[FitParamCMAES], fit_params)
        x0 = np.array([p.value() for p in params])
        scale = np.array([p.sigma() for p in params])
        lower = [(p.minimum() - p.value()) / p.sigma() for p in params]
        upper = [(p.maximum() - p.value()) / p.sigma() for p in params]

        def cost(zs: list[np.ndarray]) -> list[float]:
            return problem.cost_batch(x0 + scale * np.array(zs)).tolist()

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
        zbest, es = cma.fmin2(
            None, np.zeros(len(params)), 1.0, options,
            restarts=self._restarts, bipop=self._bipop,
            parallel_objective=cost)
        if problem.best() is None:
            raise RuntimeError("CMA-ES found no point with a finite chi2")
        stop = {str(k): v for k, v in es.stop().items()}
        # The best point of its runs (not that of the problem: e.g. a
        # multistart runs it from many starts on one problem); a success
        # if the last run converged
        return FitRun(
            solutions=[x0 + scale * np.asarray(zbest)],
            success=bool(_CONVERGED & set(stop)),
            message=f"stopped by {stop}",
            diagnostics=dict(
                iterations=int(es.result.iterations), stop=stop),
            raw=es)
