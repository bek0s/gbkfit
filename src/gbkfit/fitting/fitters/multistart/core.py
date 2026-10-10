"""
Multi-start: a local fitter run from many starting points, for problems
with several minima.
"""

import math
import typing
from typing import Any

import numpy as np
import scipy.stats

from gbkfit.fitting.base import FitRun, Fitter, fitter_parser
from gbkfit.fitting.fitparams import (
    FitParam, FitParamBounded, FitParamClass)
from gbkfit.fitting.problem import FitProblem
from gbkfit.params import ParamEvaluationError
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitterMultiStart'
]


class FitterMultiStart(Fitter):
    """
    A local fitter (e.g. least squares) run from many starting points:
    the given values, and points spread over the bounds (a scrambled
    Sobol sequence). The distinct minima it finds are the solutions,
    best first, each with its own uncertainty.

    Two minima are the same if they differ by less than 'tolerance'
    times the width of the bounds in every free element.

    The free elements take a value, and min and max (both required: the
    starts are drawn between them; see FitParamBounded).

    Parameters
    ----------
    local : Fitter
        The local fitter. Its free elements must take FitParamBounded
        options (e.g. lmfit.least_squares, scipy.least_squares).
    nstarts : int, optional
        The number of starts (the first is the given values).
    nsolutions : int, optional
        The most solutions to keep.
    tolerance : float, optional
        The distance below which two minima are the same, as a fraction
        of the width of the bounds.
    seed : int, optional
        The seed of the starts and the Laplace samples.
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the solutions.

    Raises
    ------
    ConfigError
        If the local fitter does not take FitParamBounded options, or a
        number is not positive.
    """

    @staticmethod
    def type() -> str:
        return 'multistart'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamBounded

    @classmethod
    def load(cls, info: dict[str, Any]) -> 'FitterMultiStart':
        info = dict(info)
        info['local'] = parseutils.load_option(
            fitter_parser.load, info, 'local', required=True)
        return typing.cast(FitterMultiStart, super().load(info))

    def libraries(self) -> tuple[str, ...]:
        return self._local.libraries()

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), local=fitter_parser.dump(self._local),
            nstarts=self._nstarts, nsolutions=self._nsolutions,
            tolerance=self._tolerance) | self.dump_common()

    def __init__(
            self,
            local: Fitter,
            nstarts: int = 16,
            nsolutions: int = 5,
            tolerance: float = 1e-3,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        if local.fit_param_class() is not FitParamBounded:
            raise ConfigError(
                f"the local fitter of multistart must take a value, min and "
                f"max for each free element; '{local.type()}' does not")
        for name, value in dict(
                nstarts=nstarts, nsolutions=nsolutions,
                tolerance=tolerance).items():
            if not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        self._local = local
        self._nstarts = nstarts
        self._nsolutions = nsolutions
        self._tolerance = tolerance

    def local(self) -> Fitter:
        """Return the local fitter."""
        return self._local

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        params = typing.cast(list[FitParamBounded], fit_params)
        lower = np.array([p.minimum() for p in params])
        upper = np.array([p.maximum() for p in params])
        if not np.all(np.isfinite(upper - lower)):
            raise ConfigError(
                "multistart needs min and max for every free element")
        starts = [np.array([p.value() for p in params])]
        if self._nstarts > 1:
            sobol = scipy.stats.qmc.Sobol(
                problem.ndim(), seed=int(rng.integers(2 ** 31)))
            # (Sobol points are balanced in powers of 2; drawing a few
            # more and keeping the first is fine)
            nsobol = 2 ** math.ceil(math.log2(self._nstarts - 1))
            points = sobol.random(nsobol)[:self._nstarts - 1]
            starts += list(lower + points * (upper - lower))
        # Each start: its run, or why it failed (e.g. a start where the
        # parameters cannot be evaluated), and its model evaluations
        runs: list[FitRun] = []
        starts_info: list[dict[str, Any]] = []
        for start in starts:
            start_params = [
                FitParamBounded(v, p.minimum(), p.maximum())
                for v, p in zip(start, params)]
            nevals = problem.nevals()
            try:
                run = self._local._run(problem, start_params, rng)
            except (ParamEvaluationError, ValueError) as e:
                starts_info.append(dict(
                    error=str(e), nevals=problem.nevals() - nevals))
                continue
            chi2 = problem.chi2(run.solutions[0])
            runs.append(run)
            starts_info.append(dict(
                chi2=chi2, success=run.success,
                nevals=problem.nevals() - nevals))
        if not runs:
            raise RuntimeError(
                f"every start of multistart failed; the first: "
                f"{starts_info[0]['error']}")
        # The minima, best first, and the distinct ones among them
        chi2s = [info['chi2'] for info in starts_info if 'chi2' in info]
        solutions: list[np.ndarray] = []
        for i in np.argsort(chi2s):
            x = runs[i].solutions[0]
            if not any(np.all(np.abs(x - s) <= self._tolerance
                              * (upper - lower)) for s in solutions):
                solutions.append(x)
        nfailed = len(starts) - len(runs)
        return FitRun(
            solutions=solutions[:self._nsolutions],
            success=any(run.success for run in runs),
            message=f"{len(solutions)} distinct minima from "
                    f"{len(starts)} starts ({nfailed} failed)",
            diagnostics=dict(nminima=len(solutions), starts=starts_info),
            raw=[run.raw for run in runs])
