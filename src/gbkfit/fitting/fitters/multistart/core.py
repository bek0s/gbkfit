"""
Multi-start: a local fitter run from many starting points, for problems
with several minima.
"""

import math
import typing
from typing import Annotated, Any

import numpy as np
import scipy.stats
from annotated_types import Gt

from gbkfit.fitting.base import FitRun, Fitter, fitter_parser
from gbkfit.fitting.fitparams import (
    FitParam, FitParamBounded, FitParamClass)
from gbkfit.fitting.problem import FitProblem
from gbkfit.params import EvaluationError
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError, check_options


__all__ = [
    'FitterMultiStart'
]


class FitterMultiStart(Fitter):
    """
    An optimizer (e.g. least squares, CMA-ES) run from many starting
    points: the given values, and points spread over the bounds (a
    scrambled Sobol sequence). The distinct minima it finds are the
    solutions, best first, each with its own uncertainty.

    Two minima are the same if they differ by less than 'tolerance'
    times the width of the bounds in every free element.

    The free elements take the options of the optimizer (e.g. sigma for
    CMA-ES), of which min and max are required: the starts are drawn
    between them. Each start runs the optimizer with these options, at
    another initial value (see FitParamBounded.with_value).

    Parameters
    ----------
    local : Fitter
        The optimizer: a fitter whose free elements take an initial value
        and bounds (see FitParamBounded; samplers do not). Its seed and
        uncertainty options are not used (those of multistart are); its
        merit is that of multistart (see Fitter), by which it ranks the
        minima.
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
        How to work out the uncertainty of the solutions (by default, of
        merit 'aicp': none).

    Raises
    ------
    ConfigError
        If the local fitter is not an optimizer, a number is not
        positive, or the merit and the uncertainty do not go together
        (see Fitter).
    """

    @staticmethod
    def type() -> str:
        return 'multistart'

    def fit_param_class(self) -> FitParamClass:
        # (the options of the local fitter)
        return self._local.fit_param_class()

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
            nstarts: Annotated[int, Gt(0)] = 16,
            nsolutions: Annotated[int, Gt(0)] = 5,
            tolerance: Annotated[float, Gt(0)] = 1e-3,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        check_options(FitterMultiStart.__init__, locals())
        super().__init__(seed, uncertainty, local.merit())
        if not issubclass(local.fit_param_class(), FitParamBounded):
            raise ConfigError(
                f"the local fitter of multistart must be an optimizer, which "
                f"starts from an initial value of each free element; "
                f"'{local.type()}' is not")
        self._local = local
        self._nstarts = nstarts
        self._nsolutions = nsolutions
        self._tolerance = tolerance

    def local(self) -> Fitter:
        """Return the local fitter."""
        return self._local

    def _tuning(self) -> dict[str, Any]:
        # (that of the local fitter)
        return self._local._tuning()

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
        for i, start in enumerate(starts):
            problem.set_stage(f'start {i + 1} of {len(starts)}')
            start_params = [p.with_value(v) for v, p in zip(start, params)]
            nevals = problem.nevals()
            try:
                run = self._local._run(problem, start_params, rng)
            except (EvaluationError, ValueError) as e:
                starts_info.append(dict(
                    error=str(e), nevals=problem.nevals() - nevals))
                continue
            cost = problem.cost(run.solutions[0])
            runs.append(run)
            starts_info.append(dict(
                cost=cost, success=run.success,
                nevals=problem.nevals() - nevals))
        if not runs:
            raise RuntimeError(
                f"every start of multistart failed; the first: "
                f"{starts_info[0]['error']}")
        # The minima, best first, and the distinct ones among them
        costs = [info['cost'] for info in starts_info if 'cost' in info]
        solutions: list[np.ndarray] = []
        for i in np.argsort(costs):
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
