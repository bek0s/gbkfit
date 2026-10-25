"""
The brute-force grid.
"""

import math
import typing
from collections.abc import Mapping
from typing import Any, Self

import numpy as np

from gbkfit.fitting.base import FitRun, Fitter
from gbkfit.fitting.control import FitStopped
from gbkfit.fitting.fitparams import FitParam, FitParamClass
from gbkfit.fitting.problem import FitProblem
from gbkfit.fitting.uncertainty import UncertaintyOptions
from gbkfit.params import FreeElement, Prior
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'FitParamGrid',
    'FitterGrid'
]


class FitParamGrid(FitParam):
    """
    The options of a free element for a grid: its range, the number of
    its values, evenly spaced from the minimum to the maximum (both
    included), and its prior, if any (its chi-square counts in the cost
    of each point).

    Parameters
    ----------
    minimum, maximum : float
        The range (of an element: its bounds, or the support of its
        prior).
    n : int
        The number of values, at least 2 (of an element: its own, or by
        default that of the fitter).
    prior : Prior, optional
        The prior.

    Raises
    ------
    ConfigError
        If the range is not finite or is empty, or n is not an integer of
        at least 2.
    """

    @classmethod
    def of(
            cls, element: FreeElement, defaults: Mapping[str, Any]
    ) -> Self:
        n = element.n() if element.n() is not None else defaults.get('n')
        if n is None:
            raise ConfigError(
                "a grid needs option 'n' (the number of values), of the "
                "element or of the fitter")
        return cls(element.minimum(), element.maximum(), n, element.prior())

    def __init__(
            self,
            minimum: float,
            maximum: float,
            n: int,
            prior: Prior | None = None
    ):
        if not (math.isfinite(minimum) and math.isfinite(maximum)
                and minimum < maximum):
            raise ConfigError(
                f"the range of a grid must be finite and not empty; it is "
                f"[{minimum}, {maximum}]")
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            raise ConfigError(
                f"option 'n' must be an integer of at least 2 (a fixed "
                f"value is not free); it is {n!r}")
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._n = n
        self._prior = prior

    def minimum(self) -> float:
        return self._minimum

    def maximum(self) -> float:
        return self._maximum

    def prior(self) -> Prior | None:
        return self._prior

    def values(self) -> np.ndarray:
        """Return its values on the grid."""
        return np.linspace(self._minimum, self._maximum, self._n)


class FitterGrid(Fitter):
    """
    A brute-force grid (e.g. 3D Barolo's SPACEPAR): the chi-square at
    every combination of the values of the free elements (see
    FitParamGrid), with that of the priors, if any (the cost of the
    optimizers; see FitProblem.cost), evaluated in batches. Its best
    solution is the point of the lowest. With two free elements and the
    others fixed, it maps how they trade off (e.g. the inclination and the
    rotation of a ring: free one element of each node-wise parameter).

    Its diagnostics 'grid' hold the names of the free elements, the
    values of each on the grid ('axes') and the chi-square of every point
    ('chi2', with that of the priors; of the shape
    of the grid, the first element along the first axis; inf where the
    parameters cannot be evaluated, NaN where a stopped grid did not
    evaluate it).

    Parameters
    ----------
    batch : int, optional
        The number of points evaluated together.
    max_points : int, optional
        The most points the grid may have.
    n : int, optional
        The number of values of the free elements without their own (see
        FitParamGrid).
    seed : int, optional
        The seed of the random numbers (of its uncertainty).
    uncertainty : UncertaintyOptions, optional
        How to work out the uncertainty of the best solution.

    Raises
    ------
    ConfigError
        If batch or max_points is not positive, n is not an integer of at
        least 2, or the grid has more than max_points points (when it
        runs).
    """

    @staticmethod
    def type() -> str:
        return 'grid'

    @staticmethod
    def fit_param_class() -> FitParamClass:
        return FitParamGrid

    def libraries(self) -> tuple[str, ...]:
        return ()

    def dump(self) -> dict[str, Any]:
        return dict(
            type=self.type(), batch=self._batch, max_points=self._max_points,
            n=self._n) | self.dump_common()

    def __init__(
            self,
            batch: int = 1000,
            max_points: int = 1_000_000,
            n: int | None = None,
            seed: int = 0,
            uncertainty: UncertaintyOptions | None = None
    ):
        super().__init__(seed, uncertainty)
        for name, value in dict(batch=batch, max_points=max_points).items():
            if not value > 0:
                raise ConfigError(
                    f"option '{name}' must be positive; it is {value}")
        if n is not None and (
                isinstance(n, bool) or not isinstance(n, int) or n < 2):
            raise ConfigError(
                f"option 'n' must be an integer of at least 2; it is {n!r}")
        self._batch = batch
        self._max_points = max_points
        self._n = n

    def _tuning(self) -> dict[str, Any]:
        return {} if self._n is None else dict(n=self._n)

    def _run(
            self,
            problem: FitProblem,
            fit_params: list[FitParam],
            rng: np.random.Generator
    ) -> FitRun:
        params = typing.cast(list[FitParamGrid], fit_params)
        axes = [param.values() for param in params]
        shape = tuple(len(axis) for axis in axes)
        npoints = math.prod(shape)
        if npoints > self._max_points:
            raise ConfigError(
                f"the grid has {npoints} points, more than max_points "
                f"({self._max_points}): give fewer values, or a larger "
                f"max_points")
        points = np.stack(
            np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, len(axes))
        # A grid asked to stop keeps the points it evaluated
        chi2 = np.full(npoints, np.nan)
        stopped = False
        for start in range(0, npoints, self._batch):
            end = min(start + self._batch, npoints)
            try:
                chi2[start:end] = problem.cost_batch(points[start:end])
            except FitStopped:
                stopped = True
                break
        best = problem.best()
        if best is None:
            raise RuntimeError("the grid found no point with a finite chi2")
        nevaluated = int(np.count_nonzero(~np.isnan(chi2)))
        message = f'stopped after {nevaluated} of {npoints} points' \
            if stopped else f'evaluated the {npoints} points of the grid'
        return FitRun(
            solutions=[best[0]],
            success=not stopped,
            message=message,
            diagnostics=dict(grid=dict(
                names=list(problem.names()),
                axes=[axis.tolist() for axis in axes],
                chi2=chi2.reshape(shape).tolist())))
