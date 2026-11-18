"""
The fixed point x = g(x) of a map of arrays (see fixed_point), by Anderson
acceleration: e.g. the layers of a galaxy that its mass follows while its
field sets them (see GalaxyStage).

Anderson acceleration (type II, Walker & Ni 2011) takes as the next point
the map of the combination of the last points whose residuals g(x) - x
cancel best (least squares): a secant method in many dimensions, which
needs no derivatives of the map. Plain repetition of the map (Picard)
converges only where the map contracts, and slowly where it barely does;
Anderson converges in fewer passes there (the layers of a galaxy: 8-17
passes to 1e-12 for 11-31 of Picard, measured 2026-11-18), and where the
map does not contract but has a fixed point (e.g. the local term of
Palatini f(R) gravity, where Picard oscillates and diverges: 16-38
passes). Its change is not monotonic (it can grow for a pass or two):
dropping the passes before where it grows, and taking half a step,
slowed it down more than 3 times on the layers of a galaxy, and stopped
it where it settles (measured 2026-11-18).
"""

import dataclasses
from collections.abc import Callable

import numpy as np


__all__ = [
    'FixedPoint',
    'fixed_point'
]


# The last passes whose points and residuals the next point combines
_DEPTH = 5

# The change, over the smallest so far, beyond which there is no fixed
# point near (see fixed_point): the passes that settle the layers of a
# galaxy rise to at most 2.6 times it (measured 2026-11-18)
_DIVERGED = 1e3


@dataclasses.dataclass(frozen=True)
class FixedPoint:
    """
    The result of a search for a fixed point (see fixed_point): the last
    point x, the number of passes (evaluations of the map), the change of
    the last pass (see fixed_point), whether it settled (the change is
    below the tolerance), and if not, whether the change grew beyond
    _DIVERGED times its smallest (no fixed point near).
    """
    x: np.ndarray
    passes: int
    change: float
    settled: bool
    growing: bool


def fixed_point(
        g: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        change: Callable[[np.ndarray, np.ndarray], float],
        tolerance: float,
        passes: int
) -> FixedPoint:
    """
    Return the fixed point x = g(x) of a map of flat arrays, from the
    given start, by Anderson acceleration (see the module docstring) on
    the last _DEPTH passes: the first point whose change, of it to its map
    (change(x, g(x))), is below the tolerance, within the given number of
    passes. It stops where the change grows beyond _DIVERGED times the
    smallest so far: there is no fixed point near. The points are those
    of the same start every time, so that the fixed point is a function
    of the map alone.
    """
    points: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    smallest = np.inf
    for done in range(1, passes + 1):
        mapped = g(x)
        size = change(x, mapped)
        if size < tolerance:
            return FixedPoint(x, done, size, True, False)
        smallest = min(smallest, size)
        if size > _DIVERGED * smallest:
            return FixedPoint(x, done, size, False, True)
        residual = mapped - x
        points = [*points, x][-(_DEPTH + 1):]
        residuals = [*residuals, residual][-(_DEPTH + 1):]
        x = mapped
        if len(points) > 1:
            dx = np.diff(points, axis=0).T
            dr = np.diff(residuals, axis=0).T
            gamma = np.linalg.lstsq(dr, residual, rcond=None)[0]
            x = mapped - (dx + dr) @ gamma
    return FixedPoint(x, passes, size, False, False)
