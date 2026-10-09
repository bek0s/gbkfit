"""
Timers of the steps of evaluations (e.g. of a model), whose statistics
tell users how long the steps take.
"""

import contextlib
import dataclasses
import math
import time
from collections.abc import Generator
from typing import Any


__all__ = [
    'TimerStats',
    'Timers'
]


@dataclasses.dataclass(frozen=True)
class TimerStats:
    """
    The statistics of the times of a step, in milliseconds, to four
    significant digits.

    Attributes
    ----------
    unit : str
        The unit of the times ('millisecond').
    count : int
        The number of times.
    mean, stddev : float
        The mean of the times and their (sample) standard deviation.
    min, max : float
        The shortest and the longest time.
    """
    unit: str
    count: int
    mean: float
    stddev: float
    min: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        """Return the statistics as a dict."""
        return dataclasses.asdict(self)


class _RunningStats:
    """
    The count, mean, sum of squared deviations (Welford's algorithm),
    minimum and maximum of values, updated with each value.
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def add(self, value: float) -> None:
        """Update the statistics with a value."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min = min(self.min, value)
        self.max = max(self.max, value)


class Timers:
    """
    The times of named steps (e.g. 'model_eval'). Each step keeps running
    statistics of its times, so the memory they take does not grow with
    the number of times.
    """

    def __init__(self):
        self._steps: dict[str, _RunningStats] = {}

    @contextlib.contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """
        Measure the time of the block, as a time of the step of the given
        name. A block that raises an exception is not measured.
        """
        start = time.perf_counter_ns()
        yield
        self.record(name, (time.perf_counter_ns() - start) / 1e6)

    def record(self, name: str, milliseconds: float) -> None:
        """Add a time to the step of the given name."""
        self._steps.setdefault(name, _RunningStats()).add(milliseconds)

    def clear(self) -> None:
        """Forget the times of all the steps."""
        self._steps.clear()

    def stats(self) -> dict[str, TimerStats]:
        """Return the statistics of the times of each step, by name."""
        result = {}
        for name, steps in self._steps.items():
            variance = steps.m2 / (steps.count - 1) if steps.count > 1 else 0.0
            result[name] = TimerStats(
                unit='millisecond',
                count=steps.count,
                mean=_round(steps.mean),
                stddev=_round(math.sqrt(variance)),
                min=_round(steps.min),
                max=_round(steps.max))
        return result


def _round(value: float) -> float:
    """Return a value rounded to four significant digits."""
    return float(f'{value:.4g}')
