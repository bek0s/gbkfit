
import collections
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.stats as stats


__all__ = [
    'SimpleTimer',
    'TimerStats',
    'clear_time_stats',
    'get_time_stats',
]


_log = logging.getLogger(__name__)


@dataclass
class TimerStats:
    """Statistics for timing measurements."""
    unit: str
    min: float
    max: float
    mean: float
    median: float
    stddev: float
    mad: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the TimerStats to a dictionary."""
        return asdict(self)  # type: ignore  # Silence PyCharm bug


_times: dict[str, list[int]] = collections.defaultdict(list)


class SimpleTimer:
    """
    A simple timer class for measuring execution time.
    """

    def __init__(self, name: str):
        self._name = name
        self._start_time = None

    def start(self):
        """Start the timer."""
        if self._start_time is not None:
            raise RuntimeError(
                f"SimpleTimer is running; use .stop() to stop it")
        self._start_time = time.perf_counter_ns()
        return self

    def stop(self):
        """Stop the timer and record the elapsed time."""
        if self._start_time is None:
            raise RuntimeError(
                "SimpleTimer is not running; use .start() to start it")
        elapsed = time.perf_counter_ns() - self._start_time
        _times[self._name].append(elapsed)
        self._start_time = None
        return elapsed / 1_000_000

    def __enter__(self) -> 'SimpleTimer':
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def clear_time_stats() -> None:
    """
    Clear all accumulated timing statistics.
    """
    _times.clear()


def get_time_stats(
        discard: int = 1,
        num_samples_recommended: int = 20
) -> dict[str, TimerStats]:
    """
    Get statistics for all recorded timings.
    """
    stats_ = dict()
    for key, values in _times.items():
        if not values:
            continue
        samples = values[discard:] if len(values) > discard else values
        if len(samples) < num_samples_recommended:
            _log.warning(
                f"after discarding the first {discard} samples, "
                f"only {len(samples)} samples are available for '{key}'; "
                f"recommended:{num_samples_recommended}+ measurements")
        values_ms = np.array(samples) / 1_000_000
        stats_[key] = TimerStats(
            unit='millisecond',
            min=float(np.round(np.min(values_ms), 2)),
            max=float(np.round(np.max(values_ms), 2)),
            mean=float(np.round(np.mean(values_ms), 2)),
            median=float(np.round(np.median(values_ms), 2)),
            stddev=float(np.round(np.std(values_ms), 2)),
            mad=float(np.round(stats.median_abs_deviation(values_ms), 2)),
            sample_count=len(samples))
    return stats_
