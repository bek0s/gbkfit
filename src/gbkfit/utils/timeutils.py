
import collections
import logging
import time

import numpy as np
import scipy.stats as stats


__all__ = [
    'SimpleTimer',
    'clear_time_stats',
    'get_time_stats',
]


_log = logging.getLogger(__name__)


_times = collections.defaultdict(list)


def clear_time_stats():
    _times.clear()


def get_time_stats(discard=1, num_samples_recommended=20):
    global _times
    times = _times
    stats_ = dict()
    for key, val in times.items():
        if len(val) == 0:
            continue
        if len(val) > discard:
            val = val[discard:]
        if len(val) < num_samples_recommended:
            _log.warning(
                f"after discarding {discard} samples, "
                f"only {len(val) - discard} samples are available for '{key}'; "
                f"for more accurate statistics, "
                f"{num_samples_recommended}+ measurements are recommended")
        val = np.array(val) / 1_000_000
        stats_[key] = dict(
            min=np.round(np.min(val), 2),
            max=np.round(np.max(val), 2),
            mean=np.round(np.mean(val), 2),
            median=np.round(np.median(val), 2),
            stddev=np.round(np.std(val), 2),
            mad=np.round(stats.median_absolute_deviation(val), 2))  # noqa
    return stats_


class SimpleTimer:

    def __init__(self, name):
        self._name = name
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise RuntimeError(
                "SimpleTimer is running; use .stop() to stop it")
        self._start_time = time.perf_counter_ns()

    def stop(self):
        global _times
        if self._start_time is None:
            raise RuntimeError(
                "SimpleTimer is not running; use .start() to start it")
        elapsed = time.perf_counter_ns() - self._start_time
        _times[self._name].append(elapsed)
