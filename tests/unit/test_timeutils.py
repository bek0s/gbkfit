"""
Tests for the timers.
"""

import numpy as np
import pytest

from gbkfit.utils import timeutils


def test_timers_keep_the_statistics_of_the_times():
    times = [1.0, 2.5, 3.0, 10.0, 0.004]
    timers = timeutils.Timers()
    for time_ in times:
        timers.record('step', time_)
    stats = timers.stats()['step']
    assert stats.count == 5
    assert stats.mean == pytest.approx(np.mean(times), rel=1e-3)
    assert stats.stddev == pytest.approx(np.std(times, ddof=1), rel=1e-3)
    assert (stats.min, stats.max) == (0.004, 10.0)
    timers.clear()
    assert timers.stats() == {}


def test_timers_measure_blocks_without_exceptions():
    timers = timeutils.Timers()
    with timers.measure('step'):
        pass
    with pytest.raises(ValueError):
        with timers.measure('step'):
            raise ValueError
    stats = timers.stats()['step']
    assert stats.count == 1
    assert stats.min > 0
