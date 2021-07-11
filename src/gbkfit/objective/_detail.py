
import logging

import numpy as np
import scipy.stats as stats


_log = logging.getLogger(__name__)


def time_stats(times):
    num_samples_recommended = 20
    stats_ = dict()
    for key, val in times.items():
        if len(val) == 0:
            continue
        # The first measurement is usually way off due to
        # lazy evaluation or other reasons (e.g., cold start)
        if len(val) > 1:
            val = val[1:]
        if len(val) < num_samples_recommended:
            _log.warning(
                f"only {len(val)} samples are available for '{key}'; "
                f"for more accurate statistics, "
                f"{num_samples_recommended}+ measurements are recommended")
        val = np.array(val) / 1_000_000
        stats_[key] = dict(
            min=np.round(np.min(val), 2),
            max=np.round(np.max(val), 2),
            mean=np.round(np.mean(val), 2),
            median=np.round(np.median(val), 2),
            stddev=np.round(np.std(val), 2),
            mad=np.round(stats.median_absolute_deviation(val), 2))
    return stats_
