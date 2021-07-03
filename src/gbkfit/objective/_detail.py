
import numpy as np
import scipy.stats as stats


def time_stats(times):
    tstats = dict()
    for key, val in times.items():
        if val:
            tstats[key] = dict(
                min=np.round(np.min(val), 2),
                max=np.round(np.max(val), 2),
                mean=np.round(np.mean(val), 2),
                median=np.round(np.median(val), 2),
                stddev=np.round(np.std(val), 2),
                mad=np.round(stats.median_absolute_deviation(val), 2))
    return tstats
