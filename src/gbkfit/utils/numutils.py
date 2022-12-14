
import gbkfit.utils.iterutils


def cumsum(x, origin=0, out=None):
    assert -len(x) < origin < len(x), "origin out of bounds"
    origin = gbkfit.utils.iterutils.unwrap_sequence_index(origin, len(x))
    result = [None] * len(x) if out is None else out
    prev = result[origin] = x[origin]
    for i in range(origin + 1, len(x)):
        result[i] = prev + x[i]
        prev = result[i]
    prev = x[origin]
    for i in range(origin - 1, -1, -1):
        result[i] = prev + x[i]
        prev = result[i]
    return result
