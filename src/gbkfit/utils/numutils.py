
import gbkfit.utils.iterutils

from collections.abc import Sequence
from numbers import Number
from typing import TypeVar


_NumberT = TypeVar('_NumberT', bound=Number)


def cumsum(
        x: Sequence[_NumberT],
        origin: int = 0,
        out: list[_NumberT] | None = None
) -> list[Number]:
    """
    Compute cumulative sum of a sequence starting from a specified
    origin point.
    """
    n = len(x)
    if n == 0:
        raise RuntimeError("input sequence cannot be empty")
    if not (-n < origin < n):
        raise RuntimeError(f"origin {origin} is out of bounds for sequence of length {n}")
    origin = gbkfit.utils.iterutils.unwrap_sequence_index(origin, n)
    if out is not None and len(out) != n:
        raise RuntimeError("output array must have same length as input")
    result = [0] * n if out is None else out
    result[origin] = x[origin]
    for i in range(origin + 1, n):
        result[i] = result[i - 1] + x[i]
    for i in range(origin - 1, -1, -1):
        result[i] = result[i + 1] + x[i]
    return result
