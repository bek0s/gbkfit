"""
Helpers for numbers and arrays of numbers.
"""

import numpy as np


__all__ = [
    'cumsum_from'
]


def cumsum_from(
        x: np.ndarray,
        origin: int,
        out: np.ndarray | None = None
) -> np.ndarray:
    """
    Return the cumulative sum of an array outward from an origin.

    The sum at the origin is the value there, and the sum at any other
    index is that of the values from the origin to it, on either side.

    Parameters
    ----------
    x : np.ndarray
        A one-dimensional, non-empty array.
    origin : int
        The index of the origin (negative indices count from the end).
    out : np.ndarray, optional
        The array to write the sums to, of the shape of x (it can be x
        itself). By default, a new array.

    Returns
    -------
    np.ndarray
        The sums.

    Raises
    ------
    ValueError
        If x is not one-dimensional and non-empty, or out does not have
        its shape.
    IndexError
        If the origin is outside x.
    """
    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(
            f"the array must be one-dimensional and non-empty; it has the "
            f"shape {x.shape}")
    if out is not None and out.shape != x.shape:
        raise ValueError(
            f"out must have the shape of the array {x.shape}; it has the "
            f"shape {out.shape}")
    if not -x.size <= origin < x.size:
        raise IndexError(
            f"the origin {origin} is outside an array of {x.size} values")
    origin %= x.size
    # (both sums are computed before out, which can be x, is written)
    right = np.cumsum(x[origin:])
    left = np.cumsum(x[origin::-1])[::-1]
    result = np.empty_like(x) if out is None else out
    result[origin:] = right
    result[:origin + 1] = left
    return result
