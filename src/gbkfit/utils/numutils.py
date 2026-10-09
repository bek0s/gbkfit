"""
Helpers for numbers and arrays of numbers.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np


__all__ = [
    'all_negative',
    'all_positive',
    'cumsum_from',
    'nativify'
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


def all_positive(x: Iterable[Any], include_zero: bool = False) -> bool:
    """
    Check whether all the items of an iterable are positive.

    Parameters
    ----------
    x : Iterable
        An iterable of numbers.
    include_zero : bool
        Whether 0 counts as positive (i.e. check that none is negative).

    Returns
    -------
    bool
        Whether all the items are positive.
    """
    return all(i >= 0 if include_zero else i > 0 for i in x)


def all_negative(x: Iterable[Any], include_zero: bool = False) -> bool:
    """
    Check whether all the items of an iterable are negative.

    Parameters
    ----------
    x : Iterable
        An iterable of numbers.
    include_zero : bool
        Whether 0 counts as negative (i.e. check that none is positive).

    Returns
    -------
    bool
        Whether all the items are negative.
    """
    return all(i <= 0 if include_zero else i < 0 for i in x)


def nativify(x: Any) -> Any:
    """
    Return a nested structure with numpy values as Python values.

    Parameters
    ----------
    x : Any
        A list, tuple or dict, possibly nested, or a value.

    Returns
    -------
    Any
        The structure, of lists and dicts, with numpy arrays as lists and
        numpy scalars (e.g. numbers, bools, strings) as their Python
        values, as JSON and YAML can write them.
    """
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [nativify(item) for item in x]
    if isinstance(x, dict):
        return {key: nativify(value) for key, value in x.items()}
    return x
