"""
Helpers for numbers and arrays of numbers.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt


__all__ = [
    'MAD_TO_STD',
    'all_positive',
    'cumsum_from',
    'is_even',
    'is_odd',
    'nativify',
    'roundu_even',
    'roundu_multiple',
    'roundu_odd',
    'roundu_po2'
]


# The standard deviation of a Gaussian, in its median absolute deviations
# (1 over the quantile of the normal at 3/4, to 5 digits)
MAD_TO_STD = 1.4826


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
    # Both sums are computed before out is written, as out can be x
    right = np.cumsum(x[origin:])
    left = np.cumsum(x[origin::-1])[::-1]
    # The result has the type of the sums, which is wider than that of
    # small integers and bools
    result = np.empty(x.shape, right.dtype) if out is None else out
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


def is_odd(x: npt.ArrayLike) -> np.ndarray:
    """Return whether each value of x, truncated to an integer, is odd."""
    return np.trunc(np.asarray(x)).astype(int) % 2 != 0


def is_even(x: npt.ArrayLike) -> np.ndarray:
    """Return whether each value of x, truncated to an integer, is even."""
    return np.trunc(np.asarray(x)).astype(int) % 2 == 0


def roundu_even(x: npt.ArrayLike) -> np.ndarray:
    """Return each value of x rounded up to an even number."""
    return 2 * np.ceil(0.5 * np.asarray(x))


def roundu_odd(x: float) -> float:
    """Return x rounded up to an odd number."""
    x_even = roundu_even(x)
    return x_even + 1 if x_even - 1 < x else x_even - 1


def roundu_multiple(x: float, multiple: float) -> float:
    """
    Return x rounded up to a multiple of multiple.

    Raises
    ------
    ValueError
        If multiple is not positive.
    """
    if not multiple > 0:
        raise ValueError(f"multiple must be positive; it is {multiple}")
    remainder = np.mod(x, multiple)
    return x - remainder + multiple if remainder != 0 else x


def _roundu_po2(x: float) -> int:
    """Return the smallest power of 2 at or above x (at least 1)."""
    power = 1
    while power < x:
        power *= 2
    return power


def roundu_po2(x: npt.ArrayLike) -> np.ndarray:
    """
    Return the smallest power of 2 at or above each value of x (at least
    1).
    """
    return np.vectorize(_roundu_po2)(x)
