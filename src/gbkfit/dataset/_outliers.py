"""
The screening of data for outliers: values far from the median of their
neighbours (see Dataset.screen_outliers).
"""

import numbers
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.spatial

from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .base import Dataset


# The fewest neighbours with values that a value is judged by: one with
# fewer is kept
_MIN_NEIGHBOURS = 3

# The standard deviation of a Gaussian, in its median absolute deviations
_MAD_TO_STD = 1.4826

# The most values of the windows of a grid copied at once
_CHUNK = 2 ** 22


def check_nsigma(nsigma: float) -> float:
    """Return nsigma; raise ConfigError unless it is a positive number."""
    if not (isinstance(nsigma, numbers.Real) and not isinstance(nsigma, bool)
            and nsigma > 0):
        raise ConfigError(
            f"nsigma of the outliers is a positive number; it is {nsigma!r}")
    return float(nsigma)


def neighbour_stats(
        neighbours: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the median of the neighbours (n, k; NaN for those without
    values) of n values, their robust scatter (1.4826 times their median
    absolute deviation from their median), and their number.
    """
    with warnings.catch_warnings():
        # (the values with no neighbours, which are kept)
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(neighbours, axis=1)
        scatter = _MAD_TO_STD * np.nanmedian(
            np.abs(neighbours - median[:, None]), axis=1)
    return median, scatter, np.count_nonzero(np.isfinite(neighbours), axis=1)


def outliers(
        values: np.ndarray,
        median: np.ndarray,
        scatter: np.ndarray,
        count: np.ndarray,
        nsigma: float
) -> np.ndarray:
    """
    Return where values are outliers of their neighbours, of their
    median, robust scatter and number (see neighbour_stats): further from
    the median than nsigma times the larger of their scatter and that of
    all the values, 1.4826 times the median of |x - m|. (The neighbours
    of a value are few, and their scatter, by itself, too often small by
    chance.) A value with fewer than _MIN_NEIGHBOURS neighbours is not
    one.
    """
    judged = (count >= _MIN_NEIGHBOURS) & np.isfinite(values)
    if not judged.any():
        return judged
    deviation = np.abs(values - median)
    overall = _MAD_TO_STD * np.median(deviation[judged])
    return judged & (deviation > nsigma * np.maximum(scatter, overall))


def window_size(
        size: Sequence[float], axes: Sequence[str]
) -> tuple[int, ...]:
    """
    Return the size of windows of the options, along the axes of the data
    (FITS order, e.g. x, y and the channels), in numpy order. Raise
    ConfigError unless it is an odd positive integer for each axis (so
    that a window has a centre).
    """
    size = tuple(size)
    if len(size) != len(axes) or not all(
            isinstance(n, numbers.Integral) and not isinstance(n, bool)
            and n >= 1 and n % 2 == 1 for n in size):
        names = axes[0] if len(axes) == 1 \
            else f"{', '.join(axes[:-1])} and {axes[-1]}"
        raise ConfigError(
            f"the size of the windows of these data is an odd positive "
            f"integer for each of {names}; it is {list(size)}")
    return size[::-1]


def grid_outliers(
        values: np.ndarray, size: tuple[int, ...], nsigma: float
) -> np.ndarray:
    """
    Return where the values of a grid (NaN where it has none) are
    outliers of their neighbours: the other values in a window of a size
    (numpy order) centred on them (see outliers).
    """
    padded = np.pad(
        values, [(n // 2, n // 2) for n in size], constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    nwindow = int(np.prod(size))
    nslab = int(np.prod(values.shape[1:]))
    median, scatter = np.empty(values.size), np.empty(values.size)
    count = np.empty(values.size, dtype=int)
    # (in slabs along the first axis, to bound the copies of the windows)
    step = max(1, _CHUNK // (nwindow * nslab))
    for start in range(0, len(values), step):
        neighbours = np.delete(
            windows[start:start + step].reshape(-1, nwindow), nwindow // 2,
            axis=1)
        flat = slice(start * nslab, start * nslab + len(neighbours))
        median[flat], scatter[flat], count[flat] = neighbour_stats(neighbours)
    return outliers(
        values.ravel(), median, scatter, count, nsigma).reshape(values.shape)


def region_size(
        size: Sequence[float], nregions: int, channels: bool
) -> tuple[int, int]:
    """
    Return the number of nearest regions and the size of the window of
    channels (1 without channels) of the options. Raise ConfigError
    unless the first is a positive integer below the number of regions,
    and the second, with channels, an odd positive integer.
    """
    size = tuple(size)
    shape = ("the number of nearest regions, and an odd size of the window "
             "of channels") if channels else "the number of nearest regions"
    integers = all(
        isinstance(n, numbers.Integral) and not isinstance(n, bool)
        and n >= 1 for n in size)
    if len(size) != 1 + channels or not integers or size[0] >= nregions \
            or (channels and size[1] % 2 == 0):
        raise ConfigError(
            f"the size of the neighbourhoods of these data is {shape}, "
            f"positive integers (fewer regions than the {nregions}); it is "
            f"{list(size)}")
    return size[0], size[1] if channels else 1


def region_outliers(
        values: np.ndarray,
        centres: np.ndarray,
        nnearest: int,
        window: int,
        nsigma: float
) -> np.ndarray:
    """
    Return where the values of regions (nchannels, nregions) are outliers
    of their neighbours (see outliers): the values of the region and of
    its nearest regions, by their centres (nregions, 2), in a window of
    channels centred on the value, but itself.
    """
    nchannels, nregions = values.shape
    # Each region and its nearest regions, itself first
    _, nearest = scipy.spatial.cKDTree(centres).query(centres, nnearest + 1)
    regions = np.array([
        [i, *row[row != i][:nnearest]] for i, row in enumerate(nearest)])
    half = window // 2
    padded = np.pad(values, [(half, half), (0, 0)], constant_values=np.nan)
    nneighbourhood = window * (nnearest + 1)
    median, scatter = np.empty(values.size), np.empty(values.size)
    count = np.empty(values.size, dtype=int)
    # (in slabs of channels, to bound the copies of the neighbourhoods)
    step = max(1, _CHUNK // (nneighbourhood * nregions))
    for start in range(0, nchannels, step):
        channels = np.arange(start, min(start + step, nchannels))[:, None] \
            + np.arange(window)
        neighbours = padded[
            channels[:, None, :, None], regions[None, :, None, :]]
        neighbours = np.delete(
            neighbours.reshape(-1, nneighbourhood), half * (nnearest + 1),
            axis=1)
        flat = slice(start * nregions, start * nregions + len(neighbours))
        median[flat], scatter[flat], count[flat] = neighbour_stats(neighbours)
    return outliers(
        values.ravel(), median, scatter, count, nsigma).reshape(values.shape)


def screen_grid(
        dataset: 'Dataset',
        size: Sequence[float],
        axes: Sequence[str],
        nsigma: float
) -> 'Dataset':
    """
    Return a copy of a dataset of items on a grid with the outliers of
    each item masked, in windows of a size along its axes (see
    window_size and Dataset.screen_outliers).
    """
    window = window_size(size, axes)
    nsigma = check_nsigma(nsigma)
    return dataset.with_data({
        key: np.where(
            grid_outliers(item.data(), window, nsigma), np.nan, item.data())
        for key, item in dataset.items()})
