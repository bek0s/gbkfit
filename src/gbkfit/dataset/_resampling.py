"""
The resampling of the residuals of data on grids (e.g. of pixels, or of
positions along a slit and channels), for bootstraps (see
Dataset.resample_blocks).
"""

import numbers
from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.utils.parseutils import ConfigError


def block_size(
        size: Sequence[float], axes: Sequence[str]
) -> tuple[int, ...]:
    """
    Return the size of blocks of the options, along the axes of the data
    (FITS order, e.g. x, y and the channels), in numpy order. Raise
    ConfigError unless it is a positive integer for each axis.
    """
    size = tuple(size)
    if len(size) != len(axes) or not all(
            isinstance(n, numbers.Integral) and not isinstance(n, bool)
            and n >= 1 for n in size):
        names = axes[0] if len(axes) == 1 \
            else f"{', '.join(axes[:-1])} and {axes[-1]}"
        raise ConfigError(
            f"the size of the blocks of these data is a positive integer for "
            f"each of {names}; it is {list(size)}")
    return size[::-1]


def _tiles(
        shape: tuple[int, ...], size: tuple[int, ...]
) -> tuple[np.ndarray, int, tuple[np.ndarray, ...]]:
    """
    Cut a grid of a shape into blocks of a size (numpy order): return the
    block of each pixel (its index), the number of blocks, and the
    position of each pixel in its block along each axis (as arrays that
    broadcast to the shape).
    """
    pixels = np.ix_(*(np.arange(n) for n in shape))
    counts = [-(-n // b) for n, b in zip(shape, size)]
    blocks = np.ravel_multi_index(
        [p // b for p, b in zip(pixels, size)], counts)
    positions = tuple(p % b for p, b in zip(pixels, size))
    return blocks, int(np.prod(counts)), positions


def _window_sums(x: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    """
    Return the sums of an array over the blocks of a size (numpy order),
    at each first pixel at which a block fits in it.
    """
    for axis, b in enumerate(size):
        n = x.shape[axis]
        sums = np.cumsum(x, axis=axis)
        sums = np.concatenate(
            [np.zeros_like(sums.take([0], axis)), sums], axis)
        x = sums.take(np.arange(b, n + 1), axis) \
            - sums.take(np.arange(n + 1 - b), axis)
    return x


def resample_blocks(
        residuals: Mapping[str, np.ndarray],
        size: Sequence[float],
        axes: Sequence[str],
        rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """
    Return residuals on a grid redrawn in blocks of a size along its axes
    (see block_size and Dataset.resample_blocks).
    """
    shape = next(iter(residuals.values())).shape
    size = block_size(size, axes)
    valid = np.logical_and.reduce(
        [np.isfinite(r) for r in residuals.values()]).astype(np.int64)
    origins = np.argwhere(_window_sums(valid, size) == np.prod(size))
    if not len(origins):
        raise ConfigError(
            f"no block of the size {list(size[::-1])} has residuals in all of "
            f"its pixels; make the blocks smaller")
    blocks, nblocks, positions = _tiles(shape, size)
    drawn = origins[rng.integers(len(origins), size=nblocks)][blocks]
    source = tuple(
        drawn[..., axis] + positions[axis] for axis in range(len(shape)))
    return {
        key: np.where(np.isfinite(r), r[source], np.nan)
        for key, r in residuals.items()}
