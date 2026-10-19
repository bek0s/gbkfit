"""
The resampling of the residuals of data on grids of pixels, for
bootstraps (see Dataset.resample_blocks and Dataset.resample_flips).
"""

import numbers
from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError


# The symmetries of a disk that the flips draw: unchanged, rotated by 180
# degrees about its centre, and mirrored about its major and its minor
# axis; and whether each swaps its approaching and receding sides
_SWAPS = np.array([False, True, False, True])

# The spectra of a model cube with less flux than this part of the
# brightest one have no line centre (their flux is round-off)
_NO_FLUX = 1e-6


def _block_size(
        size: Sequence[int], shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Return the size of blocks of the options (FITS order: x, y and the
    channels) in numpy order. Raise ConfigError unless it is a positive
    integer for each axis of the data.
    """
    axes = 'x, y and the channels' if len(shape) == 3 else 'x and y'
    size = tuple(size)
    if len(size) != len(shape) or not all(
            isinstance(n, numbers.Integral) and not isinstance(n, bool)
            and n >= 1 for n in size):
        raise ConfigError(
            f"the size of the blocks of these data is a positive integer for "
            f"each of {axes}; it is {list(size)}")
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
        size: Sequence[int],
        rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """
    Return residuals on a grid redrawn in blocks (see
    Dataset.resample_blocks).
    """
    shape = next(iter(residuals.values())).shape
    size = _block_size(size, shape)
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


def _mirror_pixels(
        grid: gridutils.Grid, centre: Sequence[float], posa: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the pixel nearest the image of each pixel of a grid under each
    symmetry of a disk (see _SWAPS), its y and x indices, and whether it
    is on the grid: arrays of shape (4, ny, nx).
    """
    x, y = gridutils.sky_positions(grid)
    dx, dy = x - centre[0], y - centre[1]
    # (the major axis, at posa from north through east; see
    # transform_posa)
    ux, uy = -np.sin(np.radians(posa)), np.cos(np.radians(posa))
    along = dx * ux + dy * uy
    mx, my = 2 * along * ux - dx, 2 * along * uy - dy
    images_x = centre[0] + np.stack([dx, -dx, mx, -mx])
    images_y = centre[1] + np.stack([dy, -dy, my, -my])
    matrix, offset = gridutils.sky_to_pixel(grid)
    i = np.rint(
        matrix[0, 0] * images_x + matrix[0, 1] * images_y + offset[0])
    j = np.rint(
        matrix[1, 0] * images_x + matrix[1, 1] * images_y + offset[1])
    size_x, size_y = grid.size[:2]
    on = (i >= 0) & (i < size_x) & (j >= 0) & (j < size_y)
    return (np.where(on, j, 0).astype(np.intp),
            np.where(on, i, 0).astype(np.intp), on)


def _line_centres(spectra: np.ndarray) -> np.ndarray:
    """
    Return the centre of the line of each spectrum of a model cube (nz,
    ny, nx), in channels: the mean channel weighted by its flux (its
    positive values), or that of the whole cube for the spectra without
    flux (see _NO_FLUX).
    """
    flux = np.maximum(np.nan_to_num(spectra), 0)
    channels = np.arange(len(flux))[:, None, None]
    sums = flux.sum(axis=0)
    weighted = (flux * channels).sum(axis=0)
    if not sums.sum() > 0:
        raise ConfigError(
            "the model has no flux, whose lines the flips mirror the channels "
            "about")
    has_flux = sums > _NO_FLUX * sums.max()
    return np.where(
        has_flux, weighted / np.where(has_flux, sums, 1),
        weighted.sum() / sums.sum())


def resample_flips(
        grid: gridutils.Grid,
        residuals: Mapping[str, np.ndarray],
        size: Sequence[int],
        centre: Sequence[float],
        posa: float,
        rng: np.random.Generator,
        signs: Mapping[str, int] | None = None,
        spectra: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """
    Return residuals on a grid redrawn by flipping blocks of them (see
    Dataset.resample_flips): of maps, whose items change by their sign
    where the sides of the disk swap (1 by default); or of a cube (nz,
    ny, nx), whose channels mirror about the lines of its model spectra.
    """
    shape = next(iter(residuals.values())).shape
    blocks, nblocks, _ = _tiles(shape, _block_size(size, shape))
    symmetry = rng.integers(len(_SWAPS), size=nblocks)[blocks]
    swap = _SWAPS[symmetry]
    mirror_j, mirror_i, on = _mirror_pixels(grid, centre, posa)
    j, i = np.indices(shape[-2:])
    source_j, source_i = mirror_j[symmetry, j, i], mirror_i[symmetry, j, i]
    on = on[symmetry, j, i]
    if spectra is None:
        source = (source_j, source_i)
    else:
        # The offset from the centre of the line at a pixel is that from
        # the centre at its image, mirrored where the sides swap
        centres = _line_centres(spectra)
        offset = np.arange(shape[0])[:, None, None] - centres
        k = np.rint(centres[source_j, source_i]
                    + np.where(swap, -offset, offset))
        on &= (k >= 0) & (k < shape[0])
        source = (np.where(on, k, 0).astype(np.intp), source_j, source_i)
    resampled = {}
    for key, r in residuals.items():
        sign = 1 if signs is None else signs[key]
        moved = np.where(swap, sign, 1) * r[source]
        resampled[key] = np.where(
            on & np.isfinite(moved) & np.isfinite(r), moved, r)
    return resampled
