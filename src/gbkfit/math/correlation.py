"""
Correlations of noise on grids: the nearest correlation to a shape, and
the dense matrices of correlated values and their whitening.
"""

from collections.abc import Sequence

import numpy as np
import scipy.fft
import scipy.linalg


__all__ = [
    'MIN_CONDITIONAL_VARIANCE',
    'correlation_matrix',
    'periodic_correlation',
    'whitening_matrix'
]


# The smallest variance of a value given the values before it that a
# whitening takes (see whitening_matrix): below it, the whitening would
# multiply the rounding errors of the values over a thousand times
MIN_CONDITIONAL_VARIANCE = 1e-6


def periodic_correlation(
        shape: np.ndarray, size: Sequence[int]
) -> np.ndarray:
    """
    Return the nearest correlation to a shape on a periodic grid.

    The shape is placed on the grid, centred on a separation of 0, and its
    nearest correlation is its symmetric part without the negative parts
    of its power spectrum (a correlation has none), divided by its value
    at 0: e.g. a box takes the nearest; a Gaussian stays itself.

    Parameters
    ----------
    shape : np.ndarray
        The shape along its last len(size) axes (odd sizes, centred on a
        separation of 0), one for all its other axes or one for each.
    size : Sequence of int
        The size of the grid along each of those axes (at least that of
        the shape).

    Returns
    -------
    np.ndarray
        The correlation of each separation d at the index d mod size.
    """
    naxes = len(size)
    axes = tuple(range(-naxes, 0))
    extent = shape.shape[-naxes:]
    grid = np.zeros(shape.shape[:-naxes] + tuple(size))
    grid[(..., *(slice(0, n) for n in extent))] = shape
    grid = np.roll(grid, [-(n // 2) for n in extent], axis=axes)
    power = np.maximum(scipy.fft.rfftn(grid, axes=axes).real, 0)
    correlation = scipy.fft.irfftn(power, size, axes=axes)
    return correlation / correlation[(..., *(0,) * naxes, *(None,) * naxes)]


def correlation_matrix(
        correlation: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """
    Return the correlations of the values at points of a grid.

    Parameters
    ----------
    correlation : np.ndarray
        The correlation of each separation d on the grid at the index
        d mod its shape (see periodic_correlation), which is at least
        twice the largest separation along each axis, less 1, so that
        separations do not wrap around.
    points : np.ndarray
        The indices of the points, of shape (n, ndim).

    Returns
    -------
    np.ndarray
        The matrix of their correlations, of shape (n, n).
    """
    n = len(points)
    matrix = np.empty((n, n))
    # (in blocks of rows: the separations of all pairs at once would take
    # gigabytes for tens of thousands of points)
    rows = max(1, 2 ** 22 // max(n, 1))
    for start in range(0, n, rows):
        block = points[start:start + rows, None, :] - points[None, :, :]
        index = tuple(
            np.mod(block[..., axis], correlation.shape[axis])
            for axis in range(points.shape[1]))
        matrix[start:start + rows] = correlation[index]
    return matrix


def whitening_matrix(matrix: np.ndarray, white_floor: float) -> np.ndarray:
    """
    Return the whitening of values of a correlation matrix C taken with a
    white floor f: the inverse of the Cholesky factor L of (1 - f) C + f I
    (L L^T), which turns the values into values of no correlation. The
    square of each value of the diagonal of L is the variance of a value
    given the values before it, at least f.

    Parameters
    ----------
    matrix : np.ndarray
        The correlation matrix (1 on its diagonal).
    white_floor : float
        The share of the variance taken as white, in [0, 1).

    Returns
    -------
    np.ndarray
        L^-1, lower triangular.

    Raises
    ------
    np.linalg.LinAlgError
        If the matrix is not positive definite with that floor, or nearly
        not: a variance of a value given the values before it is below
        MIN_CONDITIONAL_VARIANCE.
    """
    floored = (1 - white_floor) * matrix
    floored[np.diag_indices_from(floored)] += white_floor
    factor = np.linalg.cholesky(floored)
    if np.min(np.diag(factor), initial=1) ** 2 < MIN_CONDITIONAL_VARIANCE:
        raise np.linalg.LinAlgError("the matrix is nearly singular")
    return scipy.linalg.solve_triangular(
        factor, np.eye(len(factor)), lower=True)
