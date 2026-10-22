"""
The structure function of a map (e.g. of the residuals of a fit, as eddy
measures them): how much its values differ with the separation of their
pixels.
"""

import dataclasses

import numpy as np
import numpy.typing as npt


__all__ = [
    'StructureFunction',
    'structure_function'
]


@dataclasses.dataclass(frozen=True)
class StructureFunction:
    """
    The structure function of a map (see structure_function).

    Attributes
    ----------
    separation : np.ndarray
        The mean separation (pixels) of the pairs of each bin.
    value : np.ndarray
        S2 of each bin: the mean of (v(x) - v(x + l))^2 of its pairs (NaN
        without any).
    pairs : np.ndarray
        The number of the pairs of each bin (each pair counted once).
    """
    separation: np.ndarray
    value: np.ndarray
    pairs: np.ndarray


def structure_function(
        values: npt.ArrayLike,
        edges: npt.ArrayLike | None = None
) -> StructureFunction:
    """
    Return the second-order structure function of a regular map, S2(l) =
    <(v(x) - v(x + l))^2> over all the pairs of its pixels with values
    whose separation |l| (pixels) falls in each bin: 2 sigma^2 at every
    separation for white noise of sigma, rising with the separation for
    correlated values. Exact, by correlations through FFTs: of the mask m,
    of m v and of m v^2.

    Parameters
    ----------
    values : ArrayLike
        The map (ny, nx; NaN where it has no value).
    edges : ArrayLike, optional
        The edges of the bins of the separations (pixels): by default, one
        pixel wide, about each whole separation from 1.

    Returns
    -------
    StructureFunction
        The structure function.

    Raises
    ------
    ValueError
        If the map is not 2D.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"the map must be 2D; it is {values.ndim}D")
    ny, nx = values.shape
    mask = np.isfinite(values).astype(float)
    v = np.where(mask > 0, values, 0.0)
    # (padded, so that the correlations do not wrap around)
    shape = (2 * ny, 2 * nx)

    def correlation(f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """sum_x f(x) g(x + l) at each shift l (centred)."""
        out = np.fft.irfft2(
            np.conj(np.fft.rfft2(f, shape)) * np.fft.rfft2(g, shape), shape)
        return np.fft.fftshift(out)
    pairs = correlation(mask, mask)
    squares = correlation(mask * v ** 2, mask) + correlation(mask, mask * v ** 2)
    sums = squares - 2 * correlation(v, v)
    dy, dx = np.indices(shape)
    separation = np.hypot(dx - nx, dy - ny)
    if edges is None:
        edges = np.arange(0.5, np.hypot(nx, ny) + 1)
    edges = np.asarray(edges, dtype=float)
    which = np.digitize(separation.ravel(), edges) - 1
    inside = (which >= 0) & (which < len(edges) - 1) & (pairs.ravel() > 0.5)
    count = np.bincount(
        which[inside], weights=np.round(pairs.ravel()[inside]),
        minlength=len(edges) - 1)
    total = np.bincount(
        which[inside], weights=sums.ravel()[inside], minlength=len(edges) - 1)
    mean = np.bincount(
        which[inside],
        weights=(separation.ravel() * np.round(pairs.ravel()))[inside],
        minlength=len(edges) - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return StructureFunction(
            separation=mean / count, value=total / count, pairs=count / 2)
