"""
Gauss-Hermite series of line-of-sight velocity distributions (LOSVDs),
in the convention of van der Marel & Franx (1993):

    L(v) = gamma / sigma alpha(w) sum_m h_m H_m(w),  w = (v - V) / sigma,

with alpha(w) = exp(-w^2 / 2) / sqrt(2 pi), the Hermite polynomials H_m
orthonormal under the weight alpha^2 (up to 1 / (2 sqrt(pi))), h_0 = 1,
and h_1 = h_2 = 0 when V and sigma are those of the best fit. Then

    gamma h_m = 2 sqrt(pi) int L(v) alpha(w) H_m(w) dv,

which is linear in L: the form of the constraints of Schwarzschild models
(Rix et al. 1997).
"""

import dataclasses
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.special


__all__ = [
    'GaussHermite',
    'fit_gauss_hermite',
    'gauss_hermite_losvd',
    'hermite_polynomials',
    'hermite_weights',
    'mass_factor'
]


# The Gauss-Legendre nodes of the averages over velocity bins
_BIN_NODES = 6


@dataclasses.dataclass(frozen=True)
class GaussHermite:
    """
    A Gauss-Hermite series: its gamma, its mean V and dispersion sigma
    (the units of the velocities), and its coefficients h (h_3, h_4,
    ...).
    """
    gamma: float
    mean: float
    sigma: float
    h: tuple[float, ...]


def hermite_polynomials(w: npt.ArrayLike, order: int) -> np.ndarray:
    """
    Return the Hermite polynomials H_0 ... H_order of van der Marel &
    Franx (1993) at w.

    Parameters
    ----------
    w : array_like
        The points.
    order : int
        The highest order.

    Returns
    -------
    np.ndarray
        An array of shape (order + 1, *w.shape).
    """
    w = np.asarray(w, dtype=float)
    polynomials = np.empty((order + 1, *w.shape))
    polynomials[0] = 1
    if order > 0:
        polynomials[1] = np.sqrt(2) * w
    # H_{m+1} = (sqrt(2) w H_m - sqrt(m) H_{m-1}) / sqrt(m + 1)
    for m in range(1, order):
        polynomials[m + 1] = (
            np.sqrt(2) * w * polynomials[m]
            - np.sqrt(m) * polynomials[m - 1]) / np.sqrt(m + 1)
    return polynomials


def gauss_hermite_losvd(
        v: npt.ArrayLike,
        gamma: float,
        mean: float,
        sigma: float,
        h: Sequence[float]
) -> np.ndarray:
    """
    Return the LOSVD of a Gauss-Hermite series at the velocities v.

    Parameters
    ----------
    v : array_like
        The velocities.
    gamma, mean, sigma : float
        Its gamma, mean and dispersion.
    h : sequence of float
        Its coefficients h_3, h_4, ...

    Returns
    -------
    np.ndarray
        The LOSVD (the units of gamma over those of the velocities).
    """
    w = (np.asarray(v, dtype=float) - mean) / sigma
    polynomials = hermite_polynomials(w, 2 + len(h))
    series = 1 + sum(
        hm * polynomials[m] for m, hm in enumerate(h, start=3))
    return gamma / sigma * _alpha(w) * series


def mass_factor(h: Sequence[float]) -> float:
    """
    Return the integral of a Gauss-Hermite series of gamma = 1 (its mass
    is gamma times it): 1 + sum over even m >= 4 of h_m sqrt(m!) / m!!.

    Parameters
    ----------
    h : sequence of float
        Its coefficients h_3, h_4, ...
    """
    return 1 + sum(
        hm * np.sqrt(scipy.special.factorial(m))
        / scipy.special.factorial2(m)
        for m, hm in enumerate(h, start=3) if m % 2 == 0)


def hermite_weights(
        edges: npt.ArrayLike,
        mean: npt.ArrayLike,
        sigma: npt.ArrayLike,
        order: int
) -> np.ndarray:
    """
    Return the weights that turn velocity histograms into their sums
    gamma h_m, for the series of means and dispersions given: 2 sqrt(pi)
    times the averages of alpha(w) H_m(w) over each bin. The contents of
    a bin are taken as spread uniformly over it.

    Parameters
    ----------
    edges : array_like
        The edges of the velocity bins, increasing: an array of shape
        (bins + 1,).
    mean, sigma : array_like
        The means and dispersions of the series, broadcast together
        (e.g. one for each spatial bin).
    order : int
        The highest order m.

    Returns
    -------
    np.ndarray
        An array of shape (*shape, bins, order + 1), where shape is that
        of mean and sigma broadcast together: gamma h_m of a histogram is
        the sum of its contents times these weights over the bins.
    """
    edges = np.asarray(edges, dtype=float)
    mean, sigma = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(sigma, dtype=float))
    nodes, node_weights = np.polynomial.legendre.leggauss(_BIN_NODES)
    centres = (edges[1:] + edges[:-1]) / 2
    halves = (edges[1:] - edges[:-1]) / 2
    # The nodes in every bin (bins, nodes), then w for every series
    v = centres[:, None] + halves[:, None] * nodes
    w = (v - mean[..., None, None]) / sigma[..., None, None]
    integrand = _alpha(w) * hermite_polynomials(w, order)
    # (the average over a bin is the half-sum of the node weights)
    averages = integrand @ node_weights / 2
    return 2 * np.sqrt(np.pi) * np.moveaxis(averages, 0, -1)


def fit_gauss_hermite(
        v: npt.ArrayLike,
        losvd: npt.ArrayLike,
        order: int = 4
) -> GaussHermite:
    """
    Fit a Gauss-Hermite series to a LOSVD by least squares.

    Parameters
    ----------
    v : array_like
        The velocities, evenly spaced, wide enough to hold the LOSVD.
    losvd : array_like
        The LOSVD at the velocities.
    order : int, optional
        The highest order of the series, at least 2 (h_3 ... h_order are
        fitted).

    Returns
    -------
    GaussHermite
        The series of the best fit.
    """
    v = np.asarray(v, dtype=float)
    losvd = np.asarray(losvd, dtype=float)
    # The moments, for the first guess
    mass = losvd.sum() * (v[1] - v[0])
    mean = (losvd * v).sum() / losvd.sum()
    sigma = np.sqrt((losvd * (v - mean) ** 2).sum() / losvd.sum())

    def residuals(x: np.ndarray) -> np.ndarray:
        return gauss_hermite_losvd(v, x[0], x[1], x[2], x[3:]) - losvd

    start = np.concatenate(([mass, mean, sigma], np.zeros(order - 2)))
    lower = np.full_like(start, -np.inf)
    lower[2] = 0
    x = scipy.optimize.least_squares(
        residuals, start, bounds=(lower, np.inf),
        x_scale=np.where(start == 0, 1, np.abs(start)),
        xtol=1e-12, ftol=1e-12).x
    return GaussHermite(
        gamma=float(x[0]), mean=float(x[1]), sigma=float(x[2]),
        h=tuple(float(hm) for hm in x[3:]))


def _alpha(w: np.ndarray) -> np.ndarray:
    """Return the standard Gaussian at w."""
    return np.exp(-w ** 2 / 2) / np.sqrt(2 * np.pi)
