"""
The algebra of Gauss-Hermite series of distributions of velocities (e.g.
of the bins of pPXF): their normalisation, and their mean and dispersion.
"""

import dataclasses
import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt


__all__ = [
    'GaussHermiteMoments',
    'gauss_hermite_moments',
    'gauss_hermite_norm'
]


def _norm_term(order: int) -> float:
    """
    Return the share of the normalisation of a Gauss-Hermite series that
    its coefficient h_m of an even order m adds for each unit of it (see
    gauss_hermite_norm): sqrt(m!) / m!!.
    """
    double_factorial = math.prod(range(order, 0, -2))
    return math.sqrt(math.factorial(order)) / double_factorial


def gauss_hermite_norm(hermite: Mapping[int, npt.ArrayLike]) -> np.ndarray:
    """
    Return the normalisation N of Gauss-Hermite series of the given
    coefficients h_m (by order m, from 3): the integral of the series
    exp(-w^2 / 2) (1 + sum of h_m H_m(w)) over that of its Gaussian,
    with H_m of van der Marel & Franx (1993). It is 1 + sum over the
    even m of h_m sqrt(m!) / m!! (the odd H_m integrate to 0): of a
    series to h4, as pPXF fits, N = 1 + sqrt(6) / 4 h4.

    Parameters
    ----------
    hermite : Mapping of int to array_like
        The coefficients of each order, of the same shape (or
        broadcastable).

    Returns
    -------
    np.ndarray
        The normalisation, of the shape of the coefficients (1 without
        an even order).
    """
    norm = np.asarray(1.0)
    for order, value in hermite.items():
        if order % 2 == 0:
            norm = norm + _norm_term(order) * np.asarray(value, dtype=float)
    return norm


@dataclasses.dataclass(frozen=True)
class GaussHermiteMoments:
    """
    The mean and the dispersion of distributions of velocities given as
    Gauss-Hermite series (see gauss_hermite_moments), in the unit of their
    velocities, and their errors, if known.
    """
    mean: np.ndarray
    dispersion: np.ndarray
    mean_error: np.ndarray | None = None
    dispersion_error: np.ndarray | None = None


def gauss_hermite_moments(
        v: npt.ArrayLike,
        sigma: npt.ArrayLike,
        h3: npt.ArrayLike,
        h4: npt.ArrayLike,
        v_error: npt.ArrayLike | None = None,
        sigma_error: npt.ArrayLike | None = None,
        h3_error: npt.ArrayLike | None = None,
        h4_error: npt.ArrayLike | None = None
) -> GaussHermiteMoments:
    """
    Return the mean and the dispersion of distributions of velocities
    given by the Gauss-Hermite series of V, sigma, h3 and h4 (e.g. those
    of the bins of pPXF), exactly those of the series.

    The series is exp(-w^2 / 2) (1 + h3 H3(w) + h4 H4(w)), w = (v - V) /
    sigma, with H3 and H4 of van der Marel & Franx (1993), as pPXF and
    the method gauss_hermite of the observables of moments take it. Of
    the normalisation N = 1 + sqrt(6) / 4 h4 (see gauss_hermite_norm),
    its mean is V + sqrt(3) sigma h3 / N, and its dispersion sigma
    sqrt((1 + 27 / sqrt(24) h4) / N - 3 h3^2 / N^2) (about sigma (1 +
    sqrt(6) h4) for small h4). V and sigma are not those: e.g. the
    dispersion of a series of h4 = -0.02 is 5% below sigma. Models whose
    stars have a Gaussian distribution of velocities at each point (e.g.
    Jeans kinematics) predict the mean and the dispersion of a line of
    sight, not the shape of its distribution, and so not V, sigma, h3
    and h4: they are fitted to these. They are those of the series;
    those of the distribution it was fitted to differ by its terms
    beyond h4 (e.g. 0.2% in the dispersion of an isotropic Hernquist
    sphere).

    Parameters
    ----------
    v, sigma, h3, h4 : array_like
        The coefficients of the series, of the same shape (v and sigma in
        the same unit).
    v_error, sigma_error, h3_error, h4_error : array_like, optional
        Their errors, independent (e.g. the formal errors of pPXF):
        those of the mean and the dispersion follow to first order, with
        both v_error and sigma_error given (h3_error and h4_error 0 if
        not given).

    Returns
    -------
    GaussHermiteMoments
        The mean and the dispersion, NaN where the series has none (a
        negative variance), and their errors.
    """
    v, sigma, h3, h4 = (
        np.asarray(value, dtype=float) for value in (v, sigma, h3, h4))
    a = 27 / np.sqrt(24)
    # (the derivative of the normalisation by h4)
    b = _norm_term(4)
    c = np.sqrt(3)
    norm = gauss_hermite_norm({4: h4})
    with np.errstate(invalid='ignore', divide='ignore'):
        q = (1 + a * h4) / norm - 3 * h3 ** 2 / norm ** 2
        root = np.sqrt(q)
        mean = v + c * sigma * h3 / norm
        dispersion = sigma * root
        if v_error is None or sigma_error is None:
            return GaussHermiteMoments(mean, dispersion)
        # (the derivatives of the mean and the dispersion by v, sigma, h3
        # and h4, and the errors of those)
        errors = [
            np.asarray(error, dtype=float) if error is not None else 0.0
            for error in (v_error, sigma_error, h3_error, h4_error)]
        mean_partials = (
            1.0, c * h3 / norm, c * sigma / norm,
            -b * c * sigma * h3 / norm ** 2)
        dq_h4 = (a * norm - b * (1 + a * h4)) / norm ** 2 \
            + 6 * b * h3 ** 2 / norm ** 3
        dispersion_partials = (
            0.0, root, -3 * sigma * h3 / (norm ** 2 * root),
            sigma * dq_h4 / (2 * root))
        return GaussHermiteMoments(
            mean, dispersion,
            np.sqrt(sum((d * e) ** 2 for d, e in zip(mean_partials, errors))),
            np.sqrt(sum(
                (d * e) ** 2 for d, e in zip(dispersion_partials, errors))))
