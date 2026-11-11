"""
The numerical kernels of the masses of a lens (see the module masses):
the deflections of the SIE, the EPL and the NFW halo along their major
and minor axes, and that of an elliptical convergence, by a tanh-sinh
quadrature.
"""

import math
from collections.abc import Callable

import numpy as np


__all__ = [
    'elliptical_deflection',
    'epl_deflection',
    'nfw_convergence',
    'nfw_round_deflection',
    'sie_deflection'
]


def sie_deflection(
        u: np.ndarray,
        v: np.ndarray,
        theta_e: float,
        q: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an SIE along its major (u) and minor (v)
    axes (see LensSIE); 0 at its centre.
    """
    r = np.hypot(u, v)
    centre = r == 0
    if q > 1 - 1e-8:
        with np.errstate(invalid='ignore', divide='ignore'):
            scale = np.where(centre, 0, theta_e / r)
        return scale * u, scale * v
    f = math.sqrt(1 - q * q)
    psi = np.where(centre, 1, np.sqrt(q * q * u * u + v * v))
    scale = theta_e * math.sqrt(q) / f
    return (scale * np.arctan(f * u / psi),
            scale * np.arctanh(f * v / psi))


def epl_deflection(
        u: np.ndarray,
        v: np.ndarray,
        theta_e: float,
        slope: float,
        q: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an EPL along its major (u) and minor (v)
    axes (see LensEPL); 0 at its centre.
    """
    t = slope - 1
    b = theta_e * math.sqrt(q)
    z = q * u + 1j * v
    radius = np.abs(z)
    centre = radius == 0
    radius = np.where(centre, 1, radius)
    # The series of e^(i phi) 2F1(...): its terms fall at least as f^n, so
    # the rest after n terms is below f^n / (1 - f)
    f = (1 - q) / (1 + q)
    term = z / radius
    ratio = -f * term * term
    series = term
    n = 1
    while f ** n > 1e-12 * (1 - f):
        term = term * ratio * (2 * n - 2 + t) / (2 * n + 2 - t)
        series = series + term
        n += 1
    alpha = 2 / (1 + q) * b ** t * radius ** (1 - t) * series
    alpha = np.where(centre, 0, alpha)
    return alpha.real, alpha.imag


def _nfw_f(x: np.ndarray) -> np.ndarray:
    """
    Return F(x) of an NFW halo (see LensNFW): arccosh(1 / x) /
    sqrt(1 - x^2), and arccos(1 / x) / sqrt(x^2 - 1) for x > 1; the sum
    of (1 - x^2)^k / (2k + 1) near x = 1.
    """
    z = 1 - x * x
    near = np.abs(z) < 0.1
    series = sum(np.where(near, z, 0) ** k / (2 * k + 1) for k in range(16))
    with np.errstate(divide='ignore'):
        inverse = 1 / x
    angle = np.where(
        x < 1, np.arccosh(np.maximum(inverse, 1)),
        np.arccos(np.minimum(inverse, 1)))
    root = np.sqrt(np.abs(np.where(near, 1, z)))
    return np.where(near, series, angle / root)


def nfw_convergence(x: np.ndarray) -> np.ndarray:
    """
    Return the convergence of an NFW halo over its kappa_s at x times its
    scale radius (see LensNFW): 2 (F(x) - 1) / (1 - x^2), the sum of
    2 (1 - x^2)^(k - 1) / (2k + 1) from k = 1 near x = 1.
    """
    z = 1 - x * x
    near = np.abs(z) < 0.1
    series = sum(np.where(near, z, 0) ** (k - 1) / (2 * k + 1)
                 for k in range(1, 17))
    return 2 * np.where(near, series, (_nfw_f(x) - 1) / np.where(near, 1, z))


def nfw_round_deflection(
        u: np.ndarray,
        v: np.ndarray,
        kappa_s: float,
        r_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of a round NFW halo (see LensNFW) along u and v:
    4 kappa_s r_s h(x) / x along the radius, with x = r / r_s and
    h(x) = ln(x / 2) + F(x).
    """
    r = np.hypot(u, v)
    centre = r == 0
    x = np.where(centre, 1, r) / r_s
    h = np.log(x / 2) + _nfw_f(x)
    scale = np.where(centre, 0, 4 * kappa_s * h / (x * x))
    return scale * u, scale * v


def _tanh_sinh_nodes(
        step: float = 1 / 8, extent: float = 3.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the nodes w of the tanh-sinh quadrature over [0, 1], 1 - w
    (exact near w = 1) and their weights.
    """
    tau = np.arange(-extent, extent + step / 2, step)
    s = math.pi / 2 * np.sinh(tau)
    weights = step * math.pi / 4 * np.cosh(tau) / np.cosh(s) ** 2
    return 1 / (1 + np.exp(-2 * s)), 1 / (1 + np.exp(2 * s)), weights


_NODES, _ONE_MINUS_NODES, _WEIGHTS = _tanh_sinh_nodes()


def elliptical_deflection(
        u: np.ndarray,
        v: np.ndarray,
        q: float,
        kappa: Callable[[np.ndarray], np.ndarray],
        chunk: int = 4096
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an elliptical convergence kappa(xi) (a
    function of xi; see the module masses) along its major (u) and minor
    (v) axes, by the integrals over the ellipses within each point (Keeton
    2001, with xi^2 = q xi_K^2): alpha_u = q u J_0 and alpha_v = q v J_1,

        J_n = integral over w from 0 to 1 of kappa(xi(w)) / d^(n + 1/2),
        d = 1 - (1 - q^2) w,   xi(w)^2 = q w (u^2 + v^2 / d);

    0 at its centre. The pixels are integrated in chunks, to keep the
    memory of the nodes at each pixel small.
    """
    shape = u.shape
    u, v = u.ravel(), v.ravel()
    d = q * q + (1 - q * q) * _ONE_MINUS_NODES
    alpha_u = np.empty_like(u)
    alpha_v = np.empty_like(v)
    for start in range(0, u.size, chunk):
        part = slice(start, start + chunk)
        u_part, v_part = u[part, None], v[part, None]
        xi = np.sqrt(q * _NODES * (u_part * u_part + v_part * v_part / d))
        with np.errstate(divide='ignore', invalid='ignore'):
            weighted = _WEIGHTS * kappa(xi) / np.sqrt(d)
        alpha_u[part] = q * u[part] * weighted.sum(-1)
        alpha_v[part] = q * v[part] * (weighted / d).sum(-1)
    centre = (u == 0) & (v == 0)
    alpha_u[centre] = 0
    alpha_v[centre] = 0
    return alpha_u.reshape(shape), alpha_v.reshape(shape)
