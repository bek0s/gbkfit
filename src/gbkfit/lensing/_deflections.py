"""
The numerical kernels of the pieces of a lens (see the module pieces):
the deflections of the SIE, the EPL and the round NFW halo along their
major and minor axes, the convergence of the NFW halo, and the
deflection of a circular multipole of a power-law profile (that of an
elliptical convergence is the kernel of elliptical layers, see
_layers).
"""

import math

import numpy as np

from gbkfit.params import ModelEvaluationError


__all__ = [
    'epl_deflection',
    'multipole_deflection',
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
    Return the deflection of a round NFW halo (see LensNFW) along u and
    v: 4 kappa_s r_s h(x) / x along the radius, with x = r / r_s and
    h(x) = ln(x / 2) + F(x).
    """
    r = np.hypot(u, v)
    centre = r == 0
    x = np.where(centre, 1, r) / r_s
    h = np.log(x / 2) + _nfw_f(x)
    scale = np.where(centre, 0, 4 * kappa_s * h / (x * x))
    return scale * u, scale * v


def multipole_deflection(
        x: np.ndarray,
        y: np.ndarray,
        theta_e: float,
        t: float,
        order: int,
        a: float,
        posa: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection along x and y of a circular multipole of order
    m, at positions x and y from its centre: that of the convergence

        kappa = a / 2 * (theta_e / r)^t * cos(m (phi - phi_m)),

    the deviation from its ellipse of a host of the radial slope t of its
    convergence (1 for the SIE, slope - 1 for the EPL; e.g. order 4 for
    disky or boxy isodensity contours), at the radius r and the angle
    phi about its centre, with phi_m that of the position angle posa
    (degrees, north through east): a maximum of the convergence for
    a > 0. 0 at its centre.

    Its potential is psi = R(r) cos(m (phi - phi_m)), the solution of
    the Poisson equation (div grad psi = 2 kappa) that is a power of r:

        R = 2 k r^(2 - t) / ((2 - t)^2 - m^2),   k = a / 2 * theta_e^t,

    and for m = 1, where that power is also a solution of the Laplace
    equation at t = 1, R = 2 k r ((r^(1 - t) - theta_e^(1 - t)) /
    (1 - t)) / (3 - t): the power less the constant deflection
    2 k theta_e^(1 - t) / ((1 - t)(3 - t)), which moves only the source,
    and 2 k r ln(r / theta_e) / (3 - t) at t = 1 (the isothermal
    multipole of e.g. Van de Vyvere et al. 2022, of t = 1).

    Raises
    ------
    ModelEvaluationError
        If (2 - t)^2 = m^2 for m > 1 (order 2 of the slope t = 0, a
        uniform sheet), where no power is a solution.
    """
    if theta_e == 0 or a == 0:
        return np.zeros_like(x), np.zeros_like(y)
    m = order
    r = np.hypot(x, y)
    centre = r == 0
    r = np.where(centre, 1, r)
    cos_phi, sin_phi = x / r, y / r
    # (the angle from +x of the position angle is 90 degrees more)
    angle = m * (np.arctan2(y, x) - math.radians(posa + 90))
    k = a / 2 * theta_e ** t
    power = r ** (1 - t)
    if m == 1:
        e = 1 - t
        log_r = np.log(r / theta_e)
        # ((r^e - theta_e^e) / e, exact as e goes to 0)
        g = log_r if e == 0 else theta_e ** e * np.expm1(e * log_r) / e
        scale = 2 * k / (3 - t)
        alpha_r = scale * (g + power) * np.cos(angle)
        alpha_phi = -scale * g * np.sin(angle)
    else:
        denominator = (2 - t) ** 2 - m * m
        if denominator == 0:
            raise ModelEvaluationError(
                f"a multipole of order {m} has no deflection about a host "
                f"of the radial slope t = {t:g} of its convergence")
        c = 2 * k / denominator
        alpha_r = c * (2 - t) * power * np.cos(angle)
        alpha_phi = -m * c * power * np.sin(angle)
    alpha_x = alpha_r * cos_phi - alpha_phi * sin_phi
    alpha_y = alpha_r * sin_phi + alpha_phi * cos_phi
    return np.where(centre, 0, alpha_x), np.where(centre, 0, alpha_y)
