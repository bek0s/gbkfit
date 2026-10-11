"""
The numerical methods of mass models: integrals of Bessel functions for
axisymmetric disks, tables of spherical profiles (the deprojected Sersic
profile), and the potential of a spherical density.
"""

import dataclasses
import functools
from collections.abc import Callable

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.special

from .units import G


# The integrals of J1 (see bessel_j1_integral): integrated between the
# first zeros of J1(u), with Gauss-Legendre nodes in each interval, and
# in the first one in panels of geometric widths from u = 1e-9 (where the
# integrand of a small R has its structure)
_J1_NINTERVALS = 160
_J1_NNODES = 12
_J1_NPANELS = 40
# (the partial sums averaged repeatedly, for the alternating tail)
_J1_NAVERAGE = 12


def _j1_nodes() -> tuple[np.ndarray, np.ndarray]:
    """
    Return the nodes u of the panels between the first zeros of J1(u)
    (an array of shape (panels, nodes); the first _J1_NPANELS panels
    cover the first interval), and their weights times u J1(u).
    """
    zeros = scipy.special.jn_zeros(1, _J1_NINTERVALS)
    first = np.concatenate(
        ([0], np.geomspace(1e-9 * zeros[0], zeros[0], _J1_NPANELS)))
    edges = np.concatenate((first, zeros[1:]))
    x, w = np.polynomial.legendre.leggauss(_J1_NNODES)
    lo, hi = edges[:-1, None], edges[1:, None]
    u = (hi - lo) / 2 * x + (hi + lo) / 2
    return u, (hi - lo) / 2 * w * u * scipy.special.j1(u)


_J1_U, _J1_WEIGHTS = _j1_nodes()


def bessel_j1_integral(
        R: np.ndarray, g: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Return the integral of k J1(k R) g(k) over k from 0 to infinity, for
    each R (0 at R = 0), for a smooth g of g(0) = 1 that falls at least
    like k^-1.5 (to about 1e-10, relative, for the thin exponential disk
    at R / rd from 1e-3 to 1e3).

    With u = k R, it is the integral of u J1(u) g(u / R) / R^2: summed
    between the zeros of J1(u), where the terms alternate in sign, and
    the last partial sums averaged repeatedly (the Euler transform of
    the tail).
    """
    R = np.asarray(R, dtype=float)
    out = np.zeros_like(R)
    positive = R > 0
    radii = R[positive]
    panels = np.sum(
        _J1_WEIGHTS * g(_J1_U / radii[:, None, None]), axis=-1)
    terms = np.concatenate(
        (panels[:, :_J1_NPANELS].sum(axis=1, keepdims=True),
         panels[:, _J1_NPANELS:]), axis=1)
    sums = np.cumsum(terms, axis=-1)[:, -_J1_NAVERAGE:]
    while sums.shape[1] > 1:
        sums = (sums[:, 1:] + sums[:, :-1]) / 2
    out[positive] = sums[:, 0] / radii ** 2
    return out


# The log grid on which smooth radial functions are computed before they
# are interpolated (see on_log_grid): points per decade
_LOG_GRID_DENSITY = 32


def on_log_grid(
        x: np.ndarray, f: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """
    Return a positive function f, smooth in ln x, at x > 0: computed on a
    geometric grid that covers x (_LOG_GRID_DENSITY points per decade),
    and interpolated by a cubic spline of ln f in ln x.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = x.min() / 1.5, x.max() * 1.5
    grid = np.geomspace(lo, hi, max(
        4, int(np.ceil(_LOG_GRID_DENSITY * np.log10(hi / lo))) + 1))
    spline = scipy.interpolate.CubicSpline(np.log(grid), np.log(f(grid)))
    return np.exp(spline(np.log(x)))


@dataclasses.dataclass(frozen=True)
class SphericalTables:
    """
    The density, enclosed mass and potential of a spherical profile of
    unit total mass and scale radius (and G = 1), as cubic splines in
    ln r of ln(density) and ln(mass) and of the potential, on radii from
    r_min to r_max. Below r_min, the density and the mass follow the
    power law of the innermost radii; beyond r_max, the profile has
    neither mass nor density left (to the precision of the tables).
    """
    r_min: float
    r_max: float
    log_density: scipy.interpolate.CubicSpline
    log_mass: scipy.interpolate.CubicSpline
    potential_spline: scipy.interpolate.CubicSpline
    slope: float          # d ln(density) / d ln r at r_min
    total_mass: float     # the mass within r_max

    def density(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        inner = np.exp(self.log_density(np.log(self.r_min))) \
            * (np.maximum(r, 1e-300) / self.r_min) ** self.slope
        middle = np.exp(self.log_density(np.log(np.clip(
            r, self.r_min, self.r_max))))
        return np.where(r < self.r_min, inner,
                        np.where(r > self.r_max, 0.0, middle))

    def mass(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        inner = np.exp(self.log_mass(np.log(self.r_min))) \
            * (np.maximum(r, 0) / self.r_min) ** (3 + self.slope)
        middle = np.exp(self.log_mass(np.log(np.clip(
            r, self.r_min, self.r_max))))
        return np.where(r < self.r_min, inner,
                        np.where(r > self.r_max, self.total_mass, middle))

    def potential(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        middle = self.potential_spline(np.log(np.clip(
            r, self.r_min, self.r_max)))
        outer = -self.total_mass / np.maximum(r, self.r_max)
        return np.where(r > self.r_max, outer, middle)


def _spherical_tables(r: np.ndarray, density: np.ndarray) -> SphericalTables:
    """
    Return the tables of a spherical profile from its density on radii r
    (geometric), integrated by the trapezoidal rule in ln r.
    """
    log_r = np.log(r)
    slope = (np.log(density[1]) - np.log(density[0])) / (log_r[1] - log_r[0])
    # The mass within r: from the power law within r[0], then the exact
    # integral of a cubic spline in ln r of 4 pi r^3 rho
    mass_inner = 4 * np.pi * density[0] * r[0] ** 3 / (3 + slope)
    shell = scipy.interpolate.CubicSpline(
        log_r, 4 * np.pi * density * r ** 3).antiderivative()
    mass = mass_inner + shell(log_r) - shell(log_r[0])
    # The potential: -(M(<r) / r + 4 pi int_r^inf rho r' dr'), with
    # nothing beyond r[-1]
    outer = scipy.interpolate.CubicSpline(
        log_r, 4 * np.pi * density * r ** 2).antiderivative()
    potential = -(mass / r + outer(log_r[-1]) - outer(log_r))
    return SphericalTables(
        r_min=float(r[0]), r_max=float(r[-1]),
        log_density=scipy.interpolate.CubicSpline(
            log_r, np.log(np.maximum(density, 1e-300))),
        log_mass=scipy.interpolate.CubicSpline(log_r, np.log(mass)),
        potential_spline=scipy.interpolate.CubicSpline(log_r, potential),
        slope=float(slope), total_mass=float(mass[-1]))


# The tables of the deprojected Sersic profile: their radii, and the
# Gauss-Legendre nodes of the integral that deprojects the density
_SERSIC_NRADII = 800
_SERSIC_NNODES = 384
_SERSIC_R_MIN = 1e-6


def sersic_b(n: float) -> float:
    """
    Return b(n) of the Sersic profile exp(-b ((R / Re)^(1/n) - 1)): half
    of its light is within Re, gamma(2n, b) = Gamma(2n) / 2, exactly.
    """
    return float(scipy.special.gammaincinv(2 * n, 0.5))


@functools.lru_cache(maxsize=64)
def sersic_tables(n: float) -> SphericalTables:
    """
    Return the tables of the spherical deprojection of the Sersic profile
    of index n, of unit total mass and effective radius.

    The surface density is S(R) = S_e exp(-b (R^(1/n) - 1)), and the
    density its Abel deprojection, -1/pi int_r^inf S'(R) / sqrt(R^2 -
    r^2) dR; with R = r cosh(t), (b / (n pi)) r^(1/n - 1) int_0^inf
    S(r cosh t) cosh(t)^(1/n - 1) dt, by Gauss-Legendre quadrature up to
    the radius where S falls below e^-40 S_e.
    """
    b = sersic_b(n)
    sigma_e = 1 / (2 * np.pi * n * np.exp(b) * b ** (-2 * n)
                   * scipy.special.gamma(2 * n))
    r_max = (1 + 40 / b) ** n
    r = np.geomspace(_SERSIC_R_MIN, r_max, _SERSIC_NRADII)
    x, w = np.polynomial.legendre.leggauss(_SERSIC_NNODES)
    t_max = np.arccosh(np.maximum(r_max / r, 1))[:, None]
    t = (x + 1) / 2 * t_max
    cosh_t = np.cosh(t)
    sigma = sigma_e * np.exp(-b * ((r[:, None] * cosh_t) ** (1 / n) - 1))
    integral = np.sum(w / 2 * t_max * sigma * cosh_t ** (1 / n - 1), axis=1)
    density = b / (n * np.pi) * r ** (1 / n - 1) * integral
    # (keep the radii where the density has not underflowed)
    keep = density > 0
    return _spherical_tables(r[keep], density[keep])


def spherical_potential(
        density: Callable[[np.ndarray], np.ndarray],
        enclosed_mass: Callable[[np.ndarray], np.ndarray],
        r: np.ndarray
) -> np.ndarray:
    """
    Return the potential of a spherical density at the radii r, by
    adaptive quadrature: -G (M(<r) / r + 4 pi int_r^inf rho r' dr'), with
    0 at infinity (and the limit at r = 0).
    """
    r = np.asarray(r, dtype=float)
    unique, inverse = np.unique(r, return_inverse=True)
    outer = np.array([
        scipy.integrate.quad(
            lambda x: float(density(np.array(x))) * x, radius, np.inf,
            limit=200)[0]
        for radius in unique])
    mass_term = np.divide(
        enclosed_mass(unique), unique, out=np.zeros_like(unique),
        where=unique > 0)
    return (-G * (mass_term + 4 * np.pi * outer))[inverse].reshape(r.shape)
