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

from gbkfit.params import ModelEvaluationError
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


# The tables of the density of Sersic profiles (see sersic_density_table):
# from the radius _SERSIC_R_MIN, with the Gauss-Legendre nodes of the
# integral that deprojects the density (to about 1e-7 of the peak, relative,
# within 1e-8 of it)
_SERSIC_DENSITY_NNODES = 64


def sersic_density_table(
        n: float, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the table of the spherical deprojection f of the Sersic profile
    of index n, of the surface density exp(-b (R^(1/n) - 1)) (1 at the
    effective radius, 1): ln f at size radii u, geometric from
    _SERSIC_R_MIN to where the surface density falls below e^-40 (as
    sersic_tables, by its integral, with fewer nodes), and its range: ln u
    of the first radius, the step of ln u, and the slope of ln f in ln u
    at the first radius (the power law of f below it).
    """
    b = sersic_b(n)
    u_max = (1 + 40 / b) ** n
    u = np.geomspace(_SERSIC_R_MIN, u_max, size)
    x, w = np.polynomial.legendre.leggauss(_SERSIC_DENSITY_NNODES)
    t_max = np.arccosh(np.maximum(u_max / u, 1))[:, None]
    cosh_t = np.cosh((x + 1) / 2 * t_max)
    surface = np.exp(-b * ((u[:, None] * cosh_t) ** (1 / n) - 1))
    integral = np.sum(
        w / 2 * t_max * surface * cosh_t ** (1 / n - 1), axis=1)
    log_f = np.log(np.maximum(
        b / (n * np.pi) * u ** (1 / n - 1) * integral, 1e-300))
    step = np.log(u[1] / u[0])
    slope = (log_f[1] - log_f[0]) / step
    return log_f, np.array([np.log(u[0]), step, slope])


# The projections of spherical masses (see projected_mass and
# abel_projection): integrated along each line of sight out to the
# radius _END_FACTOR times its own, or _END_RADIUS (kpc) if farther (where
# every profile is its power law), and beyond, of that power law
_END_FACTOR = 1e6
_END_RADIUS = 1e9

# The projected mass within circles (see projected_mass): Gauss-Legendre
# nodes on panels of the angle t of these edges, then on _PROJECTED_PANELS
# panels of ln cos(t), each of this many nodes
_PROJECTED_T_EDGES = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25])
_PROJECTED_PANELS = 24
_PROJECTED_NODES = 6

# The surface density (see abel_projection): Gauss-Legendre nodes on
# panels of u (R cosh u, the radius of the line of sight) of these edges,
# then on _ABEL_PANELS panels, each of this many nodes
_ABEL_FIRST_EDGES = np.array([0, 0.01, 0.03, 0.1, 0.25])
_ABEL_PANELS = 40
_ABEL_NODES = 6


def _gauss_legendre(
        edges: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the nodes and weights of the composite Gauss-Legendre rules of
    count nodes on the panels between edges (along the last axis), flat
    over the panels of each row.
    """
    x, w = np.polynomial.legendre.leggauss(count)
    lo, hi = edges[..., :-1, None], edges[..., 1:, None]
    shape = edges.shape[:-1] + (-1,)
    return ((((hi - lo) * x + hi + lo) / 2).reshape(shape),
            ((hi - lo) * w / 2).reshape(shape))


def _end(R: np.ndarray) -> np.ndarray:
    """
    Return the radius (kpc) out to which the lines of sight of the radii R
    are integrated (see _END_FACTOR).
    """
    return np.maximum(_END_FACTOR * R, _END_RADIUS)


def _power_slope(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Return the slope d ln f / d ln r of a function f between its last two
    values, at the last two radii r (along the last axis); 0 where its
    last value is 0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = np.log(f[..., -1] / f[..., -2]) \
            / np.log(r[..., -1] / r[..., -2])
    return np.where(f[..., -1] > 0, slope, 0.0)


def projected_mass(
        enclosed_mass: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray
) -> np.ndarray:
    """
    Return the projected mass (Msun) of a spherical mass within the
    circles of the radii R (kpc, positive) about its centre, from its mass
    within spheres M(r) (a function of an array of radii): M2D(<R) =
    int_0^(pi/2) M(R / cos t) cos t dt (the mass within each sphere seen
    through the circle, M(R) plus the share of the shells beyond R that
    the cylinder holds, integrated by parts), by Gauss-Legendre quadrature
    out to the sphere of the radius r_end (see _END_FACTOR), and beyond, of
    the power law M ~ r^p there: M(r_end) c^2 / (2 - p), c = R / r_end (to
    about 1e-9, and 2e-8 near the core of a cored profile).

    Raises
    ------
    ModelEvaluationError
        If it is infinite (a mass that grows as r^2 or faster).
    """
    R = np.asarray(R, dtype=float)
    radii = R.ravel()
    n = len(radii)
    # In t near 0, then in s = ln cos(t), where the integrand M(R / c) c^2 /
    # sin(t) ds is smooth (c = cos t), and of no weight, the end and a
    # point before it, of the power law beyond
    t, t_weights = _gauss_legendre(_PROJECTED_T_EDGES, _PROJECTED_NODES)
    log_end = np.log(radii / _end(radii))[:, None]
    log_start = np.log(np.cos(_PROJECTED_T_EDGES[-1]))
    log_cos, s_weights = _gauss_legendre(
        log_end + (log_start - log_end) * np.linspace(
            0, 1, _PROJECTED_PANELS + 1), _PROJECTED_NODES)
    cos = np.concatenate((
        np.broadcast_to(np.cos(t), (n, len(t))), np.exp(log_cos),
        np.exp(log_end + np.log([1.5, 1.0]))), axis=1)
    weights = np.concatenate((
        np.broadcast_to(t_weights * np.cos(t), (n, len(t))),
        s_weights * np.exp(2 * log_cos) / np.sqrt(-np.expm1(2 * log_cos))),
        axis=1)
    r = radii[:, None] / cos
    mass = enclosed_mass(r.ravel()).reshape(r.shape)
    slope = _power_slope(mass, r)
    if np.any(slope >= 2 - 1e-6):
        raise ModelEvaluationError(
            "a spherical mass has an infinite projected mass: its mass "
            "within spheres grows as r^2 or faster")
    tail = mass[:, -1] * np.exp(2 * log_end[:, 0]) / (2 - slope)
    return (np.sum(mass[:, :-2] * weights, axis=1) + tail).reshape(R.shape)


def abel_projection(
        density: Callable[[np.ndarray], np.ndarray],
        R: np.ndarray
) -> np.ndarray:
    """
    Return the surface density (Msun/kpc^2) of a spherical density at the
    projected radii R (kpc, positive), from its density rho(r) (a function
    of an array of radii): Sigma(R) = 2 int_R^inf rho r dr / sqrt(r^2 -
    R^2) = 2 R int_0^inf rho(R cosh u) cosh u du, by Gauss-Legendre
    quadrature out to the radius r_end (see _END_FACTOR), and beyond, of
    the power law rho ~ r^p there: 2 rho(r_end) r_end / -(p + 1) (to about
    1e-10 for halo profiles; a kink of the density, e.g. at a node of a
    nodewise halo, costs some 1e-6 near it).

    Raises
    ------
    ModelEvaluationError
        If it is infinite (a density that falls as r^-1 or slower).
    """
    R = np.asarray(R, dtype=float)
    radii = R.ravel()
    end = np.arccosh(_end(radii) / radii)[:, None]
    first = _ABEL_FIRST_EDGES[-1]
    edges = np.concatenate((np.broadcast_to(
        _ABEL_FIRST_EDGES[:-1], (len(radii), len(_ABEL_FIRST_EDGES) - 1)),
        first + (end - first) * np.linspace(0, 1, _ABEL_PANELS + 1)), axis=1)
    u, weights = _gauss_legendre(edges, _ABEL_NODES)
    # (and of no weight, a point before the end and the end, of the power
    # law beyond)
    u = np.concatenate((u, end - 0.4, end), axis=1)
    r = radii[:, None] * np.cosh(u)
    rho = density(r.ravel()).reshape(r.shape)
    slope = _power_slope(rho, r)
    slope = np.where(rho[:, -1] > 0, slope, -2.0)
    if np.any(slope >= -1 - 1e-6):
        raise ModelEvaluationError(
            "a spherical density has an infinite surface density: it falls "
            "as r^-1 or slower")
    tail = rho[:, -1] * np.cosh(end[:, 0]) / -(slope + 1)
    total = np.sum(rho[:, :-2] * weights * np.cosh(u[:, :-2]), axis=1)
    return (2 * radii * (total + tail)).reshape(R.shape)


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
