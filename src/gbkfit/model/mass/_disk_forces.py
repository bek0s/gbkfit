"""
The forces of axisymmetric mass disks anywhere, the radial force F_R(R,
z) = -dPhi/dR and the vertical force F_z(R, z) = -dPhi/dz (for the
jeans equilibrium of the disks of a galaxy, see parts.JeansEquilibrium), from
the pull of their rings off their plane, as their circular velocity in
it (see _disk_gravity).

A disk is a stack of thin sheets, at the heights h of its density rho(a,
h): F(R, z) = int F_s[rho(., h)](R, z - h) dh, with the field of a sheet
of the surface density sigma(a)

    F_s,R(R, zeta) = -G / R int sigma(a) 2 a / D [K(m) - E(m) (a^2 - R^2
                     + zeta^2) / ((a - R)^2 + zeta^2)] da,
    F_s,z(R, zeta) = -G zeta int sigma(a) a 4 E(m) / (((a - R)^2 + zeta^2)
                     D) da,

D^2 = (a + R)^2 + zeta^2 and m = 4 a R / D^2 (F_s,R is the kernel of the
thick disks of _disk_gravity; F_s,z follows from int_0^2pi dphi / (A - B
cos phi)^1.5 = 4 E(m) / ((A - B) sqrt(A + B))). F_s,R is even in zeta and
continuous across the sheet: C = F_s,R. F_s,z is odd, and jumps across
the sheet from 2 pi G sigma(R) to -2 pi G sigma(R), the pull of an
infinite sheet (with the mean of the two sides where sigma jumps, e.g. at
the edge of the disk): F_s,z = -2 pi G sigma(R) sign(zeta) + C(R, zeta),
with C continuous. C is tabulated at each radius on heights of geometric
widths (_ZETAS), from the surface density at the density points of
_disk_gravity by weights that depend on the geometry only (cached, as
those of _disk_gravity), and interpolated cubically in ln |zeta|; the
jumps of the sheets of a thick disk add up to the pull of its slab, -4 pi
G int_0^z rho(R, h) dh, by quadrature on [0, z], and C to the quadrature
over the heights of the density of _disk_gravity. The integrals over a
are by Gauss-Legendre quadrature on the panels far from R and on
sub-panels of geometric widths toward R near it (those of _disk_gravity),
where the kernels peak within |zeta| of R: to about 5e-5 of the force
(the interpolation of C), or for the tables of an exponential disk, 1e-4
near it and 1e-3 far from it (where its pull is weak). The radial force
of a sheet has a kink at it, -2 pi G sigma'(R) |zeta|, between the
heights of the quadrature of a thick disk: its radial force is to about
1e-3 (3e-3 near the centre of a disk a third as thick as its scale
length, 2026-10-20; 0.15% of its rotation). In the plane of a disk (z =
0), its radial force is that of its circular velocity (see
disks.MassDisk.forces).

Far from a disk, beyond its tables (more than 5 outer radii from its
plane, or for an exponential disk, beyond 1000 scale lengths from its
axis), its force is that of the multipole series of its mass (see
_FarField), of the even orders to 16: the series converges outside the
sphere that holds the mass, and its terms fall there at least as 5^-l,
so that its error is below about 1e-11 of the force.

Alternative (not taken): the potential as a multipole series of the
density, with the analytic part of Kuijken & Dubinski (1995), Phi_s = 4 pi
G Sigma(r) H(z) (r the spherical radius, H'' the pdf of the height), which
holds the jump at the plane, split off: one series for every mass, about
0.3 s a step, and the jeans dispersion within about 1e-4 at the order 64
(the error falls as order^-2.4, from the kink of the rest at the plane),
of flaring disks too with H following their thickness. But beyond the
sphere of the radius of a sharp edge of a disk (e.g. of one that follows
a component, whose density jumps to 0 beyond its last ring), Phi_s is 0
where the disk is not, so that the series alone must follow the edge:
the dispersion there is off by up to 70%, whatever the order (2026-10-20,
a thin exponential disk cut at 15 kpc).
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy.sparse
import scipy.special

from ._disk_gravity import (
    _HEIGHT_EDGES, _HEIGHT_WEIGHTS, _HEIGHTS, _KEY_RESOLUTION, _PDF_TOLERANCE,
    _Geometry, _Panels, _thick_kernel)
from .units import G


__all__ = [
    'Axis',
    'DiskForce',
    'ExponentialForce',
    'disk_force',
    'exponential_force'
]


# The force: radial ('R', -dPhi/dR) or vertical ('z', -dPhi/dz)
Axis = Literal['R', 'z']

# The heights zeta of the tables of C (see the module docstring), relative
# to the outer radius of the disk: 32 per decade from 1e-9 to 16 (beyond
# the heights of a thick disk, _HEIGHTS, and of the points asked for);
# below, C falls linearly to 0 (odd, the vertical force) or stays (even,
# the radial force)
_ZETAS = np.geomspace(1e-9, 16, 327)
_LN_ZETAS = np.log(_ZETAS)

# The highest point (relative to the outer radius of the disk) whose force
# the tables reach, with the heights of a thick disk; beyond, the far
# field (see _FarField)
_MAX_HEIGHT = 5.0

# The far field of a disk (see _FarField): its multipoles of the even
# orders to _FAR_ORDER, and the coefficients of the solid harmonics, r^l
# P_l(cos theta) = sum_j c[l / 2, j] R^2j z^(l - 2j), of R the radius and z
# the height
_FAR_ORDER = 16
_FAR_ORDERS = np.arange(0, _FAR_ORDER + 1, 2)
_SOLID = np.array([[
    (-1) ** j * scipy.special.factorial(order)
    / (4 ** j * scipy.special.factorial(j) ** 2
       * scipy.special.factorial(order - 2 * j)) if 2 * j <= order else 0.0
    for j in range(len(_FAR_ORDERS))] for order in _FAR_ORDERS])

# The vertical moments of the heights of thick disks (of outer radius 1):
# int pdf(h) h^2p dh over both sides = _VERTICAL[p] @ pdf at _HEIGHTS
_VERTICAL = 2 * _HEIGHT_WEIGHTS * _HEIGHTS ** (2 * np.arange(
    len(_FAR_ORDERS))[:, None])


def _vertical_kernel(
        a: np.ndarray, d: np.ndarray, R: float, zeta: np.ndarray
) -> np.ndarray:
    """
    Return zeta 4 a E(m) / ((d^2 + zeta^2) sqrt((a + R)^2 + zeta^2)), the
    vertical pull of the ring at a of a sheet on the point at the height
    zeta above it (per G and surface density), d = a - R apart (to keep
    its precision near R).
    """
    near2 = d ** 2 + zeta ** 2
    far2 = (a + R) ** 2 + zeta ** 2
    return zeta * 4 * a * scipy.special.ellipe(1 - near2 / far2) \
        / (near2 * np.sqrt(far2))


def _radial_kernel(
        a: np.ndarray, d: np.ndarray, R: float, zeta: np.ndarray
) -> np.ndarray:
    """
    Return 2 a / (R D) [K(m) - E(m) (a^2 - R^2 + zeta^2) / (d^2 +
    zeta^2)], the radial pull of the ring at a of a sheet on the point at
    the height zeta above it (per G and surface density; see
    _vertical_kernel), 0 on the axis.
    """
    if R == 0:
        return np.zeros(np.broadcast_shapes(np.shape(a), np.shape(zeta)))
    return _thick_kernel(R, a, d, zeta) / (2 * R)


_KERNELS = dict(R=_radial_kernel, z=_vertical_kernel)


def _mean_basis(panels: _Panels, npoints: int, R: float) -> np.ndarray:
    """
    Return the polynomials of the panels at R, as a row over the density
    points: those of its panel, or at a knot, the mean of those of the
    panels on its two sides (none beyond the edges of the disk); at the
    centre, those of the first panel (the axis is no edge).
    """
    out = np.zeros(npoints)
    inside = np.flatnonzero((panels.lo < R) & (R < panels.hi))
    sides = inside if inside.size \
        else np.flatnonzero((panels.lo == R) | (panels.hi == R))
    share = 1.0 if inside.size or R == 0 else 0.5
    for panel in sides:
        out[panels.indices[panel]] += \
            share * panels.basis(panel, np.array([R]))[0]
    return out


@dataclasses.dataclass(frozen=True)
class _SheetWeights:
    """
    The weights of the sheets of a disk of outer radius 1 at given radii,
    for one force: C_j(zeta_n) = G weights[j, n] @ sigma at the density
    points (see the module docstring), and the mean polynomials at each
    radius (means[j] @ sigma, see _mean_basis; for the jump of the
    vertical force).
    """
    weights: np.ndarray
    means: np.ndarray


@functools.lru_cache(maxsize=16)
def _sheet_weights(
        knots_key: bytes, radii_key: bytes, degree: int, axis: Axis
) -> _SheetWeights:
    knots = np.frombuffer(knots_key) / _KEY_RESOLUTION
    radii = np.frombuffer(radii_key) / _KEY_RESOLUTION
    panels = _Panels.from_knots(knots, degree)
    npoints = len(panels.points())
    kernel = _KERNELS[axis]
    weights = np.zeros((len(radii), npoints, len(_ZETAS)))
    means = np.zeros((len(radii), npoints))
    for j, R in enumerate(radii):
        near = panels.near(R)
        a, w, basis = panels.far_quadrature(~near)
        far = kernel(a[..., None], a[..., None] - R, R, _ZETAS)
        np.add.at(
            weights[j], panels.indices[~near],
            -np.einsum('pn,pnz,nq->pqz', w, far, basis))
        for panel in np.flatnonzero(near):
            for a, d, w, basis in panels.near_quadrature(panel, R):
                values = kernel(a[:, None], d[:, None], R, _ZETAS)
                weights[j, panels.indices[panel]] -= np.einsum(
                    'n,nz,nq->qz', w, values, basis)
        means[j] = _mean_basis(panels, npoints, R)
        if axis == 'z':
            # (C: the jump of the sheet taken out, above it)
            weights[j] += 2 * np.pi * means[j][:, None]
    return _SheetWeights(
        np.ascontiguousarray(weights.transpose(0, 2, 1)), means)


def _interpolate(
        tables: np.ndarray, zeta: np.ndarray, odd: bool) -> np.ndarray:
    """
    Return C at the heights zeta (relative to the outer radius of the
    disk; any sign, C odd or even in it) from its tables at _ZETAS: an
    array of the shape (len(_ZETAS),) + S, with S of the dimensions of
    zeta and broadcast with it (a table for each point). Cubic
    (Catmull-Rom) in ln |zeta|; below _ZETAS[0], linear to 0 if odd, else
    its value there.
    """
    size = np.abs(zeta)
    step = _LN_ZETAS[1] - _LN_ZETAS[0]
    u = (np.log(np.maximum(size, _ZETAS[0])) - _LN_ZETAS[0]) / step
    i = np.clip(np.floor(u).astype(int), 1, len(_ZETAS) - 3)
    t = u - i

    def at(k: np.ndarray) -> np.ndarray:
        return np.take_along_axis(tables, k[None], axis=0)[0]
    p0, p1, p2, p3 = at(i - 1), at(i), at(i + 1), at(i + 2)
    value = p1 + t / 2 * (p2 - p0 + t * (
        2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))
    if not odd:
        return np.where(size < _ZETAS[0], tables[0], value)
    value = np.where(size < _ZETAS[0], tables[0] * size / _ZETAS[0], value)
    return np.sign(zeta) * value


def _catmull_rom(
        zeta: np.ndarray, odd: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the rows of _ZETAS and their coefficients that interpolate C at
    the heights zeta (see _interpolate): arrays of the shape of zeta and 4
    (rows i - 1 to i + 2), the sign of zeta in the coefficients if odd.
    """
    size = np.abs(zeta)
    step = _LN_ZETAS[1] - _LN_ZETAS[0]
    u = (np.log(np.maximum(size, _ZETAS[0])) - _LN_ZETAS[0]) / step
    i = np.clip(np.floor(u).astype(int), 1, len(_ZETAS) - 3)
    t = (u - i)[..., None]
    coefficients = np.concatenate((
        t * (-1 + t * (2 - t)), 2 + t * t * (-5 + 3 * t),
        t * (1 + t * (4 - 3 * t)), t * t * (t - 1)), axis=-1) / 2
    rows = i[..., None] + np.arange(-1, 3)
    low = (size < _ZETAS[0])[..., None]
    rows = np.where(low, 0, rows)
    lowest = size / _ZETAS[0] if odd else np.ones_like(size)
    coefficients = np.where(
        low, np.array([1.0, 0, 0, 0]) * lowest[..., None], coefficients)
    if not odd:
        return rows, coefficients
    return rows, coefficients * np.sign(zeta)[..., None]


@dataclasses.dataclass(frozen=True)
class _ThickPlan:
    """
    How the sheets and the slab of a thick disk are summed at the points
    of one radius (see _Stack.thick), from its geometry: the sum of C of
    its sheets at +h and -h, times the weights of the heights, as a
    sparse map of the table of C at the radius (sheets: of shape
    (points, len(_ZETAS) * len(_HEIGHTS))); and the quadrature of the
    slab between the sorted |z| of the points and the edges of the panels
    of _HEIGHTS, which resolve the density (nodes and weights on each
    interval from the last, relative to the outer radius of the disk, and
    the index of the |z| of each point among the edges of the intervals).
    """
    sheets: scipy.sparse.csr_array
    nodes: np.ndarray
    weights: np.ndarray
    order: np.ndarray


@functools.lru_cache(maxsize=64)
def _thick_plan(heights_key: bytes, odd: bool) -> _ThickPlan:
    z = np.frombuffer(heights_key) / _KEY_RESOLUTION
    npoints, nheights = len(z), len(_HEIGHTS)
    sheets = []
    for side in (-1, 1):
        rows, coefficients = _catmull_rom(z[:, None] + side * _HEIGHTS, odd)
        sheets.append(scipy.sparse.csr_array(
            ((coefficients * _HEIGHT_WEIGHTS[:, None]).ravel(), (
                np.repeat(np.arange(npoints), nheights * 4),
                (rows * nheights + np.arange(nheights)[:, None]).ravel())),
            shape=(npoints, len(_ZETAS) * nheights)))
    # (without the edges, one interval [0, |z|] far wider than the disk
    # missed its density: e.g. 4778 for -2.97 (km/s)^2/kpc at one outer
    # radius above the centre of a disk of hz / R_out = 1 / 400)
    edges, at = np.unique(
        np.concatenate((_HEIGHT_EDGES, np.abs(z))), return_inverse=True)
    lo, hi = edges[:-1, None], edges[1:, None]
    x, w = np.polynomial.legendre.leggauss(8)
    return _ThickPlan(
        sheets[0] + sheets[1], ((hi - lo) * x + hi + lo) / 2,
        (hi - lo) * w / 2, at[len(_HEIGHT_EDGES):])


class _Stack:
    """
    The points (R, z) (kpc, flat arrays) at which a force of a disk is
    evaluated from its tables (within _MAX_HEIGHT of its plane), the index
    of the radius of each among those of the tables of its sheets
    (inverse), the outer radius of the disk, the unit of the heights of the
    tables (scale), and the force (axis): the sums of the sheets of the
    disk there (see the module docstring).
    """

    def __init__(
            self,
            R: np.ndarray,
            z: np.ndarray,
            inverse: np.ndarray,
            scale: float,
            axis: Axis
    ):
        self._R = R
        self._z = z
        self._inverse = inverse
        self._scale = scale
        self._axis = axis

    def thin(self, tables: np.ndarray, means: np.ndarray) -> np.ndarray:
        """
        Return the force ((km/s)^2/kpc) of a thin disk of the tables C
        (per G, an array of shape (len(_ZETAS), radii)) and the mean
        surface densities (Msun/kpc^2) at the radii of the tables.
        """
        z = self._z
        tables = tables[:, self._inverse]
        if self._axis == 'R':
            return G * _interpolate(tables, z / self._scale, odd=False)
        return G * (
            _interpolate(tables, z / self._scale, odd=True)
            - 2 * np.pi * means[self._inverse] * np.sign(z))

    def thick(
            self,
            tables: Callable[[int], np.ndarray],
            means: np.ndarray,
            pdfs_at: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Return the force ((km/s)^2/kpc) of a thick disk of the tables C of
        the sheets at the heights _HEIGHTS (per G, an array of shape
        (len(_ZETAS), len(_HEIGHTS)) for the index of a radius of the
        tables), the mean surface densities (Msun/kpc^2) of its terms at
        the radii of the tables (an array of shape (radii, terms)), and
        the pdfs of the height of its terms at the radius R and the
        heights z (kpc) of two arrays (see DiskForce.thick; the vertical
        force only takes the last two, for the pull of the slab).
        """
        scale = self._scale
        vertical = self._axis == 'z'
        out = np.zeros_like(self._z)
        for index in np.unique(self._inverse):
            points = self._inverse == index
            z = self._z[points]
            plan = _thick_plan(
                np.round(z / scale * _KEY_RESOLUTION).tobytes(), vertical)
            # The sheets at +h and -h (without their jumps); for the
            # vertical force, the pull of the slab, int_0^|z| rho(R, h) dh
            # (with the mean densities), summed between the sorted heights
            sheets = plan.sheets @ tables(index).ravel()
            if not vertical:
                out[points] = G * scale * sheets
                continue
            pdfs = np.broadcast_to(
                pdfs_at(self._R[points][:1], scale * plan.nodes.ravel()),
                (means.shape[1], 1, plan.nodes.size))
            density = (means[index] @ pdfs[:, 0]).reshape(plan.nodes.shape)
            column = np.concatenate(
                ([0.0], np.cumsum(np.sum(density * plan.weights, axis=1))))
            out[points] = G * scale * (
                sheets - 4 * np.pi * column[plan.order] * np.sign(z))
        return out


def _legendre(mu: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the Legendre polynomials P_0 to P_n at mu, and their
    derivatives: two arrays of shape (n + 1,) + the shape of mu.
    """
    p = np.zeros((n + 1,) + mu.shape)
    dp = np.zeros_like(p)
    p[0], p[1], dp[1] = 1, mu, 1
    for k in range(1, n):
        p[k + 1] = ((2 * k + 1) * mu * p[k] - k * p[k - 1]) / (k + 1)
        dp[k + 1] = dp[k - 1] + (2 * k + 1) * p[k]
    return p, dp


def _multipoles(moments: np.ndarray) -> np.ndarray:
    """
    Return the multipoles M_l = int rho r^l P_l(cos theta) dV of a disk,
    of the orders _FAR_ORDERS, from its moments int rho R^2j z^2p dV (an
    array of j by p, of R the radius and z the height; see _SOLID).
    """
    return np.array([
        sum(_SOLID[i, j] * moments[j, i - j] for j in range(i + 1))
        for i in range(len(_FAR_ORDERS))])


@functools.lru_cache(maxsize=16)
def _radial_weights(knots_key: bytes, degree: int) -> np.ndarray:
    """
    Return the weights of the radial moments of a disk of outer radius 1:
    2 pi int sigma(a) a^(2j + 1) da = weights[j] @ sigma at its density
    points, for j to _FAR_ORDER / 2 (exact: Gauss-Legendre on its panels).
    """
    knots = np.frombuffer(knots_key) / _KEY_RESOLUTION
    panels = _Panels.from_knots(knots, degree)
    a, w, basis = panels.far_quadrature(np.ones(len(panels.lo), dtype=bool))
    powers = a[..., None] ** (2 * np.arange(len(_FAR_ORDERS)) + 1)
    weights = np.zeros((len(panels.points()), len(_FAR_ORDERS)))
    np.add.at(
        weights, panels.indices,
        2 * np.pi * np.einsum('pn,pnj,nq->pqj', w, powers, basis))
    return weights.T


class _FarField:
    """
    A force of a disk at points far from it (beyond its tables, see the
    module docstring), from its multipole series, Phi = -G sum_l M_l
    P_l(cos theta) / r^(l + 1) (of the even orders: a disk is symmetric
    about its plane), at the spherical radius r and the polar angle theta
    of each point:

        F_R = -G sum_l M_l sin(theta) P'_(l + 1)(cos theta) / r^(l + 2),
        F_z = -G sum_l (l + 1) M_l P_(l + 1)(cos theta) / r^(l + 2).
    """

    def __init__(self, R: np.ndarray, z: np.ndarray, axis: Axis):
        r = np.hypot(R, z)
        p, dp = _legendre(z / r, _FAR_ORDER + 1)
        orders = _FAR_ORDERS[:, None]
        decay = r ** -(orders + 2.0)
        if axis == 'R':
            self._terms = -R / r * dp[_FAR_ORDERS + 1] * decay
        else:
            self._terms = -(orders + 1) * p[_FAR_ORDERS + 1] * decay

    def force(self, moments: np.ndarray) -> np.ndarray:
        """
        Return the force ((km/s)^2/kpc) of the disk of the moments (see
        _multipoles; Msun kpc^(2j + 2p)).
        """
        return G * _multipoles(moments) @ self._terms


def _joined(
        near: np.ndarray, inside: np.ndarray, outside: np.ndarray
) -> np.ndarray:
    """
    Return the forces at the points near a disk (inside its tables) and
    far from it, in the order of the points.
    """
    out = np.empty(near.shape)
    out[near] = inside
    out[~near] = outside
    return out


def _check_pdfs(sigmas: np.ndarray, pdfs: np.ndarray, scale: float) -> None:
    """
    Raise ValueError if a pdf of the height (at the heights _HEIGHTS times
    scale) does not integrate to 1 (within 1e-6) where its surface
    density is not 0 (see _disk_gravity.ThickDiskGravity.vcirc2).
    """
    mass = 2 * scale * (pdfs @ _HEIGHT_WEIGHTS)
    unresolved = (np.abs(mass - 1) > _PDF_TOLERANCE) & (sigmas != 0)
    if np.any(unresolved):
        worst = mass[unresolved][np.argmax(np.abs(mass[unresolved] - 1))]
        raise ValueError(
            f"a height pdf of a thick disk integrates to {worst} over the "
            f"heights of its gravity, not 1: its heights are not resolved "
            f"(e.g. a sharp cut, heavy tails, or a scale height below 1e-8 "
            f"of the outer radius of the disk)")


class DiskForce:
    """
    A force of a disk at given points, from the pull of its rings (see the
    module docstring; build it with disk_force): from its surface
    densities at the radii of its density points (radii), and for a thick
    disk, at the heights of its density (heights, those of _disk_gravity).
    The points near it (near) take its tables, the others its far field.
    """

    def __init__(
            self,
            weights: _SheetWeights,
            radial: np.ndarray,
            geometry: _Geometry,
            R: np.ndarray,
            z: np.ndarray,
            near: np.ndarray,
            axis: Axis
    ):
        scale = geometry.scale
        powers = 2 * np.arange(len(_FAR_ORDERS))
        self._weights = weights
        self._scale = scale
        self._near = near
        self._stack = _Stack(R[near], z[near], geometry.inverse, scale, axis)
        self._far = _FarField(R[~near], z[~near], axis)
        self._radial = radial * scale ** (powers + 2.0)[:, None]
        self._vertical = _VERTICAL * scale ** (powers + 1.0)[:, None]
        self.radii = geometry.points
        self.heights = _HEIGHTS * scale

    def thin(self, sigma: np.ndarray) -> np.ndarray:
        """
        Return the force ((km/s)^2/kpc) of a thin disk of the surface
        density (Msun/kpc^2) at the density points.
        """
        weights = self._weights
        moments = np.zeros((len(_FAR_ORDERS),) * 2)
        moments[:, 0] = self._radial @ sigma
        return _joined(
            self._near,
            self._stack.thin(
                (weights.weights @ sigma).T, weights.means @ sigma),
            self._far.force(moments))

    def thick(
            self,
            sigmas: np.ndarray,
            pdfs: np.ndarray,
            pdfs_at: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Return the force ((km/s)^2/kpc) of a thick disk of the density
        sum_i sigmas[i] pdfs[i].

        Parameters
        ----------
        sigmas : np.ndarray
            The surface densities (Msun/kpc^2) of its terms at the density
            points: an array of shape (n, len(radii)).
        pdfs : np.ndarray
            The pdf of the height (1/kpc) of each term at the density
            points and the heights: an array that broadcasts to the shape
            (n, len(radii), len(heights)).
        pdfs_at : callable
            The pdfs of the height of the terms at the radii R and the
            heights z (kpc) of two arrays: an array that broadcasts to the
            shape (n, len(R), len(z)).

        Raises
        ------
        ValueError
            If a pdf does not integrate to 1 over the heights (within
            1e-6) where its surface density is not 0 (see
            _disk_gravity.ThickDiskGravity.vcirc2).
        """
        weights = self._weights
        pdfs = np.broadcast_to(pdfs, sigmas.shape + (len(_HEIGHTS),))
        _check_pdfs(sigmas, pdfs, self._scale)
        rho = np.sum(sigmas[..., None] * pdfs, axis=0)
        return _joined(
            self._near,
            self._stack.thick(
                lambda index: weights.weights[index] @ rho,
                weights.means @ sigmas.T, pdfs_at),
            self._far.force(self._radial @ rho @ self._vertical.T))


def disk_force(
        axis: Axis,
        knots: np.ndarray,
        R: np.ndarray,
        z: np.ndarray,
        degree: int = 1
) -> DiskForce:
    """
    Return the radial ('R') or the vertical ('z') force of a disk at the
    points (R, z) (kpc, of two arrays of one shape), from the pull of its
    rings: its density is a polynomial in radius of the degree on each
    panel between the knots (kpc, increasing, from the inner to the outer
    edge of the disk) and 0 outside them.
    """
    R = np.asarray(R, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    near = np.abs(z) <= _MAX_HEIGHT * float(knots[-1])
    geometry = _Geometry.from_knots(knots, R[near], degree)
    return DiskForce(
        _sheet_weights(geometry.knots_key, geometry.radii_key, degree, axis),
        _radial_weights(geometry.knots_key, degree), geometry, R, z, near,
        axis)


# The tables C of the sheets of an exponential disk of M = rd = 1 (see
# ExponentialForce): at the radii x = R / rd of _EXP_X, on the heights
# _ZETAS relative to its outer radius _EXP_EXTENT (it is cut there, below
# exp(-40) of its centre), of polynomials of degree 8 on panels of width
# rd / 2
_EXP_EXTENT = 40.0
_EXP_X = np.concatenate(([0], np.geomspace(1e-4, 1e3, 113)))
_LN_EXP_X = np.log(_EXP_X[1:])

# The radial moments of the exponential disk of M = rd = 1, int Sigma
# R^2j dA = (2j + 1)! (for its far field, see _FarField; without its cut,
# which changes the force there far below 1e-11)
_EXP_MOMENTS = scipy.special.factorial(2 * np.arange(len(_FAR_ORDERS)) + 1)


@functools.lru_cache(maxsize=2)
def _exponential_tables(axis: Axis) -> np.ndarray:
    """
    Return the tables C of the exponential disk of M = rd = 1 (per G) for
    a force: an array of shape (len(_EXP_X), len(_ZETAS)).
    """
    knots = np.arange(0, _EXP_EXTENT + 0.25, 0.5)
    geometry = _Geometry.from_knots(knots, _EXP_X, 8)
    # (its weights once, uncached: about 200 MB)
    weights = _sheet_weights.__wrapped__(
        geometry.knots_key, geometry.radii_key, 8, axis)
    sigma = np.exp(-geometry.points) / (2 * np.pi)
    return (weights.weights @ sigma)[geometry.inverse]


def _exponential_table(x: float, axis: Axis) -> np.ndarray:
    """
    Return the table C of the exponential disk of M = rd = 1 at the radius
    x for a force (see _exponential_tables): cubic (Catmull-Rom) in ln x,
    linear in x below _EXP_X[2].
    """
    tables = _exponential_tables(axis)
    if x <= _EXP_X[2]:
        k = 0 if x <= _EXP_X[1] else 1
        t = (x - _EXP_X[k]) / (_EXP_X[k + 1] - _EXP_X[k])
        return tables[k] + (tables[k + 1] - tables[k]) * t
    # (the rows of x > 0 are on a uniform grid of ln x, from row 1)
    step = _LN_EXP_X[1] - _LN_EXP_X[0]
    u = (np.log(x) - _LN_EXP_X[0]) / step + 1
    i = int(np.clip(np.floor(u), 2, len(_EXP_X) - 3))
    t = u - i
    p0, p1, p2, p3 = tables[i - 1:i + 3]
    return p1 + t / 2 * (p2 - p0 + t * (
        2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))


class ExponentialForce:
    """
    A force of an exponential disk, Sigma = M / (2 pi rd^2) exp(-R / rd),
    at given points, of any scale length rd: from the tables C of the
    disk of M = rd = 1, which scale (see _exponential_tables; the weights
    of its rings would change with rd), within 1000 rd of its axis and 5
    outer radii of its plane, and its far field beyond. Build it with
    exponential_force; heights are those of the density of a thick disk.
    """

    def __init__(self, R: np.ndarray, z: np.ndarray, rd: float, axis: Axis):
        scale = _EXP_EXTENT * rd
        near = (np.abs(z) <= _MAX_HEIGHT * scale) & (R <= _EXP_X[-1] * rd)
        radii, inverse = np.unique(R[near] / rd, return_inverse=True)
        self._tables = np.array(
            [_exponential_table(x, axis) for x in radii]
        ).reshape(len(radii), len(_ZETAS)).T
        self._sigma = np.exp(-radii) / (2 * np.pi)
        self._rd = rd
        self._scale = scale
        self._near = near
        self._stack = _Stack(R[near], z[near], inverse, scale, axis)
        self._far = _FarField(R[~near], z[~near], axis)
        powers = 2 * np.arange(len(_FAR_ORDERS))
        self._radial = _EXP_MOMENTS * rd ** powers
        self._vertical = _VERTICAL * scale ** (powers + 1.0)[:, None]
        self.heights = _HEIGHTS * scale

    def thin(self, mass: float) -> np.ndarray:
        """
        Return the force ((km/s)^2/kpc) of a thin disk of the mass (Msun).
        """
        unit = mass / self._rd ** 2
        moments = np.zeros((len(_FAR_ORDERS),) * 2)
        moments[:, 0] = mass * self._radial
        return _joined(
            self._near,
            self._stack.thin(unit * self._tables, unit * self._sigma),
            self._far.force(moments))

    def thick(
            self,
            mass: float,
            pdf: np.ndarray,
            pdf_at: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Return the force ((km/s)^2/kpc) of a thick disk of the mass (Msun)
        and the pdf of the height (1/kpc) at the heights, the same at
        every radius, also given by pdf_at (see DiskForce.thick).

        Raises
        ------
        ValueError
            If the pdf does not integrate to 1 over the heights (see
            DiskForce.thick).
        """
        unit = mass / self._rd ** 2
        _check_pdfs(np.ones(1), pdf[None], self._scale)
        return _joined(
            self._near,
            self._stack.thick(
                lambda index: unit * self._tables[:, index, None] * pdf,
                unit * self._sigma[:, None], pdf_at),
            self._far.force(
                mass * np.outer(self._radial, self._vertical @ pdf)))


def exponential_force(
        axis: Axis, R: np.ndarray, z: np.ndarray, rd: float
) -> ExponentialForce:
    """
    Return the radial ('R') or the vertical ('z') force of an exponential
    disk of the scale length rd (kpc) at the points (R, z) (kpc, of two
    arrays of one shape).
    """
    return ExponentialForce(
        np.asarray(R, dtype=float).ravel(),
        np.asarray(z, dtype=float).ravel(), rd, axis)
