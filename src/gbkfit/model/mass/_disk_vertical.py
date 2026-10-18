"""
The vertical force of axisymmetric mass disks anywhere, F_z(R, z) =
-dPhi/dz (for the jeans equilibrium of the disks of a galaxy, see
parts.Kinematics), from the pull of their rings off their plane, as their
circular velocity in it (see _disk_gravity).

A disk is a stack of thin sheets, at the heights h of its density rho(a,
h): F_z(R, z) = int F_s[rho(., h)](R, z - h) dh, with the field of a
sheet of the surface density sigma(a)

    F_s(R, zeta) = -G zeta int sigma(a) a 4 E(m) / (((a - R)^2 + zeta^2)
                   sqrt((a + R)^2 + zeta^2)) da,

m = 4 a R / ((a + R)^2 + zeta^2) (from int_0^2pi dphi / (A - B cos
phi)^1.5 = 4 E(m) / ((A - B) sqrt(A + B))). Across a sheet, F_s jumps
from 2 pi G sigma(R) to -2 pi G sigma(R), the pull of an infinite sheet
(with the mean of the two sides where sigma jumps, e.g. at the edge of
the disk): F_s = -2 pi G sigma(R) sign(zeta) + C(R, zeta), with C
continuous. C is tabulated at each radius on heights of geometric widths
(_ZETAS), from the surface density at the density points of _disk_gravity
by weights that depend on the geometry only (cached, as those of
_disk_gravity), and interpolated cubically in ln |zeta|; the jumps of the
sheets of a thick disk add up to the pull of its slab, -4 pi G int_0^z
rho(R, h) dh, by quadrature on [0, z], and C to the quadrature over the
heights of the density of _disk_gravity. The integrals over a are by
Gauss-Legendre quadrature on the panels far from R and on sub-panels of
geometric widths toward R near it (those of _disk_gravity), where the
kernel peaks within |zeta| of R: to about 1e-5 of the force (the
interpolation of C), or for the tables of an exponential disk, 1e-4
near it and 1e-3 far from it (where its pull is weak).

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

import numpy as np
import scipy.sparse
import scipy.special

from ._disk_gravity import (
    _HEIGHT_WEIGHTS, _HEIGHTS, _KEY_RESOLUTION, _PDF_TOLERANCE, _Geometry,
    _Panels)
from .units import G


__all__ = [
    'DiskVerticalForce',
    'ExponentialVerticalForce',
    'disk_vertical_force',
    'exponential_vertical_force'
]


# The heights zeta of the tables of C (see the module docstring), relative
# to the outer radius of the disk: 32 per decade from 1e-9 to 16 (beyond
# the heights of a thick disk, _HEIGHTS, and of the points asked for);
# below, C falls linearly to 0
_ZETAS = np.geomspace(1e-9, 16, 327)
_LN_ZETAS = np.log(_ZETAS)

# The highest point (relative to the outer radius of the disk) whose force
# the tables reach, with the heights of a thick disk
_MAX_HEIGHT = 5.0

# The quadrature of the pull of the slab of a thick disk on [0, z], on
# panels of geometric widths toward 0 (relative to z)
_SLAB_X, _SLAB_W = (
    lambda x, w, edges: (
        (((edges[1:, None] - edges[:-1, None]) * x
          + edges[1:, None] + edges[:-1, None]) / 2).ravel(),
        ((edges[1:, None] - edges[:-1, None]) * w / 2).ravel()))(
    *np.polynomial.legendre.leggauss(8),
    np.concatenate(([0], np.geomspace(1e-9, 1, 22))))


def _sheet_kernel(
        a: np.ndarray, d: np.ndarray, R: float, zeta: np.ndarray
) -> np.ndarray:
    """
    Return zeta 4 a E(m) / ((d^2 + zeta^2) sqrt((a + R)^2 + zeta^2)), the
    pull of the ring at a of a sheet on the point at the height zeta above
    it (per G and surface density), d = a - R apart (to keep its
    precision near R).
    """
    near2 = d ** 2 + zeta ** 2
    far2 = (a + R) ** 2 + zeta ** 2
    return zeta * 4 * a * scipy.special.ellipe(1 - near2 / far2) \
        / (near2 * np.sqrt(far2))


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
    The weights of the sheets of a disk of outer radius 1 at given radii:
    C_j(zeta_n) = G weights[j, n] @ sigma at the density points (see the
    module docstring), and the mean polynomials at each radius (means[j]
    @ sigma, see _mean_basis).
    """
    weights: np.ndarray
    means: np.ndarray


@functools.lru_cache(maxsize=8)
def _sheet_weights(
        knots_key: bytes, radii_key: bytes, degree: int) -> _SheetWeights:
    knots = np.frombuffer(knots_key) / _KEY_RESOLUTION
    radii = np.frombuffer(radii_key) / _KEY_RESOLUTION
    panels = _Panels.from_knots(knots, degree)
    npoints = len(panels.points())
    weights = np.zeros((len(radii), npoints, len(_ZETAS)))
    means = np.zeros((len(radii), npoints))
    for j, R in enumerate(radii):
        near = panels.near(R)
        a, w, basis = panels.far_quadrature(~near)
        kernel = _sheet_kernel(
            a[..., None], a[..., None] - R, R, _ZETAS)
        np.add.at(
            weights[j], panels.indices[~near],
            -np.einsum('pn,pnz,nq->pqz', w, kernel, basis))
        for panel in np.flatnonzero(near):
            for a, d, w, basis in panels.near_quadrature(panel, R):
                kernel = _sheet_kernel(a[:, None], d[:, None], R, _ZETAS)
                weights[j, panels.indices[panel]] -= np.einsum(
                    'n,nz,nq->qz', w, kernel, basis)
        means[j] = _mean_basis(panels, npoints, R)
        # (C: the jump of the sheet taken out, above it)
        weights[j] += 2 * np.pi * means[j][:, None]
    return _SheetWeights(
        np.ascontiguousarray(weights.transpose(0, 2, 1)), means)


def _interpolate(tables: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """
    Return C at the heights zeta (relative to the outer radius of the
    disk; any sign, C odd in it) from its tables at _ZETAS: an array of
    the shape (len(_ZETAS),) + S, with S of the dimensions of zeta and
    broadcast with it (a table for each point). Cubic (Catmull-Rom) in ln
    |zeta|, linear to 0 below _ZETAS[0].
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
    value = np.where(size < _ZETAS[0], tables[0] * size / _ZETAS[0], value)
    return np.sign(zeta) * value


def _catmull_rom(zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the rows of _ZETAS and their coefficients that interpolate C at
    the heights zeta (see _interpolate): arrays of the shape of zeta and 4
    (rows i - 1 to i + 2), the sign of zeta in the coefficients.
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
    low = size < _ZETAS[0]
    rows = np.where(low[..., None], 0, rows)
    coefficients = np.where(
        low[..., None], np.array([1.0, 0, 0, 0]) * (size / _ZETAS[0])[
            ..., None], coefficients)
    return rows, coefficients * np.sign(zeta)[..., None]


@dataclasses.dataclass(frozen=True)
class _ThickPlan:
    """
    How the sheets and the slab of a thick disk are summed at the points
    of one radius (see _Stack.thick), from its geometry: the sum of C of
    its sheets at +h and -h, times the weights of the heights, as a
    sparse map of the table of C at the radius (sheets: of shape
    (points, len(_ZETAS) * len(_HEIGHTS))); and the quadrature of the
    slab between the sorted |z| of the points (nodes and weights on each
    interval from the last, relative to the outer radius of the disk, and
    the index of each point among the sorted heights).
    """
    sheets: scipy.sparse.csr_array
    nodes: np.ndarray
    weights: np.ndarray
    order: np.ndarray


@functools.lru_cache(maxsize=64)
def _thick_plan(heights_key: bytes) -> _ThickPlan:
    z = np.frombuffer(heights_key) / _KEY_RESOLUTION
    npoints, nheights = len(z), len(_HEIGHTS)
    sheets = []
    for side in (-1, 1):
        rows, coefficients = _catmull_rom(z[:, None] + side * _HEIGHTS)
        sheets.append(scipy.sparse.csr_array(
            ((coefficients * _HEIGHT_WEIGHTS[:, None]).ravel(), (
                np.repeat(np.arange(npoints), nheights * 4),
                (rows * nheights + np.arange(nheights)[:, None]).ravel())),
            shape=(npoints, len(_ZETAS) * nheights)))
    sizes, order = np.unique(np.abs(z), return_inverse=True)
    lo = np.concatenate(([0], sizes[:-1]))[:, None]
    hi = sizes[:, None]
    x, w = np.polynomial.legendre.leggauss(8)
    return _ThickPlan(
        sheets[0] + sheets[1], ((hi - lo) * x + hi + lo) / 2,
        (hi - lo) * w / 2, order)


class _Stack:
    """
    The points (R, z) (kpc, flat arrays) at which the vertical force of a
    disk is evaluated, the index of the radius of each among those of the
    tables of its sheets (inverse), and the outer radius of the disk, the
    unit of the heights of the tables (scale): the sums of the sheets of
    the disk there (see the module docstring).
    """

    def __init__(
            self,
            R: np.ndarray,
            z: np.ndarray,
            inverse: np.ndarray,
            scale: float
    ):
        if np.any(np.abs(z) > _MAX_HEIGHT * scale):
            raise ValueError(
                f"the vertical force of a disk is given up to {_MAX_HEIGHT} "
                f"times its outer radius ({scale:g} kpc) from its plane; a "
                f"point is at {np.max(np.abs(z)):g} kpc")
        self._R = R
        self._z = z
        self._inverse = inverse
        self._scale = scale

    def thin(self, tables: np.ndarray, means: np.ndarray) -> np.ndarray:
        """
        Return the vertical force ((km/s)^2/kpc) of a thin disk of the
        tables C (per G, an array of shape (len(_ZETAS), radii)) and the
        mean surface densities (Msun/kpc^2) at the radii of the tables.
        """
        z = self._z
        return G * (
            _interpolate(tables[:, self._inverse], z / self._scale)
            - 2 * np.pi * means[self._inverse] * np.sign(z))

    def thick(
            self,
            tables: Callable[[int], np.ndarray],
            means: np.ndarray,
            pdfs_at: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Return the vertical force ((km/s)^2/kpc) of a thick disk of the
        tables C of the sheets at the heights _HEIGHTS (per G, an array of
        shape (len(_ZETAS), len(_HEIGHTS)) for the index of a radius of
        the tables), the mean surface densities (Msun/kpc^2) of its terms
        at the radii of the tables (an array of shape (radii, terms)), and
        the pdfs of the height of its terms at the radius R and the
        heights z (kpc) of two arrays (see DiskVerticalForce.thick).
        """
        scale = self._scale
        out = np.zeros_like(self._z)
        for index in np.unique(self._inverse):
            points = self._inverse == index
            z = self._z[points]
            plan = _thick_plan(np.round(
                z / scale * _KEY_RESOLUTION).tobytes())
            # The sheets at +h and -h, without their jumps, and the pull of
            # the slab, int_0^|z| rho(R, h) dh (with the mean densities),
            # summed between the sorted heights
            sheets = plan.sheets @ tables(index).ravel()
            pdfs = np.broadcast_to(
                pdfs_at(self._R[points][:1], scale * plan.nodes.ravel()),
                (means.shape[1], 1, plan.nodes.size))
            density = (means[index] @ pdfs[:, 0]).reshape(plan.nodes.shape)
            column = np.cumsum(np.sum(density * plan.weights, axis=1))
            out[points] = G * scale * (
                sheets - 4 * np.pi * column[plan.order] * np.sign(z))
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


class DiskVerticalForce:
    """
    The vertical force of a disk at given points, from the pull of its
    rings (see the module docstring; build it with disk_vertical_force):
    from its surface densities at the radii of its density points
    (radii), and for a thick disk, at the heights of its density
    (heights, those of _disk_gravity).
    """

    def __init__(
            self,
            weights: _SheetWeights,
            geometry: _Geometry,
            R: np.ndarray,
            z: np.ndarray
    ):
        self._weights = weights
        self._scale = geometry.scale
        self._stack = _Stack(R, z, geometry.inverse, geometry.scale)
        self.radii = geometry.points
        self.heights = _HEIGHTS * geometry.scale

    def thin(self, sigma: np.ndarray) -> np.ndarray:
        """
        Return the vertical force ((km/s)^2/kpc) of a thin disk of the
        surface density (Msun/kpc^2) at the density points.
        """
        weights = self._weights
        return self._stack.thin(
            (weights.weights @ sigma).T, weights.means @ sigma)

    def thick(
            self,
            sigmas: np.ndarray,
            pdfs: np.ndarray,
            pdfs_at: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Return the vertical force ((km/s)^2/kpc) of a thick disk of the
        density sum_i sigmas[i] pdfs[i].

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
        return self._stack.thick(
            lambda index: weights.weights[index] @ rho,
            weights.means @ sigmas.T, pdfs_at)


def disk_vertical_force(
        knots: np.ndarray, R: np.ndarray, z: np.ndarray, degree: int = 1
) -> DiskVerticalForce:
    """
    Return the vertical force of a disk at the points (R, z) (kpc, of two
    arrays of one shape), from the pull of its rings: its density is a
    polynomial in radius of the degree on each panel between the knots
    (kpc, increasing, from the inner to the outer edge of the disk) and 0
    outside them.

    Raises
    ------
    ValueError
        If a point is more than 5 times the outer radius of the disk from
        its plane.
    """
    R = np.asarray(R, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    geometry = _Geometry.from_knots(knots, R, degree)
    return DiskVerticalForce(
        _sheet_weights(geometry.knots_key, geometry.radii_key, degree),
        geometry, R, z)


# The tables C of the sheets of an exponential disk of M = rd = 1 (see
# ExponentialVerticalForce): at the radii x = R / rd of _EXP_X, on the
# heights _ZETAS relative to its outer radius _EXP_EXTENT (it is cut there,
# below exp(-40) of its centre), of polynomials of degree 8 on panels of
# width rd / 2
_EXP_EXTENT = 40.0
_EXP_X = np.concatenate(([0], np.geomspace(1e-4, 1e3, 113)))
_LN_EXP_X = np.log(_EXP_X[1:])


@functools.lru_cache(maxsize=1)
def _exponential_tables() -> np.ndarray:
    """
    Return the tables C of the exponential disk of M = rd = 1 (per G): an
    array of shape (len(_EXP_X), len(_ZETAS)).
    """
    knots = np.arange(0, _EXP_EXTENT + 0.25, 0.5)
    geometry = _Geometry.from_knots(knots, _EXP_X, 8)
    weights = _sheet_weights(geometry.knots_key, geometry.radii_key, 8)
    sigma = np.exp(-geometry.points) / (2 * np.pi)
    return (weights.weights @ sigma)[geometry.inverse]


def _exponential_table(x: float) -> np.ndarray:
    """
    Return the table C of the exponential disk of M = rd = 1 at the radius
    x (see _exponential_tables): cubic (Catmull-Rom) in ln x, linear in x
    below _EXP_X[2].
    """
    tables = _exponential_tables()
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


class ExponentialVerticalForce:
    """
    The vertical force of an exponential disk, Sigma = M / (2 pi rd^2)
    exp(-R / rd), at given points, of any scale length rd: from the tables
    C of the disk of M = rd = 1, which scale (see _exponential_tables; the
    weights of its rings would change with rd). Build it with
    exponential_vertical_force; heights are those of the density of a
    thick disk.
    """

    def __init__(self, R: np.ndarray, z: np.ndarray, rd: float):
        radii, inverse = np.unique(R / rd, return_inverse=True)
        if radii[-1] > _EXP_X[-1]:
            raise ValueError(
                f"the vertical force of an exponential disk is given within "
                f"{_EXP_X[-1]:g} times its scale length; a point is at "
                f"{radii[-1]:g} times it")
        self._tables = np.array([_exponential_table(x) for x in radii]).T
        self._sigma = np.exp(-radii) / (2 * np.pi)
        self._rd = rd
        self._scale = _EXP_EXTENT * rd
        self._stack = _Stack(R, z, inverse, self._scale)
        self.heights = _HEIGHTS * self._scale

    def thin(self, mass: float) -> np.ndarray:
        """
        Return the vertical force ((km/s)^2/kpc) of a thin disk of the
        mass (Msun).
        """
        unit = mass / self._rd ** 2
        return self._stack.thin(unit * self._tables, unit * self._sigma)

    def thick(
            self,
            mass: float,
            pdf: np.ndarray,
            pdf_at: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Return the vertical force ((km/s)^2/kpc) of a thick disk of the
        mass (Msun) and the pdf of the height (1/kpc) at the heights, the
        same at every radius, also given by pdf_at (see
        DiskVerticalForce.thick).

        Raises
        ------
        ValueError
            If the pdf does not integrate to 1 over the heights (see
            DiskVerticalForce.thick).
        """
        unit = mass / self._rd ** 2
        _check_pdfs(np.ones(1), pdf[None], self._scale)
        return self._stack.thick(
            lambda index: unit * self._tables[:, index, None] * pdf,
            unit * self._sigma[:, None], pdf_at)


def exponential_vertical_force(
        R: np.ndarray, z: np.ndarray, rd: float
) -> ExponentialVerticalForce:
    """
    Return the vertical force of an exponential disk of the scale length
    rd (kpc) at the points (R, z) (kpc, of two arrays of one shape).

    Raises
    ------
    ValueError
        If a point is beyond 1000 rd from the axis, or more than 200 rd
        from the plane.
    """
    return ExponentialVerticalForce(
        np.asarray(R, dtype=float).ravel(),
        np.asarray(z, dtype=float).ravel(), rd)
