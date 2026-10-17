"""
The gravity of axisymmetric disks of any radial profile, thin or thick:
their circular velocity in their plane as the sum of the pull of their
rings (Casertano 1983), with complete elliptic integrals K and E.

The surface density of a thin disk at the radius a, the pull of the ring
at a on the radius R is 2 G Sigma(a) [a K(m) / (a + R) - a E(m) / (a - R)]
da, with m = 4 a R / (a + R)^2, and its integral over a is v^2(R). A thick
disk is its rings at each height h, with D^2 = (a + R)^2 + h^2 and m = 4 a
R / D^2: 4 G rho(a, h) a / D [K(m) - E(m) (a^2 - R^2 + h^2) / ((a - R)^2 +
h^2)] da dh, integrated over a and h >= 0 (the factor 2 has the heights
below the plane).

The radii are cut into panels at knots, where the profile may have kinks
(e.g. the nodes of a linear profile). On each panel, the density is the
polynomial of a given degree through its values at the Gauss-Lobatto
points of the panel (degree 1: linear between the knots), and v^2 is a
linear map of the density at those points (and at fixed heights): weights
that depend on the geometry only, cached. The weights integrate the
kernel against each polynomial of the panels: by Gauss-Legendre
quadrature on panels far from R, and on panels near R on sub-panels of
geometric widths toward R, where the kernel of a thin disk has a
logarithm and a pole, whose principal value is exact (the two sides of R
cancel; an edge of the disk, where the profile jumps, pulls infinitely,
as a thin disk truly does).
"""

import dataclasses
import functools

import numpy as np
import numpy.typing as npt
import scipy.special

from .units import G


__all__ = [
    'ThickDiskGravity',
    'ThinDiskGravity',
    'thick_disk_gravity',
    'thin_disk_gravity',
    'thin_kernel'
]


def _gauss_lobatto(n: int) -> np.ndarray:
    """Return the n Gauss-Lobatto points on [-1, 1]."""
    legendre = np.polynomial.legendre.Legendre.basis(n - 1)
    inner = np.sort(legendre.deriv().roots().real)
    return np.concatenate(([-1], inner, [1]))


def _lagrange(nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Return the Lagrange polynomials of the nodes at the points x: an array
    of shape (len(x), len(nodes)).
    """
    out = np.ones((len(x), len(nodes)))
    for i, node in enumerate(nodes):
        others = np.delete(nodes, i)
        out[:, i] = np.prod(
            (x[:, None] - others) / (node - others), axis=1)
    return out


def _geometric_panels(
        edges: np.ndarray, npoints: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the Gauss-Legendre points of npoints on each panel between the
    edges, and their weights.
    """
    x, w = np.polynomial.legendre.leggauss(npoints)
    lo, hi = edges[:-1, None], edges[1:, None]
    return (((hi - lo) / 2 * x + (hi + lo) / 2).ravel(),
            ((hi - lo) / 2 * w).ravel())


# The quadrature of the far panels (Gauss-Legendre on [-1, 1]), and of the
# near ones (on [0, 1], toward 0: sub-panels of geometric widths)
_FAR_X, _FAR_W = np.polynomial.legendre.leggauss(16)
_NEAR_T, _NEAR_W = _geometric_panels(
    np.concatenate(([0], np.geomspace(1e-12, 1, 30))), 10)

# The heights of thick disks, in units of the outer radius of the disk:
# Gauss-Legendre points on panels of geometric widths from 1e-9 to 10
_HEIGHTS, _HEIGHT_WEIGHTS = _geometric_panels(
    np.concatenate(([0], np.geomspace(1e-9, 10, 22))), 8)

# The singular values of the weights of thick disks that are kept, relative
# to the largest (see _thick_weights)
_THICK_RANK_TOLERANCE = 1e-12

# How close to 1 the integral of a height pdf over the heights must be
_PDF_TOLERANCE = 1e-6

# The resolution of the cache: the radii relative to the outer radius of
# the disk are rounded to multiples of 2^-40
_KEY_RESOLUTION = 2.0 ** 40


def thin_kernel(
        R: npt.ArrayLike, a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Return 2 [a K / (a + R) - (a E - R) / d]: the pull of the ring at a
    (per G and surface density) without its pole -2 R / d, with a
    logarithm at a = R only (d = a - R, given apart to keep its precision
    near R).
    """
    p = (d / (a + R)) ** 2
    e = scipy.special.ellipe(1 - p)
    return 2 * (a * scipy.special.ellipkm1(p) / (a + R)
                - (R * (e - 1) + d * e) / d)


def _thick_kernel(
        R: float, a: np.ndarray, d: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Return 4 a / D [K - E (a^2 - R^2 + h^2) / (d^2 + h^2)]: the pull of
    the ring at a and h (per G and density), for h > 0 (d = a - R, see
    thin_kernel).
    """
    d2 = (a + R) ** 2 + h ** 2
    near2 = d ** 2 + h ** 2
    p = near2 / d2
    return 4 * a / np.sqrt(d2) * (
        scipy.special.ellipkm1(p)
        - scipy.special.ellipe(1 - p) * (d * (a + R) + h ** 2) / near2)


@dataclasses.dataclass(frozen=True)
class _Panels:
    """
    The panels between knots, with the density a polynomial of a degree on
    each: their edges, the Gauss-Lobatto points of the polynomials (on [-1,
    1]), and the index of the density point of each of them on each panel
    (the panels share their edges).
    """
    lo: np.ndarray
    hi: np.ndarray
    nodes: np.ndarray
    indices: np.ndarray

    @classmethod
    def from_knots(cls, knots: np.ndarray, degree: int) -> '_Panels':
        npanels = len(knots) - 1
        indices = np.arange(npanels)[:, None] * degree \
            + np.arange(degree + 1)
        return cls(knots[:-1], knots[1:], _gauss_lobatto(degree + 1), indices)

    def points(self) -> np.ndarray:
        """Return the radii of the density points."""
        width = self.hi - self.lo
        points = self.lo[:, None] + (self.nodes + 1) / 2 * width[:, None]
        return np.concatenate((points[:, :-1].ravel(), self.hi[-1:]))

    def basis(self, panel: int, a: np.ndarray) -> np.ndarray:
        """
        Return the polynomials of a panel at the radii a: an array of shape
        (len(a), degree + 1).
        """
        lo, hi = self.lo[panel], self.hi[panel]
        return _lagrange(self.nodes, 2 * (a - lo) / (hi - lo) - 1)

    def near(self, R: float) -> np.ndarray:
        """
        Return whether each panel is near R: not farther from R than its
        width.
        """
        distance = np.maximum(np.maximum(self.lo - R, R - self.hi), 0)
        return distance <= self.hi - self.lo

    def far_quadrature(
            self, far: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the quadrature points of the given far panels and their
        weights (arrays of shape (panels, points)), and the polynomials of
        the panels at them (an array of shape (points, degree + 1)).
        """
        lo, width = self.lo[far, None], (self.hi - self.lo)[far, None]
        return (lo + (_FAR_X + 1) / 2 * width, _FAR_W / 2 * width,
                _lagrange(self.nodes, _FAR_X))

    def near_quadrature(
            self, panel: int, R: float
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Return the quadrature of a near panel, graded toward R: in two
        pieces if R is inside it, else in one graded toward its end nearest
        R. For each piece, its points a, a - R (apart, to keep its
        precision), their weights, and the polynomials of the panel at
        them.
        """
        lo, hi = self.lo[panel], self.hi[panel]
        if R <= lo:
            pieces = [(lo, hi - lo)]
        elif R >= hi:
            pieces = [(hi, lo - hi)]
        else:
            pieces = [(R, lo - R), (R, hi - R)]
        quadrature = []
        for end, length in pieces:
            # (from the end, over the signed length)
            offset = _NEAR_T * length
            a = end + offset
            quadrature.append((
                a, (end - R) + offset, _NEAR_W * abs(length),
                self.basis(panel, a)))
        return quadrature


@dataclasses.dataclass(frozen=True)
class _ThinWeights:
    """
    The weights of a thin disk of outer radius 1: v^2 = G weights @ sigma
    at the density points, and at each radius the side of the edge of the
    disk it is on (+1 the outer, -1 the inner, 0 neither) and the density
    point there, where a jump of the density pulls infinitely.
    """
    weights: np.ndarray
    edge_sides: np.ndarray
    edge_points: np.ndarray


@dataclasses.dataclass(frozen=True)
class _ThickWeights:
    """
    The weights of a thick disk of outer radius 1, factorised by singular
    value decomposition over the heights: v^2 = G left @ (right @
    rho.T).T.ravel() for the density rho at the density points and the
    heights (left has shape (radii, points * rank)).
    """
    left: np.ndarray
    right: np.ndarray


@functools.lru_cache(maxsize=16)
def _thin_weights(
        knots_key: bytes, radii_key: bytes, degree: int) -> _ThinWeights:
    knots = np.frombuffer(knots_key) / _KEY_RESOLUTION
    radii = np.frombuffer(radii_key) / _KEY_RESOLUTION
    panels = _Panels.from_knots(knots, degree)
    npoints = len(panels.points())
    weights = np.zeros((len(radii), npoints))
    edge_sides = np.zeros(len(radii))
    edge_points = np.zeros(len(radii), dtype=int)
    for j, R in enumerate(radii):
        near = panels.near(R)
        a, w, basis = panels.far_quadrature(~near)
        np.add.at(
            weights[j], panels.indices[~near],
            (w * (thin_kernel(R, a, a - R) - 2 * R / (a - R))) @ basis)
        for panel in np.flatnonzero(near):
            lo, hi = panels.lo[panel], panels.hi[panel]
            at_R = panels.basis(panel, np.array([R]))[0] \
                if lo <= R <= hi else 0
            row = 0
            for a, d, w, basis in panels.near_quadrature(panel, R):
                # The pole by subtraction: int L / (a - R) = int (L -
                # L(R)) / (a - R) + L(R) ln|a - R| at the ends; the ln 0
                # at a knot R cancels with that of the panel on its other
                # side
                row += (w * thin_kernel(R, a, d)) @ basis \
                    - 2 * R * (w / d) @ (basis - at_R)
            if hi > R >= lo:
                row -= 2 * R * at_R * np.log(hi - R)
            if lo < R <= hi:
                row += 2 * R * at_R * np.log(R - lo)
            weights[j, panels.indices[panel]] += row
        if R > 0 and R in (knots[0], knots[-1]):
            edge_sides[j] = 1 if R == knots[-1] else -1
            edge_points[j] = 0 if R == knots[0] else npoints - 1
    return _ThinWeights(weights, edge_sides, edge_points)


@functools.lru_cache(maxsize=8)
def _thick_weights(
        knots_key: bytes, radii_key: bytes, degree: int) -> _ThickWeights:
    knots = np.frombuffer(knots_key) / _KEY_RESOLUTION
    radii = np.frombuffer(radii_key) / _KEY_RESOLUTION
    panels = _Panels.from_knots(knots, degree)
    weights = np.zeros((len(radii), len(panels.points()), len(_HEIGHTS)))
    for j, R in enumerate(radii):
        near = panels.near(R)
        a, w, basis = panels.far_quadrature(~near)
        kernel = _thick_kernel(R, a[..., None], a[..., None] - R, _HEIGHTS)
        np.add.at(
            weights[j], panels.indices[~near],
            np.einsum('pn,pnh,nq->pqh', w, kernel, basis))
        for panel in np.flatnonzero(near):
            for a, d, w, basis in panels.near_quadrature(panel, R):
                kernel = _thick_kernel(R, a[:, None], d[:, None], _HEIGHTS)
                weights[j, panels.indices[panel]] += np.einsum(
                    'n,nh,nq->qh', w, kernel, basis)
    weights *= _HEIGHT_WEIGHTS
    # Most of the weights repeat over the heights: keep the singular
    # values above a tolerance (which keeps the precision of the weights)
    u, s, vt = np.linalg.svd(
        weights.reshape(-1, len(_HEIGHTS)), full_matrices=False)
    rank = int(np.sum(s > _THICK_RANK_TOLERANCE * s[0]))
    return _ThickWeights(
        (u[:, :rank] * s[:rank]).reshape(len(radii), -1), vt[:rank])


@dataclasses.dataclass(frozen=True)
class _Geometry:
    """
    The geometry of a disk at given radii: its outer radius (scale), the
    radii of its density points, the keys of the cache of its knots and of
    its radii (each once), relative to its outer radius, and the index of
    each radius in its radii. Radii closer than the resolution of the
    cache are one.
    """
    scale: float
    points: np.ndarray
    knots_key: bytes
    radii_key: bytes
    inverse: np.ndarray

    @classmethod
    def from_knots(
            cls, knots: np.ndarray, radii: np.ndarray, degree: int
    ) -> '_Geometry':
        knots = np.asarray(knots, dtype=float)
        scale = float(knots[-1])

        def rounded(x: np.ndarray) -> np.ndarray:
            return np.round(x / scale * _KEY_RESOLUTION)
        radii_rounded, inverse = np.unique(
            rounded(np.asarray(radii, dtype=float)), return_inverse=True)
        return cls(
            scale, _Panels.from_knots(knots, degree).points(),
            rounded(knots).tobytes(), radii_rounded.tobytes(), inverse)


class ThinDiskGravity:
    """
    The circular velocity of a thin disk at given radii, from its surface
    density at the radii of its density points (radii, see the module
    docstring; build it with thin_disk_gravity).
    """

    def __init__(self, weights: _ThinWeights, geometry: _Geometry):
        self._weights = weights
        self._scale = geometry.scale
        self._inverse = geometry.inverse
        self.radii = geometry.points

    def vcirc2(self, sigma: np.ndarray) -> np.ndarray:
        """
        Return the squared circular velocity ((km/s)^2) at the radii, from
        the surface density (Msun/kpc^2) at the density points: +-inf at
        an edge of the disk where the density jumps.
        """
        weights = self._weights
        v2 = G * self._scale * (weights.weights @ sigma)
        jump = sigma[weights.edge_points]
        infinite = (weights.edge_sides != 0) & (jump != 0)
        v2[infinite] = np.copysign(
            np.inf, weights.edge_sides[infinite] * jump[infinite])
        return v2[self._inverse]


class ThickDiskGravity:
    """
    The circular velocity of a thick disk at given radii, from its density
    at the radii of its density points (radii) and at fixed heights
    (heights), given as surface densities and height pdfs (see the module
    docstring; build it with thick_disk_gravity).
    """

    def __init__(self, weights: _ThickWeights, geometry: _Geometry):
        self._weights = weights
        self._scale = geometry.scale
        self._inverse = geometry.inverse
        self.radii = geometry.points
        self.heights = _HEIGHTS * geometry.scale

    def vcirc2(self, sigmas: np.ndarray, pdfs: np.ndarray) -> np.ndarray:
        """
        Return the squared circular velocity ((km/s)^2) at the radii, of
        the density sum_i sigmas[i] pdfs[i].

        Parameters
        ----------
        sigmas : np.ndarray
            Surface densities (Msun/kpc^2) at the density points: an
            array of shape (n, len(radii)).
        pdfs : np.ndarray
            The pdf of the height (1/kpc) of each surface density, at the
            density points and the heights: an array that broadcasts to
            the shape (n, len(radii), len(heights)).

        Raises
        ------
        ValueError
            If a pdf does not integrate to 1 over the heights (within
            1e-6) where its surface density is not 0: its heights are not
            resolved (e.g. a sharp cut, heavy tails, or a scale height
            below about 1e-8 of the outer radius of the disk).
        """
        weights = self._weights
        pdfs = np.broadcast_to(
            pdfs, sigmas.shape + (len(_HEIGHTS),))
        mass = 2 * self._scale * (pdfs @ _HEIGHT_WEIGHTS)
        unresolved = (np.abs(mass - 1) > _PDF_TOLERANCE) & (sigmas != 0)
        if np.any(unresolved):
            worst = mass[unresolved][np.argmax(np.abs(
                mass[unresolved] - 1))]
            raise ValueError(
                f"a height pdf of a thick disk integrates to {worst} over "
                f"the heights of its gravity, not 1: its heights are not "
                f"resolved (e.g. a sharp cut, heavy tails, or a scale "
                f"height below 1e-8 of the outer radius of the disk)")
        rho = np.sum(sigmas[..., None] * pdfs, axis=0)
        v2 = G * self._scale ** 2 * (
            weights.left @ (weights.right @ rho.T).T.ravel())
        return v2[self._inverse]

    def height_transform(self, pdf: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Return T(k) = int p(z) exp(-k |z|) dz of a pdf of the height p
        (1/kpc) given at the heights, at the wavenumbers k >= 0 (1/kpc):
        an array of the shape of k, to the precision of the heights (see
        vcirc2).
        """
        weights = 2 * self._scale * _HEIGHT_WEIGHTS * pdf
        k = np.asarray(k, dtype=float)
        return np.exp(-k[..., None] * self.heights) @ weights


def thin_disk_gravity(
        knots: np.ndarray, radii: np.ndarray, degree: int = 1
) -> ThinDiskGravity:
    """
    Return the gravity of a thin disk at the radii (kpc): its surface
    density is a polynomial of the degree on each panel between the knots
    (kpc, increasing, from the inner to the outer edge of the disk) and 0
    outside them.
    """
    geometry = _Geometry.from_knots(knots, radii, degree)
    return ThinDiskGravity(
        _thin_weights(geometry.knots_key, geometry.radii_key, degree),
        geometry)


def thick_disk_gravity(
        knots: np.ndarray, radii: np.ndarray, degree: int = 1
) -> ThickDiskGravity:
    """
    Return the gravity of a thick disk at the radii (kpc): its density is
    a polynomial in radius of the degree on each panel between the knots
    (kpc, increasing, from the inner to the outer edge of the disk) and 0
    outside them.
    """
    geometry = _Geometry.from_knots(knots, radii, degree)
    return ThickDiskGravity(
        _thick_weights(geometry.knots_key, geometry.radii_key, degree),
        geometry)
