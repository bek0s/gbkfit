"""
The scalar field of Hu-Sawicki f(R) gravity about an axisymmetric galaxy
(see gravity.GravityHuSawicki): its quasi-static equation, solved on a
grid in the spherical radius and the latitude, and the fifth force that
it gives.

Of w = f_R / fbar_R (fbar_R < 0, the field of the background at the epoch
of the galaxy), in physical units (Hu & Sawicki 2007; Oyaizu 2008):

    laplacian(w) = -a (w^(-1/(n+1)) - 1) + b rho

with a = Rbar / (3 |fbar_R|), b = 8 pi G / (3 c^2 |fbar_R|), Rbar the Ricci
scalar of the background at the epoch and rho the density that sources
the field. Its fifth force is (c^2 / 2) grad(f_R) = -(c^2 |fbar_R| / 2)
grad(w). Where the field is unscreened (w ~ 1), it is a third of the
Newtonian force within the Compton wavelength sqrt((n+1) / a); where it
is screened (w ~ 0, where the density is high enough), it vanishes.
"""

import logging
from collections.abc import Callable

import numpy as np
import scipy.interpolate
import scipy.sparse
import scipy.sparse.linalg


__all__ = [
    'FieldGradient',
    'FieldGrid',
    'solve_field'
]


_log = logging.getLogger(__name__)

# The gradient of a field, (dw/dR, dw/dz), at points (R, z)
FieldGradient = Callable[
    [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

# The cells of the grids: log-spaced in radius from the inner radius
# (kpc); in latitude, from the plane to the pole, finer near the plane
# (b_k = pi/2 sinh(beta k / n) / sinh(beta)). The field is solved on
# grids of these widths in ln r and numbers of latitudes, coarse to fine,
# each from the solution of the previous (its screened region in place)
_RMIN = 0.01
_BETA = 5.0
_LEVELS = ((0.32, 5), (0.16, 10), (0.08, 20), (0.04, 40))

# The Newton iterations of the solution on each grid: at most this many,
# until no change of ln w exceeds the tolerance; w grows at most by a
# factor of 1 + step (linear in w), and shrinks at most by e
_MAX_ITERATIONS = 200
_TOLERANCE = 1e-9


class FieldGrid:
    """
    A grid of the field from the centre of a galaxy to the radius rmax
    (kpc), above its plane (the field below it is its mirror image): its
    cells, their volumes, the gradients between them (see flux_matrix),
    and the points that average a density over them.

    Parameters
    ----------
    rmax : float
        The outer radius (kpc), at which the field is that of the
        density there (see solve_field).
    dlnr : float
        The width of the cells in ln r.
    nlat : int
        The number of latitudes.
    """

    def __init__(self, rmax: float, dlnr: float, nlat: int):
        nr = int(np.ceil(np.log(rmax / _RMIN) / dlnr))
        self.r_edges = np.geomspace(_RMIN, rmax, nr + 1)
        self.b_edges = np.pi / 2 * np.sinh(
            _BETA * np.arange(nlat + 1) / nlat) / np.sinh(_BETA)
        self.r = np.sqrt(self.r_edges[1:] * self.r_edges[:-1])
        self.b = (self.b_edges[1:] + self.b_edges[:-1]) / 2
        self.shape = (nr, nlat)
        cubes = np.diff(self.r_edges ** 3)
        sines = np.diff(np.sin(self.b_edges))
        self.volumes = 2 * np.pi / 3 * np.outer(cubes, sines)

    def flux_matrix(self) -> tuple[scipy.sparse.csr_array, np.ndarray]:
        """
        Return the matrix K and the coefficients k of the outer boundary:
        K w + k w_out is the integral of laplacian(w) over each cell (the
        cells flat, the radius first), of w at their centres and w_out at
        the outer edge (at the latitudes of the cells), with no flux
        through the inner edge, the plane and the pole.
        """
        nr, nb = self.shape
        index = np.arange(nr * nb).reshape(nr, nb)
        sines = np.diff(np.sin(self.b_edges))
        # (between radii: area 2 pi r^2 (sin b2 - sin b1), over the
        # distance between the centres)
        radial = (2 * np.pi * self.r_edges[1:-1, None] ** 2 * sines
                  / np.diff(self.r)[:, None])
        # (between latitudes: area 2 pi cos(b) (r2 - r1), of the gradient
        # dw / (r db))
        polar = (2 * np.pi * np.cos(self.b_edges[1:-1])
                 * np.diff(self.r_edges)[:, None] / np.diff(self.b))
        rows = np.concatenate([
            index[:-1].ravel(), index[:, :-1].ravel()])
        cols = np.concatenate([
            index[1:].ravel(), index[:, 1:].ravel()])
        values = np.concatenate([radial.ravel(), polar.ravel()])
        outer = (2 * np.pi * self.r_edges[-1] ** 2 * sines
                 / (self.r_edges[-1] - self.r[-1]))
        diagonal = -np.bincount(
            np.concatenate([rows, cols]), np.concatenate([values, values]),
            minlength=nr * nb)
        diagonal[index[-1]] -= outer
        matrix = scipy.sparse.coo_array(
            (np.concatenate([values, values, diagonal]),
             (np.concatenate([rows, cols, np.arange(nr * nb)]),
              np.concatenate([cols, rows, np.arange(nr * nb)]))),
            shape=(nr * nb, nr * nb)).tocsr()
        boundary = np.zeros(nr * nb)
        boundary[index[-1]] = outer
        return matrix, boundary

    def mean(
            self, function: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Return the mean of a function of (R, z) over each cell, by 2 x 2
        Gauss-Legendre points in r^3 and sin(b) (in which the volume is
        uniform).
        """
        nodes = np.array([-1, 1]) / np.sqrt(3)

        def points(edges: np.ndarray) -> np.ndarray:
            middle, half = (edges[1:] + edges[:-1]) / 2, np.diff(edges) / 2
            return middle[:, None] + half[:, None] * nodes

        r = np.cbrt(points(self.r_edges ** 3))[:, None, :, None]
        b = np.arcsin(points(np.sin(self.b_edges)))[None, :, None, :]
        values = function(r * np.cos(b), r * np.sin(b))
        return values.mean(axis=(2, 3))

    def gradient(self, w: np.ndarray, w_out: np.ndarray) -> FieldGradient:
        """
        Return the gradient of a field (dw/dR, dw/dz) at any points (R, z),
        from its values at the centres of the cells and at the outer edge:
        linear between the centres (from 0 at the centre of the galaxy,
        and in the plane and at the pole as the mirror images give), 0
        beyond the outer edge.
        """
        nr, nb = self.shape
        # (the gradients at the edges, with none through the inner edge,
        # the plane and the pole; at the centres, their means)
        radial = np.zeros((nr + 1, nb))
        radial[1:-1] = np.diff(w, axis=0) / np.diff(self.r)[:, None]
        radial[-1] = (w_out - w[-1]) / (self.r_edges[-1] - self.r[-1])
        polar = np.zeros((nr, nb + 1))
        polar[:, 1:-1] = (np.diff(w, axis=1) / np.diff(self.b)
                          / self.r[:, None])
        dr = (radial[1:] + radial[:-1]) / 2
        db = (polar[:, 1:] + polar[:, :-1]) / 2
        cos, sin = np.cos(self.b), np.sin(self.b)
        grad_R = cos * dr - sin * db
        grad_z = sin * dr + cos * db
        # (nodes: the centre of the galaxy, the plane and the pole)
        nodes_R = np.zeros((nr + 1, nb + 2))
        nodes_z = np.zeros((nr + 1, nb + 2))
        nodes_R[1:, 1:-1], nodes_z[1:, 1:-1] = grad_R, grad_z
        nodes_R[1:, 0] = grad_R[:, 0]
        nodes_z[1:, -1] = grad_z[:, -1]
        axes = (np.concatenate([[0], self.r]),
                np.concatenate([[0], self.b, [np.pi / 2]]))
        interpolate_R = scipy.interpolate.RegularGridInterpolator(
            axes, nodes_R, bounds_error=False, fill_value=0)
        interpolate_z = scipy.interpolate.RegularGridInterpolator(
            axes, nodes_z, bounds_error=False, fill_value=0)

        def gradient(
                R: np.ndarray, z: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            R, z = np.broadcast_arrays(
                np.asarray(R, dtype=float), np.asarray(z, dtype=float))
            points = np.stack(
                [np.hypot(R, z), np.arctan2(np.abs(z), np.abs(R))], axis=-1)
            return interpolate_R(points), np.sign(z) * interpolate_z(points)

        return gradient


def solve_field(
        rmax: float,
        density: Callable[[np.ndarray, np.ndarray], np.ndarray],
        a: float,
        b: float,
        n: float
) -> tuple[FieldGrid, np.ndarray, np.ndarray]:
    """
    Return the field w = f_R / fbar_R of a density (Msun/kpc^3, a function
    of (R, z)), of the constants a (kpc^-2) and b (kpc / Msun) of the field
    equation and the index n of Hu-Sawicki (see the module), to the radius
    rmax (kpc): the grid of its solution, and w at the centres of its cells
    and at its outer edge (at the latitudes of the cells). At the outer
    edge, beyond the Compton wavelength, w is in equilibrium with the
    density there: a (w^(-1/(n+1)) - 1) = b rho.

    Raises
    ------
    RuntimeError
        If the Newton iterations do not converge.
    """
    grid, u = None, None
    for dlnr, nlat in _LEVELS:
        finer = FieldGrid(rmax, dlnr, nlat)
        initial = None if grid is None else _interpolated(grid, u, finer)
        u, w_out = _solve(finer, density, a, b, n, initial)
        grid = finer
    return grid, np.exp(u).reshape(grid.shape), w_out


def _solve(
        grid: FieldGrid,
        density: Callable[[np.ndarray, np.ndarray], np.ndarray],
        a: float,
        b: float,
        n: float,
        initial: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ln w at the centres of the cells of a grid (flat) and w at its
    outer edge (see solve_field), by Newton iterations from an initial ln
    w; by default, from the larger of the field in equilibrium with the
    density of each cell, where it is screened, and that of the equation
    linear in w - 1, where it is not: laplacian(w) = a (w - 1) / (n + 1) +
    b rho.
    """
    power = -1 / (n + 1)
    matrix, boundary = grid.flux_matrix()
    outer = grid.r_edges[-1]
    w_out = (1 + b * density(
        outer * np.cos(grid.b), outer * np.sin(grid.b)) / a) ** -(n + 1)
    boundary = boundary * np.tile(w_out, grid.shape[0])
    volumes = grid.volumes.ravel()
    source = b * grid.mean(density).ravel()
    u = initial
    if u is None:
        linear = 1 + scipy.sparse.linalg.spsolve(
            (matrix - scipy.sparse.diags_array(volumes * a / (n + 1))).tocsc(),
            volumes * source - boundary - matrix @ np.ones(volumes.size))
        u = np.maximum(-(n + 1) * np.log1p(source / a),
                       np.log(np.maximum(linear, np.finfo(float).tiny)))
    for iteration in range(1, _MAX_ITERATIONS + 1):
        w = np.exp(u)
        residual = (matrix @ w + boundary
                    + volumes * (a * (w ** power - 1) - source))
        jacobian = (matrix @ scipy.sparse.diags_array(w)
                    - scipy.sparse.diags_array(
                        volumes * a / (n + 1) * w ** power))
        step = scipy.sparse.linalg.spsolve(jacobian.tocsc(), -residual)
        u = u + np.where(
            step > 0, np.log1p(np.maximum(step, 0)), np.maximum(step, -1))
        if np.max(np.abs(step)) < _TOLERANCE:
            _log.debug(
                f"the field on {grid.shape} cells converged in {iteration} "
                f"iterations")
            return u, w_out
    raise RuntimeError(
        f"the scalar field of f(R) did not converge in {_MAX_ITERATIONS} "
        f"iterations")


def _interpolated(
        grid: FieldGrid, u: np.ndarray, finer: FieldGrid
) -> np.ndarray:
    """
    Return ln w at the centres of the cells of a finer grid (flat), from
    its values at those of a grid (flat): linear in ln r and the latitude,
    the nearest beyond its centres.
    """
    axes = (np.log(grid.r), grid.b)
    interpolate = scipy.interpolate.RegularGridInterpolator(
        axes, u.reshape(grid.shape))
    lnr, b = np.meshgrid(
        np.clip(np.log(finer.r), axes[0][0], axes[0][-1]),
        np.clip(finer.b, axes[1][0], axes[1][-1]), indexing='ij')
    return interpolate(np.stack([lnr, b], axis=-1)).ravel()
