"""
The field of QUMOND, the quasi-linear MOND of Milgrom (2010), about an
axisymmetric galaxy (see gravity.GravityMOND): its gravity is g =
-grad(Phi), of

    laplacian(Phi) = -div[nu(|g_N| / a0) g_N]

of the Newtonian gravity g_N of its mass. It is g_N plus the gravity of
the phantom of the mass, -grad(Phi_ph), of laplacian(Phi_ph) = -div[(nu -
1) g_N]: that solved here, on a grid of the meridional plane (see
FieldGrid), sourced by the Newtonian gravity on the same grid, of the
masses in its cells. Beyond the grid (_RMAX), where the mass is
spherical, the gravity of the phantom is (nu - 1) g_N, as QUMOND is
there.
"""

import functools
import threading
from collections.abc import Callable

import numpy as np
import scipy.sparse.linalg

from ._field_grid import CellMasses, FieldGradient, FieldGrid
from .mass.units import G


__all__ = [
    'phantom_gradient'
]


# The Newtonian forces per unit mass, (F_R, F_z), at points (R, z)
Forces = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

# The grid of the phantom (see FieldGrid): its outer radius (kpc), its
# width of the cells in ln r and its number of latitudes (those of the
# finest grid of f(R) gravity; twice as fine changes the forces by less
# than 1e-3)
_RMAX = 1000.0
_DLNR = 0.04
_NLAT = 40


class _Solver:
    """
    The grid of the phantom and the factors of its flux matrix (see
    FieldGrid.flux_matrix), the same in every evaluation: its solutions,
    of the integrals of the laplacian over the cells, with a field of 0 at
    the outer edge.
    """

    def __init__(self):
        self.grid = FieldGrid(_RMAX, _DLNR, _NLAT)
        matrix, self.boundary = self.grid.flux_matrix()
        self._factors = scipy.sparse.linalg.splu(matrix.tocsc())
        # (its factors are not safe to solve with in two threads at once)
        self._lock = threading.Lock()

    def solve(self, integrals: np.ndarray) -> np.ndarray:
        """
        Return the field at the centres of the cells, of shape (nr, nlat),
        of the integrals of its laplacian over them, of that shape.
        """
        with self._lock:
            w = self._factors.solve(integrals.ravel())
        return w.reshape(self.grid.shape)


@functools.cache
def _solver() -> _Solver:
    """Return the solver of the phantom (made once)."""
    return _Solver()


def phantom_gradient(
        masses: CellMasses,
        forces: Forces,
        boost: Callable[[np.ndarray], np.ndarray]
) -> FieldGradient:
    """
    Return the gradient of the potential of the phantom of a mass (see the
    module), at any points (R, z): of the masses of its cells, its
    Newtonian forces (beyond the grid), and its boost nu(|g_N| / a0) - 1
    of the size of the Newtonian gravity |g_N| ((km/s)^2/kpc; 0 where
    |g_N| is 0).

    The Newtonian potential on the grid is that of the masses of its
    cells (that within its inner radius pulls through its inner edge as
    from the centre) and, at its outer edge, of their total as from the
    centre. Its gravity across each edge between two cells is the
    difference of their potentials, along the edge their mean; the
    phantom is sourced by the flux of (nu - 1) g_N through the edges of
    each cell, and is 0 at the outer edge (its potential is that of the
    mass there, as from the centre, but for a constant).
    """
    solver = _solver()
    grid = solver.grid
    edges = np.concatenate([[0], grid.r_edges])
    cells = masses(edges, grid.b_edges)
    inner, cells = cells[0], cells[1:]
    total = inner.sum() + cells.sum()
    # (the integral of the laplacian of the Newtonian potential over a cell
    # above the plane, 4 pi G times half its mass at both sides)
    cells[0] += inner
    phi_out = -G * total / _RMAX
    newton = solver.solve(
        2 * np.pi * G * cells - (solver.boundary * phi_out).reshape(
            grid.shape))
    radial, polar = grid.edge_gradients(
        newton, np.full(grid.shape[1], phi_out))
    radial[0] = G * inner.sum() / grid.r_edges[0] ** 2
    # (the size of the gravity at the edges: across them, and along them,
    # the mean of that at the centres of the cells at their sides)
    along_radial = _means(_centres(polar, axis=1), axis=0)
    along_polar = _means(_centres(radial, axis=0), axis=1)
    flux_radial = grid.radial_areas * radial * boost(
        np.hypot(radial, along_radial))
    flux_polar = grid.polar_areas * polar * boost(
        np.hypot(polar, along_polar))
    # (none through the inner edge, the plane and the pole: there, the
    # phantom has the flux of its source, or none by symmetry)
    flux_radial[0] = 0
    sources = np.diff(flux_radial, axis=0) + np.diff(flux_polar, axis=1)
    # (of the gravity g_N = -grad(Phi_N): laplacian(Phi_ph) = div[(nu - 1)
    # grad(Phi_N)])
    phantom = solver.solve(sources)
    inside = grid.gradient(phantom, np.zeros(grid.shape[1]))

    def gradient(
            R: np.ndarray, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        R, z = np.broadcast_arrays(
            np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        grad_R, grad_z = inside(R, z)
        if np.any(beyond := np.hypot(R, z) > _RMAX):
            force_R, force_z = forces(R[beyond], z[beyond])
            factor = boost(np.hypot(force_R, force_z))
            grad_R[beyond] = -factor * force_R
            grad_z[beyond] = -factor * force_z
        return grad_R, grad_z

    return gradient


def _centres(edges: np.ndarray, axis: int) -> np.ndarray:
    """Return the means of the values at the two edges of each cell."""
    return (np.take(edges, range(1, edges.shape[axis]), axis)
            + np.take(edges, range(edges.shape[axis] - 1), axis)) / 2


def _means(centres: np.ndarray, axis: int) -> np.ndarray:
    """
    Return the values at the edges between the cells, the means of those
    at their centres, and at the first and the last edge, those of the
    first and the last cell.
    """
    first = np.take(centres, [0], axis)
    last = np.take(centres, [-1], axis)
    return np.concatenate([first, _centres(centres, axis), last], axis)
