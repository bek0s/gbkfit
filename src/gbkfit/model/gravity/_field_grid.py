"""
The grid of a field about an axisymmetric galaxy, in the spherical
radius and the latitude of its meridional plane (see FieldGrid), on which
the scalar field of f(R) gravity (see _scalar_field) and the phantom of
QUMOND (see _qumond) are solved.
"""

from collections.abc import Callable

import numpy as np
import scipy.interpolate
import scipy.sparse


__all__ = [
    'CellMasses',
    'FieldGradient',
    'FieldGrid'
]


# The gradient of a field, (dw/dR, dw/dz), at points (R, z)
FieldGradient = Callable[
    [np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

# The masses (Msun) in the cells between the spheres of radii and the
# latitudes (from the plane) of edges, at both sides of the plane
CellMasses = Callable[[np.ndarray, np.ndarray], np.ndarray]

# The cells of the grids: log-spaced in radius from the inner radius
# (kpc); in latitude, from the plane to the pole, finer near the plane
# (b_k = pi/2 sinh(beta k / n) / sinh(beta))
_RMIN = 0.01
_BETA = 5.0


class FieldGrid:
    """
    A grid of the field from near the centre of a galaxy (its inner
    radius) to the radius rmax (kpc), above its plane (the field below it
    is its mirror image): its cells, between radii and latitudes (from
    the plane), their volumes, the areas of their edges, and the
    gradients between them (see flux_matrix).

    Parameters
    ----------
    rmax : float
        The outer radius (kpc), at which the field is given (see
        flux_matrix).
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
        # (the areas of the edges: of the spheres between the radii, 2 pi
        # r^2 (sin b2 - sin b1), from the inner edge to the outer, of
        # shape (nr + 1, nlat); of the cones between the latitudes, 2 pi
        # cos(b) r (r2 - r1), from the plane to the pole, of shape (nr,
        # nlat + 1))
        self.radial_areas = 2 * np.pi * self.r_edges[:, None] ** 2 * sines
        self.polar_areas = 2 * np.pi * np.cos(self.b_edges) * (
            self.r * np.diff(self.r_edges))[:, None]

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
        # (the areas of the edges between the cells, over the distances
        # between their centres: along the radius, and along the
        # latitude, of the gradient dw / (r db))
        radial = self.radial_areas[1:-1] / np.diff(self.r)[:, None]
        polar = (self.polar_areas[:, 1:-1]
                 / (self.r[:, None] * np.diff(self.b)))
        rows = np.concatenate([
            index[:-1].ravel(), index[:, :-1].ravel()])
        cols = np.concatenate([
            index[1:].ravel(), index[:, 1:].ravel()])
        values = np.concatenate([radial.ravel(), polar.ravel()])
        outer = self.radial_areas[-1] / (self.r_edges[-1] - self.r[-1])
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

    def edge_gradients(
            self, w: np.ndarray, w_out: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the gradient of a field across the edges of the cells, from
        its values at their centres and at the outer edge: dw/dr across
        the spheres (of shape (nr + 1, nlat), from the inner edge to the
        outer), and dw / (r db) across the cones (of shape (nr, nlat + 1),
        from the plane to the pole); none across the inner edge, the plane
        and the pole.
        """
        nr, nb = self.shape
        radial = np.zeros((nr + 1, nb))
        radial[1:-1] = np.diff(w, axis=0) / np.diff(self.r)[:, None]
        radial[-1] = (w_out - w[-1]) / (self.r_edges[-1] - self.r[-1])
        polar = np.zeros((nr, nb + 1))
        polar[:, 1:-1] = (np.diff(w, axis=1) / np.diff(self.b)
                          / self.r[:, None])
        return radial, polar

    def gradient(self, w: np.ndarray, w_out: np.ndarray) -> FieldGradient:
        """
        Return the gradient of a field (dw/dR, dw/dz) at any points (R, z),
        from its values at the centres of the cells and at the outer edge:
        linear between the centres (from 0 at the centre of the galaxy,
        and in the plane and at the pole as the mirror images give), 0
        beyond the outer edge.
        """
        nr, nb = self.shape
        # (at the centres, the means of the gradients across their edges)
        radial, polar = self.edge_gradients(w, w_out)
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
