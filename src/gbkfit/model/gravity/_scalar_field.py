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

import numpy as np
import scipy.interpolate
import scipy.sparse
import scipy.sparse.linalg

from gbkfit.params import ModelEvaluationError
from ._field_grid import CellMasses, FieldGrid


__all__ = [
    'solve_field'
]


_log = logging.getLogger(__name__)

# The field is solved on grids of these widths in ln r and numbers of
# latitudes (see FieldGrid), coarse to fine, each from the solution of the
# previous (its screened region in place)
_LEVELS = ((0.32, 5), (0.16, 10), (0.08, 20), (0.04, 40))

# The Newton iterations of the solution on each grid: at most this many,
# until no change of ln w exceeds the tolerance; w grows at most by a
# factor of 1 + step (linear in w), and shrinks at most by e
_MAX_ITERATIONS = 200
_TOLERANCE = 1e-9


def solve_field(
        rmax: float,
        masses: CellMasses,
        a: float,
        b: float,
        n: float
) -> tuple[FieldGrid, np.ndarray, np.ndarray]:
    """
    Return the field w = f_R / fbar_R of a mass, of the constants a
    (kpc^-2) and b (kpc / Msun) of the field equation and the index n of
    Hu-Sawicki (see the module), to the radius rmax (kpc): the grid of its
    solution, and w at the centres of its cells and at its outer edge (at
    the latitudes of the cells). The source of each cell is its mean
    density, of its mass (that within the inner radius sources nothing).
    At the outer edge, beyond the Compton wavelength, w is in equilibrium
    with the density of the outer cells: a (w^(-1/(n+1)) - 1) = b rho.

    Raises
    ------
    RuntimeError
        If the Newton iterations do not converge.
    """
    grid, u = None, None
    for dlnr, nlat in _LEVELS:
        finer = FieldGrid(rmax, dlnr, nlat)
        initial = None if grid is None else _interpolated(grid, u, finer)
        u, w_out = _solve(finer, masses, a, b, n, initial)
        grid = finer
    return grid, np.exp(u).reshape(grid.shape), w_out


def _solve(
        grid: FieldGrid,
        masses: CellMasses,
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
    # (the masses at both sides of the plane, the volumes above it)
    density = masses(grid.r_edges, grid.b_edges) / (2 * grid.volumes)
    w_out = (1 + b * density[-1] / a) ** -(n + 1)
    boundary = boundary * np.tile(w_out, grid.shape[0])
    volumes = grid.volumes.ravel()
    source = b * density.ravel()
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
    raise ModelEvaluationError(
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
