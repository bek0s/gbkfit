"""
Schwarzschild models: the masses of the orbits of a library that best
reproduce the light of a galaxy (in shells of radius, intrinsic, and in
the bins of a field, projected) and its stellar kinematics (the
Gauss-Hermite moments of the LOSVDs of the bins, as linear constraints:
Rix et al. 1997), non-negative, regularised towards the prior masses of
the orbits (as Vasiliev & Valluri 2020).
"""

import dataclasses

import numpy as np
import scipy.linalg

from .gauss_hermite import fit_gauss_hermite, hermite_weights, mass_factor
from .kinematics import StellarKinematics
from .library import OrbitLibrary, Tracer


__all__ = [
    'SchwarzschildFit',
    'fit_weights',
    'regularised_nnls'
]


@dataclasses.dataclass(frozen=True)
class SchwarzschildFit:
    """
    The fit of a Schwarzschild model: the light of each orbit (Lsun), the
    chi-square of the kinematics (the Gauss-Hermite constraints), of the
    light in the bins and in the shells, and the kinematics of the model
    (its light and the Gauss-Hermite series fitted to its LOSVDs, with
    errors of 0).
    """
    weights: np.ndarray
    chi2: float
    chi2_light: float
    chi2_shells: float
    model: StellarKinematics


def regularised_nnls(
        matrix: np.ndarray,
        target: np.ndarray,
        penalties: np.ndarray,
        tolerance: float = 1e-8,
        max_iterations: int = 100
) -> np.ndarray:
    """
    Solve min ||A w - b||^2 + sum_k p_k w_k^2 over w >= 0, with positive
    penalties p.

    By the semismooth Newton method on its dual, in as many variables as
    rows: the solution is w = max(0, A^T y) / p, with y = b - A w (fast
    when there are fewer rows than columns, as in Schwarzschild models).
    The dual is quadratic where the set of positive w is fixed, so a full
    Newton step that keeps that set reaches the solution.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix A, of shape (rows, columns).
    target : np.ndarray
        The vector b, of shape (rows,).
    penalties : np.ndarray
        The penalties p, positive, of shape (columns,).
    tolerance : float, optional
        The tolerance of the residual of y = b - A w, relative to 1 +
        ||b|| (or a full step that keeps the positive w, see above).
    max_iterations : int, optional
        The largest number of Newton iterations.

    Returns
    -------
    np.ndarray
        The solution w.

    Raises
    ------
    RuntimeError
        If it does not converge.
    """
    # The dual, as a minimum: ||y||^2 / 2 + ||max(0, A^T y)||^2_(1/p) / 2
    # - b^T y, smooth and convex; its gradient is y + A w(y) - b
    def objective(y: np.ndarray) -> tuple[float, np.ndarray]:
        w = np.maximum(matrix.T @ y, 0) / penalties
        return float(y @ y + (penalties * w) @ w) / 2 - target @ y, w

    y = target.astype(float)
    value, w = objective(y)
    for _ in range(max_iterations):
        gradient = y + matrix @ w - target
        if np.linalg.norm(gradient) <= tolerance * (
                1 + np.linalg.norm(target)):
            return w
        active = w > 0
        columns = matrix[:, active]
        hessian = np.eye(len(y)) + (columns / penalties[active]) @ columns.T
        step = -scipy.linalg.solve(hessian, gradient, assume_a='pos')
        # (backtracking, for the kinks of max)
        length = 1.0
        while True:
            trial, trial_w = objective(y + length * step)
            if trial <= value + 1e-4 * length * (gradient @ step) \
                    or length < 1e-10:
                break
            length /= 2
        y, value, w = y + length * step, trial, trial_w
        if length == 1 and np.array_equal(w > 0, active):
            return w
    raise RuntimeError(
        f"the regularised NNLS did not converge in {max_iterations} "
        f"iterations")


def fit_weights(
        library: OrbitLibrary,
        kinematics: StellarKinematics,
        tracer: Tracer,
        scale: float = 1.0,
        regularisation: float = 1.0,
        shell_error: float = 0.01
) -> SchwarzschildFit:
    """
    Fit the masses of the orbits of a library to stellar kinematics.

    The constraints, each over its error: the light of the tracer in the
    shells of the library (of relative errors shell_error), the light in
    the bins, and, for each bin, gamma h_m of m = 1 ... M (h_1 = h_2 = 0,
    of errors dV / (sqrt(2) sigma) and dsigma / (sqrt(2) sigma)) in the
    basis of its observed mean and dispersion, gamma = light / (1 + sum
    over even m >= 4 of h_m sqrt(m!) / m!!). The penalty is regularisation
    times sum_k w_k^2 / s_k, with w the masses over the light in the
    shells and s the prior masses of the orbits (1 at the prior).

    Parameters
    ----------
    library : OrbitLibrary
        The library, of the spatial bins of the kinematics.
    kinematics : StellarKinematics
        The kinematics.
    tracer : Tracer
        The light of the galaxy, whose orbits the library holds.
    scale : float, optional
        The factor of the mass of the potential of the library: its
        velocities are scaled by sqrt(scale) (e.g. the ratio of a
        mass-to-light ratio to that of the library, when the stars and
        the black hole are scaled together).
    regularisation : float, optional
        The strength of the regularisation (positive).
    shell_error : float, optional
        The relative error of the light in the shells.

    Returns
    -------
    SchwarzschildFit
        The fit.
    """
    edges = library.velocity_edges * np.sqrt(scale)
    order = 2 + kinematics.h.shape[1]
    # The light of the orbits in the bins, and their gamma h_m (orbits,
    # bins, m = 1 ... M)
    light = library.losvds.sum(axis=2)
    weights = hermite_weights(
        edges, kinematics.mean, kinematics.sigma, order)[..., 1:]
    moments = np.einsum('kbj,bjm->kbm', library.losvds, weights)
    gamma = kinematics.flux / np.array([mass_factor(h) for h in kinematics.h])
    observed = np.column_stack((
        np.zeros((len(gamma), 2)), kinematics.h)) * gamma[:, None]
    errors = np.column_stack((
        kinematics.mean_error / (np.sqrt(2) * kinematics.sigma),
        kinematics.sigma_error / (np.sqrt(2) * kinematics.sigma),
        kinematics.h_error)) * gamma[:, None]
    shell_light = np.diff(tracer.enclosed(library.shell_edges))
    total = shell_light.sum()
    blocks = dict(
        shells=(library.shells.T, shell_light, shell_error * shell_light),
        light=(light.T, kinematics.flux, kinematics.flux_error),
        kinematics=(moments.reshape(len(moments), -1).T, observed.ravel(),
                    errors.ravel()))
    matrix = np.vstack([
        rows * total / error[:, None] for rows, _, error in blocks.values()])
    target = np.concatenate([
        values / error for _, values, error in blocks.values()])
    w = regularised_nnls(
        matrix, target, regularisation / library.prior)
    residuals = matrix @ w - target
    ends = np.cumsum([len(values) for _, values, _ in blocks.values()])
    chi2 = {name: float(np.sum(part ** 2)) for name, part in zip(
        blocks, np.split(residuals, ends[:-1]))}
    return SchwarzschildFit(
        weights=w * total, chi2=chi2['kinematics'],
        chi2_light=chi2['light'], chi2_shells=chi2['shells'],
        model=_model_kinematics(library, kinematics, w * total, edges))


def _model_kinematics(
        library: OrbitLibrary,
        kinematics: StellarKinematics,
        weights: np.ndarray,
        edges: np.ndarray
) -> StellarKinematics:
    """
    Return the kinematics of a model: the light of its bins, and the
    Gauss-Hermite series fitted to their LOSVDs (of the order of the
    kinematics), with errors of 0.
    """
    histograms = np.einsum('k,kbj->bj', weights, library.losvds)
    centres = (edges[1:] + edges[:-1]) / 2
    fits = [fit_gauss_hermite(centres, histogram / np.diff(edges),
                              2 + kinematics.h.shape[1])
            for histogram in histograms]
    zeros = np.zeros(len(histograms))
    return dataclasses.replace(
        kinematics, flux=histograms.sum(axis=1), flux_error=zeros,
        mean=np.array([fit.mean for fit in fits]), mean_error=zeros,
        sigma=np.array([fit.sigma for fit in fits]), sigma_error=zeros,
        h=np.array([fit.h for fit in fits]),
        h_error=np.zeros_like(kinematics.h))
