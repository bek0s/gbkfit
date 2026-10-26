"""
The extra outputs of the moments of stars on orbits (see
region_moments.OrbitMomentsPlan): the cube of their light in the pixels
of the grid of the bins, and the moments of their light and velocities
in the cells of their spheroid and of each orbit.
"""

from collections.abc import Sequence

import astropy.units as u
import numpy as np
import scipy.signal
import scipy.sparse
from astropy.table import Table

from gbkfit.instrument import Instrument
from gbkfit.model.components.spheroids import OrbitLibraryState, SkyGrid
from gbkfit.utils import gridutils


__all__ = [
    'CELLS_DESCRIPTION',
    'ORBITS_DESCRIPTION',
    'PROFILE_DESCRIPTION',
    'PixelCube',
    'cell_tables',
    'orbit_table'
]


_KMS = u.km / u.s

_MOMENTS_DESCRIPTION = (
    "the light of the orbits (at their solved weights), its density (per "
    "arcsec^3), their mean rotation v_phi, and their dispersions along r, "
    "theta and phi (spherical, theta from the axis) and along R and z "
    "(cylindrical), with the anisotropies beta = 1 - (sigma_theta^2 + "
    "sigma_phi^2) / (2 sigma_r^2) and beta_z = 1 - sigma_z^2 / sigma_R^2; "
    "the orbits as the sky sees them, mirrored about the plane (the means "
    "of v_theta and v_z are 0). The light is in the units of the light of "
    "the spheroid.")

CELLS_DESCRIPTION = (
    "The moments of the stars on orbits in each cell of their spheroid, "
    "between two radii and two angles from the plane (both sides of it): "
    + _MOMENTS_DESCRIPTION + " With the light of the spheroid in the cell "
    "(light_spheroid), which the orbits hold within 1%.")

PROFILE_DESCRIPTION = (
    "The moments of the stars on orbits in each spherical shell of their "
    "spheroid (its cells summed over the angles): " + _MOMENTS_DESCRIPTION)

ORBITS_DESCRIPTION = (
    "Each orbit of the library and its twin (reversed in time): its energy, "
    "its angular momentum about the axis L_z, the radius R_c of the "
    "circular orbit of its energy in the plane and its circularity "
    "lambda_z = L_z / L_c (L_c that of the circular orbit: 1 circular, 0 "
    "radial or polar, negative counter-rotating), its weight (the solved "
    "factor of its share of the light) and its light (in the units of the "
    "light of the spheroid).")


class PixelCube:
    """
    The pixels of the grid of the bins of an observation as a grid of the
    sky for orbits (see SkyGrid): the cells of the pixels at the given
    oversampling (along x and y) and around them half the size of the PSF,
    each its own target (the spectra of the targets are convolved with
    the PSF after, see cube).

    Parameters
    ----------
    grid : gridutils.Grid
        The grid of the bins.
    shape : tuple of int
        Its shape (ny, nx).
    instrument : Instrument
        The instrument of the observation.
    oversampling : Sequence of int
        The oversampling of the model (along x and y).

    Attributes
    ----------
    grid : SkyGrid
        The grid of the sky.
    shape : tuple of int
        The shape (ny, nx) of the pixels.
    """

    def __init__(
            self,
            grid: gridutils.Grid,
            shape: tuple[int, int],
            instrument: Instrument,
            oversampling: Sequence[int]
    ):
        fx, fy = oversampling[:2]
        step = grid.coords.step[:2]
        psf = instrument.psf()
        self._kernel = np.ones((1, 1)) if psf is None else psf.asarray(
            (step[0] / fx, step[1] / fy), rota=grid.coords.rota)
        self._margins = (self._kernel.shape[0] // 2,
                         self._kernel.shape[1] // 2)
        self.shape = shape
        self._oversampling = (fy, fx)
        self._area = abs(step[0] * step[1])
        my, mx = self._margins
        padded = (shape[0] * fy + 2 * my, shape[1] * fx + 2 * mx)
        # (pixel p covers [p - 1/2, p + 1/2), and f cells along each axis;
        # as RegionMoments.sky_grid)
        matrix, offset = gridutils.sky_to_pixel(grid)
        f = np.array([fx, fy], dtype=float)
        cells = np.column_stack(
            (f[:, None] * matrix, f * (offset + 0.5) + (mx, my)))
        ncells = padded[0] * padded[1]
        self.grid = SkyGrid(
            cells, padded, scipy.sparse.identity(ncells, format='csr'))

    def cube(self, spectra: np.ndarray) -> np.ndarray:
        """
        Return the cube (nchannels, ny, nx) of the pixels of the spectra of
        the targets of its grid (an array of shape (nchannels, ncells)):
        convolved with the PSF, summed over the cells of each pixel, and
        per arcsec^2 (as the model cubes of other stars).
        """
        (ny, nx), (fy, fx), (my, mx) = \
            self.shape, self._oversampling, self._margins
        nchannels = len(spectra)
        cube = spectra.reshape(nchannels, *self.grid.shape)
        cube = scipy.signal.fftconvolve(
            cube, self._kernel[None], mode='same', axes=(1, 2))
        cube = cube[:, my:my + ny * fy, mx:mx + nx * fx]
        return cube.reshape(nchannels, ny, fy, nx, fx).sum(axis=(2, 4)) \
            / self._area


def cell_tables(
        state: OrbitLibraryState,
        cell_moments: np.ndarray
) -> tuple[Table, Table]:
    """
    Return the tables of the moments of the light of the orbits of a
    library in its cells and in its spherical shells (see CELLS_DESCRIPTION
    and PROFILE_DESCRIPTION), of their sums in each cell (an array of shape
    (ncell_r, ncell_b, 9), see OrbitRecords.cell_moments).
    """
    radii, angles = state.cell_edges
    k = state.kpc_per_arcsec
    # (the volume of each cell, both sides of the plane: 4 pi / 3 (r_1^3 -
    # r_0^3) (sin b_1 - sin b_0))
    volume = 4 * np.pi / 3 * np.outer(
        np.diff(radii ** 3), np.diff(np.sin(angles)))
    r0, b0 = np.meshgrid(radii[:-1], angles[:-1], indexing='ij')
    r1, b1 = np.meshgrid(radii[1:], angles[1:], indexing='ij')
    cells = Table()
    cells['radius_min'] = r0.ravel() * u.arcsec
    cells['radius_max'] = r1.ravel() * u.arcsec
    cells['radius_min_kpc'] = r0.ravel() * k * u.kpc
    cells['radius_max_kpc'] = r1.ravel() * k * u.kpc
    cells['angle_min'] = np.degrees(b0.ravel()) * u.deg
    cells['angle_max'] = np.degrees(b1.ravel()) * u.deg
    cells['light_spheroid'] = state.cell_light.ravel()
    _add_moments(cells, cell_moments.reshape(-1, 9), volume.ravel())
    profile = Table()
    inner = radii[:-1]
    profile['radius'] = np.where(
        inner > 0, np.sqrt(inner * radii[1:]), radii[1:] / 2) * u.arcsec
    profile['radius_kpc'] = profile['radius'].value * k * u.kpc
    profile['radius_min'] = inner * u.arcsec
    profile['radius_max'] = radii[1:] * u.arcsec
    _add_moments(profile, cell_moments.sum(axis=1), volume.sum(axis=1))
    return cells, profile


def _add_moments(table: Table, sums: np.ndarray, volume: np.ndarray) -> None:
    """
    Add the moments of the light of orbits to a table, of their sums in
    each row (an array of shape (nrows, 9), see OrbitRecords.cell_moments)
    and the volume of each (arcsec^3).
    """
    light = sums[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = sums[:, 1:] / light[:, None]
    vr, vR, vphi, vr2, vtheta2, vphi2, vR2, vz2 = mean.T
    sigma_r2 = vr2 - vr ** 2
    sigma_phi2 = vphi2 - vphi ** 2
    sigma_R2 = vR2 - vR ** 2
    table['light'] = light
    table['density'] = light / volume / u.arcsec ** 3
    table['v_phi'] = vphi * _KMS
    table['sigma_r'] = np.sqrt(sigma_r2) * _KMS
    table['sigma_theta'] = np.sqrt(vtheta2) * _KMS
    table['sigma_phi'] = np.sqrt(sigma_phi2) * _KMS
    table['sigma_R'] = np.sqrt(sigma_R2) * _KMS
    table['sigma_z'] = np.sqrt(vz2) * _KMS
    with np.errstate(divide='ignore', invalid='ignore'):
        table['beta'] = 1 - (vtheta2 + sigma_phi2) / (2 * sigma_r2)
        table['beta_z'] = 1 - vz2 / sigma_R2


def orbit_table(
        state: OrbitLibraryState,
        weights: np.ndarray,
        light: np.ndarray,
        energies: np.ndarray
) -> Table:
    """
    Return the table of the orbits of a library (see ORBITS_DESCRIPTION),
    of the weights and the light of each orbit and its twin (2k and 2k +
    1), and the energy of each orbit at its start ((km/s)^2; see
    OrbitRecords.energies).
    """
    starts = state.starts
    k = state.kpc_per_arcsec
    lz = starts[:, 0] * starts[:, 4] - starts[:, 1] * starts[:, 3]
    # (the circular orbit of each energy in the plane: of the radius where
    # Phi + v_c^2 / 2 is the energy, v_c^2 = dPhi / d(ln r); linear in the
    # energy between the radii of the table, those of its ends beyond)
    table = state.potential
    plane = table.values[:, 0]
    lnr = table.lnr_zero + table.lnr_step * np.arange(len(plane))
    energy = plane[:, 0] + plane[:, 1] / 2
    rc = np.exp(np.interp(energies, energy, lnr))
    lc = rc * np.sqrt(np.interp(energies, energy, plane[:, 1]))
    orbits = Table()
    orbits['start'] = np.repeat(np.arange(len(starts)), 2)
    orbits['twin'] = np.tile([False, True], len(starts))
    orbits['energy'] = np.repeat(energies, 2) * _KMS ** 2
    signed = np.column_stack((lz, -lz)).ravel()
    orbits['L_z'] = signed * u.arcsec * _KMS
    orbits['L_z_kpc'] = signed * k * u.kpc * _KMS
    orbits['R_c'] = np.repeat(rc, 2) * u.arcsec
    orbits['R_c_kpc'] = np.repeat(rc, 2) * k * u.kpc
    orbits['circularity'] = signed / np.repeat(lc, 2)
    orbits['weight'] = np.asarray(weights, dtype=float)
    orbits['light'] = light
    return orbits
