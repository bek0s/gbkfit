"""
Stellar kinematics measured in the spatial bins of an IFU field: the
light and the Gauss-Hermite moments of the LOSVD of each bin.
"""

import dataclasses
from collections.abc import Sequence

import astropy.units as u
import numpy as np
import numpy.typing as npt


__all__ = [
    'BinnedField',
    'StellarKinematics',
    'polar_bins'
]


@dataclasses.dataclass(frozen=True)
class BinnedField:
    """
    The spatial bins of an IFU field: the bin of each pixel (an integer
    array of shape (ny, nx), -1 for pixels in none), the size of the
    pixels (arcsec) and the FWHM of its Gaussian PSF (arcsec). The
    centre of the galaxy is at the centre of the pixel grid.
    """
    bin_map: np.ndarray
    pixel: float
    psf_fwhm: float

    def nbins(self) -> int:
        """Return the number of its bins."""
        return int(self.bin_map.max()) + 1

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the sky positions (arcsec) of the centres of its pixels,
        x and y: arrays of the shape of the bin map.
        """
        ny, nx = self.bin_map.shape
        x = (np.arange(nx) - (nx - 1) / 2) * self.pixel
        y = (np.arange(ny) - (ny - 1) / 2) * self.pixel
        return np.meshgrid(x, y)

    def bins_at(self, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray:
        """
        Return the bins of the sky positions (x, y) (arcsec), broadcast
        together: -1 outside the bins.
        """
        ny, nx = self.bin_map.shape
        column = np.floor(np.asarray(x) / self.pixel + nx / 2).astype(int)
        row = np.floor(np.asarray(y) / self.pixel + ny / 2).astype(int)
        inside = (column >= 0) & (column < nx) & (row >= 0) & (row < ny)
        bins = np.full(np.broadcast(column, row).shape, -1)
        bins[inside] = self.bin_map[row[inside], column[inside]]
        return bins


@dataclasses.dataclass(frozen=True)
class StellarKinematics:
    """
    The stellar kinematics of a galaxy in the bins of a field: the light
    of each bin (Lsun) and the Gauss-Hermite series of its LOSVD (its
    mean and dispersion in km/s, and its coefficients h_3 ... h_M), each
    with its error (arrays of shape (bins,), and (bins, M - 2) for h),
    at a distance (Mpc).
    """
    field: BinnedField
    distance: float
    flux: np.ndarray
    flux_error: np.ndarray
    mean: np.ndarray
    mean_error: np.ndarray
    sigma: np.ndarray
    sigma_error: np.ndarray
    h: np.ndarray
    h_error: np.ndarray

    def kpc_per_arcsec(self) -> float:
        """Return the kpc in an arcsec at its distance."""
        return float(self.distance * u.Mpc.to(u.kpc) * u.arcsec.to(u.rad))


def polar_bins(
        shape: tuple[int, int],
        pixel: float,
        edges: Sequence[float],
        sectors: int
) -> np.ndarray:
    """
    Return a bin map of polar bins: a central disk, then annuli, each
    cut into equal sectors.

    Parameters
    ----------
    shape : tuple of int
        The shape of the pixel grid (ny, nx).
    pixel : float
        The size of its pixels (arcsec).
    edges : sequence of float
        The radii of the edges of the disk and the annuli (arcsec),
        increasing from 0.
    sectors : int
        The number of sectors of each annulus.

    Returns
    -------
    np.ndarray
        The bin of each pixel (by its centre), -1 beyond the last edge:
        the disk is bin 0, and the sectors of annulus k (from 1) are the
        bins 1 + (k - 1) sectors + s.
    """
    x, y = BinnedField(np.zeros(shape, dtype=int), pixel, 0).pixel_centres()
    radius = np.hypot(x, y)
    annulus = np.searchsorted(edges, radius, side='right') - 1
    sector = np.floor(
        (np.arctan2(y, x) % (2 * np.pi)) / (2 * np.pi) * sectors).astype(int)
    bins = np.where(annulus == 0, 0, 1 + (annulus - 1) * sectors + sector)
    return np.where(radius < edges[-1], bins, -1)
