"""
The grids of the sky for orbits (see SkyGrid) of the moments of stars on
orbits in bins (see _orbit_moments.OrbitMomentsPlan): that of the bins
(see bins_sky_grid), and that of the pixels of their grid (see
PixelCube), both through the PSF.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import scipy.signal
import scipy.sparse

from gbkfit.dataset import Regions
from gbkfit.grid import Grid, sky_to_pixel
from gbkfit.instrument import Instrument
from gbkfit.model.components.spheroids import SkyGrid


__all__ = [
    'PixelCube',
    'bins_sky_grid'
]


# The smallest share of a bin in the light of a point of the sky that the
# sky grids of orbits keep (see bins_sky_grid)
_SHARE_FLOOR = 1e-6


class _PSFCells(NamedTuple):
    """
    The PSF on the cells of a grid of the sky for orbits (see _psf_cells):
    its image (ny, nx; a point without a PSF), its half size along y and
    x (the margins of cells around the pixels), and the affine map from
    the sky to the cells (see SkyGrid).
    """
    kernel: np.ndarray
    margins: tuple[int, int]
    to_cells: np.ndarray


def _psf_cells(
        grid: Grid, instrument: Instrument, oversampling: Sequence[int]
) -> _PSFCells:
    """
    Return the PSF of an instrument on the cells of the pixels of a grid,
    of the given oversampling (along x and y, as that of the model), with
    margins of half the size of the PSF around them (see _PSFCells).
    """
    fx, fy = oversampling[:2]
    step = grid.coords.step[:2]
    psf = instrument.psf()
    kernel = np.ones((1, 1)) if psf is None else psf.asarray(
        (step[0] / fx, step[1] / fy), rota=grid.coords.rota)
    my, mx = kernel.shape[0] // 2, kernel.shape[1] // 2
    # (pixel p covers [p - 1/2, p + 1/2), and f cells along each axis)
    matrix, offset = sky_to_pixel(grid)
    f = np.array([fx, fy], dtype=float)
    to_cells = np.column_stack(
        (f[:, None] * matrix, f * (offset + 0.5) + (mx, my)))
    return _PSFCells(kernel, (my, mx), to_cells)


def bins_sky_grid(
        regions: Regions, instrument: Instrument, oversampling: Sequence[int]
) -> SkyGrid:
    """
    Return the grid of the sky of bins for orbits (see SkyGrid), through
    the PSF of an instrument: the pixels of the bins, of the given
    oversampling (along x and y, as that of the model), and around them
    the half size of the PSF; the share of a bin in the light of a point
    in a cell is its mask correlated with the PSF there (the share of the
    light of the point that the PSF puts in it), shares below 1e-6 left
    out.

    Raises
    ------
    ValueError
        If the regions are not bins (on a grid).
    """
    grid = regions.grid()
    if grid is None:
        raise ValueError(
            "orbits are recorded in bins of the pixels of a grid, not in "
            "apertures")
    fx, fy = oversampling[:2]
    psf = _psf_cells(grid, instrument, oversampling)
    my, mx = psf.margins
    index = np.repeat(np.repeat(regions.index(), fy, axis=0), fx, axis=1)
    index = np.pad(index, ((my, my), (mx, mx)), constant_values=-1)
    flipped = psf.kernel[::-1, ::-1]
    cells, bins, shares = [], [], []
    for b in range(regions.nregions()):
        ys, xs = np.nonzero(index == b)
        y0, y1 = ys.min() - my, ys.max() + my + 1
        x0, x1 = xs.min() - mx, xs.max() + mx + 1
        mask = (index[y0:y1, x0:x1] == b).astype(float)
        share = scipy.signal.fftconvolve(mask, flipped, mode='same')
        yy, xx = np.nonzero(share > _SHARE_FLOOR)
        cells.append((yy + y0) * index.shape[1] + xx + x0)
        bins.append(np.full(len(yy), b))
        shares.append(share[yy, xx])
    return SkyGrid(psf.to_cells, index.shape, scipy.sparse.csr_array(
        (np.concatenate(shares),
         (np.concatenate(cells), np.concatenate(bins))),
        shape=(index.size, regions.nregions())))


class PixelCube:
    """
    The pixels of the grid of the bins of an observation as a grid of the
    sky for orbits (see SkyGrid): the cells of the pixels at the given
    oversampling (along x and y) and around them half the size of the PSF,
    each its own target (the spectra of the targets are convolved with
    the PSF after, see cube).

    Parameters
    ----------
    grid : Grid
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
            grid: Grid,
            shape: tuple[int, int],
            instrument: Instrument,
            oversampling: Sequence[int]
    ):
        fx, fy = oversampling[:2]
        step = grid.coords.step[:2]
        psf = _psf_cells(grid, instrument, oversampling)
        self._kernel = psf.kernel
        self._margins = psf.margins
        self.shape = shape
        self._oversampling = (fy, fx)
        self._area = abs(step[0] * step[1])
        my, mx = self._margins
        padded = (shape[0] * fy + 2 * my, shape[1] * fx + 2 * mx)
        ncells = padded[0] * padded[1]
        self.grid = SkyGrid(
            psf.to_cells, padded, scipy.sparse.identity(ncells, format='csr'))

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
