"""
The binning of maps and cubes for the region datasets: Voronoi bins
of a map to a target signal-to-noise, the index of the bins of a map,
and the values, the fluxes and the spectra of the bins.
"""

import logging
from typing import Literal

import astropy.io.fits as fits
import astropy.units
import astropy.wcs
import astropy.wcs.utils
import numpy as np
import scipy.spatial

import gbkfit.units
from gbkfit.io import Beam
from .arrays import FitsArrays


__all__ = [
    'bin_index',
    'bin_spectra',
    'bin_values',
    'pixel_fluxes',
    'voronoi_bins'
]


_log = logging.getLogger(__name__)


# The bin accretion of Voronoi binning (Cappellari & Copin 2003): a bin
# stops growing above this roundness, and is kept if its S/N reaches this
# fraction of the target (the pixels of the others join the nearest bin)
_VORONOI_MAX_ROUNDNESS = 0.3
_VORONOI_KEPT_FRACTION = 0.8

# The most iterations of the tessellation of Voronoi binning
_VORONOI_MAX_ITERATIONS = 1000


def voronoi_bins(
        signal: np.ndarray,
        noise: np.ndarray,
        target: float,
        tessellation: Literal['wvt', 'cvt'] | None = 'wvt'
) -> np.ndarray:
    """
    Return Voronoi bins of a map of about a target signal-to-noise each
    (Cappellari & Copin 2003): the bin of each pixel, as the bins of
    region datasets (see bin_index and the regions 'bins'). The pixels of
    finite signal and finite positive noise are binned; the S/N of a bin
    is the sum of the signal of its pixels over the sum of their noise in
    quadrature (of uncorrelated noise).

    Bin accretion makes the bins, from the pixel of highest S/N: a bin
    grows by the pixel nearest its centroid while that pixel is next to
    it, the bin stays round (roundness at most 0.3) and its S/N gets
    closer to the target; each next bin starts at the pixel nearest the
    centroid of the pixels binned so far. The pixels of the bins below
    80% of the target join the nearest bin kept. The iterations of a
    tessellation then even out the S/N of the bins and their shapes: of a
    weighted Voronoi tessellation ('wvt', Diehl & Statler 2006), or of a
    centroidal one whose bins hold about the same (S/N)^2 ('cvt',
    Cappellari & Copin 2003); None keeps the bins of the accretion.

    Parameters
    ----------
    signal : np.ndarray
        The signal of each pixel (a map).
    noise : np.ndarray
        The noise of each pixel (its standard deviation).
    target : float
        The target S/N of the bins.
    tessellation : {'wvt', 'cvt'} or None, optional
        The tessellation of the iterations.

    Returns
    -------
    np.ndarray
        The bin of each pixel (int32), from 0; -1 for no bin.

    Raises
    ------
    ValueError
        If the signal and the noise are not maps of the same shape, the
        tessellation is unknown, or no bin reaches the target.
    """
    signal = np.asarray(signal, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if signal.ndim != 2 or signal.shape != noise.shape:
        raise ValueError(
            f"the signal and the noise must be maps of the same shape; "
            f"they have shapes {signal.shape} and {noise.shape}")
    if tessellation not in ('wvt', 'cvt', None):
        raise ValueError(
            f"the tessellation must be 'wvt', 'cvt' or None, not "
            f"{tessellation!r}")
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(signal) & np.isfinite(noise) & (noise > 0)
    total = signal[valid].sum() / np.sqrt(np.sum(noise[valid] ** 2))
    if not total >= target:
        raise ValueError(
            f"all the pixels together have a S/N of {total:.3g}, below the "
            f"target ({target})")
    labels, kept = _voronoi_accretion(signal, noise, valid, target)
    if not kept.any():
        raise ValueError(
            f"no bin of the accretion reaches "
            f"{_VORONOI_KEPT_FRACTION:.0%} of the target ({target})")
    # (np.argwhere goes through the pixels in the order of labels[valid])
    points = np.argwhere(valid).astype(float)
    labels = labels[valid]
    lost = ~kept[labels]
    labels = (np.cumsum(kept) - 1)[labels]
    generators = _centroids(points[~lost], labels[~lost])
    labels[lost] = _nearest_generators(
        points[lost], generators, np.ones(len(generators)))
    signal, noise2 = signal[valid], noise[valid] ** 2
    if tessellation is not None:
        labels = _voronoi_tessellation(
            points, signal, noise2, labels, tessellation)
    bins = np.full(valid.shape, -1, dtype=np.int32)
    bins[valid] = labels
    _log.info(f"{labels.max() + 1} Voronoi bins")
    return bins


def _voronoi_accretion(
        signal: np.ndarray,
        noise: np.ndarray,
        valid: np.ndarray,
        target: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the bins of the bin accretion of voronoi_bins: the bin of each
    pixel (from 0; -2 for the invalid pixels), and whether each bin is
    kept.
    """
    labels = np.where(valid, -1, -2)
    members = np.empty((np.count_nonzero(valid), 2))
    snr = np.where(valid, signal / np.where(valid, noise, 1), -np.inf)
    start = np.array(np.unravel_index(np.argmax(snr), snr.shape))
    binned, nbinned = np.zeros(2), 0
    kept = []
    while True:
        label, size = len(kept), 1
        members[0] = start
        labels[tuple(start)] = label
        sum_signal = signal[tuple(start)]
        sum_noise2 = noise[tuple(start)] ** 2
        sn = sum_signal / np.sqrt(sum_noise2)
        centre = members[0].copy()
        # (no pixel brings a bin at the target closer to it)
        while sn < target:
            # (a pixel next to the bin is within r_max + 1 of its centre)
            radius = np.sqrt(np.max(
                np.sum((members[:size] - centre) ** 2, axis=1))) + 1.5
            pixel = _nearest_free(labels, centre, radius)
            if pixel is None or not _touches(labels, pixel, label):
                break
            members[size] = pixel
            grown = members[:size + 1]
            grown_centre = grown.mean(axis=0)
            r_max = np.sqrt(np.max(np.sum((grown - grown_centre) ** 2, axis=1)))
            roundness = r_max / np.sqrt((size + 1) / np.pi) - 1
            pixel = tuple(pixel)
            grown_sn = (sum_signal + signal[pixel]) / np.sqrt(
                sum_noise2 + noise[pixel] ** 2)
            if (roundness > _VORONOI_MAX_ROUNDNESS
                    or abs(grown_sn - target) >= abs(sn - target)):
                break
            labels[pixel] = label
            size += 1
            sum_signal += signal[pixel]
            sum_noise2 += noise[pixel] ** 2
            sn, centre = grown_sn, grown_centre
        kept.append(sn >= _VORONOI_KEPT_FRACTION * target)
        binned += members[:size].sum(axis=0)
        nbinned += size
        free = np.argwhere(labels == -1)
        if not free.size:
            return labels, np.array(kept)
        start = free[np.argmin(
            np.sum((free - binned / nbinned) ** 2, axis=1))]


def _nearest_free(
        labels: np.ndarray, point: np.ndarray, radius: float
) -> np.ndarray | None:
    """
    Return the pixel without a bin yet (label -1) nearest a point, if one
    is within a radius of it.
    """
    lower = np.maximum(np.floor(point - radius).astype(int), 0)
    upper = np.minimum(
        np.ceil(point + radius).astype(int) + 1, labels.shape)
    window = labels[lower[0]:upper[0], lower[1]:upper[1]]
    free = np.argwhere(window == -1) + lower
    if not free.size:
        return None
    distances = np.sum((free - point) ** 2, axis=1)
    nearest = np.argmin(distances)
    return free[nearest] if distances[nearest] <= radius ** 2 else None


def _touches(labels: np.ndarray, pixel: np.ndarray, label: int) -> bool:
    """Return whether a pixel is next to one of a bin, along an axis."""
    y, x = pixel
    ny, nx = labels.shape
    return ((y > 0 and labels[y - 1, x] == label)
            or (y + 1 < ny and labels[y + 1, x] == label)
            or (x > 0 and labels[y, x - 1] == label)
            or (x + 1 < nx and labels[y, x + 1] == label))


def _voronoi_tessellation(
        points: np.ndarray,
        signal: np.ndarray,
        noise2: np.ndarray,
        labels: np.ndarray,
        tessellation: Literal['wvt', 'cvt']
) -> np.ndarray:
    """
    Return the bins of points (from 0) after the iterations of the
    tessellation of voronoi_bins from bins, until the bins stop changing;
    of the signal and noise squared of the points.
    """
    tiny = np.finfo(float).tiny
    # (of a centroidal tessellation of density rho, rho A^2 is about the
    # same for every bin of area A; so with rho = (S/N)^4 per pixel, so is
    # the (S/N)^2 of the bins, Cappellari & Copin 2003, Sect. 4)
    density = (np.maximum(signal ** 4 / noise2 ** 2, tiny)
               if tessellation == 'cvt' else np.ones(len(points)))
    for iteration in range(1, _VORONOI_MAX_ITERATIONS + 1):
        generators = _centroids(points, labels, density)
        scales = np.ones(len(generators))
        if tessellation == 'wvt':
            # (Diehl & Statler 2006, eq. 4)
            sn = (np.bincount(labels, signal)
                  / np.sqrt(np.bincount(labels, noise2)))
            scales = np.sqrt(np.bincount(labels) / np.maximum(sn, tiny))
        nearest = _nearest_generators(points, generators, scales)
        # (without the bins left empty)
        nearest = np.unique(nearest, return_inverse=True)[1]
        if np.array_equal(nearest, labels):
            _log.info(f"the Voronoi bins converged in {iteration} iterations")
            return labels
        labels = nearest
    _log.warning(
        f"the Voronoi bins did not converge in {_VORONOI_MAX_ITERATIONS} "
        f"iterations")
    return labels


def _centroids(
        points: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None = None
) -> np.ndarray:
    """Return the centroid of the points of each label, from 0, weighted."""
    weights = np.ones(len(points)) if weights is None else weights
    return np.column_stack([
        np.bincount(labels, weights * p) for p in points.T]) / np.bincount(
            labels, weights)[:, None]


def _nearest_generators(
        points: np.ndarray, generators: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """
    Return the generator nearest each point in distance over the scale of
    the generator (of a weighted Voronoi tessellation; of a Voronoi one
    with scales of 1).
    """
    tree = scipy.spatial.cKDTree(generators)
    which = np.empty(len(points), dtype=int)
    todo = np.arange(len(points))
    k = 16
    while todo.size:
        k = min(k, len(generators))
        distances, nearest = tree.query(points[todo], k)
        distances = distances.reshape(todo.size, k)
        nearest = nearest.reshape(todo.size, k)
        scaled = distances / scales[nearest]
        rows = np.arange(todo.size)
        best = np.argmin(scaled, axis=1)
        which[todo] = nearest[rows, best]
        if k == len(generators):
            break
        # (the others are at least the kth distance away: in distance over
        # scale, at least that over the largest scale; else, search more)
        sure = scaled[rows, best] <= distances[:, -1] / scales.max()
        todo = todo[~sure]
        k *= 4
    return which


def bin_index(bins: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Return the bins of a map numbered from 0 (in the order of their
    numbers; -1 for no bin), and their number.

    Parameters
    ----------
    bins : np.ndarray
        The number of the bin of each pixel (negative or NaN for no bin).

    Raises
    ------
    ValueError
        If the bins are not integers.
    """
    valid = np.isfinite(bins) & (bins >= 0)
    if np.any(bins[valid] != np.round(bins[valid])):
        raise ValueError("the bins must be integers")
    numbers = np.unique(bins[valid])
    index = np.full(bins.shape, -1, dtype=np.int32)
    index[valid] = np.searchsorted(numbers, bins[valid])
    return index, len(numbers)


def bin_values(index: np.ndarray, nbins: int, data: np.ndarray) -> np.ndarray:
    """
    Return the value of each bin of a map whose pixels hold the value of
    their bin (NaN for a bin without finite pixels).

    Raises
    ------
    ValueError
        If the finite pixels of a bin hold different values.
    """
    values = np.full(nbins, np.nan)
    for i in range(nbins):
        pixels = data[(index == i) & np.isfinite(data)]
        if pixels.size == 0:
            continue
        if not np.allclose(pixels, pixels[0], rtol=1e-6, atol=0):
            raise ValueError(
                f"the pixels of bin {i} hold different values; the map must "
                f"hold the value of each bin on its pixels")
        values[i] = pixels[0]
    return values


def pixel_fluxes(
        header: fits.Header,
        unit: gbkfit.units.AnyUnit | None,
        beam: Beam | None,
        filename: str
) -> tuple[float | np.ndarray, gbkfit.units.AnyUnit] | None:
    """
    Return how the values of the pixels of data of a header and a unit
    (e.g. those of FitsArrays) become their fluxes: a factor, and the
    unit of the fluxes. Values per pixel (e.g. of a spaxel, or per
    pix) are fluxes (factor 1); per solid angle, they are times that of a
    pixel; per beam (e.g. Jy/beam), times that of a pixel over that of
    the Gaussian beam of the data, pi / (4 ln 2) bmaj bmin: of their
    header, or of each channel of a cube of CASA (a factor for each of
    its channels, in their order; see Arrays).

    Parameters
    ----------
    header : fits.Header
        The header of the data (their world coordinates).
    unit : AnyUnit or None
        The unit of the data, if any.
    beam : Beam or None
        The beam of the data, if any.
    filename : str
        The file of the data (for messages).

    Returns
    -------
    tuple of float or np.ndarray and astropy.units.UnitBase, or None
        The factor (one, or one for each channel) and the unit of the
        fluxes; None if the data have no unit.

    Raises
    ------
    ValueError
        If the unit is a brightness temperature (whose sums over pixels
        are not fluxes), or neither light per pixel, solid angle or beam
        (see gbkfit.units.spatial_unit), or is per beam and the data have
        no beam.
    """
    u = astropy.units
    if unit is None:
        return None
    if unit.is_equivalent(u.K):
        raise ValueError(
            f"{filename}: the data are a brightness temperature "
            f"({gbkfit.units.format_unit(unit)!r}), whose sums over pixels "
            f"are not fluxes: convert them to flux density (e.g. Jy/beam) "
            f"first")
    spatial = gbkfit.units.spatial_unit(unit)
    if spatial in (u.dimensionless_unscaled, u.pix):
        return 1.0, unit * spatial
    wcs = astropy.wcs.WCS(header).celestial
    pixel = astropy.wcs.utils.proj_plane_pixel_area(wcs) * u.deg ** 2
    if spatial == u.sr:
        # (the angles of the unit, e.g. 1/arcsec^2)
        angles = u.dimensionless_unscaled
        for base, power in zip(unit.bases, unit.powers):
            if base.physical_type in ('solid angle', 'angle'):
                angles = angles * base ** power
        return float((pixel * angles).decompose()), unit / angles
    if beam is None:
        raise ValueError(
            f"{filename}: the data are per beam, but have no beam (BMAJ and "
            f"BMIN in the header, or a table BEAMS)")
    return pixel.to_value(u.arcsec ** 2) / beam.area(), unit * u.beam


def bin_spectra(
        index: np.ndarray,
        nbins: int,
        cube: FitsArrays
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return the spectrum of each bin of a cube: the sum of the valid
    spectra of its pixels (a flux, as the region datasets compare; in
    float64), and its errors (added in quadrature). A channel of a bin
    without valid pixels is NaN.

    Parameters
    ----------
    index : np.ndarray
        The bin of each pixel, from 0 (-1 for none; see bin_index).
    nbins : int
        The number of bins.
    cube : FitsArrays
        The cube (channels first).

    Returns
    -------
    np.ndarray
        The spectra, of shape (nchannels, nbins).
    np.ndarray or None
        Their errors, if the cube has some.
    """
    nchannels = cube.data.shape[0]
    spectra = np.full((nchannels, nbins), np.nan)
    errors = None if cube.error is None else spectra.copy()
    valid = cube.mask != 0
    for i in range(nbins):
        pixels = index == i
        counts = valid[:, pixels].sum(axis=1)
        sums = np.nansum(cube.data[:, pixels], axis=1, dtype=np.float64)
        spectra[:, i] = np.where(counts > 0, sums, np.nan)
        if errors is not None:
            variances = np.nansum(
                cube.error[:, pixels].astype(np.float64) ** 2, axis=1)
            errors[:, i] = np.where(counts > 0, np.sqrt(variances), np.nan)
    return spectra, errors
