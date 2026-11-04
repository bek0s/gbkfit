"""
The masks of data, and their noise: values in a range, sigma
clipping, the signal of a cube and the largest regions of a mask, grown
masks and their bounding box, the noise of the errors of data, and the
continuum of spectra.
"""

import astropy.stats
import numpy as np
import scipy.ndimage

from gbkfit.config import ConfigError


__all__ = [
    'bounding_box',
    'grown_mask',
    'largest_regions_mask',
    'noise_errors',
    'range_mask',
    'sigma_clip_mask',
    'signal_mask',
    'subtract_continuum'
]


def range_mask(
        data: np.ndarray,
        minimum: float | None = None,
        maximum: float | None = None
) -> np.ndarray:
    """
    Return the mask of the values in a range (True where kept); NaN values
    are left to the other masks (kept here).
    """
    with np.errstate(invalid='ignore'):
        kept = np.ones(data.shape, bool)
        if minimum is not None:
            kept &= ~(data < minimum)
        if maximum is not None:
            kept &= ~(data > maximum)
    return kept


def sigma_clip_mask(
        data: np.ndarray,
        mask: np.ndarray,
        sigma: float,
        maxiters: int | None = 5
) -> np.ndarray:
    """
    Return the mask of the values that sigma clipping keeps (True where
    kept), of those a mask keeps (the others do not count in its
    statistics, and are not kept).
    """
    kept = np.ma.masked_array(data, mask=(mask == 0))
    return ~astropy.stats.sigma_clip(
        kept, sigma=sigma, maxiters=maxiters).mask


def noise_errors(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Return the errors of the values of a cube from its noise: for each
    channel, the robust dispersion of its negative values (of those the
    mask keeps), 1.4826 times the median of their absolute values, which
    emission (positive) does not bias, unlike the dispersion of all the
    values, even sigma-clipped (its faint wings stay); NaN for channels
    with fewer than 5 negative values. The noise must be of mean 0 (e.g.
    of a cube without continuum, as the models are).

    Parameters
    ----------
    data : np.ndarray
        The cube, of shape (nz, ny, nx).
    mask : np.ndarray
        The mask of the values that count (nonzero).

    Returns
    -------
    np.ndarray
        The errors, of the shape of the cube: the same at every pixel of
        a channel.
    """
    errors = np.full(data.shape[0], np.nan)
    for channel, (values, kept) in enumerate(zip(data, mask)):
        negative = values[(kept != 0) & (values < 0)]
        if negative.size >= 5:
            errors[channel] = 1.4826 * np.median(-negative)
    return np.broadcast_to(errors[:, None, None], data.shape).copy()


def grown_mask(
        snr: np.ndarray,
        seed: float,
        grow: float
) -> np.ndarray:
    """
    Return the mask (True where kept) grown from seeds: the values whose
    signal-to-noise reaches grow and that connect (through neighbours
    along the axes) to one that reaches seed (seed >= grow), as the masks
    of SoFiA, 3D Barolo or qubefit keep faint emission next to bright
    emission and drop isolated noise. Of a map or a cube (or any number of
    axes); NaN values are not kept.

    Raises
    ------
    ConfigError
        If seed is below grow.
    """
    if seed < grow:
        raise ConfigError(
            f"the seed threshold ({seed}) must be at least the threshold "
            f"of growth ({grow})")
    with np.errstate(invalid='ignore'):
        seeds, allowed = snr >= seed, snr >= grow
    return scipy.ndimage.binary_propagation(seeds, mask=allowed)


def signal_mask(
        cube: np.ndarray,
        seed: float,
        grow: float,
        min_ratio: float | None = None
) -> np.ndarray:
    """
    Return the mask (True where kept) of the signal of a cube: the values
    whose signal-to-noise ratio reaches grow, connected to one that
    reaches seed (see grown_mask), the noise of each channel from its
    negative values (see noise_errors), in the regions at least min_ratio
    times the largest, if given (see largest_regions_mask). NaN values are
    not kept.

    Parameters
    ----------
    cube : np.ndarray
        The cube, of shape (nz, ny, nx), of noise of mean 0 (e.g. without
        continuum).
    seed, grow : float
        The signal-to-noise ratios of the seeds and of their growth (seed
        >= grow).
    min_ratio : float, optional
        The size of the smallest region kept, as a fraction of the size
        of the largest.

    Returns
    -------
    np.ndarray
        The mask, of the shape of the cube.

    Raises
    ------
    ConfigError
        If seed is below grow.
    """
    noise = noise_errors(cube, np.isfinite(cube))
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = grown_mask(cube / noise, seed, grow)
    if min_ratio is not None:
        mask = largest_regions_mask(mask, min_ratio=min_ratio)
    return mask


def subtract_continuum(
        data: np.ndarray,
        line: np.ndarray,
        degree: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a cube less its continuum, and the continuum: of each spectrum,
    the polynomial of the given degree in the channel fitted by least
    squares to its channels outside the line (and with values; NaN where
    a spectrum has fewer of them than the degree needs).

    Parameters
    ----------
    data : np.ndarray
        The cube, of shape (nz, ny, nx).
    line : np.ndarray
        The channels of the line (True; nz), which the fit leaves out.
    degree : int, optional
        The degree of the polynomial (0: a constant).

    Returns
    -------
    tuple of np.ndarray
        The cube less its continuum, and the continuum (both of the shape
        of the cube).

    Raises
    ------
    ConfigError
        If line is not one value per channel, or the degree is negative.
    """
    line = np.asarray(line, dtype=bool)
    if line.shape != data.shape[:1] or degree < 0:
        raise ConfigError(
            "line must have one value per channel, and the degree be at "
            "least 0")
    nz = data.shape[0]
    spectra = data.reshape(nz, -1)
    # (the channels about the middle, for a well-conditioned fit)
    z = np.arange(nz) - (nz - 1) / 2
    design = np.vander(z, degree + 1)
    continuum = np.full(spectra.shape, np.nan)
    usable = ~line[:, None] & np.isfinite(spectra)
    # (the spectra of the same usable channels share one fit)
    patterns, which = np.unique(usable.T, axis=0, return_inverse=True)
    for k, pattern in enumerate(patterns):
        columns = np.flatnonzero(which.ravel() == k)
        if pattern.sum() <= degree:
            continue
        coefficients, *_ = np.linalg.lstsq(
            design[pattern], spectra[pattern][:, columns], rcond=None)
        continuum[:, columns] = design @ coefficients
    continuum = continuum.reshape(data.shape)
    return data - continuum, continuum


def largest_regions_mask(
        mask: np.ndarray,
        count: int | None = None,
        min_pixels: int | None = None,
        min_ratio: float | None = None
) -> np.ndarray:
    """
    Return the mask (True where kept) of the largest connected regions of
    the pixels a mask keeps: the count largest, those of at least
    min_pixels pixels, and those at least min_ratio times the largest
    (each criterion if given).
    """
    # (pixels connected through their sides or corners)
    labels, nregions = scipy.ndimage.label(
        mask != 0, structure=np.ones((3,) * mask.ndim))
    areas = np.bincount(labels.ravel(), minlength=nregions + 1)[1:]
    # (the largest first; of equal areas, in the order of their labels)
    order = np.argsort(-areas, kind='stable')
    if count is not None:
        order = order[:count]
    if min_pixels is not None:
        order = order[areas[order] >= min_pixels]
    if min_ratio is not None and order.size:
        order = order[areas[order] / areas[order[0]] >= min_ratio]
    return np.isin(labels, order + 1)


def bounding_box(mask: np.ndarray) -> tuple[slice, ...]:
    """
    Return the smallest ranges of pixels (one slice per axis) that hold
    the pixels a mask keeps.

    Raises
    ------
    ConfigError
        If the mask keeps no pixel.
    """
    nonzero = np.nonzero(mask)
    if not nonzero[0].size:
        raise ConfigError(
            "no valid pixels are left to crop the data to; relax the "
            "clipping, or do not minify")
    return tuple(slice(i.min(), i.max() + 1) for i in nonzero)
