"""
The whitening of the residuals of data items whose noise is correlated
(see Whitening and whitening; LikelihoodGauss), and its application on
the device of a driver (Whitener).
"""

import dataclasses
from typing import Self

import numpy as np

from gbkfit.dataset import NoiseCorrelation
from gbkfit.driver import DeviceArray, Driver
from gbkfit.math.correlation import (
    correlation_matrix, periodic_correlation, whitening_matrix)
from gbkfit.observation import correlated_pixels


__all__ = [
    'Whitening',
    'whitening'
]


@dataclasses.dataclass(frozen=True)
class Whitening:
    """
    The whitening of the residuals of a data item whose noise is
    correlated (see LikelihoodGauss and whitening). Its values are on
    channels (the first axis of the mask it is made of, see channel_mask;
    one for data without a spectral axis) and positions (the others: the
    pixels of the sky, the positions along a slit, or the regions). The
    residuals of the positions it keeps, as a matrix R of the channels by
    those positions, are whitened into spectral R spatial^T, with the
    whitening of each axis whose noise is correlated (None for the
    others; only the pixels of the sky can be correlated). That of the
    channels is of those the mask keeps, with rows and columns of 0 for
    the others, whose residuals, 0, it leaves 0. Noise of exactly the
    correlation C of the data (without the white floor) gives them the
    chi-square tr(C C'^-1) on average, nindependent: the number of
    independent values they are worth (e.g. one in 4.5 of the pixels of
    moment maps of a beam of 6.6 by 5.4 pixels, with a floor of 0.01).

    Attributes
    ----------
    pixels : np.ndarray
        The flat indices of the positions it keeps.
    spatial : np.ndarray or None
        The whitening of the pixels it keeps, if their noise is correlated
        on the sky.
    spectral : np.ndarray or None
        The whitening of the channels, if their noise is correlated along
        the spectral axis.
    nindependent : float
        The number of independent values they are worth.
    """
    pixels: np.ndarray
    spatial: np.ndarray | None
    spectral: np.ndarray | None
    nindependent: float


def whitening(
        key: str,
        mask: np.ndarray,
        correlation: NoiseCorrelation,
        white_floor: float
) -> Whitening:
    """
    Return the whitening of the residuals of a data item whose noise is
    correlated (see LikelihoodGauss), of its mask and of the
    correlation of its noise, with a white floor.

    Parameters
    ----------
    key : str
        The name of the data item.
    mask : np.ndarray
        Its mask, of its channels by its positions (see channel_mask and
        correlated_pixels).
    correlation : NoiseCorrelation
        The correlation of its noise.
    white_floor : float
        The share of the variance of the noise taken as white (see
        LikelihoodGauss).

    Returns
    -------
    Whitening
        The whitening.

    Raises
    ------
    ValueError
        If its mask is not one of correlated noise (see correlated_pixels),
        or its correlation is not positive definite with the white floor.
    """
    pixels, channels = correlated_pixels(
        key, mask, correlation.spatial is not None)
    nchannels = mask.shape[0]
    spatial = spectral = None
    if correlation.spatial is not None:
        points = np.column_stack(np.unravel_index(pixels, mask.shape[1:]))
        spatial = _whitening(
            key, correlation.spatial, points, white_floor)
    if correlation.spectral is not None:
        spectral = np.zeros((nchannels, nchannels))
        spectral[np.ix_(channels, channels)] = _whitening(
            key, correlation.spectral, channels[:, None], white_floor)
    nindependent = _nindependent(spatial, pixels.size, white_floor) \
        * _nindependent(spectral, channels.size, white_floor)
    return Whitening(pixels, spatial, spectral, nindependent)


def _nindependent(
        whitening: np.ndarray | None, size: int, white_floor: float
) -> float:
    """
    Return the number of independent values that the values of an axis
    of a size are worth, of their whitening (see Whitening; all of them
    without one): tr(C C'^-1) of their correlation C, with C' = (1 - f) C
    + f I, which is (size - f tr(C'^-1)) / (1 - f), and tr(C'^-1) the sum
    of the squares of the whitening W (C'^-1 = W^T W).
    """
    if whitening is None:
        return size
    return (size - white_floor * np.sum(whitening ** 2)) / (1 - white_floor)


def _whitening(
        key: str,
        shape: np.ndarray,
        points: np.ndarray,
        white_floor: float
) -> np.ndarray:
    """
    Return the whitening of the values at points of a grid (see
    whitening_matrix), of the shape of the correlation of the noise of
    data item key on the separations of the grid (see NoiseCorrelation),
    with a white floor; raise ValueError if it is not positive definite.
    """
    matrix = correlation_matrix(
        periodic_correlation(shape, shape.shape), points)
    try:
        return whitening_matrix(matrix, white_floor)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"the correlation of the noise of '{key}' is singular with the "
            f"white floor {white_floor} (see LikelihoodGauss): raise it"
        ) from None


class Whitener:
    """
    The whitening of the residuals of a data item (see Whitening) on the
    device of a driver: the indices of its positions and its matrices
    (that of the sky transposed), which copies share (see shared), and
    the residuals of its positions and their products, of its own.
    """

    def __init__(
            self,
            driver: Driver,
            pixels: DeviceArray,
            spatial: DeviceArray | None,
            spectral: DeviceArray | None,
            nchannels: int,
            dtype: np.dtype
    ):
        self._driver = driver
        self._pixels = pixels
        self._spatial = spatial
        self._spectral = spectral
        self._dtype = dtype
        # (the channels by the positions; one channel without a spectral
        # axis)
        size = (nchannels, pixels.size)
        self._values = driver.mem_alloc_d(size, dtype)
        self._products = driver.mem_alloc_d(size, dtype)

    @classmethod
    def on_device(
            cls,
            driver: Driver,
            whitening_: Whitening,
            nchannels: int,
            dtype: np.dtype
    ) -> Self:
        """
        Return the whitening on the device of a driver, for residuals of a
        number of channels (see Whitening) and a type.
        """
        spatial = None if whitening_.spatial is None \
            else driver.mem_copy_h2d(np.ascontiguousarray(
                whitening_.spatial.T, dtype))
        spectral = None if whitening_.spectral is None \
            else driver.mem_copy_h2d(whitening_.spectral.astype(dtype))
        return cls(
            driver, driver.mem_copy_h2d(whitening_.pixels), spatial,
            spectral, nchannels, dtype)

    def shared(self) -> Self:
        """
        Return a copy that shares its positions and matrices, with
        residuals of its own.
        """
        return type(self)(
            self._driver, self._pixels, self._spatial, self._spectral,
            self._values.shape[0], self._dtype)

    def __call__(self, residual: DeviceArray) -> None:
        """
        Whiten the residuals of the item (of its shape, its channels
        first) in place.
        """
        driver = self._driver
        values, products = self._values, self._products
        flat = residual.reshape(values.shape[0], -1)
        driver.mem_gather(flat, self._pixels, out=values)
        if self._spatial is not None:
            driver.math_matmul(values, self._spatial, out=products)
            values, products = products, values
        if self._spectral is not None:
            driver.math_matmul(self._spectral, values, out=products)
            values, products = products, values
        driver.mem_scatter(values, self._pixels, flat)
