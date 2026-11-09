"""
The profiles of the analytic PSFs and LSFs: their 1d forms, and their
scales of a full width at half maximum.
"""

import numpy as np


def gauss_fwhm_to_sigma(fwhm: float) -> float:
    """
    Return the standard deviation of a Gaussian of the given full width
    at half maximum.
    """
    return fwhm / np.sqrt(8 * np.log(2))


def gauss_sigma_to_fwhm(sigma: float) -> float:
    """
    Return the full width at half maximum of a Gaussian of the given
    standard deviation.
    """
    return sigma * np.sqrt(8 * np.log(2))


def ggauss_fwhm_to_alpha(fwhm: float, beta: float) -> float:
    """
    Return the scale of a generalised Gaussian, exp(-(|x| / alpha)^beta),
    of the given full width at half maximum.
    """
    return fwhm / (2 * np.log(2) ** (1 / beta))


def moffat_fwhm_to_alpha(fwhm: float, beta: float) -> float:
    """
    Return the scale of a Moffat, (1 + (x / alpha)^2)^(-beta), of the
    given full width at half maximum.
    """
    return fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))


def gauss_1d_fun(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Return a Gaussian of amplitude a, centre b and standard deviation c at
    each value of x: a exp(-(x - b)^2 / (2 c^2)).
    """
    return a * np.exp(- (x - b) * (x - b) / (2 * c * c))


def ggauss_1d_fun(
        x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """
    Return a generalised Gaussian of amplitude a, centre b, scale c and
    shape d at each value of x: a exp(-(|x - b| / c)^d).
    """
    return a * np.exp(-np.power(np.abs(x - b) / c, d))


def lorentz_1d_fun(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Return a Lorentzian of amplitude a, centre b and half width at half
    maximum c at each value of x: a c^2 / ((x - b)^2 + c^2).
    """
    return a * c * c / ((x - b) * (x - b) + c * c)


def moffat_1d_fun(
        x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """
    Return a Moffat profile of amplitude a, centre b, scale c and power d
    at each value of x: a (1 + ((x - b) / c)^2)^-d.
    """
    return a * np.power(1 + np.power((x - b) / c, 2), -d)
