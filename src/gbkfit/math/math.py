

import numpy as np
import numpy.typing as npt


def is_odd(x: npt.ArrayLike) -> np.ndarray:
    """Return whether each value of x, truncated to an integer, is odd."""
    return np.trunc(np.asarray(x)).astype(int) % 2 != 0


def is_even(x: npt.ArrayLike) -> np.ndarray:
    """Return whether each value of x, truncated to an integer, is even."""
    return np.trunc(np.asarray(x)).astype(int) % 2 == 0


def roundu_even(x: npt.ArrayLike) -> np.ndarray:
    """Return each value of x rounded up to an even number."""
    return 2 * np.ceil(0.5 * np.asarray(x))


def roundu_odd(x: float) -> float:
    """Return x rounded up to an odd number."""
    x_even = roundu_even(x)
    return x_even + 1 if x_even - 1 < x else x_even - 1


def roundu_multiple(x: float, multiple: float) -> float:
    """
    Return x rounded up to a multiple of multiple.

    Raises
    ------
    ValueError
        If multiple is not positive.
    """
    if not multiple > 0:
        raise ValueError(f"multiple must be positive; it is {multiple}")
    remainder = np.mod(x, multiple)
    return x - remainder + multiple if remainder != 0 else x


def _roundu_po2(x: float) -> int:
    """Return the smallest power of 2 at or above x (at least 1)."""
    power = 1
    while power < x:
        power *= 2
    return power


def roundu_po2(x: npt.ArrayLike) -> np.ndarray:
    """
    Return the smallest power of 2 at or above each value of x (at least
    1).
    """
    return np.vectorize(_roundu_po2)(x)


def transform_lh_rotate_z(
        x: np.ndarray, y: np.ndarray, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    The same transform as the native transform_lh_rotate_z (gmodels), so
    that an angle theta is a position angle, measured from north (+y)
    through east (-x).
    """
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    out_x = - x * sintheta + y * costheta
    out_y = - x * costheta - y * sintheta
    return out_x, out_y


def gauss_fwhm_to_sigma(fwhm: float) -> float:
    return fwhm / np.sqrt(8 * np.log(2))


def gauss_1d_fun(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(- (x - b) * (x - b) / (2 * c * c))


def ggauss_1d_fun(
        x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    return a * np.exp(-np.power(np.abs(x - b) / c, d))


def lorentz_1d_fun(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * c * c / ((x - b) * (x - b) + c * c)


def moffat_1d_fun(
        x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    return a * np.power(1 + np.power((x - b) / c, 2), -d)
