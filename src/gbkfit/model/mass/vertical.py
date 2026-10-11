"""
The vertical laws of thick disks: the probability density p(z) of the
height z of their mass, of a scale height h, and its transform T(k) =
int p(z) exp(-k |z|) dz, which sets the gravity of the disk in its plane
(see disks).
"""

import numpy as np
import scipy.special

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'VERTICAL_LAWS',
    'check_vertical_law',
    'vertical_density',
    'vertical_transform'
]


# The laws: 'thin' (no thickness), 'exponential' (exp(-|z| / h) / (2 h)),
# 'sech2' (sech^2(z / h) / (2 h)) and 'gauss' (a Gaussian of dispersion h)
VERTICAL_LAWS = ('thin', 'exponential', 'sech2', 'gauss')


def check_vertical_law(law: str) -> None:
    """Raise ConfigError unless the law is one of VERTICAL_LAWS."""
    if law not in VERTICAL_LAWS:
        raise ConfigError(
            f"vertical must be one of {list(VERTICAL_LAWS)}; it is {law!r}")


def vertical_density(law: str, z: np.ndarray, h: float) -> np.ndarray:
    """
    Return the probability density p(z) of a thick law (not 'thin') at the
    heights z, for the scale height h.
    """
    z = np.abs(np.asarray(z, dtype=float))
    if law == 'exponential':
        return np.exp(-z / h) / (2 * h)
    if law == 'sech2':
        return 1 / (2 * h * np.cosh(np.minimum(z / h, 350)) ** 2)
    if law == 'gauss':
        return np.exp(-z ** 2 / (2 * h ** 2)) / (np.sqrt(2 * np.pi) * h)
    raise ValueError(f"the law {law!r} has no vertical density")


def vertical_transform(law: str, k: np.ndarray, h: float) -> np.ndarray:
    """
    Return T(k) = int p(z) exp(-k |z|) dz of a law at the wavenumbers
    k >= 0, for the scale height h: 1 for a thin disk, 1 / (1 + k h) for
    the exponential law, (a/2) (psi(a/4 + 1/2) - psi(a/4)) - 1 with a =
    k h for sech2 (psi: the digamma function), and erfcx(k h / sqrt(2))
    for the Gaussian law.
    """
    k = np.asarray(k, dtype=float)
    if law == 'thin':
        return np.ones_like(k)
    if law == 'exponential':
        return 1 / (1 + k * h)
    if law == 'sech2':
        a = k * h
        a_safe = np.where(a > 0, a, 1)
        transform = a_safe / 2 * (
            scipy.special.digamma(a_safe / 4 + 0.5)
            - scipy.special.digamma(a_safe / 4)) - 1
        return np.where(a > 0, transform, 1)
    if law == 'gauss':
        return scipy.special.erfcx(k * h / np.sqrt(2))
    raise ValueError(f"unknown vertical law {law!r}")
