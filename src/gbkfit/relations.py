"""
Scaling relations of galaxies and of their dark matter halos, which the
expressions of parameters can call (as relations.<name>, e.g.
'relations.dz_c(s1, c2)'; see gbkfit.params): functions of numbers and
arrays only. Masses are log10 of solar masses; each relation states the
mass of the halos it is of (e.g. M_200c, or the virial mass of Bryan &
Norman 1998: the overdensities '200c' and 'vir' of gbkfit.model.mass).
(Moster et al. 2018 is not here: it tabulates its stellar-to-halo mass
relation at six redshifts only, for the peak halo mass.)
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


__all__ = [
    'c200_dutton14',
    'cvir_dutton14',
    'dc14_alpha',
    'dc14_beta',
    'dc14_c',
    'dc14_gamma',
    'dz_a',
    'dz_c',
    'dz_c2',
    'dz_c2_apostle',
    'dz_s1',
    'dz_s1_apostle',
    'log_mhalo_behroozi13',
    'log_mhalo_moster13'
]


# The range of log10 halo masses (Msun) in which the inverse relations
# search, by bisection to the precision of floats
_LOG_MHALO_RANGE = (4.0, 20.0)
_BISECTIONS = 64


def dz_a(s1: npt.ArrayLike, c2: npt.ArrayLike) -> np.ndarray:
    """
    Return the inner slope a of the Dekel-Zhao halo (see
    mass.ProfileDekelZhao) of the logarithmic slope s1 of its density at
    0.01 R_vir and of the concentration c2 = R_vir / r_-2 (r_-2 where its
    slope is -2; Freundlich et al. 2020, Eq. 14): (1.5 s1 - 2 k) / (1.5 -
    k), with k = 0.1 (3.5 - s1) c2^(1/2).
    """
    s1, c2 = np.asarray(s1, dtype=float), np.asarray(c2, dtype=float)
    k = 0.1 * (3.5 - s1) * np.sqrt(c2)
    return (1.5 * s1 - 2 * k) / (1.5 - k)


def dz_c(s1: npt.ArrayLike, c2: npt.ArrayLike) -> np.ndarray:
    """
    Return the concentration c = R_vir / r_s of the Dekel-Zhao halo of s1
    and c2 (see dz_a; Freundlich et al. 2020, Eq. 15): ((s1 - 2) / (0.1
    (3.5 - s1) - 1.5 c2^(-1/2)))^2.
    """
    s1, c2 = np.asarray(s1, dtype=float), np.asarray(c2, dtype=float)
    return ((s1 - 2) / (0.1 * (3.5 - s1) - 1.5 / np.sqrt(c2))) ** 2


def _dutton14(
        log_m: npt.ArrayLike,
        z: npt.ArrayLike,
        a: tuple[float, float, float, float],
        b: tuple[float, float],
        h: float
) -> np.ndarray:
    """
    Return the concentration of Dutton & Maccio (2014, Eq. 7): log10 c =
    a(z) + b(z) log10(M / (10^12 h^-1 Msun)), with a(z) = a0 + (a1 - a0)
    exp(-a2 z^a3) and b(z) = b0 + b1 z.
    """
    z = np.asarray(z, dtype=float)
    a_z = a[0] + (a[1] - a[0]) * np.exp(-a[2] * z ** a[3])
    b_z = b[0] + b[1] * z
    return 10 ** (a_z + b_z * (np.asarray(log_m, dtype=float)
                               + np.log10(h) - 12))


def c200_dutton14(
        log_m200: npt.ArrayLike, z: npt.ArrayLike, h: float = 0.671
) -> np.ndarray:
    """
    Return the concentration c_200c of dark-matter-only NFW halos of the
    mass M_200c at the redshift z (Dutton & Maccio 2014, Eqs. 7, 10, 11;
    Planck cosmology, h = 0.671, for z from 0 to 5).
    """
    return _dutton14(
        log_m200, z, (0.520, 0.905, 0.617, 1.21), (-0.101, 0.026), h)


def cvir_dutton14(
        log_mvir: npt.ArrayLike, z: npt.ArrayLike, h: float = 0.671
) -> np.ndarray:
    """
    Return the concentration c_vir of dark-matter-only NFW halos of the
    virial mass M_vir at the redshift z (Dutton & Maccio 2014, Eqs. 7, 12,
    13; their virial overdensity is that of Mainini et al. 2003, 104.2
    times the critical density at z = 0, close to that of Bryan & Norman
    1998; Planck cosmology, h = 0.671, for z from 0 to 5).
    """
    return _dutton14(
        log_mvir, z, (0.537, 1.025, 0.718, 1.08), (-0.097, 0.024), h)


def _dz_ratio_x(log_mstar: npt.ArrayLike, log_mvir: npt.ArrayLike,
                x0: float, nu: float) -> np.ndarray:
    """Return (x / x0)^nu, x = M_star / M_vir."""
    return (10 ** (np.asarray(log_mstar, dtype=float)
                   - np.asarray(log_mvir, dtype=float)) / x0) ** nu


def dz_s1(
        log_mstar: npt.ArrayLike,
        log_mvir: npt.ArrayLike,
        z: npt.ArrayLike = 0.0,
        h: float = 0.671
) -> np.ndarray:
    """
    Return the inner slope s1 of the Dekel-Zhao halo of a galaxy of the
    stellar mass M_star and the virial mass M_vir (Freundlich et al.
    2020, Section 4.3 and Table 1, from NIHAO at z = 0): s_DMO (s1 /
    s_DMO)(x), x = M_star / M_vir, with (s1 / s_DMO)(x) = 1 / (1 + y) +
    0.32 log10(1 + y), y = (x / 1.30e-3)^2.86, and s_DMO = (1 + 0.03
    c_DMO) / (1 + 0.01 c_DMO), the slope at 0.01 R_vir of the NFW halo of
    the concentration c_DMO = cvir_dutton14(log_mvir, z, h).
    """
    y = _dz_ratio_x(log_mstar, log_mvir, 1.30e-3, 2.86)
    c_dmo = cvir_dutton14(log_mvir, z, h)
    s_dmo = (1 + 0.03 * c_dmo) / (1 + 0.01 * c_dmo)
    return s_dmo * (1 / (1 + y) + 0.32 * np.log10(1 + y))


def dz_c2(
        log_mstar: npt.ArrayLike,
        log_mvir: npt.ArrayLike,
        z: npt.ArrayLike = 0.0,
        h: float = 0.671
) -> np.ndarray:
    """
    Return the concentration c2 of the Dekel-Zhao halo of a galaxy of the
    stellar mass M_star and the virial mass M_vir (see dz_s1; Freundlich
    et al. 2020): c_DMO (c2 / c_DMO)(x), with (c2 / c_DMO)(x) = 1.14 (1 +
    (x / 2.43e-2)^1.37) - x^0.142.
    """
    x = 10 ** (np.asarray(log_mstar, dtype=float)
               - np.asarray(log_mvir, dtype=float))
    y = _dz_ratio_x(log_mstar, log_mvir, 2.43e-2, 1.37)
    return cvir_dutton14(log_mvir, z, h) * (1.14 * (1 + y) - x ** 0.142)


def dz_s1_apostle(
        log_mstar: npt.ArrayLike, log_mvir: npt.ArrayLike) -> np.ndarray:
    """
    Return the inner slope s1 of the Dekel-Zhao halo of a galaxy of the
    stellar mass M_star and the virial mass M_vir as halos of APOSTLE
    respond (Bose et al. 2019: no cores), in the fit of Jiang et al.
    (2021, SatGen): log10(20 + (x / 8.77e-3)^1.66) + 0.158, x = M_star /
    M_vir.
    """
    x = 10 ** (np.asarray(log_mstar, dtype=float)
               - np.asarray(log_mvir, dtype=float))
    return np.log10(20 + (x / 8.77e-3) ** 1.66) + 0.158


def dz_c2_apostle(
        log_mstar: npt.ArrayLike,
        log_mvir: npt.ArrayLike,
        z: npt.ArrayLike = 0.0,
        h: float = 0.671
) -> np.ndarray:
    """
    Return the concentration c2 of the Dekel-Zhao halo of a galaxy of the
    stellar mass M_star and the virial mass M_vir as halos of APOSTLE
    respond (see dz_s1_apostle): c_DMO (1 + 227 x^1.45), x = M_star /
    M_vir, with c_DMO = cvir_dutton14(log_mvir, z, h).
    """
    x = 10 ** (np.asarray(log_mstar, dtype=float)
               - np.asarray(log_mvir, dtype=float))
    return cvir_dutton14(log_mvir, z, h) * (1 + 227 * x ** 1.45)


def _dc14_x(log_mstar: npt.ArrayLike, log_mvir: npt.ArrayLike) -> np.ndarray:
    """Return X = log10(M_star / M_vir) of the DC14 relations."""
    return np.asarray(log_mstar, dtype=float) \
        - np.asarray(log_mvir, dtype=float)


def dc14_alpha(
        log_mstar: npt.ArrayLike, log_mvir: npt.ArrayLike) -> np.ndarray:
    """
    Return alpha of the Zhao halo (see mass.ProfileZhao) of a galaxy of
    the stellar mass M_star and the virial mass M_vir (DC14: Di Cintio et
    al. 2014, Eq. 3; for X = log10(M_star / M_vir) in [-4.1, -1.3]): 2.94
    - log10((10^(X + 2.33))^-1.08 + (10^(X + 2.33))^2.29).
    """
    y = 10 ** (_dc14_x(log_mstar, log_mvir) + 2.33)
    return 2.94 - np.log10(y ** -1.08 + y ** 2.29)


def dc14_beta(
        log_mstar: npt.ArrayLike, log_mvir: npt.ArrayLike) -> np.ndarray:
    """
    Return beta of the Zhao halo of a galaxy (see dc14_alpha): 4.23 +
    1.34 X + 0.26 X^2.
    """
    x = _dc14_x(log_mstar, log_mvir)
    return 4.23 + 1.34 * x + 0.26 * x ** 2


def dc14_gamma(
        log_mstar: npt.ArrayLike, log_mvir: npt.ArrayLike) -> np.ndarray:
    """
    Return gamma of the Zhao halo of a galaxy (see dc14_alpha): -0.06 +
    log10((10^(X + 2.56))^-0.68 + 10^(X + 2.56)).
    """
    y = 10 ** (_dc14_x(log_mstar, log_mvir) + 2.56)
    return -0.06 + np.log10(y ** -0.68 + y)


def dc14_c(
        log_mstar: npt.ArrayLike,
        log_mvir: npt.ArrayLike,
        z: npt.ArrayLike = 0.0,
        h: float = 0.671
) -> np.ndarray:
    """
    Return the concentration c = R_vir / r_s of the Zhao halo of a galaxy
    (see dc14_alpha; r_s of the formula of mass.ProfileZhao): that of
    r_-2 (where the slope of the density is -2), c_DMO (1 + 3e-5 exp(3.4
    (X + 4.5))) (Di Cintio et al. 2014, Eq. 6), with c_DMO =
    cvir_dutton14(log_mvir, z, h), times r_-2 / r_s = ((2 - gamma) /
    (beta - 2))^(1 / alpha).
    """
    x = _dc14_x(log_mstar, log_mvir)
    alpha = dc14_alpha(log_mstar, log_mvir)
    beta = dc14_beta(log_mstar, log_mvir)
    gamma = dc14_gamma(log_mstar, log_mvir)
    c_2 = cvir_dutton14(log_mvir, z, h) * (1 + 3e-5 * np.exp(3.4 * (x + 4.5)))
    return c_2 * ((2 - gamma) / (beta - 2)) ** (1 / alpha)


def _log_mstar_moster13(
        log_mhalo: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Return log10 M_star of the halos of the mass M_200c at the redshift z
    of Moster et al. (2013, Eqs. 2, 11-14, Table 1): m / M = 2 N ((M /
    M1)^-beta + (M / M1)^gamma)^-1.
    """
    w = z / (z + 1)
    log_m1 = 11.590 + 1.195 * w
    norm = 0.0351 - 0.0247 * w
    beta = 1.376 - 0.826 * w
    gamma = 0.608 + 0.329 * w
    x = 10 ** (log_mhalo - log_m1)
    return log_mhalo + np.log10(2 * norm / (x ** -beta + x ** gamma))


def _log_mstar_behroozi13(
        log_mhalo: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Return log10 M_star of the halos of the virial mass M_vir at the
    redshift z of Behroozi et al. (2013, Eqs. 3-4, Section 5): log10(eps
    M1) + f(log10(M / M1)) - f(0), f(x) = -log10(10^(alpha x) + 1) + delta
    log10(1 + e^x)^gamma / (1 + exp(10^-x)).
    """
    a = 1 / (1 + z)
    nu = np.exp(-4 * a ** 2)
    log_m1 = 11.514 + (-1.793 * (a - 1) - 0.251 * z) * nu
    log_eps = -1.777 + (-0.006 * (a - 1) - 0.000 * z) * nu \
        - 0.119 * (a - 1)
    alpha = -1.412 + 0.731 * (a - 1) * nu
    delta = 3.508 + (2.608 * (a - 1) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1) + 0.279 * z) * nu

    def f(x: np.ndarray) -> np.ndarray:
        # (exp(10^-x) overflows far below M1, where the term is 0)
        with np.errstate(over='ignore'):
            return -np.log10(10 ** (alpha * x) + 1) + delta * np.log10(
                1 + np.exp(x)) ** gamma / (1 + np.exp(10.0 ** -x))
    return log_eps + log_m1 + f(log_mhalo - log_m1) - f(np.zeros_like(z))


def _inverse(
        relation: Callable[[np.ndarray, np.ndarray], np.ndarray],
        log_mstar: npt.ArrayLike,
        z: npt.ArrayLike
) -> np.ndarray:
    """
    Return the log10 halo masses whose log10 stellar masses are log_mstar
    by an increasing relation(log_mhalo, z), by bisection in
    _LOG_MHALO_RANGE.
    """
    log_mstar, z = np.broadcast_arrays(
        np.asarray(log_mstar, dtype=float), np.asarray(z, dtype=float))
    lo = np.full(log_mstar.shape, _LOG_MHALO_RANGE[0])
    hi = np.full(log_mstar.shape, _LOG_MHALO_RANGE[1])
    for _ in range(_BISECTIONS):
        mid = (lo + hi) / 2
        below = relation(mid, z) < log_mstar
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def log_mhalo_moster13(
        log_mstar: npt.ArrayLike, z: npt.ArrayLike) -> np.ndarray:
    """
    Return log10 M_200c of the halo of a galaxy of the stellar mass M_star
    at the redshift z: the mean stellar-to-halo mass relation of Moster
    et al. (2013) inverted (to the precision of floats).
    """
    return _inverse(_log_mstar_moster13, log_mstar, z)


def log_mhalo_behroozi13(
        log_mstar: npt.ArrayLike, z: npt.ArrayLike) -> np.ndarray:
    """
    Return log10 M_vir (Bryan & Norman 1998) of the halo of a central
    galaxy of the stellar mass M_star at the redshift z: the median
    stellar mass-halo mass relation of Behroozi et al. (2013; h = 0.7)
    inverted (to the precision of floats).
    """
    return _inverse(_log_mstar_behroozi13, log_mstar, z)
