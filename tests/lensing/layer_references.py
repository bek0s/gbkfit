"""
Numpy references of the native kernels of the deflections of the layers
of the mass of a galaxy on the sky (see gbkfit.lensing._layers): those
of an elliptical layer and of bands of rings, written plainly.
"""

import math
from collections.abc import Callable

import numpy as np

from gbkfit.driver import Driver
from gbkfit.lensing._layers import K, tanh_sinh_nodes, unit_deflection
from gbkfit.model.mass import BandLayer, EllipticalLayer, LogTable, SkyMass


# Below this eccentricity e (e^2 = 1 - q^2), a band bends by the series of
# its integrals in e^2, of this many terms
_SERIES_ECCENTRICITY = 0.1
_SERIES_TERMS = 10

_NODES, _ONE_MINUS_NODES, _WEIGHTS = tanh_sinh_nodes()


def layers_deflection(
        mass: SkyMass, x: np.ndarray, y: np.ndarray, driver: Driver
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection angle (radians) of the mass of a galaxy on the
    sky at points (kpc): the sum of those of its layers, each scale times
    that of its unit shape, uncached (as MassDeflections keeps them).
    """
    alpha_x, alpha_y = np.zeros(np.shape(x)), np.zeros(np.shape(y))
    for layer in mass.layers():
        unit_x, unit_y = unit_deflection(layer, x, y, driver)
        alpha_x += layer.scale * unit_x
        alpha_y += layer.scale * unit_y
    return alpha_x, alpha_y


def elliptical_deflection(
        layer: EllipticalLayer, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deflection of an elliptical layer (see _elliptical)."""
    return _elliptical(
        layer, np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def band_deflection(
        layer: BandLayer, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deflection of bands of rings (see _bands)."""
    return _bands(
        layer, np.asarray(x, dtype=float), np.asarray(y, dtype=float))


def _frame(
        dx: np.ndarray, dy: np.ndarray, posa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the offsets (dx, dy) of the sky along the major axis (u) of the
    position angle posa (degrees) and the minor axis (v) 90 degrees from
    it, as those of the plane of a disk (see transform_posa).
    """
    p = np.radians(posa)
    sin_p, cos_p = np.sin(p), np.cos(p)
    return -dx * sin_p + dy * cos_p, -dx * cos_p - dy * sin_p


def _sky(
        alpha_u: np.ndarray, alpha_v: np.ndarray, posa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return deflections along u and v (see _frame) along x and y."""
    p = np.radians(posa)
    sin_p, cos_p = np.sin(p), np.cos(p)
    return (-alpha_u * sin_p - alpha_v * cos_p,
            alpha_u * cos_p - alpha_v * sin_p)


def _elliptical(
        layer: EllipticalLayer, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an elliptical layer (see EllipticalLayer):
    of each slice, at the points from its centre moved by its height along
    the normal, alpha_u = pi K u J_0 and alpha_v = pi K v J_1 along its
    axes (Keeton 2001, with the semi-major axis a(w)^2 = w (u^2 + v^2 /
    d)),

        J_n = int_0^w_end q Sigma_sky(a(w)) / d^(n + 1/2) dw,
        d = 1 - (1 - q^2) w,

    q Sigma_sky(a) = Sigma(a / sigma1) / sigma1^2, by the tanh-sinh
    quadrature of 52 nodes over [0, w_end], w_end the w of the end of its
    tables (or 1, of a point within them).
    """
    sigma1, sigma2 = layer.stretch
    q = sigma2 / sigma1
    e2 = 1 - q * q
    table = layer.sigma
    a_end = sigma1 * math.exp(
        table.start + table.step * (table.values.shape[-1] - 1))
    shape = np.shape(x)
    alpha_u, alpha_v = np.zeros(x.size), np.zeros(x.size)
    for row, height in enumerate(layer.heights):
        u, v = _frame(
            x.ravel() - layer.centre[0] - height * layer.normal[0],
            y.ravel() - layer.centre[1] - height * layer.normal[1],
            layer.posa)
        uu, vv = u * u, v * v
        # (w_end: a(w) = a_end, the root in (0, 1] of the quadratic
        # e2 u^2 w^2 - (u^2 + v^2 + e2 a_end^2) w + a_end^2)
        b = uu + vv + e2 * a_end ** 2
        root = np.sqrt(np.maximum(b * b - 4 * e2 * uu * a_end ** 2, 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            w_end = np.minimum(1, 2 * a_end ** 2 / (b + root))
        w_end = np.where(b > 0, w_end, 1)
        w = w_end[:, None] * _NODES
        d = q * q + e2 * ((1 - w_end)[:, None]
                          + w_end[:, None] * _ONE_MINUS_NODES)
        a = np.sqrt(w * (uu[:, None] + vv[:, None] / d))
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma = np.where(
                a > 0, _row(table, row)(np.where(a > 0, a, 1) / sigma1), 0)
        weighted = w_end[:, None] * _WEIGHTS * sigma / sigma1 ** 2 \
            / np.sqrt(d)
        alpha_u += math.pi * K * u * weighted.sum(-1)
        alpha_v += math.pi * K * v * (weighted / d).sum(-1)
    alpha_x, alpha_y = _sky(alpha_u, alpha_v, layer.posa)
    return alpha_x.reshape(shape), alpha_y.reshape(shape)


def _row(table: LogTable, row: int) -> LogTable:
    """Return the table of one row of a table of several."""
    return LogTable(table.start, table.step, table.values[row])


def _bands(
        layer: BandLayer, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of bands of rings (see BandLayer): the sum of
    those of each band in each slice (see _band_slice).
    """
    shape = np.shape(x)
    alpha_x, alpha_y = np.zeros(x.size), np.zeros(x.size)
    for row, height in enumerate(layer.heights):
        for k in range(len(layer.posa)):
            centre = layer.centres[k] + height * layer.normals[k]
            ax, ay = _band_slice(
                x.ravel() - centre[0], y.ravel() - centre[1],
                layer.posa[k], *layer.stretch[k], *layer.edges[k],
                layer.density[row, k])
            alpha_x += ax
            alpha_y += ay
    return alpha_x.reshape(shape), alpha_y.reshape(shape)


def _band_slice(
        dx: np.ndarray,
        dy: np.ndarray,
        posa: float,
        sigma1: float,
        sigma2: float,
        inner: float,
        outer: float,
        density: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of a band at offsets from its centre (see
    BandLayer). A thin homoeoid of the semi-major axis a and the mass dm
    bends nothing inside it, and outside, alpha* = alpha_u - i alpha_v =
    K dm / S(a), S(a) = sqrt(z^2 - e^2 a^2), z = u + i v, e^2 = 1 - q^2
    (S on the branch of z at infinity, cut on the segment between its
    foci). Of the band, dm = 2 pi q Sigma_sky(a) a da from a_in = sigma1
    R_in to min(a_out, a_ray), a_ray the ellipse of the point: alpha* = 2
    pi K sum_m d_m J_(m+1), of q Sigma_sky(a) = sum_m d_m a^m (d_m = c_m /
    sigma1^(m + 2)) and J_n = int a^n / S da, in closed form (see
    _band_integrals), or by their series for small e (see _band_series).
    """
    q = sigma2 / sigma1
    e2 = 1 - q * q
    u, v = _frame(dx, dy, posa)
    # (the ellipse of the point; of a band seen edge-on, q = 0, that of
    # a point off its line is infinite)
    a_ray = np.sqrt(u * u + np.divide(
        v * v, q * q, out=np.where(v == 0, 0.0, np.inf), where=q > 0))
    a1 = sigma1 * inner
    a2 = np.minimum(sigma1 * outer, a_ray)
    outside = a2 > a1
    alpha = np.zeros(u.shape, dtype=complex)
    if not outside.any():
        return alpha.real, alpha.real
    z = u[outside] + 1j * v[outside]
    upper = a2[outside]
    coefficients = [
        c / sigma1 ** (m + 2) for m, c in enumerate(density)]
    if math.sqrt(e2) < _SERIES_ECCENTRICITY:
        integrals = _band_series(z, a1, upper, e2)
    else:
        integrals = _band_integrals(z, a1, upper, math.sqrt(e2))
    alpha[outside] = 2 * math.pi * K * sum(
        d * j for d, j in zip(coefficients, integrals))
    alpha_u, alpha_v = alpha.real, -alpha.imag
    return _sky(alpha_u, alpha_v, posa)


def _branch_root(z: np.ndarray, e: float, a: np.ndarray) -> np.ndarray:
    """
    Return S(a) = sqrt(z^2 - e^2 a^2) on the branch of z at infinity, cut on
    the segment between the foci: sqrt(z - e a) sqrt(z + e a).
    """
    return np.sqrt(z - e * a) * np.sqrt(z + e * a)


def _band_integrals(
        z: np.ndarray, a1: float, a2: np.ndarray, e: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the integrals J_1, J_2 and J_3 of a band (see _band_slice) from
    a1 to a2, in closed form:

        J_1 = (a2^2 - a1^2) / (S1 + S2),
        J_2 = z^2 / (2 e^3) (-i) ln((S2 + i e a2) / (S1 + i e a1))
              - (a2 S2 - a1 S1) / (2 e^2),
        J_3 = [2 z^2 J_1 - (a2^2 S2 - a1^2 S1)] / (3 e^2),

    S1 = S(a1), S2 = S(a2) (J_1 as the difference of -S / e^2, without
    its cancellation; the logarithm as that of asin(e a / z), of a
    change of its argument within (-pi, pi)).
    """
    s1, s2 = _branch_root(z, e, a1), _branch_root(z, e, a2)
    j1 = (a2 * a2 - a1 * a1) / (s1 + s2)
    j2 = z * z / (2 * e ** 3) * -1j * np.log(
        (s2 + 1j * e * a2) / (s1 + 1j * e * a1)) \
        - (a2 * s2 - a1 * s1) / (2 * e * e)
    j3 = (2 * z * z * j1 - (a2 * a2 * s2 - a1 * a1 * s1)) / (3 * e * e)
    return j1, j2, j3


def _band_series(
        z: np.ndarray, a1: float, a2: np.ndarray, e2: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the integrals J_1, J_2 and J_3 of a band (see _band_slice) from
    a1 to a2 by their series in e^2: 1 / S = sum_k binom(2k, k) / 4^k e^2k
    a^2k / z^(2k + 1), so J_n = sum_k binom(2k, k) / 4^k e^2k (a2^(n + 2k
    + 1) - a1^(n + 2k + 1)) / ((n + 2k + 1) z^(2k + 1)), of terms that fall
    at least as (e / q)^2 outside the band (where |z| >= q a).
    """
    out = []
    for n in (1, 2, 3):
        total = np.zeros(z.shape, dtype=complex)
        factor = 1 / z
        for k in range(_SERIES_TERMS):
            power = n + 2 * k + 1
            total = total + math.comb(2 * k, k) / 4 ** k * factor * (
                a2 ** power - a1 ** power) / power
            factor = factor * e2 / (z * z)
        out.append(total)
    return tuple(out)


def keeton_deflection(
        u: np.ndarray,
        v: np.ndarray,
        q: float,
        kappa: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of an elliptical convergence kappa(xi) (a
    function of xi = sqrt(q u^2 + v^2 / q)) along its major (u) and minor
    (v) axes, by the integrals over the ellipses within each point (Keeton
    2001, with xi^2 = q xi_K^2): alpha_u = q u J_0 and alpha_v = q v J_1,
    J_n the integral over w from 0 to 1 of kappa(xi(w)) / d^(n + 1/2), d =
    1 - (1 - q^2) w, xi(w)^2 = q w (u^2 + v^2 / d), by the tanh-sinh
    quadrature of 52 nodes; 0 at its centre (the numpy kernel the NFW lens
    had before the native one).
    """
    u, v = np.ravel(u)[:, None], np.ravel(v)[:, None]
    d = q * q + (1 - q * q) * _ONE_MINUS_NODES
    xi = np.sqrt(q * _NODES * (u * u + v * v / d))
    with np.errstate(divide='ignore', invalid='ignore'):
        weighted = _WEIGHTS * kappa(xi) / np.sqrt(d)
    alpha_u = q * u[:, 0] * weighted.sum(-1)
    alpha_v = q * v[:, 0] * (weighted / d).sum(-1)
    centre = (u[:, 0] == 0) & (v[:, 0] == 0)
    alpha_u[centre] = 0
    alpha_v[centre] = 0
    return alpha_u, alpha_v
