"""
Numpy references of the native kernels of the deflections of the layers
of the mass of a galaxy on the sky (see gbkfit.lensing._layers): those
of an elliptical layer and of bands of rings, written plainly; thick, of
the integral over the height of their sheets by adaptive quadrature (to
1e-8, cut where the integrand bends), not the rule of the kernels.
"""

import math
from collections.abc import Callable

import numpy as np
import scipy.integrate

from gbkfit.driver import Driver
from gbkfit.lensing._layers import K, tanh_sinh_nodes, unit_deflection
from gbkfit.model.mass import (
    BandLayer, EllipticalLayer, HeightProfile, SkyMass)


# Below this eccentricity e (e^2 = 1 - q^2), a band bends by the series of
# its integrals in e^2, of this many terms
_SERIES_ECCENTRICITY = 0.1
_SERIES_TERMS = 10

# Below this d at the end of the integrals of a sheet, they are split
# there, and beyond, of Gauss-Legendre nodes in ln d, this many per e-fold,
# in panels of this many nodes at most (see gbkfit::lens_elliptical_slice)
_LINE_D = 1e-4
_LINE_DENSITY = 4
_LINE_PANEL = 32

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
    """
    Return the deflection of an elliptical layer (see _elliptical): thin,
    of its sheet; thick, the integral over the height of its sheet moved by
    the height along its normal, times its shape at the height.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    heights = layer.heights
    if heights is None:
        return _elliptical(layer, x, y)
    shape = np.shape(x)
    out = np.empty((2, x.size))
    for i, (xx, yy) in enumerate(zip(x.ravel(), y.ravel())):
        u, v = _frame(xx - layer.centre[0], yy - layer.centre[1], layer.posa)
        nu, nv = _frame(*layer.normal, layer.posa)
        sigma1, sigma2 = layer.stretch
        cuts = [*_path_cuts(u, v, nu, nv, sigma2 / sigma1)]

        def sheet(h, x0=xx, y0=yy):
            return np.ravel(_elliptical(
                layer, np.array([x0 - h * layer.normal[0]]),
                np.array([y0 - h * layer.normal[1]]),
                lambda R: knot_shape(heights, R, h)))
        out[:, i] = _over_heights(sheet, heights, cuts)
    return out[0].reshape(shape), out[1].reshape(shape)


def band_deflection(
        layer: BandLayer, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of bands of rings (see _bands): thin, of their
    first coefficients; thick, of each, the integral over the height of
    its sheet moved by the height along its normal, of its coefficients
    times the shapes at its knots at the height.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    heights = layer.heights
    if heights is None:
        return _bands(layer, x, y)
    shape = np.shape(x)
    out = np.zeros((2, x.size))
    for k in range(len(layer.posa)):
        sigma1, sigma2 = layer.stretch[k]
        q = sigma2 / sigma1
        below, above = layer.knots[k]
        for i, (xx, yy) in enumerate(zip(x.ravel(), y.ravel())):
            dx, dy = xx - layer.centres[k, 0], yy - layer.centres[k, 1]
            u, v = _frame(dx, dy, layer.posa[k])
            nu, nv = _frame(*layer.normals[k], layer.posa[k])
            cuts = [*_path_cuts(u, v, nu, nv, q, sigma1 * layer.edges[k])]

            def sheet(h, dx=dx, dy=dy, k=k, below=below, above=above):
                shapes = height_shapes(heights, h)
                density = shapes[below] * layer.density[k, 0] \
                    + shapes[above] * layer.density[k, 1]
                return np.ravel(_band_slice(
                    np.array([dx - h * layer.normals[k, 0]]),
                    np.array([dy - h * layer.normals[k, 1]]),
                    layer.posa[k], sigma1, sigma2, *layer.edges[k], density))
            out[:, i] += _over_heights(sheet, heights, cuts)
    return out[0].reshape(shape), out[1].reshape(shape)


def height_pdf(
        heights: HeightProfile, term: int, h: float
) -> np.ndarray:
    """
    Return the pdf of the height of a term of a vertical structure at
    each of its knots at the height h (see HeightProfile): cubic between
    the values of its table (4-point Lagrange, as LogTable, of the values),
    and 0 above its top.
    """
    top = heights.tops[term]
    tables = heights.pdfs[term]
    n = tables.shape[-1]
    height = abs(h)
    if height > top:
        return np.zeros(len(tables))
    x = height / top * (n - 1)
    index = min(max(int(x), 1), n - 3)
    t = x - index
    weights = (-t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2,
               -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6)
    return sum(w * tables[:, index + k - 1] for k, w in enumerate(weights))


def height_shapes(heights: HeightProfile, h: float) -> np.ndarray:
    """
    Return the shape of a vertical structure at each of its knots at the
    height h: the sum over its terms of their shares times their pdfs.
    """
    return sum(heights.shares[t] * height_pdf(heights, t, h)
               for t in range(len(heights.tops)))


def knot_shape(heights: HeightProfile, R: np.ndarray, h: float) -> np.ndarray:
    """
    Return the shape of a vertical structure at the radii R and the height
    h: linear between its knots, held beyond the first and the last.
    """
    values = height_shapes(heights, h)
    if len(values) == 1:
        return np.full(np.shape(R), values[0])
    return np.interp(R, heights.knots, values)


def _path_cuts(
        u: float, v: float, nu: float, nv: float, q: float,
        edges: tuple[float, float] | None = None
) -> list[float]:
    """
    Return the heights where the deflection of a sheet of a thick layer
    at a point bends as the sheet moves along the normal (see
    gbkfit::LensSheetPath): where the point meets its line of nodes, and
    where the ellipse of the point crosses the edges (semi-major axes) of
    a band.
    """
    a2 = q * q * nu * nu + nv * nv
    if not a2 > 0:
        return []
    h0 = (q * q * u * nu + v * nv) / a2
    a_min = abs(u * nv - v * nu) / math.sqrt(a2)
    cuts = [h0]
    for edge in edges if edges is not None else ():
        if a_min < edge:
            d = q * math.sqrt(edge * edge - a_min * a_min) / math.sqrt(a2)
            cuts += [h0 - d, h0 + d]
    return cuts


def _over_heights(
        sheet: Callable[[float], np.ndarray],
        heights: HeightProfile,
        cuts: list[float]
) -> np.ndarray:
    """
    Return the integral over the height (from -top to top, of the highest
    top of the terms of a vertical structure) of the deflection of a sheet
    at the height, cut at its plane, the tops of its terms and the given
    heights, by adaptive quadrature (to 1e-8).
    """
    top = float(np.max(heights.tops))
    points = sorted({
        c for c in [0.0, *heights.tops, *(-heights.tops), *cuts]
        if -top < c < top})
    value, _ = scipy.integrate.quad_vec(
        sheet, -top, top, points=points, epsabs=0, epsrel=1e-8,
        limit=2000)
    return value


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
        layer: EllipticalLayer,
        x: np.ndarray,
        y: np.ndarray,
        factor: Callable[[np.ndarray], np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of the sheet of an elliptical layer (see
    EllipticalLayer), times a factor of the radius (none without), at the
    points from its centre: alpha_u = pi K u J_0 and alpha_v = pi K v J_1
    along its axes (Keeton 2001, with the semi-major axis a(w)^2 = w (u^2
    + v^2 / d)),

        J_n = int_0^w_end q Sigma_sky(a(w)) / d^(n + 1/2) dw,
        d = 1 - (1 - q^2) w,

    q Sigma_sky(a) = Sigma(a / sigma1) / sigma1^2, by the tanh-sinh
    quadrature of 52 nodes over [0, w_end], w_end the w of the end of its
    table (1 - w_end exactly; 1, of a point within it), split at d =
    _LINE_D below it, of Gauss-Legendre nodes in ln d beyond.
    """
    sigma1, sigma2 = layer.stretch
    q = sigma2 / sigma1
    e2 = 1 - q * q
    table = layer.sigma
    a_end = sigma1 * math.exp(
        table.start + table.step * (len(table.values) - 1))
    shape = np.shape(x)
    u, v = _frame(x.ravel() - layer.centre[0], y.ravel() - layer.centre[1],
                  layer.posa)
    uu, vv = u * u, v * v
    # (s_end = 1 - w_end: 0 within the table, else the root in (0, 1] of
    # e2 u^2 s^2 + b s + c)
    aa = a_end * a_end
    c = q * q * (aa - uu) - vv
    b = vv + (2 * q * q - 1) * uu + e2 * aa
    with np.errstate(divide='ignore', invalid='ignore'):
        s_end = np.where(c < 0, -2 * c / (b + np.sqrt(np.maximum(
            b * b - 4 * e2 * uu * c, 0))), 0)
    d_end = q * q + e2 * s_end
    near = d_end < _LINE_D
    s_top = np.where(near, (_LINE_D - q * q) / e2, s_end)
    w_top = 1 - s_top

    def weighted(dw, d, a):
        # (the terms of J_0, of the weights dw in w)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.where(a > 0, a, 1) / sigma1
            sigma = np.where(
                a > 0, table(R) * (1 if factor is None else factor(R)), 0)
        return dw * sigma / sigma1 ** 2 / np.sqrt(d)
    w = w_top[:, None] * _NODES
    d = q * q + e2 * (s_top[:, None] + w_top[:, None] * _ONE_MINUS_NODES)
    terms = weighted(w_top[:, None] * _WEIGHTS, d,
                     np.sqrt(w * (uu[:, None] + vv[:, None] / d)))
    j0, j1 = terms.sum(-1), (terms / d).sum(-1)
    for i in np.flatnonzero(near):
        top, length = math.log(_LINE_D), math.log(_LINE_D / d_end[i])
        total = math.ceil(_LINE_DENSITY * length)
        panels = -(-total // _LINE_PANEL)
        nodes, weights = np.polynomial.legendre.leggauss(
            max(-(-total // panels), 3))
        width = length / panels
        for panel in range(panels):
            dd = np.exp(top - width * (panel + (nodes + 1) / 2))
            ww = 1 - (dd - q * q) / e2
            terms = weighted(width / 2 * weights * dd / e2, dd, np.sqrt(
                ww * (uu[i] + vv[i] / dd)))
            j0[i] += terms.sum()
            j1[i] += (terms / dd).sum()
    alpha_x, alpha_y = _sky(
        math.pi * K * u * j0, math.pi * K * v * j1, layer.posa)
    return alpha_x.reshape(shape), alpha_y.reshape(shape)


def _bands(
        layer: BandLayer, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of thin bands of rings (see BandLayer): the sum
    of those of each band (see _band_slice).
    """
    shape = np.shape(x)
    alpha_x, alpha_y = np.zeros(x.size), np.zeros(x.size)
    for k in range(len(layer.posa)):
        ax, ay = _band_slice(
            x.ravel() - layer.centres[k, 0], y.ravel() - layer.centres[k, 1],
            layer.posa[k], *layer.stretch[k], *layer.edges[k],
            layer.density[k, 0])
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
    change of its argument within (-pi, pi); S + i e a as z^2 / (S - i e
    a) where that is larger, without their cancellation).
    """
    s1, s2 = _branch_root(z, e, a1), _branch_root(z, e, a2)

    def plus(s, a):
        larger = np.abs(s + 1j * e * a) >= np.abs(s - 1j * e * a)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(
                larger, s + 1j * e * a, z * z / (s - 1j * e * a))
    j1 = (a2 * a2 - a1 * a1) / (s1 + s2)
    j2 = z * z / (2 * e ** 3) * -1j * np.log(plus(s2, a2) / plus(s1, a1)) \
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
