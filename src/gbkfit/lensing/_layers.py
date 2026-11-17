"""
The deflections of the mass of a galaxy laid out on the sky (see
model.mass.SkyMass): the deflection angle (alpha-hat, radians) of the
light that passes the plane of the galaxy at points of the plane (kpc,
in the frame of its layers), of each of its layers per unit scale (see
unit_deflection; the mass deflects as the sum of its layers, each scale
times that of its unit shape, see MassDeflections):

    alpha-hat(b) = K int Sigma(b') (b - b') / |b - b'|^2 d^2 b',

K = 4 G / c^2, of the surface density Sigma of the sky (Msun/kpc^2). A
point bends as K M b / |b|^2, a round layer as K M2D(<b) b / |b|^2, an
elliptical layer by the integrals over the ellipses within each point
(Keeton 2001; see elliptical_convergence, which the elliptical shapes of
lenses use too), and a band of rings in closed form (see _bands): the
last two by the native kernels of a driver (float64; the points and their
deflections on the host, copied to and from its device).
"""

import dataclasses
import math
from collections.abc import Callable

import astropy.constants
import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.model.mass import (
    BandLayer, EllipticalLayer, LogTable, PointLayer, RoundLayer, SkyLayer)


__all__ = [
    'SheetFrame',
    'elliptical_convergence',
    'tanh_sinh_nodes',
    'unit_deflection'
]


# K = 4 G / c^2 (kpc / Msun)
K = (4 * astropy.constants.G / astropy.constants.c ** 2).to_value(
    'kpc / solMass')


def tanh_sinh_nodes(
        step: float = 1 / 8, extent: float = 3.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the nodes w of the tanh-sinh quadrature over [0, 1], 1 - w
    (exact near w = 1) and their weights.
    """
    tau = np.arange(-extent, extent + step / 2, step)
    s = math.pi / 2 * np.sinh(tau)
    weights = step * math.pi / 4 * np.cosh(tau) / np.cosh(s) ** 2
    return 1 / (1 + np.exp(-2 * s)), 1 / (1 + np.exp(2 * s)), weights


_NODES, _ONE_MINUS_NODES, _WEIGHTS = tanh_sinh_nodes()


@dataclasses.dataclass(frozen=True)
class SheetFrame:
    """
    Where elliptical sheets are on the sky (e.g. the slices of an
    elliptical layer, see EllipticalLayer), in a unit of length or of
    angle, the same for all: an axisymmetric profile f(R), R the radius
    in the plane of a sheet, stretched by sigma1 along the position angle
    of its major axis and sigma2 across (so f(a / sigma1) / (sigma1
    sigma2) on the ellipse of the semi-major axis a), about a centre moved
    by the height of the sheet along a normal.

    Attributes
    ----------
    centre : tuple of float
        The centre.
    posa : float
        The position angle (degrees, north through east) of the major
        axis.
    stretch : tuple of float
        sigma1 and sigma2, sigma1 >= sigma2 >= 0.
    heights : np.ndarray
        The height of each sheet along the normal.
    normal : tuple of float
        The vector of the sky of the normal (per unit of height).
    """
    centre: tuple[float, float]
    posa: float
    stretch: tuple[float, float]
    heights: np.ndarray
    normal: tuple[float, float]


def unit_deflection(
        layer: SkyLayer,
        x: np.ndarray,
        y: np.ndarray,
        driver: Driver
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection angle of the unit shape of a layer of the mass
    of a galaxy on the sky (of a scale of 1).

    Parameters
    ----------
    layer : SkyLayer
        The layer.
    x, y : np.ndarray
        The points of its plane (kpc, in the frame of the layer).
    driver : Driver
        The driver whose native kernels compute the deflections of
        elliptical layers and bands (the points and their deflections on
        the host, copied to and from its device).

    Returns
    -------
    tuple of np.ndarray
        The deflection angle (radians) along x and y; 0 at the centre of a
        point or a round layer.
    """
    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    match layer:
        case PointLayer():
            return _radial(layer.centre, x, y, np.ones_like)
        case RoundLayer():
            return _radial(layer.centre, x, y, layer.mass)
        case EllipticalLayer():
            # (of the convergence pi K Sigma, per kpc)
            table = layer.sigma
            kappa = LogTable(
                table.start, table.step,
                np.atleast_2d(table.values) + math.log(math.pi * K))
            return elliptical_convergence(SheetFrame(
                layer.centre, layer.posa, layer.stretch, layer.heights,
                layer.normal), kappa, x, y, driver)
        case BandLayer():
            alpha_x, alpha_y = _bands(layer, x, y, driver)
            return 2 * math.pi * K * alpha_x, 2 * math.pi * K * alpha_y
    raise TypeError(f"no deflection for a layer of {type(layer).__name__}")


def elliptical_convergence(
        frame: SheetFrame,
        kappa: LogTable,
        x: np.ndarray,
        y: np.ndarray,
        driver: Driver
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of a convergence laid out in elliptical sheets,
    (1 / pi) int kappa(t') (t - t') / |t - t'|^2 d^2 t', by the integrals
    over the ellipses within each point (Keeton 2001): of each sheet, at
    the points from its centre moved by its height along the normal,
    alpha_u = u J_0 and alpha_v = v J_1 along its axes, with the
    semi-major axis a(w)^2 = w (u^2 + v^2 / d),

        J_n = int_0^w_end q kappa_sky(a(w)) / d^(n + 1/2) dw,
        d = 1 - (1 - q^2) w,

    q kappa_sky(a) = kappa(a / sigma1) / sigma1^2, by the tanh-sinh
    quadrature of 52 nodes over [0, w_end], w_end the w of the end of its
    table (or 1, of a point within it; the sheets end there): to about
    1e-7 (4e-7 for the axis ratio 0.02, a thin disk 1 degree from
    edge-on).

    Parameters
    ----------
    frame : SheetFrame
        Where the sheets are.
    kappa : LogTable
        The convergence of each sheet (a row of the table for each), of
        the radius in its plane: dimensionless, of points in angles (the
        deflection then in their unit), or per unit of length, of points
        in lengths (the deflection then in radians; e.g. pi K Sigma of a
        surface density Sigma, see the module).
    x, y : np.ndarray
        The points.
    driver : Driver
        The driver whose native kernels compute it (the points and their
        deflections on the host, copied to and from its device).

    Returns
    -------
    tuple of np.ndarray
        The deflection along x and y, of the shape of the points.
    """
    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    sigma1, sigma2 = frame.stretch
    q = sigma2 / sigma1
    p = math.radians(frame.posa)
    # (of the points within the tables, d and the weights over sqrt(d) of
    # the quadrature over all of [0, 1], see gbkfit::LensNodes)
    d = q * q + (1 - q * q) * _ONE_MINUS_NODES
    nodes = np.stack([
        _NODES, _ONE_MINUS_NODES, _WEIGHTS, 1 / d, _WEIGHTS / np.sqrt(d)])

    def call(
            kernels: type, x_d: DeviceArray, y_d: DeviceArray,
            alpha_x_d: DeviceArray, alpha_y_d: DeviceArray
    ) -> None:
        kernels.elliptical_deflection(
            x_d, y_d, (math.sin(p), math.cos(p), q, sigma1), frame.centre,
            frame.normal,
            *_on_device(driver, np.atleast_2d(kappa.values)), kappa.start,
            kappa.step, *_on_device(driver, frame.heights, nodes),
            alpha_x_d, alpha_y_d)
    return _native(driver, call, x, y)


def _radial(
        centre: tuple[float, float],
        x: np.ndarray,
        y: np.ndarray,
        mass: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection of a circular mass about a centre, of its
    projected mass within circles (a function of the radii): K M2D(<b) b
    / |b|^2.
    """
    dx, dy = x - centre[0], y - centre[1]
    r2 = dx * dx + dy * dy
    inside = r2 > 0
    radii = np.sqrt(np.where(inside, r2, 1))
    scale = np.where(inside, K * mass(radii) / np.where(inside, r2, 1), 0)
    return scale * dx, scale * dy


def _bands(
        layer: BandLayer, x: np.ndarray, y: np.ndarray, driver: Driver
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection over 2 pi K of bands of rings (see BandLayer):
    the sum of those of each band in each slice, at the points from its
    centre moved by the height of the slice along its normal. A thin
    homoeoid of the semi-major axis a and the mass dm bends nothing inside
    it, and outside, alpha* = alpha_u - i alpha_v = K dm / S(a), S(a) =
    sqrt(z^2 - e^2 a^2), z = u + i v, e^2 = 1 - q^2 (S on the branch of z
    at infinity, cut on the segment between its foci). Of a band, dm = 2
    pi q Sigma_sky(a) a da from a_in = sigma1 R_in to min(a_out, a_ray),
    a_ray the ellipse of the point: alpha* = 2 pi K sum_m d_m J_(m+1), of
    q Sigma_sky(a) = sum_m d_m a^m (d_m = c_m / sigma1^(m + 2)) and J_n =
    int a^n / S da, in closed form,

        J_1 = (a2^2 - a1^2) / (S1 + S2),
        J_2 = z^2 / (2 e^3) (-i) ln((S2 + i e a2) / (S1 + i e a1))
              - (a2 S2 - a1 S1) / (2 e^2),
        J_3 = [2 z^2 J_1 - (a2^2 S2 - a1^2 S1)] / (3 e^2),

    S1 = S(a1), S2 = S(a2) (J_1 as the difference of -S / e^2, without
    its cancellation; the logarithm as that of asin(e a / z)), or below e
    = 0.1, where those lose digits as 1 / e^3, by their series in e^2, of
    10 terms (1 / S = sum_k binom(2k, k) / 4^k e^2k a^2k / z^(2k + 1),
    of terms that fall at least as (e / q)^2 outside the band).
    """
    sigma1 = layer.stretch[:, 0]
    p = np.radians(layer.posa)
    frames = np.column_stack([
        np.sin(p), np.cos(p), layer.stretch[:, 1] / sigma1,
        sigma1 * layer.edges[:, 0], sigma1 * layer.edges[:, 1]])
    coefficients = layer.density / sigma1[:, None] ** np.array([2, 3, 4])

    def call(
            kernels: type, x_d: DeviceArray, y_d: DeviceArray,
            alpha_x_d: DeviceArray, alpha_y_d: DeviceArray
    ) -> None:
        kernels.band_deflection(
            x_d, y_d, *_on_device(
                driver, layer.centres, frames, layer.normals, coefficients,
                layer.heights), alpha_x_d, alpha_y_d)
    return _native(driver, call, x, y)


def _native(
        driver: Driver,
        call: Callable[..., None],
        x: np.ndarray,
        y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the deflection at points (on the host) of a call of the native
    kernels of a driver (float64; the class Lensing), on its device: of
    the kernels, the points and the arrays of their deflections along x
    and y there.
    """
    shape = x.shape
    x_d, y_d = _on_device(driver, x.ravel(), y.ravel())
    alpha_x_d = driver.mem_alloc_d((x.size,), np.float64)
    alpha_y_d = driver.mem_alloc_d((x.size,), np.float64)
    call(driver.native_class('Lensing', np.float64), x_d, y_d,
         alpha_x_d, alpha_y_d)
    return (driver.mem_copy_d2h(alpha_x_d).reshape(shape),
            driver.mem_copy_d2h(alpha_y_d).reshape(shape))


def _on_device(driver: Driver, *arrays: np.ndarray) -> list[DeviceArray]:
    """
    Return arrays of the host on the device of a driver (contiguous,
    float64).
    """
    return [
        driver.mem_copy_h2d(np.ascontiguousarray(array, dtype=np.float64))
        for array in arrays]
