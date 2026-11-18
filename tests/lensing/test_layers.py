"""
Tests of the deflections of the mass of galaxies laid out on the sky (see
gbkfit.lensing._layers and gbkfit.model.mass.sky): the native kernels
against their numpy references, the bands and ellipses against 2D brute
force integrals, and analytic lenses made of mass components (a point,
an isothermal sphere, an NFW halo, an inclined disk, a warped disk); and
thick layers to exactly edge-on against the integral over their height.
"""

import math

import numpy as np
import pytest

from gbkfit.driver.drivers.host import DriverHost
from gbkfit.lensing import LensNFW, SkyPoints, _layers
from gbkfit.lensing._deflections import nfw_convergence, nfw_round_deflection
from gbkfit.model.mass import (
    BandLayer, EllipticalLayer, Halo, LogTable, MassContext, MassDisk,
    MassModel, MassPoint, PointLayer, ProfileNFW, ProfilePseudoIsothermal,
    RingGeometry, RoundLayer, SkyPlacement, SurfaceExponential,
    SurfaceNodewise, TracedDisk, VerticalComponent, VerticalExponential,
    VerticalSech2, VerticalThin, sky)
from gbkfit.model.mass.units import G
from layer_references import (
    band_deflection, elliptical_deflection, keeton_deflection,
    layers_deflection)


K = _layers.K

# (the driver of the deflections but where a test gives one: the host)
HOST = DriverHost()

# The arcsec of a radian
ARCSEC = math.degrees(1) * 3600


def _context(**densities) -> MassContext:
    return MassContext(
        critical_density=136.0, omega_m=0.3, ricci_scalar=0.0,
        ricci_scalar_today=0.0, densities=densities, kpc_per_arcsec=5.0)


def _points(n: int, extent: float, seed: int = 0) -> tuple[
        np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-extent, extent, (2, n))


def _brute_force(
        sigma_sky, centre, posa, a_in, a_out, q, x, y, na=400, nphi=800):
    """
    Return the deflection at points of a surface density of the sky
    sigma_sky(a) on the ellipses of the semi-major axis a (from a_in to
    a_out) of the axis ratio q and the position angle posa about a centre,
    by a 2D sum over its ellipses (Gauss-Legendre in a, uniform in their
    angle).
    """
    nodes, weights = np.polynomial.legendre.leggauss(na)
    a = (a_out - a_in) / 2 * nodes + (a_out + a_in) / 2
    da = (a_out - a_in) / 2 * weights
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    p = np.radians(posa)
    eu = np.array([-np.sin(p), np.cos(p)])
    ev = np.array([-np.cos(p), -np.sin(p)])
    A, P = np.meshgrid(a, phi, indexing='ij')
    px = centre[0] + A * np.cos(P) * eu[0] + q * A * np.sin(P) * ev[0]
    py = centre[1] + A * np.cos(P) * eu[1] + q * A * np.sin(P) * ev[1]
    mass = sigma_sky(A) * q * A * da[:, None] * (2 * np.pi / nphi)
    out = np.empty((2, len(x)))
    for i, (xx, yy) in enumerate(zip(x, y)):
        dx, dy = xx - px, yy - py
        r2 = dx * dx + dy * dy
        out[:, i] = K * np.sum(mass * dx / r2), K * np.sum(mass * dy / r2)
    return out


def _band(q, density, centre=(0.3, -0.2), posa=35.0, sigma1=1.3,
          edges=(0.7, 2.9)):
    """A thin band."""
    return BandLayer(
        1.0, np.array([centre]), np.array([posa]),
        np.array([[sigma1, q * sigma1]]), np.array([edges]),
        np.array([[density, (0.0, 0.0, 0.0)]], dtype=float),
        np.zeros((1, 2), dtype=np.int32), None, np.zeros((1, 2)))


@pytest.mark.parametrize('q', [0.05, 0.3, 0.7, 0.95, 0.999, 1.0])
@pytest.mark.parametrize('density', [
    (1.0, 0.0, 0.0), (2.0, -0.3, 0.0), (1.0, 0.2, -0.05)])
def test_band_against_brute_force(q, density):
    # (points off the band, where the brute force converges)
    layer = _band(q, density)
    x, y = _points(60, 6)
    alpha = np.array(_layers.unit_deflection(layer, x, y, HOST))
    sigma1, (c0, c1, c2) = 1.3, density
    reference = _brute_force(
        lambda a: (c0 + c1 * a / sigma1 + c2 * (a / sigma1) ** 2)
        / (sigma1 * q * sigma1),
        (0.3, -0.2), 35.0, 0.7 * sigma1, 2.9 * sigma1, q, x, y,
        na=400 if q > 0.2 else 1600)
    p = np.radians(35.0)
    u = -(x - 0.3) * np.sin(p) + (y + 0.2) * np.cos(p)
    v = -(x - 0.3) * np.cos(p) - (y + 0.2) * np.sin(p)
    a_ray = np.hypot(u, v / q)
    off = (a_ray < 0.7 * sigma1 * 0.95) | (a_ray > 2.9 * sigma1 * 1.05)
    scale = np.abs(reference[:, off]).max()
    np.testing.assert_allclose(
        alpha[:, off], reference[:, off], rtol=0, atol=1e-11 * scale)


def test_band_is_zero_inside():
    layer = _band(0.4, (1.0, 0.3, -0.02))
    p = np.radians(35.0)
    # (points within the inner ellipse, a < 0.7 * 1.3)
    a = np.linspace(0, 0.9, 7)
    angle = np.linspace(0, 2 * np.pi, 11)
    A, T = np.meshgrid(a, angle)
    u, v = A * np.cos(T), 0.4 * A * np.sin(T)
    x = 0.3 - u * np.sin(p) - v * np.cos(p)
    y = -0.2 + u * np.cos(p) - v * np.sin(p)
    alpha_x, alpha_y = _layers.unit_deflection(layer, x, y, HOST)
    assert np.all(alpha_x == 0) and np.all(alpha_y == 0)


@pytest.mark.parametrize('q', [0.2, 0.6, 1.0])
def test_uniform_ellipse_against_brute_force(q):
    # A uniform ellipse: one band from the centre, of a constant density;
    # inside it, the deflection is linear in the position
    sigma = 1.0
    layer = _band(q, (sigma, 0, 0), edges=(0.0, 2.0), sigma1=1.0)
    x, y = _points(40, 5, seed=3)
    alpha = np.array(_layers.unit_deflection(layer, x, y, HOST))
    reference = _brute_force(
        lambda a: sigma / q + 0 * a, (0.3, -0.2), 35.0, 0, 2.0, q, x, y)
    p = np.radians(35.0)
    u = -(x - 0.3) * np.sin(p) + (y + 0.2) * np.cos(p)
    v = -(x - 0.3) * np.cos(p) - (y + 0.2) * np.sin(p)
    outside = np.hypot(u, v / q) > 2.1
    np.testing.assert_allclose(
        alpha[:, outside], reference[:, outside], rtol=0,
        atol=1e-12 * np.abs(reference).max())
    # (inside, of the convergence pi K Sigma_sky, Sigma_sky = sigma / q:
    # alpha_u = 2 pi K Sigma_sky q u / (1 + q), alpha_v = 2 pi K Sigma_sky
    # v / (1 + q))
    inside = np.hypot(u, v / q) < 1.9
    alpha_u = -alpha[0] * np.sin(p) + alpha[1] * np.cos(p)
    alpha_v = -alpha[0] * np.cos(p) - alpha[1] * np.sin(p)
    np.testing.assert_allclose(
        alpha_u[inside], 2 * np.pi * K * sigma * u[inside] / (1 + q),
        rtol=1e-12)
    np.testing.assert_allclose(
        alpha_v[inside], 2 * np.pi * K * sigma * v[inside] / (q * (1 + q)),
        rtol=1e-12)


def _elliptical(q, values, r_end=3.0):
    """A thin elliptical layer of the table of the values to r_end."""
    step = math.log(10) / 64
    start = math.log(r_end) - step * (len(values) - 1)
    return EllipticalLayer(
        (0.1, 0.2), 1.0, 20.0, (1.0, q), LogTable(start, step, values), None,
        (0.0, 0.0))


@pytest.mark.parametrize('q', [0.02, 0.2, 0.6, 0.95, 1.0])
def test_elliptical_against_one_band(q):
    # A quadratic surface density to the end of its table, by Keeton's
    # integral, against the band of it in closed form: inside and out (to
    # about 1e-7, the tanh-sinh quadrature of 52 nodes; 4e-7 at q = 0.02)
    n = 400
    start = math.log(3.0) - math.log(10) / 64 * (n - 1)
    R = np.exp(start + math.log(10) / 64 * np.arange(n))
    c = (1.0, 0.3, -0.05)
    ellipse = _elliptical(q, np.log(c[0] + c[1] * R + c[2] * R * R))
    band = BandLayer(
        1.0, np.array([[0.1, 0.2]]), np.array([20.0]), np.array([[1.0, q]]),
        np.array([[0.0, 3.0]]), np.array([[c, (0.0, 0.0, 0.0)]]),
        np.zeros((1, 2), dtype=np.int32), None, np.zeros((1, 2)))
    x, y = _points(3000, 5, seed=2)
    reference = np.array(_layers.unit_deflection(band, x, y, HOST))
    np.testing.assert_allclose(
        np.array(_layers.unit_deflection(ellipse, x, y, HOST)), reference,
        rtol=0, atol=5e-7 * np.abs(reference).max())


def test_host_against_numpy():
    # (thin: the numpy references of the kernels, to the last digits)
    x, y = _points(500, 8, seed=5)
    rng = np.random.default_rng(6)
    n = 300
    R = np.exp(math.log(4) - math.log(10) / 64 * np.arange(n)[::-1])
    ellipse = EllipticalLayer(
        (0.4, -0.3), 2.5, 140.0, (1.2, 0.3),
        LogTable(math.log(R[0]), math.log(10) / 64, -R / 1.3), None,
        (0.0, 0.0))
    np.testing.assert_allclose(
        np.array(_layers.unit_deflection(ellipse, x, y, HOST)),
        np.array(elliptical_deflection(ellipse, x, y)), rtol=1e-12,
        atol=1e-14 * K)
    nb = 9
    edges = np.cumsum(rng.uniform(0.2, 0.8, nb + 1))
    bands = BandLayer(
        1.7, rng.uniform(-0.3, 0.3, (nb, 2)), rng.uniform(0, 180, nb),
        np.column_stack([np.ones(nb), rng.uniform(0.02, 1, nb)])
        * rng.uniform(0.8, 1.2, (nb, 1)),
        np.column_stack([edges[:-1], edges[1:]]),
        np.stack([rng.uniform(-0.1, 1, (nb, 3)) * np.array([1, 0.3, 0.05]),
                  np.zeros((nb, 3))], axis=1),
        np.zeros((nb, 2), dtype=np.int32), None, np.zeros((nb, 2)))
    # (bands of eccentricities on both sides of the series)
    bands.stretch[0, 1] = bands.stretch[0, 0] * 0.999
    bands.stretch[1, 1] = bands.stretch[1, 0] * 0.995
    np.testing.assert_allclose(
        np.array(_layers.unit_deflection(bands, x, y, HOST)),
        np.array(band_deflection(bands, x, y)), rtol=1e-10,
        atol=1e-13 * K)


def test_point_mass():
    placement = SkyPlacement((1.0, -2.0))
    model = MassModel(MassPoint('bh'))
    mass = model.sky_mass({'mass_bh_log_m': 9.0}, _context(), placement)
    x, y = np.array([3.0, 1.0]), np.array([-2.0, 2.0])
    alpha_x, alpha_y = layers_deflection(mass, x, y, HOST)
    np.testing.assert_allclose(alpha_x, [K * 1e9 / 2, 0], rtol=1e-14)
    np.testing.assert_allclose(alpha_y, [0, K * 1e9 / 4], rtol=1e-14)


def test_isothermal_sphere_gives_the_sis_ring():
    # A pseudo-isothermal halo of a core much smaller than the rays bends
    # as an SIS of its asymptotic circular velocity: 2 pi (v / c)^2, and at
    # any radius as 2 pi (v / c)^2 (sqrt(1 + R^2 / rc^2) - 1) rc / R
    halo = Halo('halo', ProfilePseudoIsothermal('density_radius'))
    rho_s, rc = 1e9, 0.01
    params = {'mass_halo_log_rho': math.log10(rho_s), 'mass_halo_rs': rc}
    model = MassModel(halo)
    assert set(params) == set(model.pdescs())
    mass = model.sky_mass(params, _context(), SkyPlacement((0.0, 0.0)))
    v2 = 4 * math.pi * G * rho_s * rc ** 2
    c = 299792.458
    R = np.geomspace(1e-3, 50, 30)
    alpha_x, alpha_y = layers_deflection(mass, R, 0 * R, HOST)
    expected = 2 * math.pi * v2 / c ** 2 * (
        np.sqrt(1 + (R / rc) ** 2) - 1) * rc / R
    np.testing.assert_allclose(alpha_x, expected, rtol=5e-8)
    np.testing.assert_array_equal(alpha_y, 0)
    # (far from the core: the Einstein radius of the SIS, 4 pi sigma^2 /
    # c^2, sigma^2 = v^2 / 2)
    assert alpha_x[-1] == pytest.approx(
        4 * math.pi * v2 / 2 / c ** 2, rel=2 * rc / R[-1])


def test_nfw_halo_as_the_nfw_lens():
    halo = Halo('halo', ProfileNFW('density_radius'))
    rho_s, rs = 1e7, 15.0
    params = {'mass_halo_log_rho': math.log10(rho_s), 'mass_halo_rs': rs}
    mass = MassModel(halo).sky_mass(
        params, _context(), SkyPlacement((0.5, 0.0)))
    x, y = _points(200, 100, seed=7)
    alpha = np.array(layers_deflection(mass, x, y, HOST))
    # (LensNFW: kappa_s = rho_s r_s / Sigma_cr, in units of radians with
    # 1 / Sigma_cr = 4 pi G / c^2 = pi K per kpc)
    reference = np.array(nfw_round_deflection(
        x - 0.5, y, math.pi * K * rho_s * rs, rs))
    np.testing.assert_allclose(alpha, reference, rtol=1e-8, atol=1e-12)


def test_far_field_of_a_disk():
    # Far away, any mass bends as a point of its mass: 4 G M / (c^2 b)
    disk = MassDisk('disk', SurfaceExponential(), VerticalThin())
    plane = SkyPlacement(
        (0.0, 0.0), disks={'gas': _Rings(posa=30.0, incl=60.0)})
    mass = MassModel(disk).sky_mass(
        {'mass_disk_log_m': 10.0, 'mass_disk_rd': 2.0}, _context(), plane)
    b = 3000.0
    alpha_x, alpha_y = layers_deflection(
        mass, np.array([b]), np.zeros(1), HOST)
    assert alpha_x[0] == pytest.approx(K * 1e10 / b, rel=1e-5)
    assert abs(alpha_y[0]) < 1e-5 * alpha_x[0]


class _Rings:
    """A disk of rings, of a geometry linear in the radius."""

    def __init__(self, x=0.0, y=0.0, posa=0.0, incl=0.0, dposa=0.0,
                 dincl=0.0, lift=0.0, tilt=0.0):
        self._values = (x, y, posa, incl, dposa, dincl, lift, tilt)

    def rings(self, radii):
        x, y, posa, incl, dposa, dincl, lift, tilt = self._values
        z = np.zeros(len(radii))
        return RingGeometry(
            z + x, z + y, posa + dposa * radii, incl + dincl * radii,
            lift * radii, tilt * radii / 10, z)


def _disk_brute_force(
        sigma, rings, edges, x, y, nr=600, nphi=720):
    """
    Return the deflection at points of a thin disk of the surface density
    sigma(R), of rings of the geometry of rings (see _Rings) at each
    radius, as the disk is drawn: each point of its plane (R, theta) on
    the sky where the geometry of its radius puts it, by a 2D sum.
    """
    nodes, weights = np.polynomial.legendre.leggauss(nr)
    R = (edges[1] - edges[0]) / 2 * nodes + (edges[1] + edges[0]) / 2
    dR = (edges[1] - edges[0]) / 2 * weights
    theta = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    geometry = rings.rings(R)
    p, i = np.radians(geometry.posa), np.radians(geometry.incl)
    Rg, T = np.meshgrid(R, theta, indexing='ij')
    xd, yd = Rg * np.cos(T), Rg * np.sin(T)
    zd = geometry.lift[:, None] + geometry.tilt_x[:, None] * xd
    y1 = yd * np.cos(i)[:, None] - zd * np.sin(i)[:, None]
    px = geometry.x[:, None] - xd * np.sin(p)[:, None] \
        - y1 * np.cos(p)[:, None]
    py = geometry.y[:, None] + xd * np.cos(p)[:, None] \
        - y1 * np.sin(p)[:, None]
    mass = sigma(Rg) * Rg * dR[:, None] * (2 * np.pi / nphi)
    out = np.empty((2, len(x)))
    for k, (xx, yy) in enumerate(zip(x, y)):
        dx, dy = xx - px, yy - py
        r2 = dx * dx + dy * dy
        out[:, k] = K * np.sum(mass * dx / r2), K * np.sum(mass * dy / r2)
    return out


class _Density:
    """A thin disk density at knots (see mass.DiskDensity)."""

    def __init__(self, knots, sigma):
        self._knots, self._sigma = knots, sigma

    def knots(self, rstep):
        return self._knots

    def thick(self):
        return False

    def surface_densities(self, R):
        return np.interp(R, self._knots, self._sigma, right=0)[None]

    def height_cuts(self):
        return ()

    def height_tops(self):
        return ()

    def mass_per_unit(self):
        return None


def test_inclined_thin_disk_against_direct_integration():
    # An exponential disk at 70 degrees, of a table (elliptical layer),
    # against the 2D sum of its mass (rays outside it, where the sum
    # converges, to 1e-8; and within, against its bands)
    disk = MassDisk('disk', SurfaceExponential(), VerticalThin())
    rings = _Rings(x=0.2, y=-0.1, posa=30.0, incl=70.0)
    placement = SkyPlacement((0.2, -0.1), disks={'gas': rings})
    mass = MassModel(disk).sky_mass(
        {'mass_disk_log_m': 10.0, 'mass_disk_rd': 1.0}, _context(),
        placement)
    angle = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    x, y = 0.2 + 30 * np.cos(angle), -0.1 + 30 * np.sin(angle)
    alpha = np.array(layers_deflection(mass, x, y, HOST))
    reference = _disk_brute_force(
        lambda R: 1e10 / (2 * np.pi) * np.exp(-R), rings, (0, 28), x, y)
    np.testing.assert_allclose(
        alpha, reference, rtol=0, atol=1e-6 * np.abs(reference).max())
    # (within it: the elliptical layer against the bands of a fine
    # nodewise disk of the same density, to the error of their linear
    # profile)
    knots = np.concatenate(
        [np.linspace(0, 4, 801), np.linspace(4.01, 30, 2600)])
    traced = TracedDisk('traced', 'gas')
    density = _Density(knots, 1e10 / (2 * np.pi) * np.exp(-knots))
    banded = MassModel(traced).sky_mass(
        {'mass_traced_upsilon': 1.0}, _context(gas=density), placement)
    x, y = _points(300, 3, seed=11)
    np.testing.assert_allclose(
        np.array(layers_deflection(mass, x, y, HOST)),
        np.array(layers_deflection(banded, x, y, HOST)), rtol=0,
        atol=2e-6 * K * 1e10 / 1.0)


def test_warped_disk_as_rings_against_direct_integration():
    # A traced disk whose rings tilt, turn and lift with radius, as bands
    # of the geometry of their middles, against the 2D sum of its mass as
    # the disk draws it (continuous): equal to second order in the change
    # of the geometry across a band
    knots = np.linspace(0, 10, 201)
    density = _Density(knots, 1e8 * (1 - knots / 10) ** 2)
    rings = _Rings(x=0.1, y=0.3, posa=20.0, incl=50.0, dposa=3.0,
                   dincl=2.0, lift=0.02, tilt=0.03)
    traced = TracedDisk('traced', 'gas')
    placement = SkyPlacement((0.1, 0.3), disks={'gas': rings})
    mass = MassModel(traced).sky_mass(
        {'mass_traced_upsilon': 1.0}, _context(gas=density), placement)
    angle = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    x, y = 0.1 + 14 * np.cos(angle), 0.3 + 14 * np.sin(angle)
    alpha = np.array(layers_deflection(mass, x, y, HOST))
    reference = _disk_brute_force(
        lambda R: np.interp(R, knots, 1e8 * (1 - knots / 10) ** 2), rings,
        (0, 10), x, y, nr=2000)
    np.testing.assert_allclose(
        alpha, reference, rtol=0, atol=2e-5 * np.abs(reference).max())


@pytest.mark.parametrize('q', [0.3, 0.7, 0.98])
def test_elliptical_nfw_lens_as_its_convergence_by_keeton(q):
    # LensNFW by the native kernel of a table of its convergence, against
    # Keeton's integral of its closed convergence (the numpy kernel it had)
    params = dict(kappa_s=0.3, r_s=1.5, q=q, posa=25.0, xpos=0.2, ypos=-0.1)
    x, y = _points(2000, 20, seed=8)
    alpha = np.array(LensNFW().deflection(
        SkyPoints(x, y, (0.0, 0.0)), params, HOST))
    p = np.radians(25.0)
    dx, dy = x - 0.2, y + 0.1
    u = -np.sin(p) * dx + np.cos(p) * dy
    v = np.cos(p) * dx + np.sin(p) * dy
    alpha_u, alpha_v = keeton_deflection(
        u, v, q, lambda xi: 0.3 * nfw_convergence(xi / 1.5))
    reference = np.array([-np.sin(p) * alpha_u + np.cos(p) * alpha_v,
                          np.cos(p) * alpha_u + np.sin(p) * alpha_v])
    np.testing.assert_allclose(
        alpha, reference, rtol=0, atol=2e-7 * np.abs(reference).max())


def _sech2(hz):
    """The vertical structure of a sech2 law of the scale height hz."""
    return VerticalSech2().height_profile(dict(hz=hz), _context())


def _layers_of_every_kind():
    """Return a layer of each kind, thin and thick."""
    rng = np.random.default_rng(12)
    step = math.log(10) / 64
    R = np.exp(math.log(1e-3) + step * np.arange(400))
    thin = EllipticalLayer(
        (0.3, -0.2), 1.0, 30.0, (1.2, 0.5),
        LogTable(math.log(1e-3), step, -R / 2), None, (0, 0))
    thick = EllipticalLayer(
        (0.3, -0.2), 1.0, 30.0, (1.0, 0.1), thin.sigma, _sech2(0.3),
        (0.4, 0.8))
    nb = 7
    edges = np.linspace(0, 6, nb + 1)
    knots = np.linspace(0, 6, nb + 1)
    flaring = sky.height_profile(
        knots, np.ones((1, nb + 1)), [3.0],
        lambda z: 1 / (2 * (0.2 + 0.05 * knots[None, :, None])
                       * np.cosh(z / (0.2 + 0.05 * knots[None, :, None]))
                       ** 2))
    bands = BandLayer(
        1.0, rng.uniform(-0.2, 0.2, (nb, 2)), rng.uniform(0, 180, nb),
        np.column_stack([np.ones(nb), rng.uniform(0.02, 1, nb)]),
        np.column_stack([edges[:-1], edges[1:]]),
        rng.uniform(0, 1, (nb, 2, 3)) * [1, 0.2, 0.02],
        np.column_stack([np.arange(nb), np.arange(1, nb + 1)]).astype(
            np.int32), flaring, rng.uniform(-0.5, 0.5, (nb, 2)))
    # (of one plane)
    flat = BandLayer(
        1.0, np.tile([0.1, -0.1], (nb, 1)), np.full(nb, 40.0),
        np.tile([1.0, 0.05], (nb, 1)), bands.edges, bands.density,
        bands.knots, flaring, np.tile([0.6, 0.7], (nb, 1)))
    point = PointLayer((0.1, 0.1), 1.0)
    round_ = RoundLayer((0.1, 0.1), 1.0, LogTable(
        math.log(1e-3), step, np.log(1 - np.exp(-R))))
    return dict(thin=thin, thick=thick, bands=bands, flat=flat, point=point,
                round=round_)


@pytest.mark.parametrize(
    'kind', ['thin', 'thick', 'bands', 'flat', 'point', 'round'])
def test_driver_against_host(driver, kind):
    # (the kernels of the driver, the rays and deflections on the host)
    # (to 1e-12 of the largest: the device rounds and fuses differently)
    layer = _layers_of_every_kind()[kind]
    x, y = _points(3000, 8, seed=13)
    reference = np.array(_layers.unit_deflection(layer, x, y, HOST))
    np.testing.assert_allclose(
        np.array(_layers.unit_deflection(layer, x, y, driver)), reference,
        rtol=0, atol=1e-12 * np.abs(reference).max())


def test_elliptical_nfw_lens_on_the_driver(driver):
    # (its convergence on the kernels of the driver, as on the host)
    params = dict(kappa_s=0.3, r_s=1.5, q=0.6, posa=25.0, xpos=0.2, ypos=-0.1)
    points = SkyPoints(*_points(2000, 20, seed=9), (0.0, 0.0))
    reference = np.array(LensNFW().deflection(points, params, HOST))
    np.testing.assert_allclose(
        np.array(LensNFW().deflection(points, params, driver)), reference,
        rtol=0, atol=1e-12 * np.abs(reference).max())


# Thick layers, to exactly edge-on: the kernels against the integral over
# the height of the numpy references of their sheets, by adaptive
# quadrature (see layer_references); at most 1e-5 of the largest
# deflection apart (0.02 mas of a disk of 5e10 Msun at z 0.35, rays
# 0.1 to 20 arcsec from its centre)
THICK_TOLERANCE = 1e-5


def _rays(seed=0):
    """Rays (kpc) near the centre of a disk (0.5-5 kpc) and far (10-100)."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * np.pi, 8)
    radius = np.concatenate([rng.uniform(0.5, 5, 5), rng.uniform(10, 100, 3)])
    return radius * np.cos(angle), radius * np.sin(angle)


# Of a smooth layer whose shape changes with radius: the tanh-sinh
# quadrature of its sheets does not resolve the bends of its shape at its
# knots (some 5e-5 of each sheet, measured against adaptive quadrature),
# which the quadrature of the height then integrates as noise
KNOTS_TOLERANCE = 5e-5


def _against_heights(layer, x, y, reference, tolerance=THICK_TOLERANCE):
    alpha = np.array(_layers.unit_deflection(layer, x, y, HOST))
    expected = np.array(reference(layer, x, y))
    np.testing.assert_allclose(
        alpha, expected, rtol=0, atol=tolerance * np.abs(expected).max())


@pytest.mark.parametrize('incl, vertical', [
    (60.0, VerticalSech2()), (89.0, VerticalSech2()), (90.0, VerticalSech2()),
    (90.0, VerticalExponential())])
def test_thick_elliptical_disk_to_edge_on(incl, vertical):
    disk = MassDisk('disk', SurfaceExponential(), vertical)
    placement = SkyPlacement((0.0, 0.0), disks={'gas': _Rings(
        posa=30.0, incl=incl)})
    (layer,) = MassModel(disk).sky_mass(
        {'mass_disk_log_m': 0.0, 'mass_disk_rd': 3.0, 'mass_disk_hz': 0.4},
        _context(), placement).layers()
    _against_heights(layer, *_rays(), elliptical_deflection)


@pytest.mark.parametrize('incl', [60.0, 90.0])
def test_thick_bands_to_edge_on(incl):
    # (a nodewise disk, of bands in the plane of the mass, linear between
    # its nodes)
    disk = MassDisk('disk', SurfaceNodewise([0, 2, 5, 9, 14]),
                    VerticalSech2())
    placement = SkyPlacement((0.0, 0.0), disks={'gas': _Rings(
        posa=30.0, incl=incl)})
    (layer,) = MassModel(disk).sky_mass(
        {'mass_disk_log_sigma': np.array([8.0, 7.6, 7.0, 6.0]),
         'mass_disk_hz': 0.5}, _context(), placement).layers()
    _against_heights(layer, *_rays(1), band_deflection)


class _ThickDensity:
    """
    A thick disk density at knots, of two terms of sech2 pdfs of their
    own cuts, the first flaring (see mass.DiskDensity).
    """

    def __init__(self, knots):
        self._knots = knots

    def knots(self, rstep):
        return self._knots

    def thick(self):
        return True

    def surface_densities(self, R):
        return np.stack([np.interp(R, self._knots, s, right=0) for s in (
            1e8 * np.exp(-self._knots / 3), 3e7 * np.exp(-self._knots / 5))])

    def height_pdfs(self, R, z):
        h = np.stack([0.2 + 0.05 * np.asarray(R), 0.8 + 0 * np.asarray(R)])
        cut = np.array([2.0, 4.0])[:, None, None]
        h, z = h[:, :, None], np.asarray(z)[None, None, :]
        pdf = 1 / (2 * h * np.cosh(z / h) ** 2) / np.tanh(cut / h)
        return np.where(np.abs(z) <= cut, pdf, 0)

    def height_cuts(self):
        return (2.0, 4.0)

    def height_tops(self):
        return (2.0, 4.0)

    def mass_per_unit(self):
        return 1.0


def test_thick_disk_in_the_layer_of_a_component():
    # (an elliptical layer whose shape changes with radius, of terms of
    # their own cuts; edge-on as the others, of the same quadrature of the
    # height)
    incl = 60.0
    disk = MassDisk('disk', SurfaceExponential(), VerticalComponent('gas'))
    density = _ThickDensity(np.linspace(0, 12, 13))
    placement = SkyPlacement((0.0, 0.0), disks={'gas': _Rings(
        posa=30.0, incl=incl)})
    (layer,) = MassModel(disk).sky_mass(
        {'mass_disk_log_m': 0.0, 'mass_disk_rd': 3.0},
        _context(gas=density), placement).layers()
    assert len(layer.heights.knots) == 13
    _against_heights(
        layer, *_rays(2), elliptical_deflection, KNOTS_TOLERANCE)


def test_warped_thick_disk_through_edge_on():
    # A traced disk in its own layer, of rings from 85 to 93 degrees, which
    # turn and tilt: each band of its own plane, through edge-on
    density = _ThickDensity(np.linspace(0, 8, 9))
    rings = _Rings(posa=20.0, incl=85.0, dposa=2.0, dincl=1.0, tilt=0.02)
    placement = SkyPlacement((0.0, 0.0), disks={'gas': rings})
    (layer,) = MassModel(TracedDisk('traced', 'gas')).sky_mass(
        {'mass_traced_upsilon': 1.0}, _context(gas=density),
        placement).layers()
    # (its bands from 0.08 to 0.009 in axis ratio, and back, edge-on)
    ratio = layer.stretch[:, 1] / layer.stretch[:, 0]
    assert ratio.min() < 0.01 and ratio[0] > 0.07 and ratio[-1] > 0.04
    _against_heights(layer, *_rays(3), band_deflection)


def test_thin_band_seen_edge_on_near_its_line():
    # (q = cos(90 deg): S + i e a cancels near the line, where it takes
    # z^2 / (S - i e a))
    q = math.cos(math.radians(90.0))
    layer = _band(q, (1.0, 0.2, -0.03), centre=(0.0, 0.0), posa=0.0)
    u = np.concatenate([np.linspace(-0.01, 0.01, 21), [0.0, 1e-9]])
    v = np.concatenate([np.full(21, -3e-7), [-6.6e-7, -1e-8]])
    x, y = -v, u
    alpha = np.array(_layers.unit_deflection(layer, x, y, HOST))
    assert np.all(np.isfinite(alpha))
    expected = np.array(band_deflection(layer, x, y))
    np.testing.assert_allclose(
        alpha, expected, rtol=1e-9, atol=1e-12 * np.abs(expected).max())
