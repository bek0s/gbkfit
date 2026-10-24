"""
Tests for the mass of galaxies: its components against independent
formulas and integrals, their parameterisations, the distance of their
galaxy, and the velocity traits 'mass', which take their circular
velocity.
"""

import copy
import dataclasses
import functools
import logging

import astropy.cosmology
import numpy as np
import pytest
import scipy.integrate
import scipy.special

from gbkfit.model import Extras, galaxy_parser
from gbkfit.model.distance import (
    KPC_PER_ARCSEC_MPC, DistanceDirect, DistanceRedshift)
from gbkfit.model.mass import (
    DarkFraction, MassContext, MassDisk, MassModel, MassPoint, MassSpheroid,
    MassUser, ProfileBurkert, ProfileCoreNFW, ProfileDekelZhao,
    ProfileEinasto, ProfileNFW, ProfileNFWTruncated, ProfileNodewise,
    ProfilePseudoIsothermal, ProfileSersic, ProfileZhao,
    ResponseBlumenthal86, ResponseGnedin04, ResponseSIDM, ShapeFlattened,
    SurfaceComponent, SurfaceExponential, SurfaceGaussianRing,
    SurfaceNodewise, VerticalComponent, VerticalSech, VerticalExponential,
    VerticalGauss, VerticalSech2, VerticalThin, mass_component_parser)
from gbkfit.model.mass import _disk_gravity
from gbkfit.utils.parseutils import ConfigError


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The epoch of Planck 2018 at z = 0
CONTEXT = MassContext(
    critical_density=astropy.cosmology.Planck18.critical_density0.to_value(
        'solMass / kpc3'),
    omega_m=astropy.cosmology.Planck18.Om0)

# The vertical laws, by type
LAWS = dict(
    thin=VerticalThin, exponential=VerticalExponential, sech2=VerticalSech2,
    sech=VerticalSech,
    gauss=VerticalGauss)


def exponential_disk(vertical='thin'):
    """An exponential mass disk of a vertical law."""
    return MassDisk('d', SurfaceExponential(), LAWS[vertical]())


def nodewise_disk(nodes, vertical='thin'):
    """A nodewise mass disk of a vertical law."""
    return MassDisk('gas', SurfaceNodewise(nodes), LAWS[vertical]())


def test_exponential_disk_is_its_potential():
    # The circular velocity of a thin disk from its potential, by the
    # Hankel transform of its surface density (Binney & Tremaine 2.165):
    # v^2 = 2 pi G r int_0^inf S(k) k J1(k r) dk, with S(k) = sigma0 rd^2
    # (1 + k^2 rd^2)^(-3/2) for an exponential disk
    log_m, rd = 10.5, 3.0
    sigma0 = 10 ** log_m / (2 * np.pi * rd ** 2)

    def vcirc2(r):
        # (integrated between the zeros of J1, up to where the rest of
        # the integral is negligible)
        if r == 0:
            return 0
        edges = np.concatenate(([0], scipy.special.jn_zeros(1, 500) / r))
        integral = sum(
            scipy.integrate.quad(
                lambda k: k * scipy.special.j1(k * r)
                * (1 + (k * rd) ** 2) ** -1.5, a, b)[0]
            for a, b in zip(edges[:-1], edges[1:]))
        return 2 * np.pi * G * sigma0 * rd ** 2 * r * integral

    radius = np.array([0, 0.1, 1, 3, 6.6, 10])
    result = exponential_disk().vcirc2(
        radius, dict(log_m=log_m, rd=rd), CONTEXT)
    np.testing.assert_allclose(
        result, [vcirc2(r) for r in radius], rtol=1e-4)


@pytest.mark.parametrize('vertical', ['exponential', 'sech2', 'sech', 'gauss'])
def test_thick_disks_of_vanishing_thickness_are_thin(vertical):
    # (the numerical integral of a thick disk against Freeman's closed
    # form; the transforms and shares of the vertical laws against their
    # densities)
    radius = np.concatenate(([0], np.geomspace(1e-3, 1e3, 61)))
    thin = exponential_disk().vcirc2(
        radius, dict(log_m=10, rd=3), CONTEXT)
    thick = exponential_disk(vertical).vcirc2(
        radius, dict(log_m=10, rd=3, hz=1e-9), CONTEXT)
    np.testing.assert_allclose(thick, thin, rtol=1e-6)
    for k in (0, 0.3, 5):
        expected, _ = scipy.integrate.quad(
            lambda z: 2 * LAWS[vertical].pdf(z, 0.7) * np.exp(-k * z),
            0, np.inf)
        assert LAWS[vertical].transform(np.array(k), 0.7) \
            == pytest.approx(expected, rel=1e-10)
    share, _ = scipy.integrate.quad(
        lambda z: 2 * LAWS[vertical].pdf(z, 0.7), 0, 1.3)
    assert LAWS[vertical].within(np.array(1.3), 0.7) \
        == pytest.approx(share, rel=1e-10)


@pytest.mark.parametrize('vertical', ['thin', 'sech2', 'sech', 'gauss'])
def test_disks_of_any_profile_are_exponential_disks(vertical):
    # The pull of the rings of an exponential profile (up to 30 rd, where
    # it is negligible) against the exponential disk (Freeman's closed
    # form, or the Hankel transform when thick), at hz / rd from 1e-3
    from gbkfit.model.mass import surfaces
    log_m, rd = 10, 2.0
    knots = np.linspace(0, 30 * rd, 61)
    radius = knots[1:41]
    if vertical == 'thin':
        gravity = _disk_gravity.thin_disk_gravity(knots, radius, degree=5)
        result = gravity.vcirc2(
            10 ** log_m / (2 * np.pi * rd ** 2) * np.exp(-gravity.radii / rd))
        expected = exponential_disk().vcirc2(
            radius, dict(log_m=log_m, rd=rd), CONTEXT)
        np.testing.assert_allclose(result, expected, rtol=1e-8)
        return
    gravity = _disk_gravity.thick_disk_gravity(knots, radius, degree=5)
    sigma = 10 ** log_m / (2 * np.pi * rd ** 2) * np.exp(-gravity.radii / rd)
    for eta in (1e-3, 0.03, 0.3):
        result = gravity.vcirc2(
            sigma[None], LAWS[vertical].pdf(gravity.heights, eta * rd))
        expected = G * 10 ** log_m / rd * surfaces._exponential_shape_direct(
            radius / rd, functools.partial(LAWS[vertical].transform, h=eta))
        np.testing.assert_allclose(result, expected, rtol=1e-7)


NODEWISE_DISK = nodewise_disk([0, 1, 2.5, 4, 7, 10])
NODEWISE_SIGMA = np.log10([50, 80, 40, 30, 10])


def test_nodewise_disk_is_the_integral_of_its_rings():
    # Thin: 2 G int sigma(a) [a K / (a + R) - a E / (a - R)] da, its
    # principal value by subtraction of sigma(R) (the ln of the two
    # sides of R add up), on the pieces between the nodes and R
    params = dict(log_sigma=NODEWISE_SIGMA)
    nodes = NODEWISE_DISK.surface().nodes()

    def sigma(a):
        return NODEWISE_DISK.surface().surface_density(a, params, CONTEXT)

    def pull(a, R):
        # (with 1 - m = ((a - R) / (a + R))^2, which keeps K finite)
        p = ((a - R) / (a + R)) ** 2
        return a * scipy.special.ellipkm1(p) / (a + R) \
            - (a * scipy.special.ellipe(1 - p) - R) / (a - R)

    def vcirc2(R):
        pieces = np.unique(np.append(nodes, R))
        pieces = pieces[pieces <= nodes[-1]]
        integral = 0
        for a0, a1 in zip(pieces[:-1], pieces[1:]):
            integral += scipy.integrate.quad(
                lambda a: sigma(a) * pull(a, R)
                - R * (sigma(a) - sigma(R)) / (a - R), a0, a1,
                epsabs=0, epsrel=1e-12, limit=200)[0]
        log = np.log((nodes[-1] - R) / R) if R < nodes[-1] else 0
        return 2 * G * (integral - R * sigma(R) * log)

    radius = np.array([0.5, 1, 1.7, 2.5, 5.5, 7, 9.9, 10, 12, 30])
    np.testing.assert_allclose(
        NODEWISE_DISK.vcirc2(radius, params, CONTEXT),
        [vcirc2(R) for R in radius], rtol=1e-9)


@pytest.mark.parametrize('vertical', ['thin', 'sech2'])
def test_disks_far_away_are_their_multipoles(vertical):
    # In the plane, far away: v^2 = G M / R (1 + 3 (<a^2> / 4 - <z^2> / 2)
    # / R^2 + 15 / 64 <8 z^4 - 24 a^2 z^2 + 3 a^4> / R^4), with the means
    # over the mass (the multipoles 2 and 4; the rest is of order R^-6),
    # to the precision of thick disks
    hz = 0.3
    disk = nodewise_disk(NODEWISE_DISK.surface().nodes(), vertical)
    params = dict(log_sigma=NODEWISE_SIGMA, hz=hz)
    nodes = NODEWISE_DISK.surface().nodes()

    def mean(power):
        def integral(p):
            return scipy.integrate.quad(
                lambda a: 2 * np.pi * a ** (p + 1)
                * disk.surface().surface_density(a, params, CONTEXT),
                0, nodes[-1], points=nodes, epsabs=0, epsrel=1e-13)[0]
        return integral(power) / integral(0)

    mass = 2 * np.pi * scipy.integrate.quad(
        lambda a: a * disk.surface().surface_density(a, params, CONTEXT),
        0, nodes[-1], points=nodes, epsabs=0, epsrel=1e-13)[0]
    # (the moments of sech^2(z / hz) / (2 hz))
    z2, z4 = (np.pi ** 2 * hz ** 2 / 12, 7 * np.pi ** 4 * hz ** 4 / 240) \
        if vertical == 'sech2' else (0, 0)
    radius = np.array([200.0, 1000.0])
    expected = G * mass / radius * (
        1 + 3 * (mean(2) / 4 - z2 / 2) / radius ** 2
        + 15 / 64 * (8 * z4 - 24 * mean(2) * z2 + 3 * mean(4)) / radius ** 4)
    np.testing.assert_allclose(
        disk.vcirc2(radius, params, CONTEXT), expected, rtol=1e-8)


def test_disks_at_jumps_of_their_density():
    # A thin disk pulls infinitely at a jump of its surface density (the
    # edge of a hole), outwards; a thick one does not; one without a
    # jump does not (the density is 0 at the last node)
    params = dict(log_sigma=[2, 2], hz=0.1)
    radius = np.array([0.999, 1, 1.001, 3])
    thin = nodewise_disk([1, 2, 3]).vcirc2(radius, params, CONTEXT)
    thick = nodewise_disk([1, 2, 3], 'sech2').vcirc2(
        radius, params, CONTEXT)
    assert thin[1] == -np.inf
    assert np.all(np.isfinite(thin[[0, 2, 3]]))
    assert np.all(np.isfinite(thick))


def ring_vertical_force(sigma, R, z, edges):
    """
    The vertical force of a thin disk of the surface density sigma(a),
    zero outside the edges, at (R, z): its rings by adaptive quadrature,
    split at its edges and around R.
    """
    def pull(a):
        near2, far2 = (a - R) ** 2 + z ** 2, (a + R) ** 2 + z ** 2
        return -G * z * float(sigma(np.array(a))) * a * 4 \
            * scipy.special.ellipe(1 - near2 / far2) / (near2 * np.sqrt(far2))
    lo, hi = edges[0], edges[-1]
    points = sorted(
        p for p in {*edges, R, R - 10 * abs(z), R + 10 * abs(z)}
        if lo < p < hi)
    return scipy.integrate.quad(
        pull, lo, hi, points=points, limit=2000, epsabs=0, epsrel=1e-11)[0]


def test_vertical_forces_of_thin_disks_are_the_pull_of_their_rings():
    # Off the plane, the pull of their rings: a nodewise disk with a hole
    # (a jump at its first node), and an exponential one (from its
    # tables); across the plane, the pull of an infinite sheet, 2 pi G
    # Sigma (of the mean of the two sides at a jump)
    nodewise = MassDisk('d', SurfaceNodewise([1, 3, 6, 10]), VerticalThin())
    params = dict(log_sigma=[8, 7.5, 7])
    sigma = functools.partial(
        nodewise.surface_density, params=params, context=CONTEXT)
    R, z = np.meshgrid([0.5, 1, 2, 5, 10], [0.01, 0.3, 2])
    np.testing.assert_allclose(
        nodewise.vertical_force(R, z, params, CONTEXT),
        np.vectorize(lambda r, h: ring_vertical_force(
            sigma, r, h, [1, 3, 6, 10]))(R, z), rtol=2e-5)
    np.testing.assert_allclose(
        nodewise.vertical_force([2, 1], [-1e-9, 1e-9], params, CONTEXT),
        [2 * np.pi * G * sigma(2), -np.pi * G * 1e8], rtol=1e-6)
    exponential = MassDisk('d', SurfaceExponential(), VerticalThin())
    params = dict(log_m=10, rd=3)
    sigma = functools.partial(
        exponential.surface_density, params=params, context=CONTEXT)
    R, z = np.meshgrid([0, 1, 5, 20], [0.005, 0.2, 3])
    np.testing.assert_allclose(
        exponential.vertical_force(R, z, params, CONTEXT),
        np.vectorize(lambda r, h: ring_vertical_force(
            sigma, r, h, [0, 120]))(R, z), rtol=1e-3)


def test_vertical_forces_of_thick_disks_are_their_stacked_sheets():
    # int p(h) F_thin(R, z - h) dh, of an exponential disk of an
    # exponential vertical law
    thick = MassDisk('d', SurfaceExponential(), VerticalExponential())
    params = dict(log_m=10, rd=3, hz=0.3)
    sigma = functools.partial(
        thick.surface_density, params=params, context=CONTEXT)

    def stacked(R, z):
        return scipy.integrate.quad(
            lambda h: VerticalExponential.pdf(h, 0.3) * ring_vertical_force(
                sigma, R, z - h, [0, 120]), -12, 12, points=[0, z],
            limit=200, epsabs=0, epsrel=1e-9)[0]
    R, z = np.array([2, 5, 10]), np.array([0.1, 0.5, 0.05])
    np.testing.assert_allclose(
        thick.vertical_force(R, z, params, CONTEXT),
        [stacked(*point) for point in zip(R, z)], rtol=5e-4)


def ring_radial_force(sigma, R, z, edges):
    """
    The radial force of a thin disk (see ring_vertical_force): the pull of
    its rings, -G sigma(a) 2 a / (R D) [K(m) - E(m) (a^2 - R^2 + z^2) / ((a
    - R)^2 + z^2)] da, D^2 = (a + R)^2 + z^2, m = 4 a R / D^2.
    """
    def pull(a):
        far2 = (a + R) ** 2 + z ** 2
        m = 4 * a * R / far2
        return -G * float(sigma(np.array(a))) * 2 * a / (R * np.sqrt(far2)) \
            * (scipy.special.ellipk(m) - scipy.special.ellipe(m)
               * (a ** 2 - R ** 2 + z ** 2) / ((a - R) ** 2 + z ** 2))
    lo, hi = edges[0], edges[-1]
    points = sorted(
        p for p in {*edges, R, R - 10 * abs(z), R + 10 * abs(z)}
        if lo < p < hi)
    return scipy.integrate.quad(
        pull, lo, hi, points=points, limit=2000, epsabs=0, epsrel=1e-11)[0]


def test_radial_forces_of_disks_are_the_pull_of_their_rings():
    # Off the plane, the pull of their rings: a thin nodewise disk with a
    # hole, a thin exponential one (from its tables), and a thick one, its
    # stacked sheets (to about 1e-3, see _disk_forces); in the plane, that
    # of the circular velocity, which they approach
    nodewise = MassDisk('d', SurfaceNodewise([1, 3, 6, 10]), VerticalThin())
    params = dict(log_sigma=[8, 7.5, 7])
    sigma = functools.partial(
        nodewise.surface_density, params=params, context=CONTEXT)
    R, z = np.meshgrid([0.5, 1, 2, 5, 10, 14], [0.01, 0.3, 2, -1])
    np.testing.assert_allclose(
        nodewise.forces(R, z, params, CONTEXT)[0],
        np.vectorize(lambda r, h: ring_radial_force(
            sigma, r, h, [1, 3, 6, 10]))(R, z), rtol=1e-4)
    R = np.array([2.0, 5.0])
    in_plane = -nodewise.vcirc2(R, params, CONTEXT) / R
    np.testing.assert_array_equal(
        nodewise.forces(R, 0, params, CONTEXT)[0], in_plane)
    np.testing.assert_allclose(
        nodewise.forces(R, 1e-6, params, CONTEXT)[0], in_plane, rtol=1e-5)
    exponential = MassDisk('d', SurfaceExponential(), VerticalThin())
    params = dict(log_m=10, rd=3)
    sigma = functools.partial(
        exponential.surface_density, params=params, context=CONTEXT)
    R, z = np.meshgrid([1, 5, 20], [0.005, 0.2, 3])
    np.testing.assert_allclose(
        exponential.forces(R, z, params, CONTEXT)[0],
        np.vectorize(lambda r, h: ring_radial_force(
            sigma, r, h, [0, 120]))(R, z), rtol=5e-4)
    thick = MassDisk('d', SurfaceExponential(), VerticalExponential())
    params = dict(log_m=10, rd=3, hz=0.3)

    def stacked(R, z):
        return scipy.integrate.quad(
            lambda h: VerticalExponential.pdf(h, 0.3) * ring_radial_force(
                sigma, R, z - h, [0, 120]), -12, 12, points=[0, z],
            limit=200, epsabs=0, epsrel=1e-9)[0]
    R, z = np.array([2, 5, 10]), np.array([0.1, 0.5, 0.05])
    np.testing.assert_allclose(
        thick.forces(R, z, params, CONTEXT)[0],
        [stacked(*point) for point in zip(R, z)], rtol=1e-3)


def test_unresolved_heights_are_errors():
    disk = nodewise_disk([0, 1, 2], 'sech2')
    with pytest.raises(ValueError, match="heights are not resolved"):
        disk.vcirc2([0.5, 1], dict(log_sigma=[2, 2], hz=1e-12), CONTEXT)


@pytest.mark.parametrize('nodes', [[1], [0, 2, 1], [-1, 1], [0, np.inf]])
def test_nodes_of_nodewise_disks_are_checked(nodes):
    with pytest.raises(
            ConfigError, match="the nodes of a nodewise surface density"):
        SurfaceNodewise(nodes)


@pytest.mark.parametrize('fwhm', [2.0, 0.1])
def test_gaussian_rings_are_the_pull_of_their_rings(fwhm):
    # Against the pull of the rings of a polynomial of degree 5 on fine
    # panels within 10 dispersions of the peak (thin_disk_gravity); far
    # away, v^2 = G M / R (1 + 3 <a^2> / (4 R^2)), to its next multipole
    ring = MassDisk('ring', SurfaceGaussianRing(), VerticalThin())
    params = dict(log_m=10, r_peak=5, fwhm=fwhm)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    radius = np.concatenate((np.geomspace(0.05, 30, 60), [5.0]))
    knots = np.linspace(max(0, 5 - 10 * sigma), 5 + 10 * sigma, 801)
    gravity = _disk_gravity.thin_disk_gravity(knots, radius, degree=5)
    expected = gravity.vcirc2(ring.surface_density(
        gravity.radii, params, CONTEXT))
    result = ring.vcirc2(radius, params, CONTEXT)
    np.testing.assert_allclose(
        result, expected, rtol=0, atol=1e-8 * np.abs(expected).max())
    assert np.any(result < 0)
    a = np.linspace(knots[0], knots[-1], 20001)
    weight = a * ring.surface_density(a, params, CONTEXT)
    mean_a2 = np.trapezoid(weight * a ** 2, a) / np.trapezoid(weight, a)
    assert ring.vcirc2(np.array([1e3]), params, CONTEXT)[0] \
        == pytest.approx(G * 1e10 / 1e3 * (1 + 3 * mean_a2 / 4e6), rel=1e-8)


class Layers:
    """
    The density of a component (see mass.ComponentDensity) between 0.5
    and 10 kpc: a sech2 layer of the scale height 0.3 + flare * 0.04 R,
    and, if flare, a Gaussian one of dispersion 0.5 (kpc).
    """

    def __init__(self, flare):
        self._flare = flare

    def knots(self, rstep):
        return np.linspace(0.5, 10, 20)

    def thick(self):
        return True

    def surface_densities(self, R):
        R = np.asarray(R, dtype=float)
        return np.array([np.exp(-R / 3), self._flare * 0.3 * np.exp(-R / 1.5)])

    def height_pdfs(self, R, z):
        h = 0.3 + self._flare * 0.04 * np.asarray(R, dtype=float)[:, None]
        sech2 = VerticalSech2.pdf(z, h)
        gauss = VerticalGauss.pdf(z, 0.5) * np.ones_like(h)
        return np.array([sech2, gauss])

    def mass_per_unit(self):
        return None


def test_mass_disks_follow_one_component():
    disk = MassDisk('d', SurfaceExponential(), VerticalComponent('hi'))
    assert disk.followed_components() == ('hi',)
    with pytest.raises(ConfigError, match="follows at most one component"):
        MassDisk('d', SurfaceComponent('hi'), VerticalComponent('co'))


def test_vertical_structure_of_a_component():
    # The shape of its density at its subnodes; where it has none, that
    # of the nearest subnodes with density, interpolated between them
    class Gap(Layers):
        def surface_densities(self, R):
            return super().surface_densities(R) * (np.abs(R - 5) > 0.3)
    context = MassContext(1, 1, dict(hi=Gap(flare=1)))
    knots = Gap(flare=1).knots(None)
    z = np.array([0, 0.2, 1])
    shapes = VerticalComponent('hi').shapes(
        np.array([0, 4.5, 5.0, 5.5, 20]), z, context)
    sigmas = Gap(flare=1).surface_densities(knots)
    pdfs = Gap(flare=1).height_pdfs(knots, z)
    total = sigmas.sum(axis=0)
    full = np.einsum('tk,tkz->kz', sigmas, pdfs) \
        / np.where(total > 0, total, 1)[:, None]
    # (the subnode 5.0 is empty, between 4.5 and 5.5)
    np.testing.assert_allclose(
        knots[[0, 8, 9, 10, 19]], [0.5, 4.5, 5, 5.5, 10])
    np.testing.assert_allclose(shapes[0], full[0])
    np.testing.assert_allclose(shapes[1], full[8])
    np.testing.assert_allclose(shapes[2], (full[8] + full[10]) / 2)
    np.testing.assert_allclose(shapes[3], full[10])
    np.testing.assert_allclose(shapes[4], full[19])


@pytest.mark.parametrize('surface', ['exponential', 'nodewise'])
@pytest.mark.parametrize('flare', [0, 1])
def test_disks_as_thick_as_a_component(surface, flare):
    # Of one layer of constant thickness, the disk of its vertical law;
    # flaring, of two layers, the pull of its rings out to 40 rd (where
    # the exponential is negligible)
    context = MassContext(1, 1, dict(hi=Layers(flare)))
    radius = np.geomspace(0.05, 30, 40)
    vertical = VerticalComponent('hi')
    if surface == 'exponential':
        params = dict(log_m=10.3, rd=0.4 if flare else 2.0)
        disk = MassDisk('d', SurfaceExponential(), vertical)
        outer = 40 * params['rd']
    else:
        params = dict(log_sigma=NODEWISE_SIGMA)
        disk = MassDisk('d', NODEWISE_DISK.surface(), vertical)
        outer = NODEWISE_DISK.surface().nodes()[-1]
    result = disk.vcirc2(radius, params, context)
    if not flare:
        law = MassDisk('d', disk.surface(), VerticalSech2())
        np.testing.assert_allclose(
            result, law.vcirc2(radius, params | dict(hz=0.3), context),
            rtol=1e-6)
        return
    knots = np.union1d(np.linspace(0, outer, 801), vertical.knots(context))
    gravity = _disk_gravity.thick_disk_gravity(knots, radius, degree=5)
    expected = gravity.vcirc2(
        disk.surface().surface_density(gravity.radii, params, context)[None],
        vertical.shapes(gravity.radii, gravity.heights, context)[None])
    np.testing.assert_allclose(result, expected, rtol=1e-6)


SPHERES = [
    (ProfileNFW('density_radius'), dict(log_rho=7, rs=10)),
    (ProfileBurkert('density_radius'), dict(log_rho=7.5, rs=4)),
    (ProfilePseudoIsothermal('density_radius'), dict(log_rho=7.5, rs=2)),
    (ProfileEinasto('density_radius'), dict(log_rho=6.5, rs=15, alpha=0.17)),
    (ProfileZhao('density_radius'),
     dict(log_rho=6.5, rs=8, alpha=2, beta=4, gamma=0.5)),
    (ProfileDekelZhao('density_radius'), dict(log_rho=7, rs=5, a=0.5)),
    (ProfileDekelZhao('density_radius'), dict(log_rho=7, rs=5, a=1.5)),
    (ProfileSersic(), dict(log_m=10, re=1.5, n=0.6)),
    (ProfileSersic(), dict(log_m=10, re=1.5, n=4)),
    (ProfileCoreNFW('density_radius'), dict(log_rho=7, rs=10, rc=1.5, n=0.8)),
    (ProfileNFWTruncated('density_radius'), dict(log_rho=7, rs=10, rt=30))]


def test_cored_and_truncated_nfw_halos_change_the_nfw_halo():
    # Of the scales of the NFW halo of their parameters: cored (Read et al.
    # 2016), its mass times tanh(r / r_c)^n, the NFW halo without a core
    # (n = 0); truncated, the NFW halo far within r_t
    nfw = ProfileNFW()
    params = dict(log_m=12, c=8)
    radius = np.array([0.3, 2, 15, 80])
    core = ProfileCoreNFW()
    for n in (0.0, 0.7):
        cored = params | dict(rc=2.0, n=n)
        assert core.scales(cored, CONTEXT) == nfw.scales(params, CONTEXT)
        np.testing.assert_allclose(
            core.enclosed_mass(radius, cored, CONTEXT),
            nfw.enclosed_mass(radius, params, CONTEXT)
            * np.tanh(radius / 2) ** n, rtol=1e-12)
    truncated = ProfileNFWTruncated()
    np.testing.assert_allclose(
        truncated.enclosed_mass(radius, params | dict(rt=1e7), CONTEXT),
        nfw.enclosed_mass(radius, params, CONTEXT), rtol=1e-5)


@pytest.mark.parametrize('profile, params', SPHERES)
def test_spheres_are_their_densities(profile, params):
    # Their enclosed mass is the integral of their density (including
    # radii where the halos use their series), and their circular
    # velocity G M(<r) / r
    sphere = MassSpheroid('s', profile)
    radius = np.array([1e-4, 5e-4, 2e-3, 0.5, 2, 10, 100])
    expected = [
        scipy.integrate.quad(
            lambda r: 4 * np.pi * r ** 2 * float(profile.density(
                np.array(r), params, CONTEXT)),
            0, radius_, points=[1e-5, 1e-3, 0.1], limit=400)[0]
        for radius_ in radius]
    np.testing.assert_allclose(
        profile.enclosed_mass(radius, params, CONTEXT), expected, rtol=1e-6)
    np.testing.assert_allclose(
        sphere.vcirc2(radius, params, CONTEXT),
        G * np.array(expected) / radius, rtol=1e-6)


@pytest.mark.parametrize('sphere, params', [
    (MassPoint('p'), dict(log_m=8))] + [
        (MassSpheroid('s', profile), params) for profile, params in SPHERES
        if profile.type() != 'pseudo_isothermal'])
def test_spheres_are_their_potentials(sphere, params):
    # The forces are -grad(potential), off the plane too
    R, z, step = np.array([0.7, 3, 12]), np.array([0.4, 2, 5]), 1e-4

    def potential(R, z):
        return sphere.potential(R, z, params, CONTEXT)
    force_R, force_z = sphere.forces(R, z, params, CONTEXT)
    np.testing.assert_allclose(
        force_R, -(potential(R + step, z) - potential(R - step, z))
        / (2 * step), rtol=1e-5)
    np.testing.assert_allclose(
        force_z, -(potential(R, z + step) - potential(R, z - step))
        / (2 * step), rtol=1e-5)


def test_the_potential_of_a_pseudo_isothermal_sphere_is_infinite():
    with pytest.raises(ValueError, match="infinite"):
        MassSpheroid('h', ProfilePseudoIsothermal()).potential(
            1, 0, dict(log_m=12, c=10), CONTEXT)


def test_sersic_of_index_1_is_the_deprojected_exponential():
    # The Abel deprojection of an exponential surface density S0
    # exp(-R / rd) is S0 / (pi rd) K0(r / rd) (Binney & Tremaine 2.6),
    # and the total mass is that of the surface density
    b = scipy.special.gammaincinv(2, 0.5)
    mass, re = 1e10, 2.0
    rd = re / b
    sigma0 = mass / (2 * np.pi * rd ** 2)
    radius = np.geomspace(1e-3, 10, 9)
    params = dict(log_m=np.log10(mass), re=re, n=1)
    sersic = ProfileSersic()
    np.testing.assert_allclose(
        sersic.density(radius, params, CONTEXT),
        sigma0 / (np.pi * rd) * scipy.special.k0(radius / rd), rtol=1e-6)
    assert sersic.enclosed_mass(1e3, params, CONTEXT) \
        == pytest.approx(mass, rel=1e-8)


NODEWISE_NODES = [0, 0.5, 1.5, 3, 6]
NODEWISE_PARAMS = dict(log_sigma=[9.5, 9.0, 8.4, 7.6])


def test_nodewise_spheroids_are_the_deprojections_of_their_profile():
    # Projected again, its density is its surface density; its mass is the
    # integral of its density, and in all that of its surface density; its
    # potential pulls with G M(<r) / r^2, and is -G M / r beyond its nodes
    profile = ProfileNodewise(NODEWISE_NODES)
    params = NODEWISE_PARAMS

    def density(r):
        return float(profile.density(np.array([r]), params, CONTEXT)[0])
    for R in (0.2, 1.0, 4.0):
        projected = scipy.integrate.quad(
            lambda t: 2 * R * np.cosh(t) * density(R * np.cosh(t)),
            0, np.arccosh(6 / R), epsrel=1e-10, limit=400)[0]
        assert projected == pytest.approx(
            profile.surface_density(R, params, CONTEXT), rel=1e-7)
    for r in (0.7, 2.5):
        mass = scipy.integrate.quad(
            lambda s: 4 * np.pi * s ** 2 * density(s), 0, r,
            points=[n for n in NODEWISE_NODES if n < r], epsrel=1e-10,
            limit=400)[0]
        assert profile.enclosed_mass(r, params, CONTEXT) == pytest.approx(
            mass, rel=1e-7)
    total = scipy.integrate.quad(
        lambda R: 2 * np.pi * R * profile.surface_density(R, params, CONTEXT),
        0, 6, points=NODEWISE_NODES, epsrel=1e-12)[0]
    assert profile.enclosed_mass(10.0, params, CONTEXT) == pytest.approx(
        total, rel=1e-10)
    r, step = np.array([0.3, 2.0, 5.0]), 1e-5
    pull = (profile.potential(r + step, params, CONTEXT)
            - profile.potential(r - step, params, CONTEXT)) / (2 * step)
    np.testing.assert_allclose(
        pull, G * profile.enclosed_mass(r, params, CONTEXT) / r ** 2,
        rtol=1e-6)
    assert profile.potential(np.array([20.0]), params, CONTEXT)[0] \
        == pytest.approx(-G * total / 20, rel=1e-10)


def test_nodewise_spheroids_of_a_sampled_sersic_are_the_sersic():
    # Of its projected density at 400 nodes (to 20 re), the Sersic spheroid
    # (to the linear sampling), spherical and flattened
    sersic = dict(log_m=10.5, re=1.5, n=2.0)
    nodes = np.concatenate(([0], np.geomspace(1e-3, 30, 400)))
    params = dict(log_sigma=np.log10(
        ProfileSersic().surface_density(nodes[:-1], sersic, CONTEXT)))
    radius = np.array([0.2, 1, 3, 8])
    for shape, q in ((None, {}), (ShapeFlattened(), dict(q=0.6))):
        np.testing.assert_allclose(
            MassSpheroid('b', ProfileNodewise(nodes), shape).vcirc2(
                radius, params | q, CONTEXT),
            MassSpheroid('b', ProfileSersic(), shape).vcirc2(
                radius, sersic | q, CONTEXT), rtol=1e-3)


def test_nodewise_spheroids_load_and_check_their_nodes():
    info = dict(type='spheroid', name='bulge',
                profile=dict(type='nodewise', nodes=NODEWISE_NODES))
    spheroid = mass_component_parser.load(copy.deepcopy(info))
    assert mass_component_parser.dump(spheroid)['profile'] == info['profile']
    assert spheroid.pdescs()['log_sigma'].size() == 4
    for nodes in ([0.5, 1], [0, 2, 1], [0]):
        with pytest.raises(
                ConfigError, match="the nodes of a nodewise profile"):
            ProfileNodewise(nodes)


def test_flattened_spheroids_of_axis_ratio_1_are_spheres():
    # (to 1e-11 for the halos; the tables of the Sersic profile, of its
    # density and of its mass, agree to about 2e-8)
    radius = np.array([0.01, 0.3, 2, 15, 120])
    for profile, params in SPHERES:
        sphere = MassSpheroid('s', profile)
        flat = MassSpheroid('s', profile, ShapeFlattened())
        np.testing.assert_allclose(
            flat.vcirc2(radius, params | dict(q=1), CONTEXT),
            sphere.vcirc2(radius, params, CONTEXT), rtol=1e-7)


def test_flattened_spheroids_are_their_potential_integrals():
    # The circular velocity of a homoeoid density rho(m) / q (Binney &
    # Tremaine 2.88): v^2 = 2 pi G R^2 int_0^inf rho(R / sqrt(1 + u)) du /
    # ((1 + u)^2 sqrt(q^2 + u)), against the integral over m, oblate and
    # prolate
    profile, params = SPHERES[1]
    radius = np.array([0.5, 4, 30])
    for q in (0.3, 2.0):
        flat = MassSpheroid('s', profile, ShapeFlattened())

        def vcirc2(R):
            return 2 * np.pi * G * R ** 2 * scipy.integrate.quad(
                lambda u: float(profile.density(
                    np.array(R / np.sqrt(1 + u)), params, CONTEXT))
                / ((1 + u) ** 2 * np.sqrt(q ** 2 + u)), 0, np.inf,
                epsabs=0, epsrel=1e-10, limit=400)[0]
        np.testing.assert_allclose(
            flat.vcirc2(radius, params | dict(q=q), CONTEXT),
            [vcirc2(R) for R in radius], rtol=1e-7)


@pytest.mark.parametrize('q', [0.3, 1.0, 2.0])
def test_flattened_spheroids_give_their_forces_anywhere(q):
    # Off their plane, those of a homoeoid density rho(m) / q (Binney &
    # Tremaine 2.140): F_R = -2 pi G R int rho du / ((1 + u)^2 (q^2 +
    # u)^(1/2)) and F_z = -2 pi G z int rho du / ((1 + u) (q^2 +
    # u)^(3/2)), m^2 = R^2 / (1 + u) + z^2 / (q^2 + u); a sphere at q = 1
    profile, params = SPHERES[1]
    flat = MassSpheroid('s', profile, ShapeFlattened())
    R, z = np.array([0.5, 4, 30, 3]), np.array([0.2, 3, 10, 0])

    def force(R, z, power, axis):
        def integrand(u):
            m = np.sqrt(R ** 2 / (1 + u) + z ** 2 / (q ** 2 + u))
            return float(profile.density(np.array(m), params, CONTEXT)) \
                / ((1 + u) ** (2.5 - power) * (q ** 2 + u) ** power)
        return -2 * np.pi * G * axis * scipy.integrate.quad(
            integrand, 0, np.inf, epsabs=0, epsrel=1e-10, limit=400)[0]
    force_R, force_z = flat.forces(R, z, params | dict(q=q), CONTEXT)
    np.testing.assert_allclose(
        force_R, [force(*point, 0.5, point[0]) for point in zip(R, z)],
        rtol=1e-7)
    np.testing.assert_allclose(
        force_z, [force(*point, 1.5, point[1]) for point in zip(R, z)],
        rtol=1e-7, atol=1e-12)
    if q == 1:
        sphere = MassSpheroid('s', profile).forces(R, z, params, CONTEXT)
        np.testing.assert_allclose(force_R, sphere[0], rtol=1e-7)
        np.testing.assert_allclose(force_z, sphere[1], rtol=1e-7)


def test_contracted_halos_pull_toward_their_centre():
    # Spherical shells: -v^2(r) / r^2 toward the centre, of the contracted
    # circular velocity at the spherical radius r
    model = mass_model(
        BARYONS + [responding_halo(response=dict(type='blumenthal86'))])
    halo = model.component('halo')
    values = BARYON_VALUES | dict(mass_halo_log_m=12, mass_halo_c=5)
    params = model.component_params('halo', values)
    context = model._context(halo, values, CONTEXT)
    R, z = np.array([0.5, 3, 20]), np.array([0.3, 4, 0])
    r = np.hypot(R, z)
    force_R, force_z = halo.forces(R, z, params, context)
    scale = -halo.vcirc2(r, params, context) / r ** 2
    np.testing.assert_allclose(force_R, scale * R, rtol=1e-12)
    np.testing.assert_allclose(force_z, scale * z, rtol=1e-12)


def test_flat_sersic_spheroids_are_thin_disks():
    # Squashed to a sheet, a Sersic spheroid of index 1 is the thin
    # exponential disk of its mass and of the scale length re / b(1): seen
    # face-on, it is its profile at any axis ratio
    from gbkfit.model.mass import _numerics
    params = dict(log_m=10, re=3, n=1)
    flat = MassSpheroid('b', ProfileSersic(), ShapeFlattened())
    radius = np.array([0.5, 2, 5, 12, 30])
    np.testing.assert_allclose(
        flat.vcirc2(radius, params | dict(q=1e-6), CONTEXT),
        exponential_disk().vcirc2(
            radius, dict(log_m=10, rd=3 / _numerics.sersic_b(1)), CONTEXT),
        rtol=1e-4)
    column = scipy.integrate.quad(
        lambda z: float(flat.density(2.0, z, params | dict(q=0.4), CONTEXT)),
        -np.inf, np.inf, epsrel=1e-10)[0]
    sphere = scipy.integrate.quad(
        lambda z: float(ProfileSersic().density(
            np.hypot(2.0, z), params, CONTEXT)), -np.inf, np.inf,
        epsrel=1e-10)[0]
    assert column == pytest.approx(sphere, rel=1e-6)


def test_spheroid_shapes_load_and_dump():
    info = dict(
        type='spheroid', name='bulge', profile=dict(type='sersic'),
        shape=dict(type='flattened'))
    bulge = mass_component_parser.load(copy.deepcopy(info))
    assert mass_component_parser.dump(bulge) == info
    assert list(bulge.pdescs()) == ['log_m', 're', 'n', 'q']
    assert not bulge.spherical()
    assert MassSpheroid('halo', ProfileNFW()).spherical()


# Baryons that halos respond to: a thin exponential disk and a bulge
BARYONS = [
    dict(type='disk', name='stars', surface=dict(type='exponential'),
         vertical=dict(type='thin')),
    dict(type='spheroid', name='bulge', profile=dict(type='sersic'))]
BARYON_VALUES = dict(
    mass_stars_log_m=10.6, mass_stars_rd=3, mass_bulge_log_m=10,
    mass_bulge_re=1, mass_bulge_n=4)


def responding_halo(**options):
    """The configuration of an NFW halo that responds to BARYONS."""
    return dict(
        type='spheroid', name='halo', profile=dict(type='nfw'),
        baryons=['stars', 'bulge']) | options


def test_contracted_halos_keep_the_dark_matter_of_their_shells():
    # Within r, the dark matter the halo had within r', with r' M(r') =
    # r (M(r') + M_b(r)) (in units of G: M = v^2 r of the halo, and the
    # mass of the baryons within the sphere), against r' found by root
    # finding at each radius
    model = mass_model(
        BARYONS + [responding_halo(response=dict(type='blumenthal86'))])
    params = BARYON_VALUES | dict(mass_halo_log_m=12, mass_halo_c=5)
    radius = np.array([0.5, 2, 5, 20, 100])
    parts = model.vcirc2_components(radius, params, CONTEXT)
    halo = MassSpheroid('h', ProfileNFW())

    def halo_mass(r):
        return float(halo.vcirc2(
            np.array([r]), dict(log_m=12, c=5), CONTEXT)[0]) * r
    masses = model.enclosed_mass_components(radius, params, CONTEXT)
    baryons = G * (masses['stars'] + masses['bulge'])
    for r, baryon, v2 in zip(radius, baryons, parts['halo']):
        initial = scipy.optimize.brentq(
            lambda x: x * halo_mass(x) - r * (halo_mass(x) + baryon),
            r, 1e4 * r, xtol=1e-14, rtol=1e-15)
        assert v2 == pytest.approx(halo_mass(initial) / r, rel=1e-10)


def test_gnedin_and_cautun_contractions():
    # Gnedin et al. (2004): within R = r_bar, the dark matter the halo had
    # within r_bar' of the shell r', with r' M(r_bar') = r (M(r_bar') +
    # M_b(R)), r_bar = 0.85 r_D (r / r_D)^0.8, against r' found by root
    # finding; it contracts the halo less than Blumenthal et al. Cautun et
    # al. (2020): its formula, of the baryon fraction of the context
    params = BARYON_VALUES | dict(mass_halo_log_m=12, mass_halo_c=5)
    radius = np.array([0.5, 2, 5, 20])
    context = dataclasses.replace(CONTEXT, baryon_fraction=0.16)
    models = {
        name: mass_model(BARYONS + [responding_halo(response=dict(type=name))])
        for name in ('blumenthal86', 'gnedin04', 'cautun20')}
    parts = {
        name: model.vcirc2_components(radius, params, context)
        for name, model in models.items()}
    halo, halo_params = MassSpheroid('h', ProfileNFW()), dict(log_m=12, c=5)
    r_d = 5 * ProfileNFW().scales(halo_params, context)[1]

    def averaged(r):
        return 0.85 * r_d * (r / r_d) ** 0.8

    def mass(r):
        return float(halo.vcirc2(
            np.array([averaged(r)]), halo_params, context)[0]) * averaged(r)
    masses = models['gnedin04'].enclosed_mass_components(
        radius, params, context)
    baryons = G * (masses['stars'] + masses['bulge'])
    for R, baryon, v2 in zip(radius, baryons, parts['gnedin04']['halo']):
        r = r_d * (R / (0.85 * r_d)) ** (1 / 0.8)
        initial = scipy.optimize.brentq(
            lambda x: x * mass(x) - r * (mass(x) + baryon),
            r, 1e4 * r, xtol=1e-14, rtol=1e-15)
        assert v2 == pytest.approx(mass(initial) / R, rel=1e-10)
    before = halo.vcirc2(radius, halo_params, context)
    assert np.all(before < parts['gnedin04']['halo'])
    assert np.all(parts['gnedin04']['halo'] < parts['blumenthal86']['halo'])
    eta = baryons / radius * 0.84 / (0.16 * before)
    np.testing.assert_allclose(
        parts['cautun20']['halo'],
        before * (0.45 + 0.38 * (eta + 1.16) ** 0.53), rtol=1e-10)
    with pytest.raises(ValueError, match="cosmic baryon fraction"):
        mass_model(BARYONS + [responding_halo(response=dict(type='cautun20'))]
                   ).vcirc2(radius, params, CONTEXT)
    with pytest.raises(ConfigError, match="takes the radius r_D"):
        MassSpheroid('h', ProfileNFW('density_radius'), baryons=['stars'],
                     response=ResponseGnedin04())


@pytest.mark.parametrize('profile, solve, bounds, given', [
    ('nfw', 'log_m', [9, 15], dict(mass_halo_c=8)),
    ('burkert', 'c', [1, 100], dict(mass_halo_log_m=11.8))])
@pytest.mark.parametrize(
    'response', [{}, dict(response=dict(type='blumenthal86'))])
def test_dark_fractions_set_a_parameter_of_the_halo(
        profile, solve, bounds, given, response):
    model = mass_model(BARYONS + [responding_halo(
        profile=dict(type=profile),
        dark_fraction=dict(solve=solve, bounds=bounds)) | response])
    assert f'mass_halo_{solve}' not in model.pdescs()
    params = BARYON_VALUES | given | dict(mass_halo_fdm=0.3, mass_halo_r_fdm=5)
    parts = model.vcirc2_components(np.array([5.0]), params, CONTEXT)
    v2, baryon_v2 = parts['halo'][0], parts['stars'][0] + parts['bulge'][0]
    assert v2 / (v2 + baryon_v2) == pytest.approx(0.3, rel=1e-10)
    with pytest.raises(ValueError, match="no value of"):
        model.vcirc2(
            np.array([5.0]), params | dict(mass_halo_fdm=0.999), CONTEXT)


def test_responding_halos_are_checked(tmp_path):
    # (of baryons that give their masses within spheres: not those of user
    # components without a function of it, nor tables)
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n    return 100 * R\n")
    with pytest.raises(ConfigError, match=r"\['mine'\] do not give them"):
        mass_model([
            dict(type='user', name='mine', file=str(tmp_path / 'curve.py'),
                 func='vcirc2'),
            responding_halo(
                baryons=['mine'], response=dict(type='blumenthal86'))])
    with pytest.raises(ConfigError, match="needs the baryons"):
        MassSpheroid('h', ProfileNFW(), response=ResponseBlumenthal86())
    with pytest.raises(ConfigError, match="spherical shape"):
        MassSpheroid(
            'h', ProfileNFW(), ShapeFlattened(), baryons=['stars'],
            response=ResponseBlumenthal86())
    with pytest.raises(ConfigError, match="not a parameter"):
        MassSpheroid('h', ProfileNFW(), baryons=['stars'],
                     dark_fraction=DarkFraction('q', [0.1, 1]))
    with pytest.raises(ConfigError, match="does not have"):
        mass_model([responding_halo(response=dict(type='blumenthal86'))])
    with pytest.raises(ConfigError, match="in a loop"):
        mass_model([
            responding_halo(name='a', baryons=['b']),
            responding_halo(name='b', baryons=['a'])])
    info = responding_halo(
        response=dict(type='blumenthal86'),
        dark_fraction=dict(solve='log_m', bounds=[9.0, 15.0]))
    dumped = mass_component_parser.dump(
        mass_component_parser.load(copy.deepcopy(info)))
    assert {key: dumped[key] for key in (
        'baryons', 'response', 'dark_fraction')} == {
        key: info[key] for key in ('baryons', 'response', 'dark_fraction')}
    assert mass_component_parser.dump(
        mass_component_parser.load(copy.deepcopy(dumped))) == dumped


# The units of the scattering of SIDM: cm^2/g in kpc^2/Msun, and km/s
# times Gyr in kpc
SIGMA_UNIT = 2.0883573729600e-10
TIME_UNIT = 1.0227121650456948


@pytest.mark.parametrize('with_baryons', [False, True])
def test_sidm_cores(with_baryons):
    # The isothermal core of SIDM (Jiang et al. 2023): at r1, one
    # scattering in the age, (4 / sqrt(pi)) rho v sigma_m t_age = 1 (rho
    # and the Jeans velocity dispersion v of the halo before, in the
    # potential of the halo and of the baryons); within, the isothermal
    # Jeans-Poisson equation (solved here by solve_ivp), of the density and
    # the mass of the halo before at r1; outside, the halo before
    baryons = [BARYONS[0]] if with_baryons else []
    halo = dict(type='spheroid', name='halo', profile=dict(type='nfw'),
                baryons=['stars'] if with_baryons else [],
                response=dict(type='sidm'))
    model = mass_model(baryons + [halo])
    params = BARYON_VALUES | dict(
        mass_halo_log_m=11, mass_halo_c=15, mass_halo_sigma_m=1,
        mass_halo_t_age=10)
    own = dict(log_m=11, c=15, sigma_m=1, t_age=10)
    before = MassSpheroid('h', ProfileNFW())

    def halo_v2(r):
        return before.vcirc2(r, own, CONTEXT)

    stars = mass_model(baryons) if with_baryons else None

    def baryons_mass(r):
        if stars is None:
            return np.zeros_like(r)
        return stars.enclosed_mass(r, params, CONTEXT)

    def baryon_mass(r):
        return float(baryons_mass(np.array([r]))[0])

    core = model.component('halo').response().core(
        halo_v2, baryons_mass, ProfileNFW(), own, CONTEXT)
    r1 = core.r1

    def mass(r, v2):
        return v2(np.array([r]))[0] * r / G

    rho1 = ProfileNFW().density(np.array([r1]), own, CONTEXT)[0]
    pressure = scipy.integrate.quad(
        lambda r: ProfileNFW().density(np.array([r]), own, CONTEXT)[0] * G
        * (mass(r, halo_v2) + baryon_mass(r)) / r ** 2,
        r1, np.inf, epsrel=1e-10, limit=400)[0]
    scatterings = (4 / np.sqrt(np.pi) * rho1 * np.sqrt(pressure / rho1)
                   * SIGMA_UNIT * 10 * TIME_UNIT)
    assert scatterings == pytest.approx(1, rel=1e-3)

    def isothermal(r, state):
        y, m = state
        return [G * (m + baryon_mass(r)) / (r ** 2 * core.v0 ** 2),
                4 * np.pi * r ** 2 * core.rho0 * np.exp(-y)]

    start = 1e-4 * r1
    m0 = 4 * np.pi / 3 * core.rho0 * start ** 3
    radii = np.array([0.1, 0.5, 0.9]) * r1
    solution = scipy.integrate.solve_ivp(
        isothermal, (start, r1),
        [G * (m0 + baryon_mass(start)) / (2 * start * core.v0 ** 2), m0],
        t_eval=np.append(radii, r1), rtol=1e-11, atol=0)
    y, m = solution.y
    assert core.rho0 * np.exp(-y[-1]) == pytest.approx(rho1, rel=1e-4)
    assert m[-1] == pytest.approx(mass(r1, halo_v2), rel=1e-5)
    R = np.append(radii, [1.5 * r1, 3 * r1])
    v2 = model.vcirc2_components(R, params, CONTEXT)['halo']
    np.testing.assert_allclose(v2[:3], G * m[:3] / radii, rtol=1e-5)
    np.testing.assert_array_equal(v2[3:], halo_v2(R[3:]))
    without = model.vcirc2_components(
        R, params | dict(mass_halo_sigma_m=0), CONTEXT)['halo']
    np.testing.assert_array_equal(without, halo_v2(R))


def test_sidm_responses_are_checked():
    # Of dark matter alone, or with a contraction of the halo before, which
    # needs baryons and is not SIDM
    info = responding_halo(response=dict(
        type='sidm', contraction=dict(type='gnedin04')))
    dumped = mass_component_parser.dump(
        mass_component_parser.load(copy.deepcopy(info)))
    assert dumped['response'] == info['response']
    MassSpheroid('h', ProfileNFW(), response=ResponseSIDM())
    with pytest.raises(ConfigError, match="needs the baryons"):
        MassSpheroid('h', ProfileNFW(), response=ResponseSIDM(
            ResponseBlumenthal86()))
    with pytest.raises(ConfigError, match="is not SIDM"):
        ResponseSIDM(ResponseSIDM())


HALOS = [
    ProfileNFW, ProfileBurkert, ProfilePseudoIsothermal, ProfileEinasto,
    ProfileZhao, ProfileDekelZhao]
SHAPES = dict(
    einasto=dict(alpha=0.17), zhao=dict(alpha=1, beta=3, gamma=1.3),
    dekel_zhao=dict(a=1))


@pytest.mark.parametrize('halo_class', HALOS)
@pytest.mark.parametrize('overdensity', ['200c', '500c', '200m', 'vir'])
def test_halos_have_their_mass_and_concentration(halo_class, overdensity):
    # The mean density within r_D is D rho_ref, the mass within it M_D,
    # and its velocity V_D: given by (M_D, c) or (V_D, c), a halo is the
    # same
    profile = halo_class('mass_concentration', overdensity)
    halo = MassSpheroid('h', profile)
    shape = SHAPES.get(profile.type(), {})
    params = dict(log_m=12, c=9) | shape
    rho_s, r_s = profile.scales(params, CONTEXT)
    r_d = 9 * r_s
    mass = 1e12
    reference = dict(
        c=CONTEXT.critical_density,
        m=CONTEXT.omega_m * CONTEXT.critical_density)
    if overdensity == 'vir':
        x = CONTEXT.omega_m - 1
        density = (18 * np.pi ** 2 + 82 * x - 39 * x ** 2) \
            * CONTEXT.critical_density
    else:
        density = float(overdensity[:-1]) * reference[overdensity[-1]]
    assert mass / (4 / 3 * np.pi * r_d ** 3) == pytest.approx(density)
    assert profile.enclosed_mass(r_d, params, CONTEXT) == pytest.approx(mass)
    by_velocity = MassSpheroid(
        'h', halo_class('velocity_concentration', overdensity))
    radius = np.array([0.5, 5, 50])
    np.testing.assert_allclose(
        by_velocity.vcirc2(
            radius, dict(v=np.sqrt(G * mass / r_d), c=9) | shape, CONTEXT),
        halo.vcirc2(radius, params, CONTEXT), rtol=1e-10)


def test_dekel_zhao_has_the_slopes_of_its_shape():
    # Its logarithmic slope is (a + 3.5 x^(1/2)) / (1 + x^(1/2)): a at the
    # centre, -3.5 far out, and -2 at ((2 - a) / 1.5)^2 r_s
    profile = ProfileDekelZhao('density_radius')
    params = dict(log_rho=7, rs=5, a=0.7)
    x = np.array([1e-6, 0.3, ((2 - 0.7) / 1.5) ** 2, 4, 1e6])
    step = 1e-5

    def log_density(r):
        return np.log(profile.density(r, params, CONTEXT))
    slope = (log_density(5 * x * (1 + step))
             - log_density(5 * x * (1 - step))) \
        / (np.log1p(step) - np.log1p(-step))
    np.testing.assert_allclose(
        -slope, (0.7 + 3.5 * np.sqrt(x)) / (1 + np.sqrt(x)), rtol=1e-6)


def test_zhao_is_nfw_for_its_shape():
    radius = np.array([1e-3, 0.5, 5, 50, 500])
    np.testing.assert_allclose(
        MassSpheroid('h', ProfileZhao()).vcirc2(
            radius, dict(log_m=12, c=9, alpha=1, beta=3, gamma=1), CONTEXT),
        MassSpheroid('h', ProfileNFW()).vcirc2(
            radius, dict(log_m=12, c=9), CONTEXT),
        rtol=1e-10)


@pytest.mark.parametrize('options, message', [
    (dict(parameters='mass'), "parameters must be one of"),
    (dict(overdensity='200x'), "overdensity must be"),
    (dict(parameters='density_radius', overdensity='200c'),
     "take no overdensity")])
def test_halo_options_are_checked(options, message):
    with pytest.raises(ConfigError, match=message):
        ProfileNFW(**options)


def sphere_integral(density, radius, epsrel=1e-10):
    """
    The integral of a density(R, z) over the sphere of a radius about the
    centre (in spherical coordinates, and even in z): 4 pi int_0^radius s^2
    int_0^1 density(s sqrt(1 - mu^2), s mu) dmu ds.
    """
    def shell(mu, s):
        return s ** 2 * float(density(s * np.sqrt(1 - mu ** 2), s * mu))
    return 4 * np.pi * scipy.integrate.dblquad(
        shell, 0, radius, 0, 1, epsabs=0, epsrel=epsrel)[0]


@pytest.mark.parametrize(
    'vertical', ['thin', 'exponential', 'sech2', 'sech', 'gauss'])
def test_disks_enclose_the_mass_of_their_density_within_spheres(vertical):
    # Thin: M (1 - (1 + x) exp(-x)), x = r / rd; thick: the integral of
    # its density over the sphere
    disk = exponential_disk(vertical)
    params = dict(log_m=10, rd=2, hz=0.4)
    radius = np.array([0.5, 3, 12])
    if vertical == 'thin':
        x = radius / 2
        expected = 1e10 * (1 - (1 + x) * np.exp(-x))
    else:
        expected = [sphere_integral(
            lambda R, z: disk.density(R, z, params, CONTEXT), r)
            for r in radius]
    np.testing.assert_allclose(
        disk.enclosed_mass(radius, params, CONTEXT), expected, rtol=1e-8)


@pytest.mark.parametrize('vertical', ['thin', 'exponential'])
def test_nodewise_disks_enclose_their_mass_of_every_radius(vertical):
    # Linear between its nodes, where it bends: 2 pi int Sigma(R) s(R,
    # sqrt(r^2 - R^2)) R dR, between the nodes; all of it beyond them
    nodes = [0.0, 0.3, 0.6, 1.2]
    disk = MassDisk('d', SurfaceNodewise(nodes), LAWS[vertical]())
    params = dict(log_sigma=[8, 7.8, 7.5], hz=0.05)
    radius = np.array([0.1, 0.45, 0.9, 1.2, 50.0, 1e4])

    def mass(r):
        def ring(R):
            sigma = disk.surface().surface_density(R, params, CONTEXT)
            height = np.sqrt(max(r ** 2 - R ** 2, 0))
            share = disk.vertical().share_within(R, height, params, CONTEXT)
            return 2 * np.pi * R * sigma * share
        return scipy.integrate.quad(
            ring, 0, min(r, nodes[-1]), epsabs=0, epsrel=1e-12, limit=200,
            points=[node for node in nodes[1:-1] if node < r])[0]
    np.testing.assert_allclose(
        disk.enclosed_mass(radius, params, CONTEXT),
        [mass(r) for r in radius], rtol=1e-10)


def test_disks_as_thick_as_a_component_enclose_their_density():
    # Of two flaring layers: the integral of its density over the sphere,
    # 2 pi int R int_-h^h rho dz dR (h = sqrt(r^2 - R^2)), with R = r
    # sin(phi), by Gauss-Legendre rules between its subnodes, where its
    # shape bends
    context = MassContext(1, 1, dict(hi=Layers(flare=1)))
    vertical = VerticalComponent('hi')
    disk = MassDisk('d', SurfaceExponential(), vertical)
    params = dict(log_m=10, rd=2)
    radius = 4.0
    t, w = np.polynomial.legendre.leggauss(60)
    knots = vertical.knots(context)
    edges = np.arcsin(np.union1d([0, radius], knots[knots < radius]) / radius)
    expected = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        phi = (a + b) / 2 + (b - a) / 2 * t
        R, h = radius * np.sin(phi), radius * np.cos(phi)
        z = h[:, None] * (1 + t) / 2
        rho = disk.density(
            np.broadcast_to(R[:, None], z.shape), z, params, context)
        column = h * (rho @ w)
        expected += 2 * np.pi * (b - a) / 2 * np.sum(w * R * column * h)
    np.testing.assert_allclose(
        disk.enclosed_mass(radius, params, context), expected, rtol=1e-7)


@pytest.mark.parametrize('q', [0.3, 1.0, 2.0])
def test_flattened_spheroids_enclose_the_mass_of_their_density(q):
    # Oblate, a sphere and prolate: the integral of the density over the
    # sphere; at q = 1, the mass of the profile
    profile, params = SPHERES[1]
    flat = MassSpheroid('s', profile, ShapeFlattened())
    params = params | dict(q=q)
    radius = np.array([0.5, 4, 30])
    expected = [sphere_integral(
        lambda R, z: flat.density(R, z, params, CONTEXT), r) for r in radius]
    np.testing.assert_allclose(
        flat.enclosed_mass(radius, params, CONTEXT), expected, rtol=1e-8)
    if q == 1:
        np.testing.assert_allclose(
            flat.enclosed_mass(radius, params, CONTEXT),
            profile.enclosed_mass(radius, params, CONTEXT), rtol=1e-12)


def test_user_component_gives_its_enclosed_mass_by_a_function(tmp_path):
    (tmp_path / 'point.py').write_text(
        "import numpy as np\n"
        "def vcirc2(R, params):\n"
        "    return params['gm'] / R\n"
        "def mass(r, params):\n"
        "    return np.full_like(r, params['gm'] / 4.30091727e-6)\n")
    path = str(tmp_path / 'point.py')
    component = MassUser('mine', path, 'vcirc2', ['gm'], mass='mass')
    assert component.gives_enclosed_mass()
    np.testing.assert_allclose(component.enclosed_mass(
        np.array([1.0, 3.0]), dict(gm=G * 1e10), CONTEXT), 1e10)
    without = MassUser('mine', path, 'vcirc2', ['gm'])
    assert not without.gives_enclosed_mass()
    with pytest.raises(NotImplementedError, match="no function"):
        without.enclosed_mass(np.array([1.0]), dict(gm=1), CONTEXT)


def test_user_component_is_its_function(tmp_path):
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n"
        "    return params['v'] ** 2 * R / (R + params['r'])\n")
    component = MassUser(
        'mine', str(tmp_path / 'curve.py'), 'vcirc2', ['v', 'r'])
    assert list(component.pdescs()) == ['v', 'r']
    radius = np.array([0, 1, 10])
    np.testing.assert_allclose(
        component.vcirc2(radius, dict(v=100, r=1), CONTEXT),
        1e4 * radius / (radius + 1))
    with pytest.raises(ConfigError, match="does not define a function"):
        MassUser('mine', str(tmp_path / 'curve.py'), 'other')


def test_user_component_gives_its_forces_by_a_function(tmp_path):
    # Those of a point mass; without the function, in the plane only
    (tmp_path / 'point.py').write_text(
        "import numpy as np\n"
        "def vcirc2(R, params):\n"
        "    return params['gm'] / R\n"
        "def forces(R, z, params):\n"
        "    scale = -params['gm'] / np.hypot(R, z) ** 3\n"
        "    return scale * R, scale * z\n")
    path = str(tmp_path / 'point.py')
    component = MassUser('mine', path, 'vcirc2', ['gm'], forces='forces')
    params = dict(gm=G * 1e10)
    R, z = np.array([1.0, 3.0]), np.array([0.5, -2.0])
    expected = MassPoint('bh').forces(R, z, dict(log_m=10), CONTEXT)
    np.testing.assert_allclose(
        component.forces(R, z, params, CONTEXT), expected, rtol=1e-10)
    assert component.gives_forces()
    in_plane = MassUser('mine', path, 'vcirc2', ['gm'])
    assert not in_plane.gives_forces()
    with pytest.raises(NotImplementedError, match="in its plane"):
        in_plane.forces(R, z, params, CONTEXT)


def test_mass_model_adds_its_components():
    model = MassModel([
        MassDisk('stars', SurfaceExponential(), VerticalThin()),
        MassSpheroid('halo', ProfileNFW())])
    assert list(model.pdescs()) == [
        'mass_stars_log_m', 'mass_stars_rd', 'mass_halo_log_m', 'mass_halo_c']
    params = dict(
        mass_stars_log_m=10.5, mass_stars_rd=3, mass_halo_log_m=12,
        mass_halo_c=8)
    radius = np.array([0, 1, 5, 20])
    expected = (
        exponential_disk().vcirc2(radius, dict(log_m=10.5, rd=3), CONTEXT)
        + MassSpheroid('h', ProfileNFW()).vcirc2(
            radius, dict(log_m=12, c=8), CONTEXT))
    np.testing.assert_allclose(
        model.vcirc2(radius, params, CONTEXT), expected, rtol=1e-12)


def test_distances_give_the_length_of_an_arcsec():
    # At 10 Mpc, an arcsec is 48.48 pc; by a redshift, the angular
    # diameter distance of the cosmology; a distance sets no redshift
    # The redshift of the galaxy (0.5 here) gives the distance of
    # DistanceRedshift, unless it has its own (for the distance only), and
    # the epoch of both
    direct = DistanceDirect()
    assert list(direct.pdescs()) == ['distance']
    assert direct.kpc_per_arcsec(dict(distance=10), 0.5) \
        == pytest.approx(0.0484813681)
    assert direct.epoch(dict(distance=10), 0.5) == 0.5
    by_redshift = DistanceRedshift(cosmology='WMAP9')
    assert by_redshift.pdescs() == {}

    def kpc_per_arcsec(redshift):
        distance = astropy.cosmology.WMAP9.angular_diameter_distance(redshift)
        return distance.to_value('kpc') * np.pi / (180 * 3600)
    assert by_redshift.kpc_per_arcsec({}, 0.5) == pytest.approx(
        kpc_per_arcsec(0.5))
    assert by_redshift.epoch({}, 0.5) == 0.5
    own = DistanceRedshift(redshift=0.4, cosmology='WMAP9')
    assert own.kpc_per_arcsec({}, 0.5) == pytest.approx(kpc_per_arcsec(0.4))
    assert own.epoch({}, 0.5) == 0.4
    assert own.dump()['redshift'] == 0.4
    with pytest.raises(ConfigError, match="redshift cannot be negative"):
        DistanceRedshift(redshift=-1)
    with pytest.raises(ConfigError, match="cosmology must be one of"):
        DistanceRedshift(cosmology='Planck99')


MASS = [
    dict(type='disk', name='stars', surface=dict(type='exponential'),
         vertical=dict(type='thin')),
    dict(type='disk', name='gas',
         surface=dict(type='nodewise', nodes=[0.0, 0.3, 0.6, 1.2]),
         vertical=dict(type='thin')),
    dict(type='spheroid', name='halo', profile=dict(
        type='pseudo_isothermal', parameters='density_radius'),
        shape=dict(type='sphere'))]

DISTANCE = dict(type='direct', cosmology='Planck18')

MASS_PROPERTIES = dict(
    distance=10, mass_stars_log_m=10.5, mass_stars_rd=0.5,
    mass_gas_log_sigma=[8, 7.8, 7.5], mass_halo_log_rho=8, mass_halo_rs=1)

DISK = dict(
    loose=False, tilted=False, rnodes=list(range(0, 21, 2)), rstep=0.5,
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(dptraits=dict(type='uniform')),
    emission=dict(type='lines'))

DISK_PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=4, dpt_a=15)


def test_the_mass_of_a_galaxy_at_a_direct_distance_is_of_its_epoch():
    # The critical density that defines its halos is that of its redshift
    # (5.5 times that of today at z = 1.5), as by a distance of redshift
    galaxy = galaxy_parser.load(dict(
        components=[DISK | dict(type='smdisk', kinematics=dict(
            vptraits=dict(type='mass'), dptraits=dict(type='uniform')))],
        mass=copy.deepcopy(MASS), distance=copy.deepcopy(DISTANCE),
        redshift=1.5))
    mass = galaxy.plan_stage().mass(DISK_PROPERTIES | MASS_PROPERTIES)
    assert mass.context.critical_density == pytest.approx(
        astropy.cosmology.Planck18.critical_density(1.5).to_value(
            'solMass / kpc3'))


def with_traits(component, **traits):
    """
    The configuration of a disk component with the given traits of its
    density (ptraits, htraits) and kinematics (vptraits, dptraits)
    instead of its own.
    """
    density = {
        key: value for key, value in traits.items()
        if key in ('ptraits', 'htraits')}
    kinematics = {
        key: value for key, value in traits.items()
        if key in ('vptraits', 'dptraits')}
    return component | dict(
        density=component['density'] | density,
        kinematics=component['kinematics'] | kinematics)


def disk_galaxy(
        disk_type, vptraits, mass=None, distance=None, thick=False,
        driver=None):
    """
    A galaxy of a disk (DISK) of the given type and velocity polar
    traits, thick (of sech2 height traits) if thick, with the given mass
    and distance, and the given driver of its gravity (the host if None).
    """
    disk = with_traits(DISK, vptraits=vptraits) | dict(type=disk_type)
    if thick:
        disk = with_traits(disk, htraits=dict(type='sech2', zmax=8))
    if disk_type == 'mcdisk':
        disk |= dict(cflux=1e-3)
    info = dict(components=[disk])
    if mass is not None:
        info |= dict(mass=copy.deepcopy(mass))
    if distance is not None:
        info |= dict(distance=copy.deepcopy(distance))
    return galaxy_parser.load(
        info, drivers={None: driver} if driver is not None else None)


def mass_model(mass):
    """The mass model of the given mass components."""
    return MassModel(mass_component_parser.load(copy.deepcopy(mass)))


def evaluate(driver, galaxy, line_of_sight, properties):
    """
    Evaluate a spectral cube of a galaxy, along the given line of sight
    ('flat' or 'grid').
    """
    from gbkfit.observation import (
        ObservationGroup, ObservationSky, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import ParamSpace
    group = ObservationGroup([galaxy], [ObservationSky(
        PixelSpectra(size=(32, 32, 51), step=(1, 1, 10)),
        line_of_sight_parser.load(dict(type=line_of_sight)), driver)])
    params = ParamSpace(group.pdescs(), properties)
    return group.model_h(params.evaluate())[0]['spectra']['d'].copy()


@pytest.mark.parametrize('line_of_sight, disk_type', [
    ('flat', 'smdisk'),
    ('grid', 'smdisk'),
    ('grid', 'mcdisk')])
def test_mass_trait_is_the_circular_velocity_at_the_subnodes(
        driver, line_of_sight, disk_type):
    # A disk with the trait 'mass' is the disk whose node-wise tangential
    # velocity at its subnodes is the circular velocity of the mass of its
    # galaxy, at the radii of the subnodes in kpc
    thick = line_of_sight == 'grid'
    massive = disk_galaxy(
        disk_type, dict(type='mass'), MASS, DISTANCE, thick=thick)
    given = disk_galaxy(
        disk_type, dict(type='nw_uniform', direction='tan', sampling='subrings'),
        thick=thick)
    subrnodes = np.array(given.constants()['subrnodes'])
    vcirc = np.sqrt(mass_model(MASS).vcirc2(
        subrnodes * 10 * 0.0048481368, MASS_PROPERTIES, CONTEXT))
    assert vcirc.max() > 100
    properties = dict(DISK_PROPERTIES)
    if thick:
        properties |= dict(ht_s=1)
    np.testing.assert_allclose(
        evaluate(
            driver, massive, line_of_sight, properties | MASS_PROPERTIES),
        evaluate(
            driver, given, line_of_sight, properties | dict(vpt_vt=vcirc)),
        rtol=1e-6, atol=1e-7)


def test_mass_trait_parameters_and_round_trip():
    galaxy = disk_galaxy(
        'smdisk', dict(type='mass'), MASS, DISTANCE, thick=True)
    # The trait has no parameters: those of the distance and the mass are
    # the galaxy's
    assert not any(name.startswith('vpt') for name in galaxy.pdescs())
    assert set(MASS_PROPERTIES) <= set(galaxy.pdescs())
    dumped = galaxy_parser.dump(galaxy)
    assert dumped['mass'] == MASS
    assert dumped['distance'] == DISTANCE
    assert galaxy_parser.dump(galaxy_parser.load(copy.deepcopy(dumped))) \
        == dumped


def test_mass_trait_mass_and_distance_need_each_other():
    # (mass that no component uses is for profiles or derived values: see
    # ObservationGroup, which warns of it unused)
    with pytest.raises(ConfigError, match="need the mass of their galaxy"):
        disk_galaxy('smdisk', dict(type='mass'))
    assert not disk_galaxy(
        'smdisk', dict(type='arctan', direction='tan'), MASS, DISTANCE
    ).components_use_mass()
    with pytest.raises(ConfigError, match="needs its distance"):
        disk_galaxy('smdisk', dict(type='mass'), MASS)
    with pytest.raises(ConfigError, match="used by its mass only"):
        disk_galaxy('smdisk', dict(type='arctan', direction='tan'), distance=DISTANCE)


def test_tables_of_thick_disks_are_their_integrals():
    # Inside the range of the table of a vertical law (hz / rd from 1e-3
    # to 1), against the integral computed directly (beyond it)
    from gbkfit.model.mass import surfaces
    rng = np.random.default_rng(1)
    x = np.exp(rng.uniform(np.log(1e-3), np.log(1e3), 50))
    for eta in (2e-3, 0.08, 0.9):
        np.testing.assert_allclose(
            surfaces._exponential_shape(x, VerticalSech2, eta),
            surfaces._exponential_shape_direct(
                x, functools.partial(VerticalSech2.transform, h=eta)),
            rtol=1e-6)


# The length of an arcsec at the distance of DISTANCE (10 Mpc)
KPC_PER_ARCSEC = 10 * KPC_PER_ARCSEC_MPC


def test_disk_densities_are_azimuthal_means(driver):
    # The surface density of each density polar trait at the subnodes: an
    # exponential; a harmonic (mean 0); a circular blob, whose mean is a
    # exp(-(R - r)^2 / (2 s^2)) i0e(R r / s^2), narrow enough to need
    # more azimuths; none outside the ring that the disk selects. The
    # circular velocity that the disk takes from its galaxy is not needed
    from gbkfit.model import SMDisk
    from gbkfit.model.components._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import DiskDensity
    component = SMDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        density=dict(ptraits=[
            dict(type='exponential'), dict(type='nw_harmonic', order=2),
            dict(type='mixture_gauss', nblobs=1)]),
        kinematics=dict(
            vptraits=dict(type='mass'), dptraits=dict(type='uniform')),
        sptraits=dict(type='rrange'),
        emission=dict(type='lines')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=4,
        pt1_a=np.full(11, 0.5), pt1_p=np.zeros(11), pt2_r=15, pt2_t=40,
        pt2_a=0.5, pt2_s=0.3, pt2_q=1, pt2_p=0, dpt_a=10, spt_rmin=0,
        spt_rmax=18)
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float32), None, None, True,
        None, None, 0.0)
    density = DiskDensity(
        component.density_disk(), plan.disk_plan(), params, KPC_PER_ARCSEC)
    radius = np.array(component.density_disk().subrnodes())
    selected = radius < 18
    blob = 0.5 * np.exp(-(radius - 15) ** 2 / (2 * 0.3 ** 2)) \
        * scipy.special.i0e(radius * 15 / 0.3 ** 2)
    np.testing.assert_allclose(
        density.surface_densities(radius * KPC_PER_ARCSEC),
        [np.exp(-radius / 4) * selected, 0 * radius, blob * selected],
        rtol=1e-5, atol=2e-6)


def test_disk_densities_are_exact_at_jumps(driver):
    # A map, and a selection of 93 degrees of azimuth (which the trapezoid
    # rule took as 66 / 256 of a turn): the means of the exponential and
    # of the map where it is selected, against 2^21 azimuths per circle
    from gbkfit.model import SMDisk
    from gbkfit.model.components._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import DiskDensity
    component = SMDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        density=dict(ptraits=[
            dict(type='exponential'),
            dict(type='map', size=[6, 4], step=2.0)]),
        kinematics=dict(
            vptraits=dict(type='uniform', direction='tan'), dptraits=dict(type='uniform')),
        sptraits=dict(type='azrange'),
        emission=dict(type='lines')))
    cells = np.random.default_rng(3).normal(size=24)
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=4,
        pt1_a=cells, vpt_vt=100, dpt_a=10, spt_p=17, spt_s=93)
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float32), None, None, True,
        None, None, 0.0)
    density = DiskDensity(
        component.density_disk(), plan.disk_plan(), params, KPC_PER_ARCSEC)
    radius = np.array(component.density_disk().subrnodes())
    theta = (np.arange(2 ** 21) + 0.5) * 360 / 2 ** 21
    selected = np.abs((theta - 17 + 180) % 360 - 180) < 93 / 2
    expected = []
    for r in radius:
        i = np.floor(r * np.cos(np.deg2rad(theta)) / 2 + 3).astype(int)
        j = np.floor(r * np.sin(np.deg2rad(theta)) / 2 + 2).astype(int)
        inside = (i >= 0) & (i < 6) & (j >= 0) & (j < 4)
        values = np.where(inside, cells[np.clip(j, 0, 3) * 6 + np.clip(
            i, 0, 5)], 0)
        expected.append([
            np.exp(-r / 4) * np.mean(selected),
            np.mean(values * selected)])
    np.testing.assert_allclose(
        density.surface_densities(radius * KPC_PER_ARCSEC),
        np.transpose(expected), rtol=1e-5, atol=1e-6)


def test_disk_densities_see_small_blobs(driver):
    # Structure far narrower than the spacing of 4096 azimuths, which the
    # trapezoid rule could miss: a round Gaussian blob of 0.01 arcsec at
    # r = 15, whose mean is a exp(-(R - r)^2 / (2 s^2)) i0e(R r / s^2); a
    # round cuspy blob (a generalised Gaussian of shape 0.5) at r = 10,
    # by quadrature split at its cusp; and a distortion of width 0.01 along
    # the rings, whose mean is a s erf(pi R / (s sqrt(2))) / (R sqrt(2 pi))
    from gbkfit.model import SMDisk
    from gbkfit.model.components._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import DiskDensity
    component = SMDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        density=dict(ptraits=[
            dict(type='mixture_gauss', nblobs=1),
            dict(type='mixture_ggauss', nblobs=1),
            dict(type='nw_distortion')]),
        kinematics=dict(
            vptraits=dict(type='uniform', direction='tan'), dptraits=dict(type='uniform')),
        emission=dict(type='lines')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, vpt_vt=100, dpt_a=10,
        pt_r=15, pt_t=40, pt_a=1, pt_s=0.01, pt_q=1, pt_p=0,
        pt1_r=10, pt1_t=-70, pt1_a=1, pt1_s=0.005, pt1_b=0.5, pt1_q=1,
        pt1_p=0, pt2_a=np.ones(11), pt2_p=np.full(11, 40.0),
        pt2_s=np.full(11, 0.01))
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float64), None, None, True,
        None, None, 0.0)
    density = DiskDensity(
        component.density_disk(), plan.disk_plan(), params, KPC_PER_ARCSEC)
    radius = np.array([10, 10.002, 15, 15.005, 15.02])

    def cusp(R):
        def blob(dtheta):
            rho = np.sqrt(max(R ** 2 + 100 - 20 * R * np.cos(dtheta), 0))
            return np.exp(-np.sqrt(rho / 0.005))
        return scipy.integrate.quad(
            blob, -np.pi, np.pi, points=[0], epsabs=1e-16, epsrel=1e-8,
            limit=400)[0] / (2 * np.pi)
    expected = [
        np.exp(-(radius - 15) ** 2 / (2 * 0.01 ** 2))
        * scipy.special.i0e(radius * 15 / 0.01 ** 2),
        [cusp(R) for R in radius],
        0.01 * scipy.special.erf(np.pi * radius / (0.01 * np.sqrt(2)))
        / (radius * np.sqrt(2 * np.pi))]
    np.testing.assert_allclose(
        density.surface_densities(radius * KPC_PER_ARCSEC), expected,
        rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize('line_of_sight', ['flat', 'grid'])
def test_disk_of_a_component_is_the_nodewise_disk_of_its_density(
        driver, line_of_sight):
    # A disk whose mass follows a node-wise density (linear between the
    # subnodes, and 0 at the last) is the nodewise disk of that surface
    # density at the subnodes, and of the scale height of the density
    # (in kpc) if thick
    from gbkfit.model.components._nodal import subring_radii
    rnodes = list(range(0, 21, 2))
    brightness = np.append(np.linspace(1, 0.2, 10), 0)
    upsilon = 2e8
    subrnodes = np.array(subring_radii(rnodes, DISK['rstep']))
    thick = line_of_sight == 'grid'
    disk = with_traits(
        DISK, ptraits=dict(type='nw_uniform'), vptraits=dict(type='mass'))
    disk |= dict(type='smdisk', name='hi')
    properties = {
        f'hi_{name}': value for name, value in DISK_PROPERTIES.items()
        if name != 'pt_s'} | dict(hi_pt_a=brightness)
    if thick:
        disk = with_traits(disk, htraits=dict(type='sech2', zmax=8))
        properties |= dict(hi_ht_s=1)
    halo = dict(type='spheroid', name='halo', profile=dict(
        type='pseudo_isothermal', parameters='density_radius'))
    halo_properties = dict(distance=10, mass_halo_log_rho=8, mass_halo_rs=1)

    def galaxy(gas):
        return galaxy_parser.load(dict(
            components=[copy.deepcopy(disk)], mass=[gas, halo],
            distance=copy.deepcopy(DISTANCE)))

    following = galaxy(dict(
        type='disk', name='gas',
        surface=dict(type='component', component='hi'),
        vertical=dict(type='component', component='hi')))
    nodewise = galaxy(dict(
        type='disk', name='gas',
        surface=dict(
            type='nodewise', nodes=(subrnodes * KPC_PER_ARCSEC).tolist()),
        vertical=dict(type='sech2' if thick else 'thin')))
    sigma = upsilon * np.interp(subrnodes[:-1], rnodes, brightness)
    np.testing.assert_allclose(
        evaluate(
            driver, following, line_of_sight,
            properties | halo_properties | dict(mass_gas_upsilon=upsilon)),
        evaluate(
            driver, nodewise, line_of_sight,
            properties | halo_properties | dict(
                mass_gas_log_sigma=np.log10(sigma),
                mass_gas_hz=KPC_PER_ARCSEC)),
        rtol=1e-5, atol=1e-7)


def test_disks_follow_components_of_their_galaxy():
    mass = [
        dict(type='disk', name='gas',
             surface=dict(type='component', component='hi', rstep=1.0),
             vertical=dict(type='component', component='hi', rstep=None))]
    loaded = mass_component_parser.load(copy.deepcopy(mass))
    assert mass_component_parser.dump(loaded) == mass
    with pytest.raises(ConfigError, match="follows the component 'hi'"):
        disk_galaxy('smdisk', dict(type='mass'), mass, DISTANCE)


def test_disk_follows_a_component_without_light():
    # The mass of a disk can follow a component whose light no observation
    # sees, here dust: thin, its gravity is that of the nodewise disk of
    # the surface density of the dust (as in test_disk_of_a_component_is_
    # the_nodewise_disk_of_its_density), at the subnodes of the disk it
    # moves
    from gbkfit.model.components._nodal import subring_radii
    rnodes = list(range(0, 21, 2))
    opacity = np.append(np.linspace(1, 0.2, 10), 0)
    upsilon = 2e8
    stars = with_traits(DISK, vptraits=dict(type='mass')) \
        | dict(type='smdisk', name='stars')
    dust = dict(
        type='smdisk', name='dust', loose=False, tilted=False,
        rnodes=rnodes, rstep=DISK['rstep'],
        density=dict(
            ptraits=dict(type='nw_uniform'),
            htraits=dict(type='sech2', zmax=8)),
        opacity=dict(type='dust'))
    galaxy = galaxy_parser.load(dict(
        components=[stars, dust],
        mass=[dict(
            type='disk', name='gas',
            surface=dict(type='component', component='dust'),
            vertical=dict(type='thin'))],
        distance=copy.deepcopy(DISTANCE)))
    params = {
        f'{name}_{key}': value for name in ('stars', 'dust')
        for key, value in DISK_PROPERTIES.items()
        if f'{name}_{key}' in galaxy.pdescs()}
    params |= dict(
        dust_pt_a=opacity, dust_ht_s=1, distance=10,
        mass_gas_upsilon=upsilon)
    subrnodes = np.array(subring_radii(rnodes, DISK['rstep']))
    nodewise = mass_model([dict(
        type='disk', name='gas',
        surface=dict(
            type='nodewise', nodes=(subrnodes * KPC_PER_ARCSEC).tolist()),
        vertical=dict(type='thin'))])
    sigma = upsilon * np.interp(subrnodes[:-1], rnodes, opacity)
    np.testing.assert_allclose(
        galaxy.plan_stage().evaluate(params).given[0]['vpt_vt'],
        np.sqrt(nodewise.vcirc2(
            subrnodes * KPC_PER_ARCSEC,
            dict(mass_gas_log_sigma=np.log10(sigma)), CONTEXT)),
        rtol=1e-6)


def test_pressure_support_is_the_jeans_equation(driver):
    # An exponential density (scale 4) and dispersion (scale 10) of a
    # thin disk: (R / rho) d(rho sigma^2)/dR = -sigma^2 R (1 / 4 + 2 / 10),
    # to the precision of the three-point derivative on the subnodes
    from gbkfit.model import SMDisk
    from gbkfit.model.components._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import pressure_support
    component = SMDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=dict(type='mass'), dptraits=dict(type='exponential')),
        emission=dict(type='lines')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=4,
        dpt_a=30, dpt_s=10)
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float32), None, None, True,
        None, None, 0.0)
    disk = component.density_disk()
    radius = np.array(disk.subrnodes())
    np.testing.assert_allclose(
        pressure_support(disk, plan.disk_plan(), params, radius, 'jeans'),
        -(30 * np.exp(-radius / 10)) ** 2 * radius * (1 / 4 + 2 / 10),
        rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize('ptrait, surface', [
    (dict(type='exponential'), lambda r: np.exp(-r / 4)),
    (dict(type='ggauss'), lambda r: np.exp(-(r / 4) ** 1.5))])
def test_pressure_support_of_self_gravitating_disks(driver, ptrait, surface):
    # R d(S^2)/dR sigma^2 / S^2, whatever the thickness of the disk, with
    # the three-point derivative on the subnodes, as the disk takes it: its
    # limit 2 sigma^2 dln S / dln R is, of an exponential S (scale 4),
    # -2 sigma^2 R / 4 (DysmalPy's type 1, -3.36 sigma^2 R / R_e with R_e =
    # 1.678 x 4), and of a Sersic one (a ggauss of shape 1.5 = 1 / n),
    # -2 sigma^2 1.5 (R / 4)^1.5 (type 2), here of a dispersion of scale 10
    from gbkfit.model import SMDisk
    from gbkfit.model.components._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components import traits
    from gbkfit.model.components.disks._profiles import pressure_support
    component = SMDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        density=dict(ptraits=ptrait, htraits=dict(type='sech2', zmax=8)),
        kinematics=dict(
            vptraits=dict(
                type='mass', pressure_support='self_gravitating'),
            dptraits=dict(type='exponential')),
        emission=dict(type='lines')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=4, ht_s=1,
        dpt_a=30, dpt_s=10) | (dict(pt_b=1.5) if 'pt_b' in component.pdescs()
                               else {})
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float64), None, None, True,
        None, None, 0.0)
    disk = component.density_disk()
    assert disk.pressure_supported_params() == {
        'vpt_vt': traits.PressureSupport('self_gravitating')}
    radius = np.array(disk.subrnodes())
    squared = surface(radius) ** 2
    np.testing.assert_allclose(
        pressure_support(
            disk, plan.disk_plan(), params, radius, 'self_gravitating'),
        radius * np.gradient(squared, radius, edge_order=2)
        * (30 * np.exp(-radius / 10)) ** 2 / squared, rtol=1e-10)


@pytest.mark.parametrize('line_of_sight', ['flat', 'grid'])
def test_mass_trait_with_pressure_support(driver, line_of_sight):
    # The trait 'mass' with pressure support is the node-wise velocity
    # sqrt(v_c^2 + (R / rho) d(rho sigma^2)/dR) at the subnodes, with the
    # derivative by the three-point formula: thin, rho is the surface
    # density; thick and flaring (sech2, node-wise scale height s), rho is
    # the surface density / (2 s)
    from gbkfit.model.components._nodal import subring_radii
    rnodes = list(range(0, 21, 2))
    flaring = np.linspace(0.5, 2, 11)
    disk = with_traits(
        DISK, dptraits=dict(type='exponential'),
        vptraits=dict(type='mass', pressure_support='jeans'))
    disk |= dict(type='smdisk')
    properties = DISK_PROPERTIES | dict(dpt_a=30, dpt_s=10)
    subrnodes = np.array(subring_radii(rnodes, DISK['rstep']))
    density = np.exp(-subrnodes / 4)
    if line_of_sight == 'grid':
        disk = with_traits(
            disk, htraits=dict(type='sech2', zmax=8, rnodes=True))
        properties |= dict(ht_s=flaring)
        density /= 2 * np.interp(subrnodes, rnodes, flaring)
    pressure = density * (30 * np.exp(-subrnodes / 10)) ** 2
    support = subrnodes * np.gradient(pressure, subrnodes, edge_order=2) \
        / density
    vcirc2 = mass_model(MASS).vcirc2(
        subrnodes * KPC_PER_ARCSEC, MASS_PROPERTIES, CONTEXT)
    supported = galaxy_parser.load(dict(
        components=[disk], mass=copy.deepcopy(MASS),
        distance=copy.deepcopy(DISTANCE)))
    given = galaxy_parser.load(dict(components=[with_traits(
        disk, vptraits=dict(type='nw_uniform', direction='tan', sampling='subrings'))]))
    assert np.all(vcirc2 + support >= 0) and np.max(-support) > 500
    np.testing.assert_allclose(
        evaluate(
            driver, supported, line_of_sight, properties | MASS_PROPERTIES),
        evaluate(driver, given, line_of_sight, properties | dict(
            vpt_vt=np.sqrt(vcirc2 + support))),
        rtol=1e-5, atol=1e-7)


def test_pressure_support_option():
    from gbkfit.model.components import traits
    trait = traits.vpt_parser.load(
        dict(type='mass', pressure_support='jeans'))
    assert traits.vpt_parser.dump(trait) == dict(
        type='mass', pressure_support='jeans')
    assert traits.vpt_parser.dump(traits.VPTraitMass()) \
        == dict(type='mass')
    with pytest.raises(ConfigError, match="pressure_support must be"):
        traits.VPTraitMass('asymmetric')
    trait = traits.vpt_parser.load(dict(
        type='mass', pressure_support='jeans', pressure_density='gas'))
    assert traits.vpt_parser.dump(trait) == dict(
        type='mass', pressure_support='jeans', pressure_density='gas')
    with pytest.raises(ConfigError, match="needs pressure_support"):
        traits.VPTraitMass(pressure_density='gas')
    several = dict(
        type='mass', pressure_support='jeans', pressure_density=['hi', 'h2'])
    assert traits.vpt_parser.dump(traits.vpt_parser.load(several)) == several
    with pytest.raises(ConfigError, match="each once"):
        traits.VPTraitMass('jeans', ['hi', 'hi'])


@pytest.mark.parametrize('mode', ['jeans', 'self_gravitating'])
def test_pressure_support_takes_the_density_of_a_mass_component(mode):
    # The density at its midplane of a thick exponential gas disk
    # ('jeans', as DysmalPy's type 3), or the surface density of a
    # flattened Sersic disk of index 1 ('self_gravitating', as its types 1
    # and 2 with the effective radius of the mass disk), both of the scale
    # length 1 kpc: (R / rho) d(rho sigma^2)/dR on the subnodes, with rho
    # ~ exp(-R), or ~ exp(-2 R) / sigma^2, for the dispersion of the disk
    from gbkfit.model.mass import _numerics
    mass = [
        dict(type='disk', name='gas', surface=dict(type='exponential'),
             vertical=dict(type='sech2')),
        dict(type='spheroid', name='stars', profile=dict(type='sersic'),
             shape=dict(type='flattened')),
        dict(type='spheroid', name='halo', profile=dict(
            type='pseudo_isothermal', parameters='density_radius'))]
    density = dict(jeans='gas', self_gravitating='stars')[mode]
    galaxy = disk_galaxy('smdisk', dict(
        type='mass', pressure_support=mode, pressure_density=density),
        mass, DISTANCE)
    params = DISK_PROPERTIES | dict(
        distance=10, mass_gas_log_m=9, mass_gas_rd=1, mass_gas_hz=0.1,
        mass_stars_log_m=10, mass_stars_re=_numerics.sersic_b(1),
        mass_stars_n=1, mass_stars_q=0.2, mass_halo_log_rho=7,
        mass_halo_rs=3)
    extra = Extras()
    galaxy.plan_stage().evaluate(params, extra)
    table = extra.get('vpt_vt_rotation', component='component0')
    r = np.asarray(table['radius'])
    profile = np.exp(-np.asarray(table['radius_kpc'])) \
        ** dict(jeans=1, self_gravitating=2)[mode]
    np.testing.assert_allclose(
        table['pressure'], DISK_PROPERTIES['dpt_a'] ** 2 * r
        * np.gradient(profile, r, edge_order=2) / profile, rtol=1e-5)


def test_pressure_support_adds_the_densities_of_mass_components():
    # Two thick exponential gas disks of one thickness (e.g. the atomic
    # and the molecular gas), of the scale lengths 2 and 1 kpc: the
    # 'jeans' pressure of the sum of their densities at their midplane,
    # ~ M_hi / rd_hi^2 exp(-R / rd_hi) + M_h2 / rd_h2^2 exp(-R / rd_h2)
    mass = [
        dict(type='disk', name=name, surface=dict(type='exponential'),
             vertical=dict(type='sech2')) for name in ('hi', 'h2')] + [
        dict(type='spheroid', name='halo', profile=dict(
            type='pseudo_isothermal', parameters='density_radius'))]
    galaxy = disk_galaxy('smdisk', dict(
        type='mass', pressure_support='jeans',
        pressure_density=['hi', 'h2']), mass, DISTANCE)
    params = DISK_PROPERTIES | dict(
        distance=10, mass_hi_log_m=9.5, mass_hi_rd=2, mass_hi_hz=0.1,
        mass_h2_log_m=9, mass_h2_rd=1, mass_h2_hz=0.1, mass_halo_log_rho=7,
        mass_halo_rs=3)
    extra = Extras()
    galaxy.plan_stage().evaluate(params, extra)
    table = extra.get('vpt_vt_rotation', component='component0')
    r, R = np.asarray(table['radius']), np.asarray(table['radius_kpc'])
    profile = 10 ** 9.5 / 4 * np.exp(-R / 2) + 10 ** 9 * np.exp(-R)
    np.testing.assert_allclose(
        table['pressure'], DISK_PROPERTIES['dpt_a'] ** 2 * r
        * np.gradient(profile, r, edge_order=2) / profile, rtol=1e-5)


@pytest.mark.parametrize('mode', ['jeans', 'self_gravitating'])
def test_pressure_support_takes_the_density_of_a_mass_disk_of_its_disk(mode):
    # A mass disk that follows the disk (its surface density and vertical
    # structure) has its density at its subnodes, upsilon times it, which
    # the pressure does not see: the support of the density of the disk
    disk = with_traits(
        DISK, vptraits=dict(type='mass', pressure_support=mode),
        htraits=dict(type='sech2', zmax=8)) | dict(type='smdisk', name='gas')
    mass = [
        dict(type='disk', name='hi', surface=dict(
            type='component', component='gas'),
            vertical=dict(type='component', component='gas')),
        dict(type='spheroid', name='halo', profile=dict(
            type='pseudo_isothermal', parameters='density_radius'))]

    def rotation(density):
        support = disk | dict(kinematics=disk['kinematics'] | dict(
            vptraits=disk['kinematics']['vptraits'] | density))
        galaxy = galaxy_parser.load(dict(
            components=[support], mass=copy.deepcopy(mass),
            distance=copy.deepcopy(DISTANCE)))
        params = {f'gas_{key}': value
                  for key, value in DISK_PROPERTIES.items()} | dict(
            gas_ht_s=0.5, distance=10, mass_hi_upsilon=1e8,
            mass_halo_log_rho=7, mass_halo_rs=3)
        return galaxy.plan_stage().evaluate(params).given[0]['vpt_vt']
    np.testing.assert_allclose(
        rotation(dict(pressure_density='hi')), rotation({}), rtol=1e-10)


def test_pressure_density_is_a_mass_component_that_gives_it():
    vptraits = dict(
        type='mass', pressure_support='jeans', pressure_density='dust')
    with pytest.raises(ConfigError, match="which its galaxy does not have"):
        disk_galaxy('smdisk', vptraits, MASS, DISTANCE)
    # (a thin disk has no density at its midplane)
    galaxy = disk_galaxy(
        'smdisk', vptraits | dict(pressure_density='stars'), MASS, DISTANCE)
    with pytest.raises(NotImplementedError, match="a thin disk"):
        galaxy.plan_stage().evaluate(DISK_PROPERTIES | MASS_PROPERTIES)


def rotation_at_heights(vptraits, mass, params):
    """
    The galaxy of a thick disk (DISK, of sech2 height traits to zmax 8)
    of the velocity traits vptraits and 'mass' at each height, and of the
    given mass, and the values that its gravity gives the disk at the
    values params of its parameters.
    """
    disk = with_traits(
        DISK, vptraits=vptraits, htraits=dict(type='sech2', zmax=8))
    disk['kinematics'] |= dict(vhtraits=dict(type='mass', zmax=8))
    galaxy = galaxy_parser.load(dict(
        components=[disk | dict(type='smdisk')], mass=copy.deepcopy(mass),
        distance=copy.deepcopy(DISTANCE)))
    return galaxy, galaxy.plan_stage().evaluate(params).given[0]


def test_rotation_at_each_height_of_a_point_mass(driver):
    # v(R, z)^2 = G M R^2 / r^3, plus the jeans pressure support of the
    # disk, of a constant height profile and a uniform dispersion: R
    # d(rho sigma^2)/dR / rho at every height, of the exponential surface
    # density (by the three-point derivative on the subnodes, as the disk
    # takes it); v = 0 where v^2 < 0 (the pressure exceeds the gravity),
    # at the midplane and at the heights; the factors v(R, z) / v(R, 0) at
    # the subnodes and the 33 heights of the table (1 where v(R, 0) = 0),
    # which the disk interpolates linearly
    from gbkfit.model.components._component_set import IMAGE_SPECTRAL_AXIS
    params = DISK_PROPERTIES | dict(
        ht_s=1, dpt_a=100, distance=10, mass_bh_log_m=10)
    galaxy, given = rotation_at_heights(
        dict(type='mass', pressure_support='jeans'),
        [dict(type='point', name='bh')], params)
    component = galaxy.components()[0]
    r = np.array(component.density_disk().subrnodes())[:, None]
    z = np.linspace(0, 8, 33)
    density = np.exp(-r / 4)
    support = r * np.gradient(
        density * 100 ** 2, r[:, 0], axis=0, edge_order=2) / density
    R = r[1:] * KPC_PER_ARCSEC
    v2 = G * 1e10 * R ** 2 / np.hypot(R, z * KPC_PER_ARCSEC) ** 3 \
        + support[1:]
    v = np.sqrt(np.maximum(v2, 0))
    assert np.any(v2[:, 0] < 0) and np.any((v2[:, 0] > 0) & (v2[:, -1] < 0))
    np.testing.assert_allclose(given['vpt_vt'][1:], v[:, 0], rtol=1e-8)
    factors = np.array([given[f'vht_f{k}'] for k in range(33)]).T
    np.testing.assert_allclose(
        factors[1:], np.divide(
            v, v[:, :1], out=np.ones_like(v), where=v[:, :1] > 0),
        rtol=1e-8, atol=1e-7)
    np.testing.assert_array_equal(factors[0], 1)
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float64), None, None, True,
        None, None, 0.0).disk_plan()
    values = {p: params[p] for p in component.pdescs()} | given
    heights = [0.125, 4.125, 7.875, 9]
    np.testing.assert_allclose(
        plan.evaluate_height_traits(values, 'vht', r[10], heights)[0],
        np.interp(heights, z, factors[10]), rtol=1e-12)
    radius = (r[10] + r[11]) / 2
    np.testing.assert_allclose(
        plan.evaluate_height_traits(values, 'vht', radius, z[5])[0],
        (factors[10, 5] + factors[11, 5]) / 2, rtol=1e-12)


def test_self_gravitating_rotation_at_each_height():
    # In the isothermal sheet of the surface density S of an exponential
    # mass disk (scale length 1 kpc) and a uniform dispersion sigma, rho
    # sigma^2 ~ S^2 sech^2(z / z0), z0 = sigma^2 / (pi G S): R d(rho
    # sigma^2)/dR / rho = sigma^2 R (-2 + 2 u tanh(u)), u = z / z0, R in
    # kpc, to the precision of the three-point derivative on the subnodes
    params = DISK_PROPERTIES | dict(
        ht_s=1, distance=10, mass_gas_log_m=9.5, mass_gas_rd=1,
        mass_bh_log_m=10.5)
    mass = [
        dict(type='disk', name='gas', surface=dict(type='exponential'),
             vertical=dict(type='thin')),
        dict(type='point', name='bh')]
    galaxy, given = rotation_at_heights(dict(
        type='mass', pressure_support='self_gravitating',
        pressure_density='gas'), mass, params)
    disk = galaxy.components()[0].density_disk()
    R = np.array(disk.subrnodes())[1:, None] * KPC_PER_ARCSEC
    z = np.linspace(0, 8, 33) * KPC_PER_ARCSEC
    surface = 10 ** 9.5 / (2 * np.pi) * np.exp(-R)
    u = np.pi * G * surface * z / 15 ** 2
    model = mass_model(mass)
    force_R = model.forces(R, z, dict(
        mass_gas_log_m=9.5, mass_gas_rd=1, mass_bh_log_m=10.5), CONTEXT)[0]
    v2 = -R * force_R + 15 ** 2 * R * (-2 + 2 * u * np.tanh(u))
    factors = np.array([given[f'vht_f{k}'] for k in range(33)]).T[1:]
    assert np.all(v2 > 0) and np.max(u) > 1
    np.testing.assert_allclose(
        factors, np.sqrt(v2 / v2[:, :1]), rtol=2e-3)


@pytest.mark.parametrize('vptraits, user, message', [
    (dict(type='arctan', direction='tan'), False,
     "pairs with the velocity trait 'mass' only, not 'arctan'"),
    (dict(type='mass', pressure_support='self_gravitating'), False,
     "give pressure_density"),
    (dict(type='mass'), True,
     r"\['mine'\] do not give their forces off their plane")])
def test_rotation_at_each_height_is_checked(
        tmp_path, vptraits, user, message):
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n    return 100 * R\n")
    mass = [dict(
        type='user', name='mine', file=str(tmp_path / 'curve.py'),
        func='vcirc2')] if user else MASS
    disk = with_traits(
        DISK, vptraits=vptraits, htraits=dict(type='sech2', zmax=8)
    ) | dict(type='smdisk')
    disk['kinematics'] |= dict(vhtraits=dict(type='mass', zmax=8))
    with pytest.raises(ConfigError, match=message):
        galaxy_parser.load(dict(
            components=[disk], mass=copy.deepcopy(mass),
            distance=copy.deepcopy(DISTANCE)))


def test_rotation_at_each_height_option():
    from gbkfit.model.components import traits
    trait = traits.vht_parser.load(dict(type='mass', zmax=3))
    assert traits.vht_parser.dump(trait) == dict(type='mass', zmax=3.0)
    assert list(trait.rotation_factor_params().values())[-1] == 3.0
    with pytest.raises(ConfigError, match="zmax must be a positive length"):
        traits.VHTraitMass(0)


def test_one_mass_trait_per_disk():
    from gbkfit.model.components import traits
    from gbkfit.model.components.parts import Kinematics
    Kinematics(
        [traits.VPTraitMass(), traits.VPTraitNWUniform('rad')],
        traits.DPTraitUniform())
    with pytest.raises(ConfigError, match="at most one velocity trait"):
        Kinematics(
            [traits.VPTraitMass(), traits.VPTraitMass('jeans')],
            traits.DPTraitUniform())


def test_observations_of_a_galaxy_share_its_gravity(driver, monkeypatch):
    # The gravity of a galaxy is evaluated once, on its driver, for all
    # the observations of it, whatever their drivers and dtypes: here one
    # on the host and one on the driver of the galaxy
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.model.components._stage import GalaxyStage
    from gbkfit.observation import (
        ObservationGroup, ObservationSky, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import ParamSpace
    calls = []
    evaluate_stage = GalaxyStage.evaluate

    def counted(self, params, *args):
        calls.append(params)
        return evaluate_stage(self, params, *args)
    monkeypatch.setattr(GalaxyStage, 'evaluate', counted)
    # (pressure support samples the profiles of the disk on that driver)
    galaxy = disk_galaxy(
        'smdisk', dict(type='mass', pressure_support='jeans'), MASS,
        DISTANCE, driver=driver)
    assert galaxy.driver() is driver
    group = ObservationGroup([galaxy], [
        ObservationSky(
            PixelSpectra(size=(32, 32, 51), step=(1, 1, 10)),
            line_of_sight_parser.load(dict(type='flat')), obs_driver,
            dtype=dtype)
        for obs_driver, dtype in ((DriverHost(), 'float32'),
                                  (driver, 'float64'))])
    params = ParamSpace(
        group.pdescs(), DISK_PROPERTIES | MASS_PROPERTIES).evaluate()
    single, double = group.model_h(params)
    assert len(calls) == 1
    assert np.isfinite(single['spectra']['d']).all()
    np.testing.assert_allclose(
        single['spectra']['d'], double['spectra']['d'], rtol=1e-5,
        atol=1e-7)


@pytest.mark.parametrize('tilt, warns', [(10, False), (30, True)])
def test_the_gravity_warns_of_a_disk_far_from_flat(caplog, tilt, warns):
    # The gravity takes a warped disk flat: a tilt of 10 degrees across its
    # rings changes its circular velocity by about 0.2%, one of 30 by
    # about 1.5%, and a warning says so (with extras)
    disk = with_traits(DISK, vptraits=dict(type='mass')) | dict(
        type='smdisk', tilted=True)
    galaxy = galaxy_parser.load(dict(
        components=[disk], mass=copy.deepcopy(MASS),
        distance=copy.deepcopy(DISTANCE)))
    nodes = len(DISK['rnodes'])
    params = DISK_PROPERTIES | MASS_PROPERTIES | dict(
        posa=np.full(nodes, 30.0), incl=np.linspace(60, 60 + tilt, nodes))
    caplog.set_level(logging.WARNING)
    galaxy.plan_stage().evaluate(params, Extras())
    assert ('tilted by' in caplog.text) == warns


def test_the_gravity_writes_the_rotation_it_gives():
    # With extras, a table of the radii of the rotation it gives a disk,
    # the circular velocity of each mass component and of all (signed;
    # in Newtonian gravity, vnewton and vcirc are the same), the pressure
    # support, and the rotation: sqrt(vcirc^2 + pressure)
    galaxy = disk_galaxy(
        'smdisk', dict(type='mass', pressure_support='jeans'), MASS,
        DISTANCE)
    params = DISK_PROPERTIES | MASS_PROPERTIES
    extra = Extras()
    given = galaxy.plan_stage().evaluate(params, extra).given
    table = extra.get('vpt_vt_rotation', component='component0')
    assert table.colnames == [
        'radius', 'radius_kpc', 'vcirc_stars', 'vcirc_gas', 'vcirc_halo',
        'vnewton', 'vcirc', 'pressure', 'vrot']
    np.testing.assert_allclose(
        table['radius'], galaxy.components()[0].density_disk().subrnodes())
    np.testing.assert_allclose(
        table['radius_kpc'], table['radius'] * KPC_PER_ARCSEC)
    vcirc2 = sum(np.sign(table[f'vcirc_{name}']) * table[f'vcirc_{name}'] ** 2
                 for name in ('stars', 'gas', 'halo'))
    np.testing.assert_allclose(
        np.sign(table['vcirc']) * table['vcirc'] ** 2, vcirc2, rtol=1e-12)
    np.testing.assert_array_equal(table['vnewton'], table['vcirc'])
    np.testing.assert_allclose(table['vrot'], given[0]['vpt_vt'])
    np.testing.assert_allclose(
        table['vrot'] ** 2, vcirc2 + table['pressure'], rtol=1e-9)


@pytest.mark.parametrize('vertical', [None, 'thin', 'sech2'])
def test_cell_masses_add_up(vertical):
    # The masses of the cells between spheres and latitudes add up to the
    # mass between the inner and the outer sphere: of a halo (of its
    # enclosed mass), and of a thin disk (in the cells next to its plane)
    # or a thick one (of its density)
    if vertical is None:
        component = dict(type='spheroid', name='c', profile=dict(
            type='nfw', parameters='density_radius'))
        params = dict(mass_c_log_rho=7, mass_c_rs=10)
    else:
        component = dict(type='disk', name='c', surface=dict(
            type='exponential'), vertical=dict(type=vertical))
        params = dict(mass_c_log_m=10, mass_c_rd=3, mass_c_hz=0.3)
    model = MassModel.load(dict(components=[component]))
    r_edges = np.geomspace(0.01, 200, 250)
    b_edges = np.pi / 2 * np.linspace(0, 1, 41) ** 2
    masses = model.cell_masses(r_edges, b_edges, params, CONTEXT)
    enclosed = model.enclosed_mass(r_edges[[0, -1]], params, CONTEXT)
    np.testing.assert_allclose(
        masses.sum(), enclosed[1] - enclosed[0], rtol=1e-4)
    if vertical == 'thin':
        assert np.all(masses[:, 1:] == 0)
