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

import gbkfit.cosmology
from gbkfit.config import ConfigError
from gbkfit.grid import SpectralAxis
from gbkfit.model import Extras, galaxy_parser
from gbkfit.model.distance import (
    KPC_PER_ARCSEC_MPC, DistanceCosmological, DistanceParameter)
from gbkfit.model.mass import (
    ContractionBlumenthal86, ContractionGnedin04, Halo, HaloProfile,
    MassContext, MassDisk, MassModel, MassPoint, MassSpheroid, MassUser,
    Overdensity, ProfileBurkert, ProfileCoreNFW, ProfileDekelZhao,
    ProfileEinasto, ProfileNFW, ProfileNodewise, ProfileNodewiseHalo,
    ProfilePseudoIsothermal, ProfileSersic, ProfileTruncatedNFW, ProfileZhao,
    ShapeFlattened, SurfaceExponential, SurfaceGaussianRing, SurfaceNodewise,
    TracedDisk, VerticalComponent, VerticalExponential, VerticalGauss,
    VerticalSech, VerticalSech2, VerticalThin, mass_component_parser,
    sidm_core)
from gbkfit.model.mass import _disk_gravity
from gbkfit.params import ModelEvaluationError


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The epoch of Planck 2018 at z = 0
CONTEXT = MassContext(
    critical_density=astropy.cosmology.Planck18.critical_density0.to_value(
        'solMass / kpc3'),
    omega_m=astropy.cosmology.Planck18.Om0,
    ricci_scalar=gbkfit.cosmology.DEFAULT.ricci_scalar(0.0),
    ricci_scalar_today=gbkfit.cosmology.DEFAULT.ricci_scalar(0.0))

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


def test_vertical_forces_of_thick_disks_far_above_them():
    # At a point far above a thick disk, alone at its radius (no other
    # heights between it and the plane), that of the disk made thin:
    # the pull of its slab takes all of its column (it took 8 points over
    # [0, z], and missed a disk 400 times thinner: +4778 for -2.97)
    params = dict(log_m=10, rd=3, hz=0.3)
    R, z = np.array([0.0, 10.0]), np.array([120.0, 120.0])
    np.testing.assert_allclose(
        MassDisk('d', SurfaceExponential(), VerticalSech2()).vertical_force(
            R, z, params, CONTEXT),
        MassDisk('d', SurfaceExponential(), VerticalThin()).vertical_force(
            R, z, params, CONTEXT), rtol=1e-4)


@pytest.mark.parametrize('vertical', [VerticalThin(), VerticalSech2()])
def test_disks_give_their_forces_far_from_them(vertical):
    # Beyond their tables (5 outer radii from their plane: 600 kpc for an
    # exponential disk of rd = 3 kpc; and 1000 rd from its axis), the
    # multipole series of their mass: across the edge of the tables as
    # within them, and far away, the pull of a point of their mass
    disk = MassDisk('d', SurfaceExponential(), vertical)
    params = dict(log_m=10, rd=3, hz=0.3)
    R = np.array([0.0, 60.0])
    inside = disk.forces(R, np.full(2, 600 - 1e-6), params, CONTEXT)
    outside = disk.forces(R, np.full(2, 600 + 1e-6), params, CONTEXT)
    np.testing.assert_allclose(outside, inside, rtol=1e-4)
    R, z = np.array([5e3, 4e3]), np.array([7e3, 0.0])
    point = -G * 1e10 * np.array([R, z]) / np.hypot(R, z) ** 3
    np.testing.assert_allclose(
        disk.forces(R, z, params, CONTEXT), point, rtol=1e-5, atol=1e-12)


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
            ValueError, match="the nodes of a nodewise surface density"):
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


def test_mass_disks_follow_one_component_or_two():
    # (e.g. H2 of a CO surface density in the layer of the HI)
    disk = MassDisk('d', SurfaceExponential(), VerticalComponent('hi'))
    assert disk.followed_components() == ('hi',)
    assert TracedDisk('gas', 'hi').followed_components() == ('hi',)
    h2 = TracedDisk('h2', 'co', vertical=VerticalComponent('hi'))
    assert h2.followed_components() == ('co', 'hi')
    with pytest.raises(ValueError, match="leave vertical out"):
        TracedDisk('gas', 'hi', vertical=VerticalComponent('hi'))


def test_a_traced_disk_in_its_layer_has_the_density_of_its_gravity():
    # Between two subnodes, the sum over the terms of their surface
    # density times the pdf of their height, linear in R, as its gravity
    # takes it (was the product of the linear surface density and the
    # linear shape, which differs for a flaring disk of two terms); its
    # column is its surface density, and 0 beyond the subnodes
    layers = Layers(flare=1)
    context = MassContext(1, 1, 0, 0, dict(hi=layers))
    disk = TracedDisk('gas', 'hi')
    params = dict(upsilon=2.0)
    knots = layers.knots(None)
    z = np.array([0.0, 0.2, 0.7, 1.5])

    def column(k):
        pdfs = layers.height_pdfs(knots[k:k + 1], z)
        return 2.0 * layers.surface_densities(knots[k:k + 1])[:, 0] \
            @ pdfs[:, 0]
    R = 0.25 * knots[3] + 0.75 * knots[4]
    np.testing.assert_allclose(
        disk.density(R, z, params, context),
        0.25 * column(3) + 0.75 * column(4), rtol=1e-12)
    old = disk.surface_density(R, params, context) \
        * VerticalComponent('hi').height_pdf(R, z, params, context)
    assert not np.allclose(disk.density(R, z, params, context), old,
                           rtol=1e-6)
    heights = np.linspace(-6, 6, 4001)
    np.testing.assert_allclose(
        np.trapezoid(disk.density(R, heights, params, context), heights),
        disk.surface_density(R, params, context), rtol=1e-6)
    assert np.all(disk.density(11.0, z, params, context) == 0)
    # (its mass within a height: that of the same interpolant)
    within = disk._column_within(np.array([R]), np.array([0.7]), params,
                                 context)
    inside = np.linspace(-0.7, 0.7, 2001)
    np.testing.assert_allclose(
        within, np.trapezoid(
            disk.density(R, inside, params, context), inside), rtol=1e-6)


def test_a_traced_disk_in_the_layer_of_another_component():
    # Its surface density, in the shape of the other component's layer:
    # as a disk of that surface density as a law in that layer
    co, hi = Layers(flare=0), Layers(flare=1)
    context = MassContext(1, 1, 0, 0, dict(co=co, hi=hi))
    h2 = TracedDisk('h2', 'co', vertical=VerticalComponent('hi'))
    params = dict(upsilon=1.0)
    knots = co.knots(None)
    # (a node-wise surface of the same values, 0 just beyond the last)
    nodewise = MassDisk(
        'n', SurfaceNodewise([*knots, knots[-1] + 1e-6]),
        VerticalComponent('hi'))
    log_sigma = np.log10(h2.surface_density(knots, params, context))
    R = np.array([1.0, 3.0, 6.0])
    np.testing.assert_allclose(
        h2.vcirc2(R, params, context),
        nodewise.vcirc2(R, dict(log_sigma=log_sigma), context), rtol=1e-4)
    own = TracedDisk('h2', 'co')
    assert not np.allclose(
        h2.vcirc2(R, params, context), own.vcirc2(R, params, context),
        rtol=1e-3)


def test_vertical_structure_of_a_component():
    # The shape of its density at its subnodes; where it has none, that
    # of the nearest subnodes with density, interpolated between them
    class Gap(Layers):
        def surface_densities(self, R):
            return super().surface_densities(R) * (np.abs(R - 5) > 0.3)
    context = MassContext(1, 1, 0, 0, dict(hi=Gap(flare=1)))
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
    context = MassContext(1, 1, 0, 0, dict(hi=Layers(flare)))
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


# The density of the overdensity 200c (that of halos by default) at the
# epoch of CONTEXT
DENSITY_200C = Overdensity('200c').density(CONTEXT)


def at(profile, params):
    """
    The profile of a halo at the values of its parameters (see
    HaloProfile.at), of the overdensity 200c; or a profile of a spheroid,
    as it is. Both take the values of the parameters and the context.
    """
    if isinstance(profile, HaloProfile):
        return profile.at(params, DENSITY_200C)
    return profile


def spheroid(profile, shape=None):
    """The mass spheroid of a profile: a halo, or a MassSpheroid."""
    kind = Halo if isinstance(profile, HaloProfile) else MassSpheroid
    return kind('s', profile, shape=shape)


SPHERES = [
    (ProfileNFW('density_radius'), dict(log_rho=7, rs=10)),
    (ProfileBurkert('density_radius'), dict(log_rho=7.5, rs=4)),
    (ProfilePseudoIsothermal('density_radius'), dict(log_rho=7.5, rs=2)),
    (ProfileEinasto('density_radius'), dict(log_rho=6.5, rs=15, alpha=0.17)),
    (ProfileZhao('density_radius'),
     dict(log_rho=6.5, rs=8, alpha=2, beta=4, gamma=0.5)),
    (ProfileDekelZhao('density_radius'), dict(log_rho=7, rs=5, gamma=0.5)),
    (ProfileDekelZhao('density_radius'), dict(log_rho=7, rs=5, gamma=1.5)),
    (ProfileSersic(), dict(log_m=10, re=1.5, n=0.6)),
    (ProfileSersic(), dict(log_m=10, re=1.5, n=4)),
    (ProfileCoreNFW('density_radius'), dict(log_rho=7, rs=10, rc=1.5, n=0.8)),
    (ProfileTruncatedNFW('density_radius'), dict(log_rho=7, rs=10, rt=30)),
    (ProfileNodewiseHalo([0.5, 2, 8, 30]), dict(log_rho=[8.5, 7.6, 6.5, 5]))]


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
        assert core.scales(cored, DENSITY_200C) \
            == nfw.scales(params, DENSITY_200C)
        np.testing.assert_allclose(
            at(core, cored).enclosed_mass(radius),
            at(nfw, params).enclosed_mass(radius) * np.tanh(radius / 2) ** n,
            rtol=1e-12)
    np.testing.assert_allclose(
        at(ProfileTruncatedNFW(), params | dict(rt=1e7)).enclosed_mass(radius),
        at(nfw, params).enclosed_mass(radius), rtol=1e-5)


@pytest.mark.parametrize('profile, params', SPHERES)
def test_spheres_are_their_densities(profile, params):
    # Their enclosed mass is the integral of their density (including
    # radii where the halos use their series), and their circular
    # velocity G M(<r) / r
    sphere = spheroid(profile)
    density = at(profile, params)
    radius = np.array([1e-4, 5e-4, 2e-3, 0.5, 2, 10, 100])
    expected = [
        scipy.integrate.quad(
            lambda r: 4 * np.pi * r ** 2 * float(density.density(
                np.array(r), params, CONTEXT)),
            0, radius_, points=[1e-5, 1e-3, 0.1, 0.5, 2, 8, 30], limit=400)[0]
        for radius_ in radius]
    np.testing.assert_allclose(
        density.enclosed_mass(radius, params, CONTEXT), expected, rtol=1e-6)
    np.testing.assert_allclose(
        sphere.vcirc2(radius, params, CONTEXT),
        G * np.array(expected) / radius, rtol=1e-6)


@pytest.mark.parametrize('sphere, params', [
    (MassPoint('p'), dict(log_m=8))] + [
        (spheroid(profile), params) for profile, params in SPHERES])
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


@pytest.mark.parametrize('profile, params', [
    SPHERES[-1], SPHERES[2],
    (SPHERES[-1][0], dict(log_rho=[8.5, 7.6, 6.5, 6]))])
def test_potentials_are_zero_at_infinity_or_at_the_centre(profile, params):
    # 0 at infinity: -int_r^inf G M / s^2 ds; of a mass that grows so
    # fast that this is infinite (the pseudo-isothermal halo, and a
    # nodewise halo whose density falls less steeply than r^-2 beyond its
    # last node), 0 at the centre: int_0^r G M / s^2 ds
    halo = at(profile, params)
    radius = np.array([0.1, 0.7, 3, 20, 60])

    def pull(s):
        return G * float(halo.enclosed_mass(np.array(s))) / s ** 2
    if halo.zero_at_infinity():
        expected = [-scipy.integrate.quad(
            pull, r, np.inf, epsrel=1e-12, limit=400)[0] for r in radius]
    else:
        expected = [scipy.integrate.quad(
            pull, 0, r, points=[0.5, 2, 8, 30], epsrel=1e-12, limit=400)[0]
            for r in radius]
        assert halo.potential(np.array([0.0]))[0] == 0
    assert halo.zero_at_infinity() == (params is SPHERES[-1][1])
    np.testing.assert_allclose(halo.potential(radius), expected, rtol=1e-9)


def test_nodewise_halos_are_power_laws_between_their_nodes():
    # Its density at its nodes, a power law between them, continued within
    # the first node and beyond the last
    profile, params = SPHERES[-1]
    halo = at(profile, params)
    nodes, log_rho = profile.nodes(), np.array(params['log_rho'], float)
    np.testing.assert_allclose(
        np.log10(halo.density(nodes)), log_rho, rtol=1e-12)
    np.testing.assert_allclose(
        np.log10(halo.density(np.sqrt(nodes[:-1] * nodes[1:]))),
        (log_rho[:-1] + log_rho[1:]) / 2, rtol=1e-12)
    slope = (log_rho[1] - log_rho[0]) / np.log10(nodes[1] / nodes[0])
    assert halo.density(np.array([0.05]))[0] == pytest.approx(
        10 ** log_rho[0] * 0.1 ** slope, rel=1e-12)
    slope = (log_rho[-1] - log_rho[-2]) / np.log10(nodes[-1] / nodes[-2])
    assert halo.density(np.array([300.0]))[0] == pytest.approx(
        10 ** log_rho[-1] * 10 ** slope, rel=1e-12)
    info = dict(type='halo', name='h',
                profile=dict(type='nodewise', nodes=[0.5, 2, 8, 30]))
    loaded = mass_component_parser.load(copy.deepcopy(info))
    assert mass_component_parser.dump(loaded)['profile'] == info['profile']
    assert loaded.pdescs()['log_rho'].size() == 4
    for nodes in ([0, 1], [2, 1], [1]):
        with pytest.raises(ValueError, match="the nodes of a nodewise halo"):
            ProfileNodewiseHalo(nodes)
    # (a density that falls as r^-3 or steeper has no finite mass)
    with pytest.raises(ModelEvaluationError, match="infinite mass"):
        profile.at(dict(log_rho=[9, 5, 4, 3]), DENSITY_200C)


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
                ValueError, match="the nodes of a nodewise profile"):
            ProfileNodewise(nodes)


def test_flattened_spheroids_of_axis_ratio_1_are_spheres():
    # (to 1e-11 for the halos; the tables of the Sersic profile, of its
    # density and of its mass, agree to about 2e-8; the kinks of the
    # nodewise halo at its nodes, to some 1e-3)
    radius = np.array([0.01, 0.3, 2, 15, 120])
    for profile, params in SPHERES:
        sphere = spheroid(profile)
        flat = spheroid(profile, ShapeFlattened())
        np.testing.assert_allclose(
            flat.vcirc2(radius, params | dict(q=1), CONTEXT),
            sphere.vcirc2(radius, params, CONTEXT),
            rtol=1e-3 if isinstance(profile, ProfileNodewiseHalo) else 1e-7)


def test_flattened_spheroids_are_their_potential_integrals():
    # The circular velocity of a homoeoid density rho(m) / q (Binney &
    # Tremaine 2.88): v^2 = 2 pi G R^2 int_0^inf rho(R / sqrt(1 + u)) du /
    # ((1 + u)^2 sqrt(q^2 + u)), against the integral over m, oblate and
    # prolate
    profile, params = SPHERES[1]
    density = at(profile, params)
    radius = np.array([0.5, 4, 30])
    for q in (0.3, 2.0):
        flat = spheroid(profile, ShapeFlattened())

        def vcirc2(R):
            return 2 * np.pi * G * R ** 2 * scipy.integrate.quad(
                lambda u: float(density.density(
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
    density = at(profile, params)
    flat = spheroid(profile, ShapeFlattened())
    R, z = np.array([0.5, 4, 30, 3]), np.array([0.2, 3, 10, 0])

    def force(R, z, power, axis):
        def integrand(u):
            m = np.sqrt(R ** 2 / (1 + u) + z ** 2 / (q ** 2 + u))
            return float(density.density(np.array(m), params, CONTEXT)) \
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
        sphere = spheroid(profile).forces(R, z, params, CONTEXT)
        np.testing.assert_allclose(force_R, sphere[0], rtol=1e-7)
        np.testing.assert_allclose(force_z, sphere[1], rtol=1e-7)


def test_contracted_halos_pull_toward_their_centre():
    # Spherical shells: -v^2(r) / r^2 toward the centre, of the contracted
    # circular velocity at the spherical radius r
    model = mass_model(
        BARYONS + [responding_halo(contraction=dict(type='blumenthal86'))])
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


@pytest.mark.parametrize('responses', [
    dict(contraction=dict(type='blumenthal86')),
    dict(contraction=dict(type='gnedin04'), sidm=True, baryons=['stars']),
    dict(contraction=dict(type='blumenthal86'),
         profile=dict(type='pseudo_isothermal'))])
def test_responding_halos_have_their_density_and_potential(responses):
    # Of their mass within spheres: their density, dM/dr / (4 pi r^2),
    # whose integral is their mass (to the precision of the spline of the
    # mass of an SIDM core, some 1e-5); their potential, whose gradient is
    # v^2 / r (to some 3e-6 of SIDM, of the edge of its core on the grid
    # of the integral), and that far out of the halo before (0 at
    # infinity, or at the centre for the pseudo-isothermal halo)
    model = mass_model(BARYONS + [responding_halo(**responses)])
    halo = model.component('halo')
    values = BARYON_VALUES | {
        name: desc.default() for name, desc in model.pdescs().items()
        if name.startswith('mass_halo_')} | dict(
        mass_halo_log_m=12, mass_halo_c=5)
    params = model.component_params('halo', values)
    context = model._context(halo, values, CONTEXT)
    radii = np.geomspace(1e-5, 200, 4000)
    density = halo.density(radii, np.zeros_like(radii), params, context)
    mass = scipy.integrate.cumulative_simpson(
        4 * np.pi * radii ** 3 * density, x=np.log(radii), initial=0)
    r = np.array([0.3, 2, 10, 50])
    np.testing.assert_allclose(
        np.interp(r, radii, mass), halo.enclosed_mass(r, params, context),
        rtol=1e-4 if halo.sidm() else 1e-5)
    step = 1e-5

    def potential(radius):
        return halo.potential(radius, np.zeros_like(radius), params, context)
    pull = (potential(r * np.exp(step)) - potential(r * np.exp(-step))) \
        / (2 * np.sinh(step) * r)
    np.testing.assert_allclose(
        pull, halo.vcirc2(r, params, context) / r,
        rtol=1e-5 if halo.sidm() else 1e-6)
    # (0 at infinity: far out, where the halo barely responds; from the
    # centre, that far out differs by the integral of the difference)
    if at(halo.profile(), params).zero_at_infinity():
        far = np.array([1e4])
        assert potential(far)[0] == pytest.approx(
            spheroid(halo.profile()).potential(far, 0, params, context)[0],
            rel=1e-3)


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
    assert Halo('halo', ProfileNFW()).spherical()


def test_halos_and_spheroids_take_their_own_profiles():
    # (no NFW bulge, no Sersic halo)
    with pytest.raises(ConfigError, match="unknown type 'nfw'"):
        mass_component_parser.load(
            dict(type='spheroid', name='b', profile=dict(type='nfw')))
    with pytest.raises(ConfigError, match="unknown type 'sersic'"):
        mass_component_parser.load(
            dict(type='halo', name='h', profile=dict(type='sersic')))


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
        type='halo', name='halo', profile=dict(type='nfw'),
        baryons=['stars', 'bulge']) | options


def virial_radius(halo_mass, context):
    """
    The radius (kpc) within which the mean density of a halo of the mass
    halo_mass(r) (G M, of the radius) is 180 times the mean matter
    density of the context, by root finding.
    """
    target = 180 * context.omega_m * context.critical_density
    return scipy.optimize.brentq(
        lambda r: 3 * halo_mass(r) / (4 * np.pi * G * r ** 3) - target,
        1.0, 1e4, xtol=1e-12, rtol=1e-15)


def test_contracted_halos_keep_the_dark_matter_of_their_shells():
    # Within r, the dark matter the halo had within r', with r' M(r') /
    # (1 - f_b) = r (M(r') + M_b(r)) (in units of G: M = v^2 r of the
    # halo, and the mass of the baryons within the sphere; f_b the share
    # of the baryons within the virial radius, of 180 times the mean
    # matter density), against r' found by root finding at each radius
    model = mass_model(
        BARYONS + [responding_halo(contraction=dict(type='blumenthal86'))])
    params = BARYON_VALUES | dict(mass_halo_log_m=12, mass_halo_c=5)
    radius = np.array([0.5, 2, 5, 20, 100])
    parts = model.vcirc2_components(radius, params, CONTEXT)
    halo = Halo('h', ProfileNFW())

    def halo_mass(r):
        return float(halo.vcirc2(
            np.array([r]), dict(log_m=12, c=5), CONTEXT)[0]) * r
    r_vir = virial_radius(halo_mass, CONTEXT)

    def baryons_at(r):
        masses = model.enclosed_mass_components(
            np.array([r]), params, CONTEXT)
        return G * float(masses['stars'][0] + masses['bulge'][0])
    keep = 1 - baryons_at(r_vir) / (baryons_at(r_vir) + halo_mass(r_vir))
    assert 0.9 < keep < 1
    for r, v2 in zip(radius, parts['halo']):
        baryon = baryons_at(r)
        initial = scipy.optimize.brentq(
            lambda x: x * halo_mass(x) / keep - r * (halo_mass(x) + baryon),
            r, 1e4 * r, xtol=1e-14, rtol=1e-15)
        assert v2 == pytest.approx(halo_mass(initial) / r, rel=1e-9)


@pytest.mark.parametrize('a0, w', [(0.85, 0.8), (1.6, 0.7)])
def test_gnedin_and_cautun_contractions(a0, w):
    # Gnedin et al. (2004): within R = r_bar, the dark matter the halo had
    # within r_bar' of the shell r', with r' M(r_bar') / (1 - f_b) = r
    # (M(r_bar') + M_b(R)), r_bar = a0 r_vir (r / r_vir)^w (by default 0.85
    # and 0.8), of the virial radius of their calibration (180 times the
    # mean matter density), for any parameterisation, against r' found by
    # root finding; by default, it contracts the halo less than Blumenthal
    # et al. Cautun et al. (2020): its formula, of the baryon fraction of
    # the context
    params = BARYON_VALUES | dict(
        mass_halo_log_m=12, mass_halo_c=5, mass_halo_a0=a0, mass_halo_w=w)
    radius = np.array([0.5, 2, 5, 20])
    context = dataclasses.replace(CONTEXT, baryon_fraction=0.16)
    models = {
        name: mass_model(BARYONS + [
            responding_halo(contraction=dict(type=name))])
        for name in ('blumenthal86', 'gnedin04', 'cautun20')}
    parts = {
        name: model.vcirc2_components(
            radius, {key: value for key, value in params.items()
                     if key in model.pdescs()}, context)
        for name, model in models.items()}
    halo, halo_params = Halo('h', ProfileNFW()), dict(log_m=12, c=5)

    def halo_mass(r):
        return float(halo.vcirc2(np.array([r]), halo_params, context)[0]) * r
    r_vir = virial_radius(halo_mass, context)

    def averaged(r):
        return a0 * r_vir * (r / r_vir) ** w

    def mass(r):
        return halo_mass(averaged(r))

    def baryons_at(r):
        masses = models['gnedin04'].enclosed_mass_components(
            np.array([r]), params, context)
        return G * float(masses['stars'][0] + masses['bulge'][0])
    keep = 1 - baryons_at(r_vir) / (baryons_at(r_vir) + halo_mass(r_vir))
    baryons = np.array([baryons_at(R) for R in radius])
    for R, baryon, v2 in zip(radius, baryons, parts['gnedin04']['halo']):
        r = r_vir * (R / (a0 * r_vir)) ** (1 / w)
        initial = scipy.optimize.brentq(
            lambda x: x * mass(x) / keep - r * (mass(x) + baryon),
            r, 1e4 * r, xtol=1e-14, rtol=1e-15)
        assert v2 == pytest.approx(mass(initial) / R, rel=1e-9)
    before = halo.vcirc2(radius, halo_params, context)
    if (a0, w) == (0.85, 0.8):
        assert np.all(before < parts['gnedin04']['halo'])
        assert np.all(
            parts['gnedin04']['halo'] < parts['blumenthal86']['halo'])
    eta = baryons / radius * 0.84 / (0.16 * before)
    np.testing.assert_allclose(
        parts['cautun20']['halo'],
        before * (0.45 + 0.38 * (eta + 1.16) ** 0.53), rtol=1e-10)
    # (of a cosmology of Omega_b: when the galaxy is planned)
    galaxy = galaxy_parser.load(dict(
        mass=BARYONS + [responding_halo(contraction=dict(type='cautun20'))],
        distance=dict(type='parameter')))
    with pytest.raises(ValueError, match="take the cosmic baryon fraction"):
        galaxy.plan_stage(
            cosmology=gbkfit.cosmology.CosmologyFlatLambdaCDM(70, 0.3))
    # (any parameterisation: the virial radius is that of the profile)
    Halo('h', ProfileNFW('density_radius'), baryons=['stars'],
         contraction=ContractionGnedin04())


@pytest.mark.parametrize('profile, parameterisation, given', [
    ('nfw', 'dark_fraction_concentration', dict(mass_halo_c=8)),
    ('burkert', 'dark_fraction_radius', dict(mass_halo_rs=6))])
@pytest.mark.parametrize(
    'responses', [{}, dict(contraction=dict(type='blumenthal86'))])
def test_dark_fractions_set_the_scale_of_the_halo(
        profile, parameterisation, given, responses):
    # The scale that gives the fraction: the mass of the halo at its
    # concentration, or its scale density at its scale radius, of the
    # halo after its contraction if it has one
    model = mass_model(BARYONS + [responding_halo(profile=dict(
        type=profile, parameterisation=parameterisation)) | responses])
    names = set(model.pdescs())
    assert {'mass_halo_fdm', 'mass_halo_rfdm'} <= names
    assert not {'mass_halo_log_m', 'mass_halo_log_rho'} & names
    params = BARYON_VALUES | given | dict(mass_halo_fdm=0.3, mass_halo_rfdm=5)
    parts = model.vcirc2_components(np.array([5.0]), params, CONTEXT)
    v2, baryon_v2 = parts['halo'][0], parts['stars'][0] + parts['bulge'][0]
    assert v2 / (v2 + baryon_v2) == pytest.approx(0.3, rel=1e-10)
    # (where no halo gives it, no model: a point fitters cannot evaluate)
    for fdm in (0.0, 1.0):
        with pytest.raises(ModelEvaluationError, match=r"within \(0, 1\)"):
            model.vcirc2(
                np.array([5.0]), params | dict(mass_halo_fdm=fdm), CONTEXT)


def test_responding_halos_are_checked(tmp_path):
    # (of baryons that give their masses within spheres: not those of user
    # components without a function of it, nor tables)
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n    return 100 * R\n")
    with pytest.raises(ValueError, match=r"\['mine'\] do not give them"):
        mass_model([
            dict(type='user', name='mine', file=str(tmp_path / 'curve.py'),
                 func='vcirc2'),
            responding_halo(
                baryons=['mine'], contraction=dict(type='blumenthal86'))])
    with pytest.raises(ValueError, match="a contraction needs the baryons"):
        Halo('h', ProfileNFW(), contraction=ContractionBlumenthal86())
    with pytest.raises(ValueError, match="a dark fraction needs the baryons"):
        Halo('h', ProfileNFW('dark_fraction_radius'))
    for options in (dict(baryons=['stars'],
                         contraction=ContractionBlumenthal86()),
                    dict(sidm=True)):
        with pytest.raises(ValueError, match="spherical shape"):
            Halo('h', ProfileNFW(), shape=ShapeFlattened(), **options)
    with pytest.raises(ValueError, match="does not have"):
        mass_model([responding_halo(contraction=dict(type='blumenthal86'))])
    with pytest.raises(ValueError, match="in a loop"):
        mass_model([
            responding_halo(name='a', baryons=['b'],
                            contraction=dict(type='blumenthal86')),
            responding_halo(name='b', baryons=['a'],
                            contraction=dict(type='blumenthal86'))])
    # (baryons that nothing uses)
    with pytest.raises(ValueError, match="it has none of these"):
        Halo('h', ProfileNFW(), baryons=['stars'])

    class Clash(ContractionBlumenthal86):
        def pdescs(self):
            return dict(c=ProfileNFW().pdescs()['c'])
    with pytest.raises(ValueError, match=r"repeat: \['c'\]"):
        Halo('h', ProfileNFW(), baryons=['stars'], contraction=Clash())
    # (SIDM of dark matter alone)
    assert 'sigma_m' in Halo('h', ProfileNFW(), sidm=True).pdescs()
    info = responding_halo(
        profile=dict(type='nfw',
                     parameterisation='dark_fraction_concentration'),
        overdensity='200m', contraction=dict(type='gnedin04'), sidm=True)
    dumped = mass_component_parser.dump(
        mass_component_parser.load(copy.deepcopy(info)))
    assert dumped == info | dict(shape=dict(type='sphere'))
    assert mass_component_parser.dump(
        mass_component_parser.load(copy.deepcopy(dumped))) == dumped


# The units of the scattering of SIDM: cm^2/g in kpc^2/Msun, and km/s
# times Gyr in kpc
SIGMA_UNIT = 2.0883573729600e-10
TIME_UNIT = 1.0227121650456948


@pytest.mark.parametrize('with_baryons', [False, True])
def test_sidm_cores(with_baryons):
    # The isothermal core of SIDM (Jiang et al. 2023): at r1, one
    # scattering in the age, (4 / sqrt(pi)) rho v sigma_m age = 1 (rho
    # and the Jeans velocity dispersion v of the halo before, in the
    # potential of the halo and of the baryons); within, the isothermal
    # Jeans-Poisson equation (solved here by solve_ivp), of the density and
    # the mass of the halo before at r1; outside, the halo before
    baryons = [BARYONS[0]] if with_baryons else []
    halo = dict(type='halo', name='halo', profile=dict(type='nfw'),
                baryons=['stars'] if with_baryons else [], sidm=True)
    model = mass_model(baryons + [halo])
    params = BARYON_VALUES | dict(
        mass_halo_log_m=11, mass_halo_c=15, mass_halo_sigma_m=1,
        mass_halo_age=10)
    own = dict(log_m=11, c=15, sigma_m=1, age=10)
    before = Halo('h', ProfileNFW())

    def halo_v2(r):
        return before.vcirc2(r, own, CONTEXT)

    stars = mass_model(baryons) if with_baryons else None

    def baryons_mass(r):
        if stars is None:
            return np.zeros_like(r)
        return stars.enclosed_mass(r, params, CONTEXT)

    def baryon_mass(r):
        return float(baryons_mass(np.array([r]))[0])

    core = sidm_core(halo_v2, baryons_mass, own)
    r1 = core.r1

    def mass(r, v2):
        return v2(np.array([r]))[0] * r / G

    nfw = at(ProfileNFW(), own)
    rho1 = nfw.density(np.array([r1]))[0]
    pressure = scipy.integrate.quad(
        lambda r: nfw.density(np.array([r]))[0] * G
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


HALOS = [
    ProfileNFW, ProfileBurkert, ProfilePseudoIsothermal, ProfileEinasto,
    ProfileZhao, ProfileDekelZhao]
SHAPES = dict(
    einasto=dict(alpha=0.17), zhao=dict(alpha=1, beta=3, gamma=1.3),
    dekel_zhao=dict(gamma=1))


@pytest.mark.parametrize('halo_class', HALOS)
@pytest.mark.parametrize('overdensity', ['200c', '500c', '200m', 'vir'])
def test_halos_have_their_mass_and_concentration(halo_class, overdensity):
    # The mean density within r_D is D rho_ref, the mass within it M_D,
    # and its velocity V_D: given by (M_D, c) or (V_D, c), a halo is the
    # same
    profile = halo_class('mass_concentration')
    halo = Halo('h', profile, overdensity=overdensity)
    shape = SHAPES.get(profile.type(), {})
    params = dict(log_m=12, c=9) | shape
    rho_s, r_s = profile.scales(
        params, Overdensity(overdensity).density(CONTEXT))
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
    assert halo.enclosed_mass(r_d, params, CONTEXT) == pytest.approx(mass)
    by_velocity = Halo(
        'h', halo_class('velocity_concentration'), overdensity=overdensity)
    radius = np.array([0.5, 5, 50])
    np.testing.assert_allclose(
        by_velocity.vcirc2(
            radius, dict(v=np.sqrt(G * mass / r_d), c=9) | shape, CONTEXT),
        halo.vcirc2(radius, params, CONTEXT), rtol=1e-10)
    # (an overdensity for every parameterisation)
    assert Halo('h', halo_class('density_radius'), overdensity=overdensity
                ).overdensity().density(CONTEXT) == pytest.approx(density)


def test_dekel_zhao_has_the_slopes_of_its_shape():
    # Its logarithmic slope is (a + 3.5 x^(1/2)) / (1 + x^(1/2)): a at the
    # centre, -3.5 far out, and -2 at ((2 - a) / 1.5)^2 r_s
    params = dict(log_rho=7, rs=5, gamma=0.7)
    halo = at(ProfileDekelZhao('density_radius'), params)
    x = np.array([1e-6, 0.3, ((2 - 0.7) / 1.5) ** 2, 4, 1e6])
    step = 1e-5

    def log_density(r):
        return np.log(halo.density(r))
    slope = (log_density(5 * x * (1 + step))
             - log_density(5 * x * (1 - step))) \
        / (np.log1p(step) - np.log1p(-step))
    np.testing.assert_allclose(
        -slope, (0.7 + 3.5 * np.sqrt(x)) / (1 + np.sqrt(x)), rtol=1e-6)


def test_zhao_is_nfw_for_its_shape():
    radius = np.array([1e-3, 0.5, 5, 50, 500])
    np.testing.assert_allclose(
        Halo('h', ProfileZhao()).vcirc2(
            radius, dict(log_m=12, c=9, alpha=1, beta=3, gamma=1), CONTEXT),
        Halo('h', ProfileNFW()).vcirc2(
            radius, dict(log_m=12, c=9), CONTEXT),
        rtol=1e-10)


def test_halo_options_are_checked():
    with pytest.raises(ValueError, match="'parameterisation' must be one of"):
        ProfileNFW(parameterisation='mass')
    with pytest.raises(ValueError, match="overdensity must be"):
        Halo('h', ProfileNFW(), overdensity='200x')
    # (a dark fraction is the halo's to solve)
    with pytest.raises(ValueError, match="solved for by its halo"):
        ProfileNFW('dark_fraction_radius').at(
            dict(fdm=0.5, rfdm=5, rs=10), DENSITY_200C)


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
    context = MassContext(1, 1, 0, 0, dict(hi=Layers(flare=1)))
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
    flat = spheroid(profile, ShapeFlattened())
    params = params | dict(q=q)
    radius = np.array([0.5, 4, 30])
    expected = [sphere_integral(
        lambda R, z: flat.density(R, z, params, CONTEXT), r) for r in radius]
    np.testing.assert_allclose(
        flat.enclosed_mass(radius, params, CONTEXT), expected, rtol=1e-8)
    if q == 1:
        np.testing.assert_allclose(
            flat.enclosed_mass(radius, params, CONTEXT),
            at(profile, params).enclosed_mass(radius, params, CONTEXT),
            rtol=1e-12)


def test_user_component_gives_its_enclosed_mass_by_a_function(tmp_path):
    (tmp_path / 'point.py').write_text(
        "import numpy as np\n"
        "def vcirc2(R, params):\n"
        "    return params['gm'] / R\n"
        "def mass(r, params):\n"
        "    return np.full_like(r, params['gm'] / 4.30091727e-6)\n")
    path = str(tmp_path / 'point.py')
    component = MassUser.from_file('mine', path, 'vcirc2', ['gm'], mass='mass')
    assert component.gives_enclosed_mass()
    np.testing.assert_allclose(component.enclosed_mass(
        np.array([1.0, 3.0]), dict(gm=G * 1e10), CONTEXT), 1e10)
    without = MassUser.from_file('mine', path, 'vcirc2', ['gm'])
    assert not without.gives_enclosed_mass()
    with pytest.raises(NotImplementedError, match="no function"):
        without.enclosed_mass(np.array([1.0]), dict(gm=1), CONTEXT)


def test_user_component_is_its_function(tmp_path):
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n"
        "    return params['v'] ** 2 * R / (R + params['r'])\n")
    component = MassUser.from_file(
        'mine', str(tmp_path / 'curve.py'), 'vcirc2', ['v', 'r'])
    assert list(component.pdescs()) == ['v', 'r']
    radius = np.array([0, 1, 10])
    np.testing.assert_allclose(
        component.vcirc2(radius, dict(v=100, r=1), CONTEXT),
        1e4 * radius / (radius + 1))
    with pytest.raises(ValueError, match="does not define a function"):
        MassUser.from_file('mine', str(tmp_path / 'curve.py'), 'other')


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
    component = MassUser.from_file(
        'mine', path, 'vcirc2', ['gm'], forces='forces')
    params = dict(gm=G * 1e10)
    R, z = np.array([1.0, 3.0]), np.array([0.5, -2.0])
    expected = MassPoint('bh').forces(R, z, dict(log_m=10), CONTEXT)
    np.testing.assert_allclose(
        component.forces(R, z, params, CONTEXT), expected, rtol=1e-10)
    assert component.gives_forces()
    in_plane = MassUser.from_file('mine', path, 'vcirc2', ['gm'])
    assert not in_plane.gives_forces()
    with pytest.raises(NotImplementedError, match="in its plane"):
        in_plane.forces(R, z, params, CONTEXT)


def test_mass_model_adds_its_components():
    model = MassModel([
        MassDisk('stars', SurfaceExponential(), VerticalThin()),
        Halo('halo', ProfileNFW())])
    assert list(model.pdescs()) == [
        'mass_stars_log_m', 'mass_stars_rd', 'mass_halo_log_m', 'mass_halo_c']
    params = dict(
        mass_stars_log_m=10.5, mass_stars_rd=3, mass_halo_log_m=12,
        mass_halo_c=8)
    radius = np.array([0, 1, 5, 20])
    expected = (
        exponential_disk().vcirc2(radius, dict(log_m=10.5, rd=3), CONTEXT)
        + Halo('h', ProfileNFW()).vcirc2(
            radius, dict(log_m=12, c=8), CONTEXT))
    np.testing.assert_allclose(
        model.vcirc2(radius, params, CONTEXT), expected, rtol=1e-12)


def test_distances_give_the_length_of_an_arcsec():
    # At 10 Mpc, an arcsec is 48.48 pc; cosmological, at the angular
    # diameter distance of the redshift of the galaxy in the cosmology; a
    # distance sets no redshift
    parameter = DistanceParameter()
    assert list(parameter.pdescs()) == ['distance']
    planck = gbkfit.cosmology.DEFAULT
    assert parameter.kpc_per_arcsec(dict(distance=10), 0.5, planck) \
        == pytest.approx(0.0484813681)
    cosmological = DistanceCosmological()
    assert cosmological.pdescs() == {}
    wmap = gbkfit.cosmology.CosmologyPublished('WMAP9')
    distance = astropy.cosmology.WMAP9.angular_diameter_distance(0.5)
    assert cosmological.kpc_per_arcsec({}, 0.5, wmap) == pytest.approx(
        distance.to_value('kpc') * np.pi / (180 * 3600))
    # (of a galaxy without a redshift, or blueshifted: no length)
    for redshift in ({}, dict(redshift=-0.001)):
        with pytest.raises(ValueError, match="needs the galaxy to have"):
            galaxy_parser.load(dict(
                components=[], mass=[dict(type='point', name='bh')],
                distance=dict(type='cosmological')) | redshift)


def test_cosmologies_give_their_epochs():
    # Published (Planck18 by default), or flat Lambda-CDM of given
    # parameters, as astropy's; Omega_m with the massive neutrinos, as
    # Planck's
    planck = gbkfit.cosmology.cosmology_parser.load(dict(type='published'))
    assert planck.dump() == dict(type='published', name='Planck18')
    reference = astropy.cosmology.Planck18
    assert planck.omega_m(0.0) == pytest.approx(
        reference.Om0 + reference.Onu0)
    assert planck.critical_density(1.5) == pytest.approx(
        reference.critical_density(1.5).to_value('solMass / kpc3'))
    info = dict(type='flat_lambda_cdm', hubble_constant=70,
                omega_matter=0.3, omega_baryon=0.05)
    flat = gbkfit.cosmology.cosmology_parser.load(copy.deepcopy(info))
    assert flat.dump() == info
    assert flat.h() == pytest.approx(0.7)
    assert flat.baryon_fraction() == pytest.approx(0.05 / 0.3)
    reference = astropy.cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
    assert flat.angular_diameter_distance(0.5) == pytest.approx(
        reference.angular_diameter_distance(0.5).to_value('Mpc'))
    # (8 pi G (rho_m + 4 rho_Lambda) / c^2, of Lambda-CDM)
    density = reference.critical_density(0.5).to_value('solMass / kpc3') * (
        reference.Om(0.5) + 4 * reference.Ode(0.5))
    assert flat.ricci_scalar(0.5) == pytest.approx(
        8 * np.pi * G * density / 299792.458 ** 2, rel=1e-6)
    assert gbkfit.cosmology.CosmologyFlatLambdaCDM(
        70, 0.3).baryon_fraction() is None
    with pytest.raises(ValueError, match="cannot exceed omega_matter"):
        gbkfit.cosmology.CosmologyFlatLambdaCDM(70, 0.3, 0.4)
    with pytest.raises(ValueError, match="'name' must be one of"):
        gbkfit.cosmology.CosmologyPublished('Planck99')

MASS = [
    dict(type='disk', name='stars', surface=dict(type='exponential'),
         vertical=dict(type='thin')),
    dict(type='disk', name='gas',
         surface=dict(type='nodewise', nodes=[0.0, 0.3, 0.6, 1.2]),
         vertical=dict(type='thin')),
    dict(type='halo', name='halo', profile=dict(
        type='pseudo_isothermal', parameterisation='density_radius'),
        overdensity='200c', shape=dict(type='sphere'))]

DISTANCE = dict(type='parameter')

MASS_PROPERTIES = dict(
    distance=10, mass_stars_log_m=10.5, mass_stars_rd=0.5,
    mass_gas_log_sigma=[8, 7.8, 7.5], mass_halo_log_rho=8, mass_halo_rs=1)

DISK = dict(
    loose=False, tilted=False, rnodes=list(range(0, 21, 2)), rstep=0.5,
    rptraits=dict(type='exponential'),
    dptraits=dict(type='uniform'),
    emission=dict(type='lines'))

DISK_PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, rpt_a=1, rpt_s=4, dpt_a=15)


def test_the_mass_of_a_galaxy_is_of_its_epoch():
    # The critical density that defines its halos is that of its redshift
    # (5.5 times that of today at z = 1.5), in its cosmology (Planck18
    # by default)
    galaxy = galaxy_parser.load(dict(
        components=[DISK | dict(
            type='thin_disk', vptraits=dict(type='mass'),
            dptraits=dict(type='uniform'))],
        mass=copy.deepcopy(MASS), distance=copy.deepcopy(DISTANCE),
        redshift=1.5))
    mass = galaxy.plan_stage().mass(DISK_PROPERTIES | MASS_PROPERTIES)
    planck = astropy.cosmology.Planck18
    assert mass.context.critical_density == pytest.approx(
        planck.critical_density(1.5).to_value('solMass / kpc3'))
    # (Omega_m with the massive neutrinos, as Planck's: 0.3111 today)
    assert mass.context.omega_m == pytest.approx(
        planck.Om(1.5) + planck.Onu(1.5))
    assert planck.Om0 + planck.Onu0 == pytest.approx(0.3111, abs=1e-4)
    assert mass.context.baryon_fraction == pytest.approx(
        planck.Ob0 / (planck.Om0 + planck.Onu0))
    # (in the cosmology of its setup)
    wmap = gbkfit.cosmology.CosmologyPublished('WMAP9')
    mass = galaxy.plan_stage(cosmology=wmap).mass(
        DISK_PROPERTIES | MASS_PROPERTIES)
    assert mass.context.critical_density == pytest.approx(
        astropy.cosmology.WMAP9.critical_density(1.5).to_value(
            'solMass / kpc3'))


def with_traits(component, **traits):
    """
    The configuration of a disk component with the given traits of its
    density (rptraits, rhtraits) and kinematics (vptraits, dptraits)
    instead of its own: a thick disk if its density has height traits, a
    thin one if not.
    """
    component = component | traits
    return component | dict(
        type='thick_disk' if 'rhtraits' in component else 'thin_disk')


def disk_galaxy(
        disk_type, vptraits, mass=None, distance=None, thick=False):
    """
    A galaxy of a disk (DISK) of the given type and velocity polar
    traits, thick (of sech2 height traits) if thick, with the given mass
    and distance.
    """
    disk = with_traits(DISK, vptraits=vptraits)
    if thick:
        disk = with_traits(disk, rhtraits=dict(type='sech2', zmax=8))
    if disk_type == 'clouds':
        disk |= dict(clouds=dict(cflux=1e-3))
    info = dict(components=[disk])
    if mass is not None:
        info |= dict(mass=copy.deepcopy(mass))
    if distance is not None:
        info |= dict(distance=copy.deepcopy(distance))
    return galaxy_parser.load(info)


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
        PixelSpectra(
            size=(32, 32), step=(1, 1),
            spectral=SpectralAxis(
                size=51, step=10, convention='relativistic')),
        line_of_sight_parser.load(dict(type=line_of_sight)))])
    params = ParamSpace.from_properties(group.pdescs(), properties)
    return group.plan(driver).model_h(
        params.evaluate())[0]['spectra']['d'].copy()


@pytest.mark.parametrize('line_of_sight, disk_type', [
    ('flat', 'smooth'),
    ('grid', 'smooth'),
    ('grid', 'clouds')])
def test_mass_trait_is_the_circular_velocity_at_the_subnodes(
        driver, line_of_sight, disk_type):
    # A disk with the trait 'mass' is the disk whose node-wise tangential
    # velocity at its subnodes is the circular velocity of the mass of its
    # galaxy, at the radii of the subnodes in kpc (a disk of no light that
    # matters beyond its last ring, where the one given at the subnodes
    # keeps its last value, and the one of the mass goes on)
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
    properties = DISK_PROPERTIES | dict(rpt_s=1.2)
    if thick:
        properties |= dict(rht_s=1)
    np.testing.assert_allclose(
        evaluate(
            driver, massive, line_of_sight, properties | MASS_PROPERTIES),
        evaluate(
            driver, given, line_of_sight, properties | dict(vpt_vt=vcirc)),
        rtol=1e-6, atol=1e-7)


def test_mass_trait_parameters_and_round_trip():
    galaxy = disk_galaxy(
        'smooth', dict(type='mass'), MASS, DISTANCE, thick=True)
    # The trait has no parameters: those of the distance and the mass are
    # the galaxy's
    assert not any(name.startswith('vpt') for name in galaxy.pdescs())
    assert set(MASS_PROPERTIES) <= set(galaxy.pdescs())
    dumped = galaxy_parser.dump(galaxy)
    assert dumped['mass'] == MASS
    assert dumped['distance'] == DISTANCE
    assert galaxy_parser.dump(galaxy_parser.load(copy.deepcopy(dumped))) \
        == dumped


def test_mass_trait_mass_and_distance():
    # (mass that no component uses is for profiles or derived values: see
    # ObservationGroup, which warns of it unused)
    with pytest.raises(ValueError, match="need the mass of their galaxy"):
        disk_galaxy('smooth', dict(type='mass'))
    assert not disk_galaxy(
        'smooth', dict(type='arctan', direction='tan'), MASS, DISTANCE
    ).components_use_mass()
    with pytest.raises(ValueError, match="needs its distance"):
        disk_galaxy('smooth', dict(type='mass'), MASS)
    # (a distance without mass gives the length of an arcsec to the
    # derived values, e.g. of an arctan disk at z ~ 1.5: audit 2026-10-31
    # #49)
    galaxy = disk_galaxy(
        'smooth', dict(type='arctan', direction='tan'), distance=DISTANCE)
    functions = galaxy.plan_functions((0.0, 0.0)).bind(dict(distance=10))
    assert functions['kpc_per_arcsec'] == pytest.approx(0.048481, rel=1e-4)


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
    from gbkfit.model import ThinDisk
    from gbkfit.model._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import NodalDiskDensity
    component = ThinDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        rptraits=[
            dict(type='exponential'), dict(type='nw_harmonic', order=2),
            dict(type='mixture_gauss', nblobs=1)],
        vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
        sptraits=dict(type='rrange'),
        emission=dict(type='lines')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, rpt_a=1, rpt_s=4,
        rpt2_a=np.full(11, 0.5), rpt2_p=np.zeros(11), rpt3_r=15, rpt3_t=40,
        rpt3_a=0.5, rpt3_s=0.3, rpt3_q=1, rpt3_p=0, dpt_a=10, spt_rmin=0,
        spt_rmax=18)
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float32), None, None, True,
        None, None, 0.0)
    density = NodalDiskDensity(
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
    from gbkfit.model import ThinDisk
    from gbkfit.model._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import NodalDiskDensity
    component = ThinDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        rptraits=[
            dict(type='exponential'),
            dict(type='map', size=[6, 4], step=2.0)],
        vptraits=dict(type='uniform', direction='tan'),
        dptraits=dict(type='uniform'),
        sptraits=dict(type='azrange'),
        emission=dict(type='lines')))
    cells = np.random.default_rng(3).normal(size=24)
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, rpt_a=1, rpt_s=4,
        rpt2_a=cells, vpt_vt=100, dpt_a=10, spt_p=17, spt_s=93)
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float32), None, None, True,
        None, None, 0.0)
    density = NodalDiskDensity(
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
    from gbkfit.model import ThinDisk
    from gbkfit.model._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import NodalDiskDensity
    component = ThinDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        rptraits=[
            dict(type='mixture_gauss', nblobs=1),
            dict(type='mixture_ggauss', nblobs=1),
            dict(type='nw_distortion')],
        vptraits=dict(type='uniform', direction='tan'),
        dptraits=dict(type='uniform'),
        emission=dict(type='lines')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, vpt_vt=100, dpt_a=10,
        rpt_r=15, rpt_t=40, rpt_a=1, rpt_s=0.01, rpt_q=1, rpt_p=0,
        rpt2_r=10, rpt2_t=-70, rpt2_a=1, rpt2_s=0.005, rpt2_b=0.5, rpt2_q=1,
        rpt2_p=0, rpt3_a=np.ones(11), rpt3_p=np.full(11, 40.0),
        rpt3_s=np.full(11, 0.01))
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float64), None, None, True,
        None, None, 0.0)
    density = NodalDiskDensity(
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
        DISK, rptraits=dict(type='nw_uniform'), vptraits=dict(type='mass'))
    disk |= dict(name='hi')
    properties = {
        f'hi_{name}': value for name, value in DISK_PROPERTIES.items()
        if name != 'rpt_s'} | dict(hi_rpt_a=brightness)
    if thick:
        disk = with_traits(disk, rhtraits=dict(type='sech2', zmax=8))
        properties |= dict(hi_rht_s=1)
    halo = dict(type='halo', name='halo', profile=dict(
        type='pseudo_isothermal', parameterisation='density_radius'))
    halo_properties = dict(distance=10, mass_halo_log_rho=8, mass_halo_rs=1)

    def galaxy(gas):
        return galaxy_parser.load(dict(
            components=[copy.deepcopy(disk)], mass=[gas, halo],
            distance=copy.deepcopy(DISTANCE)))

    following = galaxy(dict(type='traced_disk', name='gas', component='hi'))
    nodewise = galaxy(dict(
        type='disk', name='gas',
        surface=dict(
            type='nodewise', nodes=(subrnodes * KPC_PER_ARCSEC).tolist()),
        vertical=dict(type='sech2' if thick else 'thin')))
    sigma = upsilon * np.interp(subrnodes[:-1], rnodes, brightness)
    nodewise_properties = dict(mass_gas_log_sigma=np.log10(sigma))
    if thick:
        nodewise_properties |= dict(mass_gas_hz=KPC_PER_ARCSEC)
    np.testing.assert_allclose(
        evaluate(
            driver, following, line_of_sight,
            properties | halo_properties | dict(mass_gas_upsilon=upsilon)),
        evaluate(
            driver, nodewise, line_of_sight,
            properties | halo_properties | nodewise_properties),
        rtol=1e-5, atol=1e-7)


def test_a_disk_rotates_in_its_own_thin_mass_disk():
    # A thin mass disk that follows the density of its disk: the density
    # goes on as its tail beyond the last ring, to 0 at the end of the
    # tail, with no jump where a thin disk would pull infinitely: the
    # rotation is finite at every radius of its table
    disk = with_traits(DISK, vptraits=dict(type='mass')) | dict(name='gas')
    galaxy = galaxy_parser.load(dict(
        components=[disk],
        mass=[dict(type='traced_disk', name='hi', component='gas',
                   vertical=dict(type='thin')),
              dict(type='halo', name='halo', profile=dict(
                  type='pseudo_isothermal',
                  parameterisation='density_radius'))],
        distance=copy.deepcopy(DISTANCE)))
    params = {f'gas_{key}': value
              for key, value in DISK_PROPERTIES.items()} | dict(
        distance=10, mass_hi_upsilon=1e8, mass_halo_log_rho=7,
        mass_halo_rs=3)
    state = galaxy.plan_stage().evaluate(params).states[0]
    assert np.all(np.isfinite(state.rotation['vpt_vt']))


def test_disks_follow_components_of_their_galaxy():
    mass = [
        dict(type='traced_disk', name='gas', component='hi', rstep=1.0)]
    loaded = mass_component_parser.load(copy.deepcopy(mass))
    assert mass_component_parser.dump(loaded) == mass
    with pytest.raises(ValueError, match="has no component 'hi'"):
        disk_galaxy('smooth', dict(type='mass'), mass, DISTANCE)


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
        | dict(name='stars')
    dust = dict(
        type='thick_disk', name='dust', loose=False, tilted=False,
        rnodes=rnodes, rstep=DISK['rstep'],
        rptraits=dict(type='nw_uniform'),
        rhtraits=dict(type='sech2', zmax=8),
        opacity=dict(type='dust'))
    galaxy = galaxy_parser.load(dict(
        components=[stars, dust],
        mass=[dict(type='traced_disk', name='gas', component='dust',
                   vertical=dict(type='thin'))],
        distance=copy.deepcopy(DISTANCE)))
    params = {
        f'{name}_{key}': value for name in ('stars', 'dust')
        for key, value in DISK_PROPERTIES.items()
        if f'{name}_{key}' in galaxy.pdescs()}
    params |= dict(
        dust_rpt_a=opacity, dust_rht_s=1, distance=10,
        mass_gas_upsilon=upsilon)
    subrnodes = np.array(subring_radii(rnodes, DISK['rstep']))
    nodewise = mass_model([dict(
        type='disk', name='gas',
        surface=dict(
            type='nodewise', nodes=(subrnodes * KPC_PER_ARCSEC).tolist()),
        vertical=dict(type='thin'))])
    sigma = upsilon * np.interp(subrnodes[:-1], rnodes, opacity)
    # (at the subnodes; then at the radii of the tail)
    rotation = galaxy.plan_stage().evaluate(params).states[0].rotation
    np.testing.assert_allclose(
        rotation['vpt_vt'][:len(subrnodes)],
        np.sqrt(nodewise.vcirc2(
            subrnodes * KPC_PER_ARCSEC,
            dict(mass_gas_log_sigma=np.log10(sigma)), CONTEXT)),
        rtol=1e-6)


def test_pressure_support_is_the_jeans_equation(driver):
    # An exponential density (scale 4) and dispersion (scale 10) of a
    # thin disk: (R / rho) d(rho sigma^2)/dR = -sigma^2 R (1 / 4 + 2 / 10),
    # to the precision of the three-point derivative on the subnodes
    from gbkfit.model import ThinDisk
    from gbkfit.model._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import pressure_support
    component = ThinDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        rptraits=dict(type='exponential'),
        vptraits=dict(type='mass'), dptraits=dict(type='exponential'),
        emission=dict(type='lines')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, rpt_a=1, rpt_s=4,
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
    from gbkfit.model import ThickDisk
    from gbkfit.model._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.model.components.disks._profiles import pressure_support
    component = ThickDisk.load(dict(
        rnodes=list(range(0, 21, 2)), rstep=0.5,
        rptraits=ptrait, rhtraits=dict(type='sech2', zmax=8),
        vptraits=dict(
            type='mass', pressure_support=dict(type='self_gravitating')),
        dptraits=dict(type='exponential'),
        emission=dict(type='lines')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, vsys=0, rpt_a=1, rpt_s=4, rht_s=1,
        dpt_a=30, dpt_s=10) | (dict(rpt_b=1.5) if 'rpt_b' in component.pdescs()
                               else {})
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float64), None, None, True,
        None, None, 0.0)
    disk = component.density_disk()
    (name, support), = disk.pressure_supported_params().items()
    assert name == 'vpt_vt' and support.type() == 'self_gravitating' \
        and support.mass() is None
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
        vptraits=dict(type='mass', pressure_support=dict(type='jeans')))
    # (a disk of no light that matters beyond its last ring, where the one
    # given at the subnodes keeps its last value)
    properties = DISK_PROPERTIES | dict(rpt_s=1.2, dpt_a=30, dpt_s=10)
    subrnodes = np.array(subring_radii(rnodes, DISK['rstep']))
    density = np.exp(-subrnodes / 1.2)
    if line_of_sight == 'grid':
        disk = with_traits(
            disk, rhtraits=dict(type='sech2', zmax=8, nodewise=True))
        properties |= dict(rht_s=flaring)
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
        dict(type='mass', pressure_support=dict(type='jeans')))
    assert traits.vpt_parser.dump(trait) == dict(
        type='mass', pressure_support=dict(type='jeans'))
    assert traits.vpt_parser.dump(traits.VPTraitMass()) \
        == dict(type='mass')
    with pytest.raises(ConfigError, match="unknown type 'asymmetric'"):
        traits.vpt_parser.load(dict(
            type='mass', pressure_support=dict(type='asymmetric')))
    trait = traits.vpt_parser.load(dict(
        type='mass', pressure_support=dict(type='jeans', mass='gas')))
    assert traits.vpt_parser.dump(trait) == dict(
        type='mass', pressure_support=dict(type='jeans', mass='gas'))
    several = dict(type='mass', pressure_support=dict(
        type='self_gravitating', mass=['hi', 'h2']))
    assert traits.vpt_parser.dump(traits.vpt_parser.load(several)) == several
    with pytest.raises(ValueError, match="each once"):
        traits.PressureSupportJeans(['hi', 'hi'])


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
        dict(type='halo', name='halo', profile=dict(
            type='pseudo_isothermal', parameterisation='density_radius'))]
    density = dict(jeans='gas', self_gravitating='stars')[mode]
    galaxy = disk_galaxy('smooth', dict(
        type='mass', pressure_support=dict(type=mode, mass=density)),
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
        dict(type='halo', name='halo', profile=dict(
            type='pseudo_isothermal', parameterisation='density_radius'))]
    galaxy = disk_galaxy('smooth', dict(
        type='mass', pressure_support=dict(type='jeans',
        mass=['hi', 'h2'])), mass, DISTANCE)
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
        DISK, vptraits=dict(type='mass', pressure_support=dict(type=mode)),
        rhtraits=dict(type='sech2', zmax=8)) \
        | dict(name='gas')
    mass = [
        dict(type='traced_disk', name='hi', component='gas'),
        dict(type='halo', name='halo', profile=dict(
            type='pseudo_isothermal', parameterisation='density_radius'))]

    def rotation(density):
        support = disk | dict(dict(
            vptraits=disk['vptraits'] | density))
        galaxy = galaxy_parser.load(dict(
            components=[support], mass=copy.deepcopy(mass),
            distance=copy.deepcopy(DISTANCE)))
        params = {f'gas_{key}': value
                  for key, value in DISK_PROPERTIES.items()} | dict(
            gas_rht_s=0.5, distance=10, mass_hi_upsilon=1e8,
            mass_halo_log_rho=7, mass_halo_rs=3)
        state = galaxy.plan_stage().evaluate(params).states[0]
        return state.rotation['vpt_vt']
    np.testing.assert_allclose(
        rotation(dict(pressure_support=dict(type=mode, mass='hi'))),
        rotation({}), rtol=1e-10)


def test_pressure_density_is_a_mass_component_that_gives_it():
    vptraits = dict(
        type='mass', pressure_support=dict(type='jeans', mass='dust'))
    with pytest.raises(ValueError, match="which its galaxy does not have"):
        disk_galaxy('smooth', vptraits, MASS, DISTANCE)
    # (a thin disk has no density at its midplane: refused at load, not at
    # the first evaluation of a fit, audit 2026-10-31 #30a)
    with pytest.raises(ValueError, match="does not give its density"):
        disk_galaxy(
            'smooth', vptraits | dict(
                pressure_support=dict(type='jeans', mass='stars')),
            MASS,
            DISTANCE)


def rotation_at_heights(vptraits, mass, params):
    """
    The galaxy of a thick disk (DISK, of sech2 height traits to zmax 8)
    of the velocity traits vptraits and 'mass' at each height, and of the
    given mass, and what its gravity gives the disk (its state) at the
    values params of its parameters.
    """
    disk = with_traits(
        DISK, vptraits=vptraits, rhtraits=dict(type='sech2', zmax=8))
    disk |= dict(vhtraits=dict(type='mass', zmax=8))
    galaxy = galaxy_parser.load(dict(
        components=[disk], mass=copy.deepcopy(mass),
        distance=copy.deepcopy(DISTANCE)))
    return galaxy, galaxy.plan_stage().evaluate(params).states[0]



def test_a_self_gravitating_density_needs_a_projection():
    # (an NFW halo has no surface density: refused at load, not at the
    # first evaluation of a fit: audit 2026-10-31 #30a)
    disk = with_traits(
        DISK, vptraits=dict(
            type='mass', pressure_support=dict(type='self_gravitating',
            mass='halo'))) | dict(name='gas')
    with pytest.raises(ValueError, match="does not give its surface"):
        galaxy_parser.load(dict(
            components=[disk], mass=[dict(
                type='halo', name='halo', profile=dict(type='nfw'))],
            distance=copy.deepcopy(DISTANCE)))

def test_the_table_of_rotation_at_heights_stays_within_the_density():
    # (above the zmax of the density height traits, pressure support
    # divided by a density of 0: the table was NaN there, and the disk
    # with it)
    disk = with_traits(
        DISK, vptraits=dict(type='mass', pressure_support=dict(type='jeans')),
        rhtraits=dict(type='sech2', zmax=6))
    disk |= dict(vhtraits=dict(type='mass', zmax=7))
    with pytest.raises(ValueError, match="must not be above the zmax"):
        galaxy_parser.load(dict(
            components=[disk], mass=[dict(
                type='point', name='bh')], distance=copy.deepcopy(DISTANCE)))


def test_rotation_at_each_height_of_a_point_mass(driver):
    # v(R, z)^2 = G M R^2 / r^3, plus the jeans pressure support of the
    # disk, of a constant height profile and a uniform dispersion: R
    # d(rho sigma^2)/dR / rho at every height, of the exponential surface
    # density (by the three-point derivative on the subnodes, as the disk
    # takes it); v = 0 where v^2 < 0 (the pressure exceeds the gravity),
    # at the midplane and at the heights; the factors v(R, z) / v(R, 0) at
    # the subnodes and the 33 heights of the table (1 where v(R, 0) = 0),
    # which the disk interpolates linearly
    from gbkfit.model._component_set import IMAGE_SPECTRAL_AXIS
    params = DISK_PROPERTIES | dict(
        rht_s=1, dpt_a=100, distance=10, mass_bh_log_m=10)
    galaxy, state = rotation_at_heights(
        dict(type='mass', pressure_support=dict(type='jeans')),
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
    np.testing.assert_allclose(
        state.rotation['vpt_vt'][1:len(r)], v[:, 0], rtol=1e-8)
    factors = np.array(
        [state.rotation_factors[f'vht_f{k}'] for k in range(33)]).T
    np.testing.assert_allclose(
        factors[1:], np.divide(
            v, v[:, :1], out=np.ones_like(v), where=v[:, :1] > 0),
        rtol=1e-8, atol=1e-7)
    np.testing.assert_array_equal(factors[0], 1)
    plan = component.plan(
        driver, IMAGE_SPECTRAL_AXIS, np.dtype(np.float64), None, None, True,
        None, None, 0.0).disk_plan()
    values = {p: params[p] for p in component.pdescs()} | state.inputs()
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
        rht_s=1, distance=10, mass_gas_log_m=9.5, mass_gas_rd=1,
        mass_bh_log_m=10.5)
    mass = [
        dict(type='disk', name='gas', surface=dict(type='exponential'),
             vertical=dict(type='thin')),
        dict(type='point', name='bh')]
    galaxy, state = rotation_at_heights(dict(
        type='mass', pressure_support=dict(type='self_gravitating',
        mass='gas')), mass, params)
    disk = galaxy.components()[0].density_disk()
    R = np.array(disk.subrnodes())[1:, None] * KPC_PER_ARCSEC
    z = np.linspace(0, 8, 33) * KPC_PER_ARCSEC
    surface = 10 ** 9.5 / (2 * np.pi) * np.exp(-R)
    u = np.pi * G * surface * z / 15 ** 2
    model = mass_model(mass)
    force_R = model.forces(R, z, dict(
        mass_gas_log_m=9.5, mass_gas_rd=1, mass_bh_log_m=10.5), CONTEXT)[0]
    v2 = -R * force_R + 15 ** 2 * R * (-2 + 2 * u * np.tanh(u))
    factors = np.array(
        [state.rotation_factors[f'vht_f{k}'] for k in range(33)]).T[1:]
    assert np.all(v2 > 0) and np.max(u) > 1
    np.testing.assert_allclose(
        factors, np.sqrt(v2 / v2[:, :1]), rtol=2e-3)


@pytest.mark.parametrize('vptraits, user, message', [
    (dict(type='arctan', direction='tan'), False,
     "pairs with the velocity trait 'mass' only, not 'arctan'"),
    (dict(type='mass', pressure_support=dict(type='self_gravitating')), False,
     "give the mass of the pressure support"),
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
        DISK, vptraits=vptraits, rhtraits=dict(type='sech2', zmax=8))
    disk |= dict(vhtraits=dict(type='mass', zmax=8))
    with pytest.raises(ValueError, match=message):
        galaxy_parser.load(dict(
            components=[disk], mass=copy.deepcopy(mass),
            distance=copy.deepcopy(DISTANCE)))


def test_rotation_at_each_height_option():
    from gbkfit.model.components import traits
    trait = traits.vht_parser.load(dict(type='mass', zmax=3))
    assert traits.vht_parser.dump(trait) == dict(type='mass', zmax=3.0)
    assert trait.from_galaxy().heights[-1] == 3.0
    with pytest.raises(ValueError, match="'zmax' must be finite and greater than 0"):
        traits.VHTraitMass(0)


def test_one_mass_trait_per_disk():
    from gbkfit.model.components import traits
    from gbkfit.model.components.parts import KinematicTraits
    KinematicTraits(
        [traits.VPTraitMass(), traits.VPTraitNWUniform('rad')],
        traits.DPTraitUniform())
    with pytest.raises(ValueError, match="at most one velocity trait"):
        KinematicTraits(
            [traits.VPTraitMass(),
             traits.VPTraitMass(traits.PressureSupportJeans())],
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
        'smooth', dict(type='mass', pressure_support=dict(type='jeans')), MASS,
        DISTANCE)
    group = ObservationGroup([galaxy], [
        ObservationSky(
            PixelSpectra(
                size=(32, 32), step=(1, 1),
                spectral=SpectralAxis(
                    size=51, step=10, convention='relativistic')),
            line_of_sight_parser.load(dict(type='flat')), dtype=dtype,
            name=name)
        for name, dtype in (('single', 'float32'), ('double', 'float64'))])
    params = ParamSpace.from_properties(
        group.pdescs(), DISK_PROPERTIES | MASS_PROPERTIES).evaluate()
    plan = group.plan(driver, dict(single=DriverHost()))
    assert plan.galaxy_drivers() == (driver,)
    single, double = plan.model_h(params)
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
        tilted=True)
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
    from gbkfit.model.components._nodal import tail_radii
    galaxy = disk_galaxy(
        'smooth', dict(type='mass', pressure_support=dict(type='jeans')), MASS,
        DISTANCE)
    params = DISK_PROPERTIES | MASS_PROPERTIES
    extra = Extras()
    states = galaxy.plan_stage().evaluate(params, extra).states
    table = extra.get('vpt_vt_rotation', component='component0')
    assert table.colnames == [
        'radius', 'radius_kpc', 'vcirc_stars', 'vcirc_gas', 'vcirc_halo',
        'vnewton', 'vcirc', 'pressure', 'vrot']
    # (at the subnodes, and beyond, at the radii of the tail)
    subrnodes = galaxy.components()[0].density_disk().subrnodes()
    np.testing.assert_allclose(
        table['radius'], subrnodes + tail_radii(subrnodes))
    np.testing.assert_allclose(
        table['radius_kpc'], table['radius'] * KPC_PER_ARCSEC)
    vcirc2 = sum(np.sign(table[f'vcirc_{name}']) * table[f'vcirc_{name}'] ** 2
                 for name in ('stars', 'gas', 'halo'))
    np.testing.assert_allclose(
        np.sign(table['vcirc']) * table['vcirc'] ** 2, vcirc2, rtol=1e-12)
    np.testing.assert_array_equal(table['vnewton'], table['vcirc'])
    np.testing.assert_allclose(table['vrot'], states[0].rotation['vpt_vt'])
    # (0 where the pressure exceeds the gravity: far in the tail, where the
    # density falls and the dispersion keeps its value at the last ring)
    np.testing.assert_allclose(
        table['vrot'] ** 2, np.maximum(vcirc2 + table['pressure'], 0),
        rtol=1e-9)


@pytest.mark.parametrize('vertical', [None, 'thin', 'sech2'])
def test_cell_masses_add_up(vertical):
    # The masses of the cells between spheres and latitudes add up to the
    # mass between the inner and the outer sphere: of a halo (of its
    # enclosed mass), and of a thin disk (in the cells next to its plane)
    # or a thick one (of its density)
    if vertical is None:
        component = dict(type='halo', name='c', profile=dict(
            type='nfw', parameterisation='density_radius'))
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
