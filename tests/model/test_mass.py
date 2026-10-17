"""
Tests for the mass of galaxies: its components against independent
formulas and integrals, their parameterisations, the distance of their
galaxy, and the velocity traits 'tan_mass', which take their circular
velocity.
"""

import copy
import functools
import logging

import astropy.cosmology
import numpy as np
import pytest
import scipy.integrate
import scipy.special

from gbkfit.model import galaxy_parser
from gbkfit.model.distance import (
    KPC_PER_ARCSEC_MPC, DistanceDirect, DistanceRedshift)
from gbkfit.model.mass import (
    MassContext, MassDisk, MassModel, MassPoint, MassSpheroid, MassUser,
    ProfileBurkert, ProfileEinasto, ProfileNFW, ProfilePseudoIsothermal,
    ProfileSersic, ProfileZhao, SurfaceComponent, SurfaceExponential,
    SurfaceNodewise, VerticalComponent, VerticalExponential, VerticalGauss,
    VerticalSech2, VerticalThin, mass_component_parser)
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


@pytest.mark.parametrize('vertical', ['exponential', 'sech2', 'gauss'])
def test_thick_disks_of_vanishing_thickness_are_thin(vertical):
    # (the numerical integral of a thick disk against Freeman's closed
    # form; the transforms of the vertical laws against their densities)
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


@pytest.mark.parametrize('vertical', ['thin', 'sech2', 'gauss'])
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


def test_unresolved_heights_are_errors():
    disk = nodewise_disk([0, 1, 2], 'sech2')
    with pytest.raises(ValueError, match="heights are not resolved"):
        disk.vcirc2([0.5, 1], dict(log_sigma=[2, 2], hz=1e-12), CONTEXT)


@pytest.mark.parametrize('nodes', [[1], [0, 2, 1], [-1, 1], [0, np.inf]])
def test_nodes_of_nodewise_disks_are_checked(nodes):
    with pytest.raises(
            ConfigError, match="the nodes of a nodewise surface density"):
        SurfaceNodewise(nodes)


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
    (ProfileSersic(), dict(log_m=10, re=1.5, n=0.6)),
    (ProfileSersic(), dict(log_m=10, re=1.5, n=4))]


@pytest.mark.parametrize('profile, params', SPHERES)
def test_spheres_are_their_densities(profile, params):
    # Their enclosed mass is the integral of their density (including
    # radii where the halos use their series), and their circular
    # velocity G M(<r) / r
    sphere = MassSpheroid('s', profile)
    radius = np.array([1e-4, 5e-4, 2e-3, 0.5, 2, 10, 100])
    expected = [
        scipy.integrate.quad(
            lambda r: 4 * np.pi * r ** 2 * float(sphere.density_spherical(
                np.array(r), params, CONTEXT)),
            0, radius_, points=[1e-5, 1e-3, 0.1], limit=400)[0]
        for radius_ in radius]
    np.testing.assert_allclose(
        sphere.enclosed_mass(radius, params, CONTEXT), expected, rtol=1e-6)
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
    sersic = MassSpheroid('b', ProfileSersic())
    np.testing.assert_allclose(
        sersic.density_spherical(radius, params, CONTEXT),
        sigma0 / (np.pi * rd) * scipy.special.k0(radius / rd), rtol=1e-6)
    assert sersic.enclosed_mass(1e3, params, CONTEXT) \
        == pytest.approx(mass, rel=1e-8)


HALOS = [
    ProfileNFW, ProfileBurkert, ProfilePseudoIsothermal, ProfileEinasto,
    ProfileZhao]
SHAPES = dict(einasto=dict(alpha=0.17), zhao=dict(alpha=1, beta=3, gamma=1.3))


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
    assert halo.enclosed_mass(r_d, params, CONTEXT) == pytest.approx(mass)
    by_velocity = MassSpheroid(
        'h', halo_class('velocity_concentration', overdensity))
    radius = np.array([0.5, 5, 50])
    np.testing.assert_allclose(
        by_velocity.vcirc2(
            radius, dict(v=np.sqrt(G * mass / r_d), c=9) | shape, CONTEXT),
        halo.vcirc2(radius, params, CONTEXT), rtol=1e-10)


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
    # DistanceRedshift, unless it has its own (for the distance only)
    direct = DistanceDirect()
    assert list(direct.pdescs()) == ['distance']
    assert direct.kpc_per_arcsec(dict(distance=10), 0.5) \
        == pytest.approx(0.0484813681)
    assert direct.epoch(dict(distance=10), 0.5) == 0
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
    with pytest.raises(ConfigError, match="epoch cannot be negative"):
        DistanceDirect(epoch=-1)


MASS = [
    dict(type='disk', name='stars', surface=dict(type='exponential'),
         vertical=dict(type='thin')),
    dict(type='disk', name='gas',
         surface=dict(type='nodewise', nodes=[0.0, 0.3, 0.6, 1.2]),
         vertical=dict(type='thin')),
    dict(type='spheroid', name='halo', profile=dict(
        type='pseudo_isothermal', parameters='density_radius'))]

DISTANCE = dict(type='direct', cosmology='Planck18', epoch=0.0)

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
        Observation, ObservationGroup, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import ParamSpace
    group = ObservationGroup([galaxy], [Observation(
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
    # A disk with the trait 'tan_mass' is the disk whose node-wise tangential
    # velocity at its subnodes is the circular velocity of the mass of its
    # galaxy, at the radii of the subnodes in kpc
    thick = line_of_sight == 'grid'
    massive = disk_galaxy(
        disk_type, dict(type='tan_mass'), MASS, DISTANCE, thick=thick)
    given = disk_galaxy(
        disk_type, dict(type='nw_tan_uniform', sampling='subrings'),
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
        'smdisk', dict(type='tan_mass'), MASS, DISTANCE, thick=True)
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
    with pytest.raises(ConfigError, match="need the mass of their galaxy"):
        disk_galaxy('smdisk', dict(type='tan_mass'))
    with pytest.raises(ConfigError, match="no component uses"):
        disk_galaxy('smdisk', dict(type='tan_arctan'), MASS, DISTANCE)
    with pytest.raises(ConfigError, match="needs its distance"):
        disk_galaxy('smdisk', dict(type='tan_mass'), MASS)
    with pytest.raises(ConfigError, match="used by its mass only"):
        disk_galaxy('smdisk', dict(type='tan_arctan'), distance=DISTANCE)


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
            vptraits=dict(type='tan_mass'), dptraits=dict(type='uniform')),
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


@pytest.mark.parametrize('line_of_sight', ['flat', 'grid'])
def test_disk_of_a_component_is_the_nodewise_disk_of_its_density(
        driver, line_of_sight):
    # A disk whose mass follows a node-wise density (linear between the
    # subnodes, and 0 at the last) is the nodewise disk of that surface
    # density at the subnodes, and of the scale height of the density
    # (in kpc) if thick
    from gbkfit.model.components.disks._disk import subring_radii
    rnodes = list(range(0, 21, 2))
    brightness = np.append(np.linspace(1, 0.2, 10), 0)
    upsilon = 2e8
    subrnodes = np.array(subring_radii(rnodes, DISK['rstep']))
    thick = line_of_sight == 'grid'
    disk = with_traits(
        DISK, ptraits=dict(type='nw_uniform'), vptraits=dict(type='tan_mass'))
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
        disk_galaxy('smdisk', dict(type='tan_mass'), mass, DISTANCE)


def test_disk_follows_a_component_without_light():
    # The mass of a disk can follow a component whose light no observation
    # sees, here dust: thin, its gravity is that of the nodewise disk of
    # the surface density of the dust (as in test_disk_of_a_component_is_
    # the_nodewise_disk_of_its_density), at the subnodes of the disk it
    # moves
    from gbkfit.model.components.disks._disk import subring_radii
    rnodes = list(range(0, 21, 2))
    opacity = np.append(np.linspace(1, 0.2, 10), 0)
    upsilon = 2e8
    stars = with_traits(DISK, vptraits=dict(type='tan_mass')) \
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
        galaxy.plan_gravity().evaluate(params)[0]['vpt_vt'],
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
            vptraits=dict(type='tan_mass'), dptraits=dict(type='exponential')),
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
        pressure_support(disk, plan.disk_plan(), params, radius),
        -(30 * np.exp(-radius / 10)) ** 2 * radius * (1 / 4 + 2 / 10),
        rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize('line_of_sight', ['flat', 'grid'])
def test_mass_trait_with_pressure_support(driver, line_of_sight):
    # The trait 'tan_mass' with pressure support is the node-wise velocity
    # sqrt(v_c^2 + (R / rho) d(rho sigma^2)/dR) at the subnodes, with the
    # derivative by the three-point formula: thin, rho is the surface
    # density; thick and flaring (sech2, node-wise scale height s), rho is
    # the surface density / (2 s)
    from gbkfit.model.components.disks._disk import subring_radii
    rnodes = list(range(0, 21, 2))
    flaring = np.linspace(0.5, 2, 11)
    disk = with_traits(
        DISK, dptraits=dict(type='exponential'),
        vptraits=dict(type='tan_mass', pressure_support='jeans'))
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
        disk, vptraits=dict(type='nw_tan_uniform', sampling='subrings'))]))
    assert np.all(vcirc2 + support >= 0) and np.max(-support) > 500
    np.testing.assert_allclose(
        evaluate(
            driver, supported, line_of_sight, properties | MASS_PROPERTIES),
        evaluate(driver, given, line_of_sight, properties | dict(
            vpt_vt=np.sqrt(vcirc2 + support))),
        rtol=1e-5, atol=1e-7)


def test_pressure_support_option():
    from gbkfit.model.components.disks import traits
    trait = traits.vpt_parser.load(
        dict(type='tan_mass', pressure_support='jeans'))
    assert traits.vpt_parser.dump(trait) == dict(
        type='tan_mass', pressure_support='jeans')
    assert traits.vpt_parser.dump(traits.VPTraitTanMass()) \
        == dict(type='tan_mass')
    with pytest.raises(ConfigError, match="pressure_support must be"):
        traits.VPTraitTanMass('asymmetric')


def test_observations_of_a_galaxy_share_its_gravity(driver, monkeypatch):
    # The gravity of a galaxy is evaluated once, on its driver, for all
    # the observations of it, whatever their drivers and dtypes: here one
    # on the host and one on the driver of the galaxy
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.model.components._gravity import GravityPlan
    from gbkfit.observation import (
        Observation, ObservationGroup, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import ParamSpace
    calls = []
    evaluate_gravity = GravityPlan.evaluate

    def counted(self, params, out_extra=None):
        calls.append(params)
        return evaluate_gravity(self, params, out_extra)
    monkeypatch.setattr(GravityPlan, 'evaluate', counted)
    # (pressure support samples the profiles of the disk on that driver)
    galaxy = disk_galaxy(
        'smdisk', dict(type='tan_mass', pressure_support='jeans'), MASS,
        DISTANCE, driver=driver)
    assert galaxy.driver() is driver
    group = ObservationGroup([galaxy], [
        Observation(
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
    disk = with_traits(DISK, vptraits=dict(type='tan_mass')) | dict(
        type='smdisk', tilted=True)
    galaxy = galaxy_parser.load(dict(
        components=[disk], mass=copy.deepcopy(MASS),
        distance=copy.deepcopy(DISTANCE)))
    nodes = len(DISK['rnodes'])
    params = DISK_PROPERTIES | MASS_PROPERTIES | dict(
        posa=np.full(nodes, 30.0), incl=np.linspace(60, 60 + tilt, nodes))
    caplog.set_level(logging.WARNING)
    galaxy.plan_gravity().evaluate(params, {})
    assert ('tilted by' in caplog.text) == warns


def test_the_gravity_writes_the_rotation_it_gives():
    # With extras, a table of the radii of the rotation it gives a disk,
    # the circular velocity of each mass component and of all (signed),
    # the pressure support, and the rotation: sqrt(vcirc^2 + pressure)
    galaxy = disk_galaxy(
        'smdisk', dict(type='tan_mass', pressure_support='jeans'), MASS,
        DISTANCE)
    params = DISK_PROPERTIES | MASS_PROPERTIES
    extra = {}
    given = galaxy.plan_gravity().evaluate(params, extra)
    table = extra['component0_vpt_vt_rotation']
    assert table.colnames == [
        'radius', 'radius_kpc', 'vcirc_stars', 'vcirc_gas', 'vcirc_halo',
        'vcirc', 'pressure', 'vrot']
    np.testing.assert_allclose(
        table['radius'], galaxy.components()[0].density_disk().subrnodes())
    np.testing.assert_allclose(
        table['radius_kpc'], table['radius'] * KPC_PER_ARCSEC)
    vcirc2 = sum(np.sign(table[f'vcirc_{name}']) * table[f'vcirc_{name}'] ** 2
                 for name in ('stars', 'gas', 'halo'))
    np.testing.assert_allclose(
        np.sign(table['vcirc']) * table['vcirc'] ** 2, vcirc2, rtol=1e-12)
    np.testing.assert_allclose(table['vrot'], given[0]['vpt_vt'])
    np.testing.assert_allclose(
        table['vrot'] ** 2, vcirc2 + table['pressure'], rtol=1e-9)
