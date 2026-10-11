"""
Tests for mass models: their components against independent formulas
and integrals, their parameterisations, the distance of their model, and
the velocity traits 'mass', which take their circular velocity.
"""

import copy

import astropy.cosmology
import numpy as np
import pytest
import scipy.integrate
import scipy.special

from gbkfit.model import model_parser
from gbkfit.model.distance import DistanceDirect, DistanceRedshift
from gbkfit.model.mass import (
    MassBurkert, MassContext, MassEinasto, MassExponentialDisk, MassModel,
    MassNFW, MassPointMass, MassPseudoIsothermal, MassSersic, MassUser,
    MassZhao, mass_model_parser, vertical_density, vertical_transform)
from gbkfit.utils.parseutils import ConfigError


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The epoch of Planck 2018 at z = 0
CONTEXT = MassContext(
    critical_density=astropy.cosmology.Planck18.critical_density0.to_value(
        'solMass / kpc3'),
    omega_m=astropy.cosmology.Planck18.Om0)


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
    result = MassExponentialDisk('stars').vcirc2(
        radius, dict(log_m=log_m, rd=rd), CONTEXT)
    np.testing.assert_allclose(
        result, [vcirc2(r) for r in radius], rtol=1e-4)


@pytest.mark.parametrize('vertical', ['exponential', 'sech2', 'gauss'])
def test_thick_disks_of_vanishing_thickness_are_thin(vertical):
    # (the numerical integral of a thick disk against Freeman's closed
    # form; the transforms of the vertical laws against their densities)
    radius = np.concatenate(([0], np.geomspace(1e-3, 1e3, 61)))
    thin = MassExponentialDisk('d').vcirc2(
        radius, dict(log_m=10, rd=3), CONTEXT)
    thick = MassExponentialDisk('d', vertical).vcirc2(
        radius, dict(log_m=10, rd=3, hz=1e-9), CONTEXT)
    np.testing.assert_allclose(thick, thin, rtol=1e-6)
    for k in (0, 0.3, 5):
        expected, _ = scipy.integrate.quad(
            lambda z: 2 * vertical_density(vertical, z, 0.7) * np.exp(-k * z),
            0, np.inf)
        assert vertical_transform(vertical, np.array(k), 0.7) \
            == pytest.approx(expected, rel=1e-10)


SPHERES = [
    (MassNFW('h', 'density_radius'), dict(log_rho=7, rs=10)),
    (MassBurkert('h', 'density_radius'), dict(log_rho=7.5, rs=4)),
    (MassPseudoIsothermal('h', 'density_radius'), dict(log_rho=7.5, rs=2)),
    (MassEinasto('h', 'density_radius'), dict(log_rho=6.5, rs=15, alpha=0.17)),
    (MassZhao('h', 'density_radius'),
     dict(log_rho=6.5, rs=8, alpha=2, beta=4, gamma=0.5)),
    (MassSersic('b'), dict(log_m=10, re=1.5, n=0.6)),
    (MassSersic('b'), dict(log_m=10, re=1.5, n=4))]


@pytest.mark.parametrize('sphere, params', SPHERES)
def test_spheres_are_their_densities(sphere, params):
    # Their enclosed mass is the integral of their density (including
    # radii where the halos use their series), and their circular
    # velocity G M(<r) / r
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
    (MassPointMass('p'), dict(log_m=8))] + [
        case for case in SPHERES if case[0].type() != 'pseudo_isothermal'])
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
        MassPseudoIsothermal('h').potential(
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
    np.testing.assert_allclose(
        MassSersic('b').density_spherical(radius, params, CONTEXT),
        sigma0 / (np.pi * rd) * scipy.special.k0(radius / rd), rtol=1e-6)
    assert MassSersic('b').enclosed_mass(1e3, params, CONTEXT) \
        == pytest.approx(mass, rel=1e-8)


HALOS = [MassNFW, MassBurkert, MassPseudoIsothermal, MassEinasto, MassZhao]
SHAPES = dict(einasto=dict(alpha=0.17), zhao=dict(alpha=1, beta=3, gamma=1.3))


@pytest.mark.parametrize('halo_class', HALOS)
@pytest.mark.parametrize('overdensity', ['200c', '500c', '200m', 'vir'])
def test_halos_have_their_mass_and_concentration(halo_class, overdensity):
    # The mean density within r_D is D rho_ref, the mass within it M_D,
    # and its velocity V_D: given by (M_D, c) or (V_D, c), a halo is the
    # same
    halo = halo_class('h', 'mass_concentration', overdensity)
    shape = SHAPES.get(halo.type(), {})
    params = dict(log_m=12, c=9) | shape
    rho_s, r_s = halo.scales(params, CONTEXT)
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
    by_velocity = halo_class('h', 'velocity_concentration', overdensity)
    radius = np.array([0.5, 5, 50])
    np.testing.assert_allclose(
        by_velocity.vcirc2(
            radius, dict(v=np.sqrt(G * mass / r_d), c=9) | shape, CONTEXT),
        halo.vcirc2(radius, params, CONTEXT), rtol=1e-10)


def test_zhao_is_nfw_for_its_shape():
    radius = np.array([1e-3, 0.5, 5, 50, 500])
    np.testing.assert_allclose(
        MassZhao('h').vcirc2(
            radius, dict(log_m=12, c=9, alpha=1, beta=3, gamma=1), CONTEXT),
        MassNFW('h').vcirc2(radius, dict(log_m=12, c=9), CONTEXT),
        rtol=1e-10)


@pytest.mark.parametrize('options, message', [
    (dict(parameters='mass'), "parameters must be one of"),
    (dict(overdensity='200x'), "overdensity must be"),
    (dict(parameters='density_radius', overdensity='200c'),
     "take no overdensity")])
def test_halo_options_are_checked(options, message):
    with pytest.raises(ConfigError, match=message):
        MassNFW('h', **options)


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
    model = MassModel([MassExponentialDisk('stars'), MassNFW('halo')])
    assert list(model.pdescs()) == [
        'mass_stars_log_m', 'mass_stars_rd', 'mass_halo_log_m', 'mass_halo_c']
    params = dict(
        mass_stars_log_m=10.5, mass_stars_rd=3, mass_halo_log_m=12,
        mass_halo_c=8)
    radius = np.array([0, 1, 5, 20])
    expected = (
        MassExponentialDisk('s').vcirc2(
            radius, dict(log_m=10.5, rd=3), CONTEXT)
        + MassNFW('h').vcirc2(radius, dict(log_m=12, c=8), CONTEXT))
    np.testing.assert_allclose(
        model.vcirc2(radius, params, CONTEXT), expected, rtol=1e-12)


def test_distances_give_the_length_of_an_arcsec():
    # At 10 Mpc, an arcsec is 48.48 pc; by a redshift, the angular
    # diameter distance of the cosmology; a distance sets no redshift
    direct = DistanceDirect()
    assert list(direct.pdescs()) == ['distance']
    assert direct.kpc_per_arcsec(dict(distance=10)) \
        == pytest.approx(0.0484813681)
    assert direct.epoch(dict(distance=10)) == 0
    by_redshift = DistanceRedshift('WMAP9')
    expected = astropy.cosmology.WMAP9.angular_diameter_distance(0.5)
    assert by_redshift.kpc_per_arcsec(dict(redshift=0.5)) == pytest.approx(
        expected.to_value('kpc') * np.pi / (180 * 3600))
    assert by_redshift.epoch(dict(redshift=0.5)) == 0.5
    with pytest.raises(ConfigError, match="cosmology must be one of"):
        DistanceRedshift('Planck99')
    with pytest.raises(ConfigError, match="epoch cannot be negative"):
        DistanceDirect(epoch=-1)


MASS_MODEL = dict(components=[
    dict(type='exponential_disk', name='stars', vertical='thin'),
    dict(type='pseudo_isothermal', name='halo',
         parameters='density_radius')])

DISTANCE = dict(type='direct', cosmology='Planck18', epoch=0.0)

MASS_PROPERTIES = dict(
    distance=10, mass_stars_log_m=10.5, mass_stars_rd=0.5,
    mass_halo_log_rho=8, mass_halo_rs=1)

DISK = dict(
    loose=False, tilted=False, rnodes=list(range(0, 21, 2)), rstep=0.5,
    bptraits=dict(type='exponential'), dptraits=dict(type='uniform'))

DISK_PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, bpt_a=1, bpt_s=4, dpt_a=15)


def kinematics(
        model_type, disk_type, vptraits, mass_model=None, distance=None):
    disk = DISK | dict(type=disk_type, vptraits=vptraits)
    if model_type == 'kinematics_3d':
        disk |= dict(bhtraits=dict(type='sech2'))
    if disk_type == 'mcdisk':
        disk |= dict(cflux=1e-3)
    info = dict(type=model_type, components=[disk])
    if mass_model is not None:
        info |= dict(mass_model=copy.deepcopy(mass_model))
    if distance is not None:
        info |= dict(distance=copy.deepcopy(distance))
    return model_parser.load(info)


def evaluate(driver, model, properties):
    from gbkfit.observation import Observation, ObservationGroup, PixelSpectra
    from gbkfit.params import ParamSpace
    group = ObservationGroup([model], [Observation(
        PixelSpectra(size=(32, 32, 51), step=(1, 1, 10)), driver)])
    params = ParamSpace(group.pdescs(), properties)
    return group.model_h(params.evaluate())[0]['spectra']['d'].copy()


@pytest.mark.parametrize('model_type, disk_type', [
    ('kinematics_2d', 'smdisk'),
    ('kinematics_3d', 'smdisk'),
    ('kinematics_3d', 'mcdisk')])
def test_mass_trait_is_the_circular_velocity_at_the_subnodes(
        driver, model_type, disk_type):
    # A disk with the trait 'mass' is the disk whose node-wise tangential
    # velocity at its subnodes is the circular velocity of the mass model,
    # at the radii of the subnodes in kpc
    massive = kinematics(
        model_type, disk_type, dict(type='mass'), MASS_MODEL, DISTANCE)
    given = kinematics(
        model_type, disk_type,
        dict(type='nw_tan_uniform', sampling='subrings'))
    subrnodes = np.array(given.constants()['subrnodes'])
    vcirc = np.sqrt(mass_model_parser.load(copy.deepcopy(MASS_MODEL)).vcirc2(
        subrnodes * 10 * 0.0048481368, MASS_PROPERTIES, CONTEXT))
    assert vcirc.max() > 100
    properties = DISK_PROPERTIES
    if model_type == 'kinematics_3d':
        properties |= dict(bht_s=1)
    np.testing.assert_allclose(
        evaluate(driver, massive, properties | MASS_PROPERTIES),
        evaluate(driver, given, properties | dict(vpt_vt=vcirc)),
        rtol=1e-6, atol=1e-7)


def test_mass_trait_parameters_and_round_trip():
    model = kinematics(
        'kinematics_3d', 'smdisk', dict(type='mass'), MASS_MODEL, DISTANCE)
    # The trait has no parameters: those of the distance and the mass
    # model are the model's
    assert not any(name.startswith('vpt') for name in model.pdescs())
    assert set(MASS_PROPERTIES) <= set(model.pdescs())
    dumped = model_parser.dump(model)
    assert dumped['mass_model'] == MASS_MODEL
    assert dumped['distance'] == DISTANCE
    assert model_parser.dump(model_parser.load(copy.deepcopy(dumped))) \
        == dumped


def test_mass_trait_mass_model_and_distance_need_each_other():
    with pytest.raises(ConfigError, match="need the mass_model"):
        kinematics('kinematics_2d', 'smdisk', dict(type='mass'))
    with pytest.raises(ConfigError, match="no component uses"):
        kinematics(
            'kinematics_2d', 'smdisk', dict(type='tan_arctan'), MASS_MODEL,
            DISTANCE)
    with pytest.raises(ConfigError, match="needs its distance"):
        kinematics('kinematics_2d', 'smdisk', dict(type='mass'), MASS_MODEL)
    with pytest.raises(ConfigError, match="used by its mass_model only"):
        kinematics(
            'kinematics_2d', 'smdisk', dict(type='tan_arctan'),
            distance=DISTANCE)


def test_tables_of_thick_disks_are_their_integrals():
    # Inside the range of the table of a vertical law (hz / rd from 1e-3
    # to 1), against the integral computed directly (beyond it)
    from gbkfit.model.mass import disks
    rng = np.random.default_rng(1)
    x = np.exp(rng.uniform(np.log(1e-3), np.log(1e3), 50))
    for eta in (2e-3, 0.08, 0.9):
        np.testing.assert_allclose(
            disks._exponential_shape(x, 'sech2', eta),
            disks._exponential_shape_direct(x, 'sech2', eta), rtol=1e-6)
