"""
Tests of the gravity of galaxies (see Gravity): how their mass pulls, in
Newtonian gravity or in MOND.
"""

import copy

import numpy as np
import pytest

from gbkfit.model import GravityNewton, galaxy_parser
from gbkfit.utils.parseutils import ConfigError


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The length of a kpc (m)
KPC = 3.0856775814913673e19

# A thick disk that rotates in the mass of its galaxy, held up by the
# central equilibrium
DISK = dict(
    type='smdisk', name='gas', loose=False, tilted=False,
    rnodes=list(range(0, 21, 2)), rstep=0.5,
    density=dict(
        ptraits=dict(type='exponential'),
        htraits=dict(type='gauss', zmax=10)),
    kinematics=dict(
        vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
        equilibrium='central'),
    emission=dict(type='lines'))
MASS = [dict(
    type='disk', name='stars', surface=dict(type='exponential'),
    vertical=dict(type='thin'))]
POINT = [dict(type='point', name='bh')]
DISTANCE = dict(type='direct', cosmology='Planck18')
PARAMS = dict(
    gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60, gas_vsys=0,
    gas_pt_a=1, gas_pt_s=4, gas_ht_s=0.5, gas_dpt_a=10, distance=10,
    mass_bh_log_m=10, gravity_a0=1.2)

# The interpolation functions of MOND, of y = g_N / a0
NU = dict(
    simple=lambda y: 0.5 + np.sqrt(0.25 + 1 / y),
    standard=lambda y: np.sqrt(0.5 + np.sqrt(0.25 + 1 / y ** 2)),
    rar=lambda y: 1 / (1 - np.exp(-np.sqrt(y))))


def galaxy(disk=DISK, **info):
    """A galaxy of the disk, and the given options."""
    return galaxy_parser.load(
        dict(components=[copy.deepcopy(disk)]) | copy.deepcopy(info))


def with_kinematics(**kinematics):
    """The disk with the given options of its kinematics instead."""
    return DISK | dict(kinematics=DISK['kinematics'] | kinematics)


def test_newtonian_gravity_by_default():
    assert isinstance(galaxy(mass=MASS, distance=DISTANCE).gravity(),
                      GravityNewton)
    given = galaxy(mass=MASS, distance=DISTANCE, gravity=dict(type='newton'))
    assert galaxy_parser.dump(given)['gravity'] == dict(type='newton')
    assert 'gravity' not in galaxy_parser.dump(
        galaxy(mass=MASS, distance=DISTANCE))


def test_gravity_without_mass_refused():
    disk = DISK | dict(kinematics=DISK['kinematics'] | dict(
        vptraits=dict(type='arctan', direction='tan')))
    with pytest.raises(ConfigError, match='give the galaxy mass, or no grav'):
        galaxy_parser.load(dict(
            components=[copy.deepcopy(disk)], gravity=dict(type='newton')))


@pytest.mark.parametrize('interpolation', ['simple', 'standard', 'rar'])
def test_mond_circular_velocity(interpolation):
    # Of a point mass, v^2 = nu(y) G M / R, y = G M / (R^2 a0): in the
    # functions of the galaxy (vnewton: Newtonian), the rotation it gives
    # its disk, and the central equilibrium, h v / R
    mond = galaxy(
        mass=POINT, distance=DISTANCE,
        gravity=dict(type='mond', interpolation=interpolation))
    R = np.array([0.1, 2.0, 10.0, 50.0])
    a0 = 1.2e-10 * KPC / 1e6
    vnewton2 = G * 1e10 / R
    vcirc = np.sqrt(NU[interpolation](vnewton2 / (R * a0)) * vnewton2)
    bound = mond.plan_functions().bind(PARAMS)
    np.testing.assert_allclose(bound['vcirc'](R), vcirc, rtol=1e-6)
    for name in ('vnewton', 'vcirc_bh'):
        np.testing.assert_allclose(
            bound[name](R), np.sqrt(vnewton2), rtol=1e-6)
    extra = {}
    given = mond.plan_gravity().evaluate(PARAMS, extra)
    table = extra['gas_vpt_vt_rotation']
    np.testing.assert_allclose(
        table['vcirc'].value, bound['vcirc'](table['radius_kpc'].value),
        rtol=1e-12)
    np.testing.assert_allclose(given[0]['vpt_vt'], table['vcirc'].value)
    equilibrium = extra['gas_equilibrium']
    kpc_per_arcsec = mond.distance().kpc_per_arcsec(PARAMS, 0)
    np.testing.assert_allclose(
        equilibrium['vcirc'].value[1:],
        bound['vcirc'](equilibrium['radius'].value[1:] * kpc_per_arcsec),
        rtol=1e-12)
    assert galaxy_parser.dump(mond)['gravity'] == dict(
        type='mond', interpolation=interpolation)


def test_mond_without_a0_is_newtonian_gravity():
    mond = galaxy(
        mass=POINT, distance=DISTANCE,
        gravity=dict(type='mond', interpolation='simple'))
    bound = mond.plan_functions().bind(PARAMS | dict(gravity_a0=0))
    R = np.array([0.5, 5.0, 50.0])
    np.testing.assert_array_equal(bound['vcirc'](R), bound['vnewton'](R))


@pytest.mark.parametrize('disk, mass, message', [
    (with_kinematics(equilibrium='jeans'), POINT,
     "jeans equilibrium of the component 'gas' takes the vertical force"),
    (with_kinematics(
        equilibrium='self_gravitating', equilibrium_density='stars'), MASS,
     "self_gravitating equilibrium of the component 'gas'"),
    (with_kinematics(
        vptraits=dict(type='mass', pressure_support='self_gravitating')),
     POINT, "self_gravitating pressure support of the component 'gas'"),
    (DISK, MASS + [dict(
        type='spheroid', name='halo', profile=dict(type='nfw'),
        baryons=['stars'],
        dark_fraction=dict(solve='log_m', bounds=[9.0, 15.0]))],
     "the mass component 'halo' takes the shares"),
])
def test_what_mond_does_not_give_refused(disk, mass, message):
    with pytest.raises(ConfigError, match=message):
        galaxy(disk, mass=mass, distance=DISTANCE,
               gravity=dict(type='mond', interpolation='simple'))
    # (Newtonian gravity gives them)
    galaxy(disk, mass=mass, distance=DISTANCE)


def test_mond_needs_an_interpolation_function():
    with pytest.raises(ConfigError, match='interpolation'):
        galaxy(mass=POINT, distance=DISTANCE, gravity=dict(type='mond'))
    with pytest.raises(ConfigError, match="must be one of"):
        galaxy(mass=POINT, distance=DISTANCE,
               gravity=dict(type='mond', interpolation='best'))
