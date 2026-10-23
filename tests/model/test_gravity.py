"""
Tests of the gravity of galaxies (see Gravity): how their mass pulls, in
Newtonian gravity, in MOND or in f(R) gravity.
"""

import copy

import numpy as np
import pytest

from gbkfit.model import Extras, GravityHuSawicki, GravityNewton, galaxy_parser
from gbkfit.model.mass import MassContext, MassModel
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
    extra = Extras()
    given = mond.plan_gravity().evaluate(PARAMS, extra)
    table = extra.get('vpt_vt_rotation', component='gas')
    np.testing.assert_allclose(
        table['vcirc'].value, bound['vcirc'](table['radius_kpc'].value),
        rtol=1e-12)
    np.testing.assert_allclose(given[0]['vpt_vt'], table['vcirc'].value)
    equilibrium = extra.get('equilibrium', component='gas')
    kpc_per_arcsec = mond.distance().kpc_per_arcsec(PARAMS, 0)
    np.testing.assert_allclose(
        equilibrium['vcirc'].value[1:],
        bound['vcirc'](equilibrium['radius'].value[1:] * kpc_per_arcsec),
        rtol=1e-12)
    assert galaxy_parser.dump(mond)['gravity'] == dict(
        type='mond', interpolation=interpolation)


def test_mond_forces_anywhere():
    # Of a point mass, nu(y) g_N, y = |g_N| / a0, g_N = -G M (R, z) / r^3;
    # in the plane, those of its circular velocity
    mond = galaxy(
        mass=POINT, distance=DISTANCE,
        gravity=dict(type='mond', interpolation='simple'))
    field = mond.plan_gravity().mass(PARAMS).field
    R, z = np.array([0.1, 2.0, 10.0, 5.0]), np.array([0.5, -1.0, 20.0, 0])
    a0 = 1.2e-10 * KPC / 1e6
    r = np.hypot(R, z)
    nu = NU['simple'](G * 1e10 / (r ** 2 * a0))
    force_R, force_z = field.forces(R, z)
    np.testing.assert_allclose(force_R, -nu * G * 1e10 * R / r ** 3, rtol=1e-6)
    np.testing.assert_allclose(force_z, -nu * G * 1e10 * z / r ** 3, rtol=1e-6)
    np.testing.assert_array_equal(field.vertical_force(R, z), force_z)
    np.testing.assert_allclose(
        -R[3] * force_R[3], field.vcirc2(R[3:]), rtol=1e-12)


def test_mond_without_a0_is_newtonian_gravity():
    mond = galaxy(
        mass=POINT, distance=DISTANCE,
        gravity=dict(type='mond', interpolation='simple'))
    bound = mond.plan_functions().bind(PARAMS | dict(gravity_a0=0))
    R = np.array([0.5, 5.0, 50.0])
    np.testing.assert_array_equal(bound['vcirc'](R), bound['vnewton'](R))


@pytest.mark.parametrize('disk, mass, message', [
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


# A halo and a thick disk of stars, for f(R) gravity
HALO = dict(
    type='spheroid', name='halo',
    profile=dict(type='nfw', parameters='density_radius'))
THICK = [dict(
    type='disk', name='stars', surface=dict(type='exponential'),
    vertical=dict(type='exponential')), HALO]
THICK_PARAMS = PARAMS | dict(
    mass_stars_log_m=10.7, mass_stars_rd=3, mass_stars_hz=0.3,
    mass_halo_log_rho=7, mass_halo_rs=10)

# The squared velocity ((km/s)^2) of the fifth force of the halo (rho_s =
# 1e7 Msun/kpc^3, r_s = 10 kpc) at 5 to 200 kpc, by the solver of SPAM
# (Naik et al. 2019: its Grid1D), in its background (h = 0.7, Omega_m =
# 0.308, at z = 0), of log10 |f_R0| and of the environment (log10 of
# -Phi_ext / c^2; its sphere of radius lambda_C = 3.45 Mpc): partly
# screened, of its 175 cells to 5 Mpc; and screened by the environment,
# of 1400 cells to 10 lambda_C (converged: its 5 Mpc are too near)
SPAM_RADII = np.array([5.0, 20, 50, 100, 200])
SPAM_V5 = {
    (-6.5, None): [1876.37, 3709.77, 3378.31, 2644.74, 1854.35],
    (-7.0, None): [0.0, 0.01, 1500.78, 1689.12, 1357.58],
    (-6.0, -5.5): [0.0, 1970.5, 2676.4, 2289.4, 1713.5]}


@pytest.mark.parametrize('log_fr0, log_phi_ext', SPAM_V5)
def test_hu_sawicki_as_spam(log_fr0, log_phi_ext):
    # (to the precision of its solver: some 1% of v5^2)
    model = MassModel.load(dict(components=[copy.deepcopy(HALO)]))
    params = dict(mass_halo_log_rho=7, mass_halo_rs=10,
                  gravity_log_fr0=log_fr0, gravity_log_phi_ext=log_phi_ext)
    ricci = 3 * 0.308 * (0.07 / 299792.458) ** 2 * (1 + 4 * 0.692 / 0.308)
    context = MassContext(1.0, 0.308, ricci_scalar=ricci,
                          ricci_scalar_today=ricci)
    gravity = GravityHuSawicki(environment=log_phi_ext is not None)
    field = gravity.field(model, params, context)
    v5 = (field.vcirc2(SPAM_RADII)
          - model.vcirc2(SPAM_RADII, params, context))
    np.testing.assert_allclose(
        v5, SPAM_V5[log_fr0, log_phi_ext], rtol=0.015, atol=1)


def test_hu_sawicki_unscreened_and_screened():
    # Unscreened (|f_R0| = 1e-5, Compton wavelength 11 Mpc), the forces
    # are 4/3 of the Newtonian ones, anywhere; screened (1e-9), or by an
    # environment of a deep potential (-Phi_ext / c^2 = 1e-5, of 1e-6),
    # Newtonian; with that of a shallow one (1e-9), as without
    galaxy_ = galaxy(mass=THICK, distance=DISTANCE,
                     gravity=dict(type='hu_sawicki', environment=True))
    assert galaxy_parser.dump(galaxy_)['gravity'] == dict(
        type='hu_sawicki', n=1.0, screened=[], environment=True)
    R, z = np.array([1.0, 5, 20]), np.array([0.5, 0, 5])
    params = THICK_PARAMS | dict(gravity_log_fr0=-5, gravity_log_phi_ext=-9)
    mass = galaxy_.plan_gravity().mass(params)
    newton = mass.model.forces(R, z, mass.params, mass.context)
    np.testing.assert_allclose(
        mass.field.forces(R, z), 4 / 3 * np.array(newton), rtol=2e-3)
    for log_fr0, log_phi_ext in ((-9, -9), (-6, -5)):
        bound = galaxy_.plan_functions().bind(params | dict(
            gravity_log_fr0=log_fr0, gravity_log_phi_ext=log_phi_ext))
        np.testing.assert_allclose(
            bound['vcirc'](R), bound['vnewton'](R), rtol=1e-6)


def test_hu_sawicki_thin_disks():
    # A thin disk sources the field as the limit of thick ones: its fifth
    # force adds to the circular velocity as that of one of a scale height
    # of 30 pc, to 0.1 km/s, unscreened or partly screened
    thin = [MASS[0], HALO]
    galaxies = [
        galaxy(mass=mass, distance=DISTANCE, gravity=dict(type='hu_sawicki'))
        for mass in (thin, THICK)]
    R = np.array([5.0, 10, 20, 50])
    for log_fr0 in (-5, -6.3):
        params = THICK_PARAMS | dict(
            gravity_log_fr0=log_fr0, mass_stars_hz=0.03)
        added = []
        for galaxy_ in galaxies:
            bound = galaxy_.plan_functions().bind(params)
            added.append(bound['vcirc'](R) - bound['vnewton'](R))
        np.testing.assert_allclose(added[0], added[1], atol=0.1)


def test_hu_sawicki_options():
    # The screened components are of the mass model; n is positive; the
    # depth of the potential of the environment is a parameter with it
    with pytest.raises(ConfigError, match="screened mass components"):
        galaxy(mass=THICK, distance=DISTANCE,
               gravity=dict(type='hu_sawicki', screened=['gas']))
    with pytest.raises(ConfigError, match="n must be positive"):
        galaxy(mass=THICK, distance=DISTANCE,
               gravity=dict(type='hu_sawicki', n=0))
    screened = galaxy(mass=THICK, distance=DISTANCE,
                      gravity=dict(type='hu_sawicki', screened=['stars']))
    assert screened.gravity().screened() == ('stars',)
    assert 'gravity_log_phi_ext' not in screened.gravity().pdescs()
