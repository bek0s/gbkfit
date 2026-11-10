"""
Tests of the gravity of galaxies (see Gravity): how their mass pulls, in
Newtonian gravity, in MOND or in f(R) gravity (of Hu & Sawicki, and
Palatini).
"""

import copy

import numpy as np
import pytest

from gbkfit.config import ConfigError
from gbkfit.model import Extras, GravityHuSawicki, GravityNewton, galaxy_parser
from gbkfit.model.mass import MassContext, MassModel


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The length of a kpc (m)
KPC = 3.0856775814913673e19

# A thick disk that rotates in the mass of its galaxy
DISK = dict(
    type='thick_disk', name='gas', loose=False, tilted=False,
    rnodes=list(range(0, 21, 2)), rstep=0.5,
    rptraits=dict(type='exponential'),
    rhtraits=dict(type='gauss', zmax=10),
    vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
    emission=dict(type='lines'))
MASS = [dict(
    type='disk', name='stars', surface=dict(type='exponential'),
    vertical=dict(type='thin'))]
POINT = [dict(type='point', name='bh')]
DISTANCE = dict(type='parameter')
PARAMS = dict(
    gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60, gas_vsys=0,
    gas_rpt_a=1, gas_rpt_s=4, gas_rht_s=0.5, gas_dpt_a=10, distance=10,
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
    return DISK | dict(kinematics)


def balanced_by(gravity, **options):
    """
    The disk in the vertical equilibrium of the given gravity, with the
    given options of its kinematics.
    """
    return with_kinematics(dptraits=[
        dict(type='uniform'), dict(type='equilibrium', gravity=gravity)],
        **options)


def test_newtonian_gravity_by_default():
    assert isinstance(galaxy(mass=MASS, distance=DISTANCE).gravity(),
                      GravityNewton)
    given = galaxy(mass=MASS, distance=DISTANCE, gravity=dict(type='newton'))
    assert galaxy_parser.dump(given)['gravity'] == dict(type='newton')
    assert 'gravity' not in galaxy_parser.dump(
        galaxy(mass=MASS, distance=DISTANCE))


def test_gravity_without_mass_refused():
    disk = DISK | dict(dict(
        vptraits=dict(type='arctan', direction='tan')))
    with pytest.raises(ValueError, match='give the galaxy mass, or no grav'):
        galaxy_parser.load(dict(
            components=[copy.deepcopy(disk)], gravity=dict(type='newton')))


@pytest.mark.parametrize('interpolation', ['simple', 'standard', 'rar'])
@pytest.mark.parametrize('form, rtol', [('qumond', 1e-3), ('algebraic', 1e-6)])
def test_mond_circular_velocity(interpolation, form, rtol):
    # Of a point mass, v^2 = nu(y) G M / R, y = G M / (R^2 a0), in either
    # form (QUMOND to the precision of its grid): in the functions of the
    # galaxy (vnewton: Newtonian) and the rotation it gives its disk
    mond = galaxy(
        mass=POINT, distance=DISTANCE,
        gravity=dict(type='mond', interpolation=interpolation, form=form))
    R = np.array([0.1, 2.0, 10.0, 50.0])
    a0 = 1.2e-10 * KPC / 1e6
    vnewton2 = G * 1e10 / R
    vcirc = np.sqrt(NU[interpolation](vnewton2 / (R * a0)) * vnewton2)
    bound = mond.plan_functions().bind(PARAMS)
    np.testing.assert_allclose(bound['vcirc'](R), vcirc, rtol=rtol)
    for name in ('vnewton', 'vcirc_bh'):
        np.testing.assert_allclose(
            bound[name](R), np.sqrt(vnewton2), rtol=1e-6)
    extra = Extras()
    states = mond.plan_stage().evaluate(PARAMS, extra).states
    table = extra.get('vpt_vt_rotation', component='gas')
    np.testing.assert_allclose(
        table['vcirc'].value, bound['vcirc'](table['radius_kpc'].value),
        rtol=1e-12)
    np.testing.assert_allclose(
        states[0].rotation['vpt_vt'], table['vcirc'].value)
    assert galaxy_parser.dump(mond)['gravity'] == dict(
        type='mond', interpolation=interpolation, form=form)


def test_mond_forces_anywhere():
    # Of a point mass in QUMOND, nu(y) g_N, y = |g_N| / a0, g_N = -G M (R,
    # z) / r^3 (to the precision of its grid, near the pole some 1e-3;
    # beyond it, at 1000 kpc, exactly); in the plane,
    # those of its circular velocity. The algebraic form gives them in
    # the plane only
    mond = galaxy(
        mass=POINT, distance=DISTANCE,
        gravity=dict(type='mond', interpolation='simple'))
    field = mond.plan_stage().mass(PARAMS).field
    R = np.array([0.1, 2.0, 10.0, 5.0, 2000.0])
    z = np.array([0.5, -1.0, 20.0, 0, 500.0])
    a0 = 1.2e-10 * KPC / 1e6
    r = np.hypot(R, z)
    nu = NU['simple'](G * 1e10 / (r ** 2 * a0))
    force_R, force_z = field.forces(R, z)
    np.testing.assert_allclose(force_R, -nu * G * 1e10 * R / r ** 3, rtol=2e-3)
    np.testing.assert_allclose(force_z, -nu * G * 1e10 * z / r ** 3, rtol=2e-3)
    np.testing.assert_array_equal(field.vertical_force(R, z), force_z)
    np.testing.assert_allclose(
        -R[3] * force_R[3], field.vcirc2(R[3:4]), rtol=1e-12)
    algebraic = galaxy(
        mass=POINT, distance=DISTANCE,
        gravity=dict(type='mond', interpolation='simple', form='algebraic'))
    field = algebraic.plan_stage().mass(PARAMS).field
    with pytest.raises(NotImplementedError, match='plane'):
        field.forces(R, z)


class Kuzmin:
    """
    A Kuzmin disk of mass M and scale a (kpc): above its plane, the
    Newtonian field of a point of its mass at a below it.
    """

    def __init__(self, M, a):
        self.M, self.a = M, a

    def forces(self, R, z):
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        d3 = (R ** 2 + (self.a + np.abs(z)) ** 2) ** 1.5
        return (-G * self.M * R / d3,
                -G * self.M * (self.a + np.abs(z)) * np.sign(z) / d3)

    def cell_masses(self, r_edges, b_edges):
        masses = np.zeros((len(r_edges) - 1, len(b_edges) - 1))
        masses[:, 0] = -np.diff(self.M * self.a / np.hypot(r_edges, self.a))
        return masses


@pytest.mark.parametrize('M, a', [(1e9, 2.0), (1e11, 3.0)])
def test_qumond_of_a_kuzmin_disk(M, a):
    # Exactly, nu(|g_N| / a0) g_N above its plane (Brada & Milgrom 1995),
    # of the field of a point; in its plane, of |g_N| just above it, of its
    # vertical pull 2 pi G Sigma too: some 60% less than nu(|F_R| / a0) F_R
    # at 0.25 kpc of the light one
    from gbkfit.model.gravity._qumond import phantom_gradient
    disk = Kuzmin(M, a)
    a0 = 1.2e-10 * KPC / 1e6

    def boost(size):
        return NU['simple'](size / a0) - 1

    gradient = phantom_gradient(disk.cell_masses, disk.forces, boost)
    R = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32])
    z = 0.5 * R
    force_R, force_z = disk.forces(R, z)
    grad_R, grad_z = gradient(R, z)
    nu = NU['simple'](np.hypot(force_R, force_z) / a0)
    np.testing.assert_allclose(force_R - grad_R, nu * force_R, rtol=6e-3)
    np.testing.assert_allclose(force_z - grad_z, nu * force_z, rtol=1e-3)
    force_R, _ = disk.forces(R, 0)
    grad_R, _ = gradient(R, 0)
    nu = NU['simple'](G * M / (R ** 2 + a ** 2) / a0)
    np.testing.assert_allclose(force_R - grad_R, nu * force_R, rtol=4e-3)


@pytest.mark.parametrize('vertical', [
    dict(type='thin'), dict(type='exponential')])
def test_qumond_solves_newtonian_gravity(vertical):
    # The Newtonian gravity on its grid, of the masses of its cells, is
    # that of the mass: of a boost of 1, the phantom is the mass
    from gbkfit.model.gravity._qumond import phantom_gradient
    model = MassModel.load(dict(components=[dict(
        type='disk', name='stars', surface=dict(type='exponential'),
        vertical=vertical)]))
    params = dict(mass_stars_log_m=10, mass_stars_rd=3, mass_stars_hz=0.3)
    context = MassContext(1.0, 0.308, 0.0, 0.0)

    def masses(r_edges, b_edges):
        return model.cell_masses(r_edges, b_edges, params, context)

    def forces(R, z):
        return model.forces(R, z, params, context)

    gradient = phantom_gradient(masses, forces, np.ones_like)
    R = np.array([0.5, 1, 2, 4, 8, 16, 32])
    for z in (0 * R, 0.3 * R):
        np.testing.assert_allclose(
            np.negative(gradient(R, z)), forces(R, z), rtol=5e-3, atol=1e-9)


@pytest.mark.parametrize('form', ['qumond', 'algebraic'])
def test_mond_without_a0_is_newtonian_gravity(form):
    mond = galaxy(
        mass=POINT, distance=DISTANCE,
        gravity=dict(type='mond', interpolation='simple', form=form))
    bound = mond.plan_functions().bind(PARAMS | dict(gravity_a0=0))
    R = np.array([0.5, 5.0, 50.0])
    np.testing.assert_array_equal(bound['vcirc'](R), bound['vnewton'](R))


@pytest.mark.parametrize('disk, mass, message', [
    (balanced_by('sheet', fluid='stars'), MASS,
     "the vertical equilibrium 'sheet' of the component 'gas'"),
    (with_kinematics(pressure_support=dict(type='self_gravitating')),
     POINT, "self_gravitating pressure support of the component 'gas'"),
    (DISK, MASS + [dict(
        type='halo', name='halo', profile=dict(
            type='nfw', parameterisation='dark_fraction_concentration'),
        baryons=['stars'])],
     "the mass component 'halo' takes the shares"),
    (DISK, MASS + [dict(
        type='table', name='gas', radii=dict(value=[1, 2, 4], unit='kpc'),
        velocities=[20, 30, 40])],
     r"the mass components \['gas'\] do not give it"),
])
def test_what_mond_does_not_give_refused(disk, mass, message):
    with pytest.raises(ValueError, match=message):
        galaxy(disk, mass=mass, distance=DISTANCE,
               gravity=dict(type='mond', interpolation='simple'))
    # (Newtonian gravity gives them)
    galaxy(disk, mass=mass, distance=DISTANCE)


def test_algebraic_mond_gives_the_plane_only():
    # It takes tables of the circular velocity, but not the forces off the
    # plane that the equilibrium 'galaxy' takes
    table = dict(type='table', name='gas',
                 radii=dict(value=[1, 2, 4], unit='kpc'),
                 velocities=[20, 30, 40])
    gravity = dict(type='mond', interpolation='simple', form='algebraic')
    galaxy(mass=MASS + [table], distance=DISTANCE, gravity=gravity)
    with pytest.raises(ValueError, match="in the plane only"):
        galaxy(balanced_by('galaxy'), mass=MASS, distance=DISTANCE,
               gravity=gravity)


def test_mond_needs_an_interpolation_function():
    with pytest.raises(ConfigError, match='interpolation'):
        galaxy(mass=POINT, distance=DISTANCE, gravity=dict(type='mond'))
    with pytest.raises(ConfigError, match="must be one of"):
        galaxy(mass=POINT, distance=DISTANCE,
               gravity=dict(type='mond', interpolation='best'))
    with pytest.raises(ConfigError, match="must be one of"):
        galaxy(mass=POINT, distance=DISTANCE,
               gravity=dict(type='mond', interpolation='simple', form='aqual'))


# A halo and a thick disk of stars, for f(R) gravity
HALO = dict(
    type='halo', name='halo',
    profile=dict(type='nfw', parameterisation='density_radius'))
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
    mass = galaxy_.plan_stage().mass(params)
    newton = mass.model.forces(R, z, mass.params, mass.context)
    np.testing.assert_allclose(
        mass.field.forces(R, z), 4 / 3 * np.array(newton), rtol=2e-3)
    for log_fr0, log_phi_ext in ((-9, -9), (-6, -5)):
        bound = galaxy_.plan_functions().bind(params | dict(
            gravity_log_fr0=log_fr0, gravity_log_phi_ext=log_phi_ext))
        np.testing.assert_allclose(
            bound['vcirc'](R), bound['vnewton'](R), rtol=1e-6)


def test_hu_sawicki_sourced_by_a_contracted_halo():
    # A halo that responds to its baryons sources the field by the mass of
    # its spherical shells: unscreened, the forces are 4/3 of the
    # Newtonian ones, as of a halo that does not
    contracted = HALO | dict(
        baryons=['stars'], contraction=dict(type='blumenthal86'))
    galaxy_ = galaxy(mass=[THICK[0], contracted], distance=DISTANCE,
                     gravity=dict(type='hu_sawicki'))
    R, z = np.array([1.0, 5, 20]), np.array([0.5, 0, 5])
    mass = galaxy_.plan_stage().mass(
        THICK_PARAMS | dict(gravity_log_fr0=-5))
    newton = mass.model.forces(R, z, mass.params, mass.context)
    np.testing.assert_allclose(
        mass.field.forces(R, z), 4 / 3 * np.array(newton), rtol=2e-3)


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


def test_hu_sawicki_options(tmp_path):
    # The screened components are of the mass model; those not screened
    # give where their mass is (not user components); n is positive; the
    # depth of the potential of the environment is a parameter with it
    with pytest.raises(ValueError, match="screened mass components"):
        galaxy(mass=THICK, distance=DISTANCE,
               gravity=dict(type='hu_sawicki', screened=['gas']))
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n    return 100 * R\n")
    user = THICK + [dict(type='user', name='mine',
                         file=str(tmp_path / 'curve.py'), func='vcirc2')]
    with pytest.raises(ValueError, match=r"\['mine'\] do not give where"):
        galaxy(mass=user, distance=DISTANCE, gravity=dict(type='hu_sawicki'))
    galaxy(mass=user, distance=DISTANCE,
           gravity=dict(type='hu_sawicki', screened=['mine']))
    with pytest.raises(ConfigError, match="'n' must be greater than 0"):
        galaxy(mass=THICK, distance=DISTANCE,
               gravity=dict(type='hu_sawicki', n=0))
    screened = galaxy(mass=THICK, distance=DISTANCE,
                      gravity=dict(type='hu_sawicki', screened=['stars']))
    assert screened.gravity().screened() == ('stars',)
    assert 'gravity_log_phi_ext' not in screened.gravity().pdescs()


def test_palatini_corrections_of_the_density_gradient():
    # Of a thick exponential disk, rho = Sigma(R) exp(-|z| / hz) / (2 hz):
    # v^2 = v_N^2 + 8 pi G alpha R drho/dR, and the forces the Newtonian
    # ones minus 8 pi G alpha grad(rho), with drho/dR = -rho / rd and
    # drho/dz = -sign(z) rho / hz; none with alpha = 0
    disk = [dict(type='disk', name='stars', surface=dict(type='exponential'),
                 vertical=dict(type='exponential'))]
    galaxy_ = galaxy(mass=disk, distance=DISTANCE,
                     gravity=dict(type='palatini'))
    assert galaxy_parser.dump(galaxy_)['gravity'] == dict(type='palatini')
    params = PARAMS | dict(mass_stars_log_m=10, mass_stars_rd=3,
                           mass_stars_hz=0.3, gravity_alpha=-0.5)
    mass = galaxy_.plan_stage().mass(params)
    R, z = np.array([1.0, 5, 10]), np.array([0.2, -0.5, 1])
    scale = 8 * np.pi * G * -0.5

    def rho(z):
        sigma = 1e10 / (2 * np.pi * 9) * np.exp(-R / 3)
        return sigma * np.exp(-np.abs(z) / 0.3) / 0.6

    newton = mass.model.vcirc2(R, mass.params, mass.context)
    np.testing.assert_allclose(
        mass.field.vcirc2(R), newton - scale * R * rho(0) / 3, rtol=1e-7)
    force_R, force_z = mass.model.forces(R, z, mass.params, mass.context)
    expected = (force_R + scale * rho(z) / 3,
                force_z + scale * np.sign(z) * rho(z) / 0.3)
    np.testing.assert_allclose(mass.field.forces(R, z), expected, rtol=1e-7)
    bound = galaxy_.plan_functions().bind(params | dict(gravity_alpha=0))
    np.testing.assert_array_equal(bound['vcirc'](R), bound['vnewton'](R))


# A thin disk that rotates in the mass of its galaxy, and a mass disk
# that follows the gas
THIN_DISK = {
    key: value for key, value in DISK.items()
    if key != 'rhtraits'} \
    | dict(type='thin_disk', rptraits=dict(type='exponential'),
           vptraits=dict(type='mass'), dptraits=dict(type='uniform'))
GAS_MASS = dict(type='traced_disk', name='gas', component='gas')


@pytest.mark.parametrize('disk, mass, message', [
    (DISK, MASS, r"\['stars'\] do not give it"),
    (DISK, 'user', r"\['mine'\] do not give it"),
    (THIN_DISK, [GAS_MASS], r"\['gas'\] do not give it")])
def test_palatini_needs_the_density(tmp_path, disk, mass, message):
    # Of every mass component, when the galaxy is made: thin disks have
    # none in their plane (a disk that follows a thin component is thin);
    # user components do not give it; and a disk that follows a thick
    # component does, as do halos that respond to their baryons
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n    return 100 * R\n")
    if mass == 'user':
        mass = [dict(type='user', name='mine',
                     file=str(tmp_path / 'curve.py'), func='vcirc2')]
    with pytest.raises(ValueError, match=message):
        galaxy(disk, mass=mass, distance=DISTANCE,
               gravity=dict(type='palatini'))
    galaxy(mass=[GAS_MASS], distance=DISTANCE, gravity=dict(type='palatini'))
    galaxy(mass=[THICK[0], HALO | dict(
        baryons=['stars'], contraction=dict(type='blumenthal86'))],
        distance=DISTANCE, gravity=dict(type='palatini'))


def test_palatini_takes_newtonian_shares():
    # (those of the dark fraction)
    with pytest.raises(ValueError, match="dark fraction"):
        galaxy(mass=[THICK[0], dict(
            type='halo', name='halo', profile=dict(
                type='nfw', parameterisation='dark_fraction_concentration'),
            baryons=['stars'])],
            distance=DISTANCE, gravity=dict(type='palatini'))
