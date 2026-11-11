"""
Tests of the vertical equilibrium of disks (see
traits.DPTraitEquilibrium): the dispersion that holds a thick disk up in
a gravity, in quadrature with that of its other dispersion traits.
"""

import copy

import numpy as np
import pytest
import scipy.integrate

from gbkfit.config import ConfigError
from gbkfit.grid import SpectralAxis
from gbkfit.model import Extras, galaxy_parser


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The length of an arcsec (kpc) at 10 Mpc
KPC_PER_ARCSEC = 1e4 * np.pi / 648000

# A thick disk of a Gaussian height profile (its rms height is its s, 1
# arcsec: its cut at zmax is 1e-22 of it), its rotation of arctan traits,
# in the equilibrium of its rotation
DISK = dict(
    type='thick_disk', loose=False, tilted=False,
    rnodes=list(range(0, 21, 2)), rstep=0.5,
    rptraits=dict(type='exponential'),
    rhtraits=dict(type='gauss', zmax=10),
    vptraits=dict(type='arctan', direction='tan'),
    dptraits=[
        dict(type='uniform'), dict(type='equilibrium', gravity='rotation')],
    emission=dict(type='lines'))

PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, rpt_a=1, rpt_s=4, rht_s=1,
    vpt_rt=3, vpt_vt=200, dpt_a=0)


def galaxy(disk, **info):
    """A galaxy of the given disk, and the given options (e.g. mass)."""
    return galaxy_parser.load(
        dict(components=[copy.deepcopy(disk)]) | copy.deepcopy(info))


def with_kinematics(disk, **kinematics):
    """The disk with the given options of its kinematics instead."""
    return disk | dict(kinematics)


def balanced_by(disk, gravity, **options):
    """
    The disk in the equilibrium of the given gravity (and options): the
    last of its dispersion traits.
    """
    fitted = [trait for trait in disk['dptraits']
              if trait['type'] != 'equilibrium']
    return disk | dict(dptraits=fitted + [
        dict(type='equilibrium', gravity=gravity) | options])


def unbalanced(disk, dptraits):
    """The disk of the given dispersion traits instead, in no equilibrium."""
    return disk | dict(dptraits=dptraits)


def evaluate(driver, galaxy_, properties):
    """Evaluate a spectral cube of a galaxy, along a line of sight grid."""
    from gbkfit.observation import (
        ObservationGroup, ObservationSky, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import ParamSpace
    group = ObservationGroup([galaxy_], [ObservationSky(
        PixelSpectra(
            size=(32, 32), step=(1, 1),
            spectral=SpectralAxis(
                size=51, step=10, convention='relativistic')),
        line_of_sight_parser.load(dict(type='grid')))])
    params = ParamSpace.from_properties(group.pdescs(), properties)
    return group.plan(driver).model_h(
        params.evaluate())[0]['spectra']['d'].copy()


@pytest.mark.parametrize('disk_type', ['smooth', 'clouds'])
def test_rotation_equilibrium_is_the_dispersion_of_the_rotation(
        driver, disk_type):
    # The disk is that of node-wise dispersion traits at its subnodes of
    # h V / R, its rotation V (of its traits) and its rms height h; at the
    # centre, h V / R of the next subnode
    disk = dict(DISK)
    if disk_type == 'clouds':
        disk |= dict(clouds=dict(cflux=1e-3))
    balanced = galaxy(disk)
    given = galaxy(unbalanced(
        disk, dict(type='nw_uniform', sampling='subrings')))
    radii = np.array(given.constants()['subrnodes'])
    rotation = 200 * 2 / np.pi * np.arctan(radii / 3)
    sigma = np.zeros_like(radii)
    sigma[1:] = 1 * rotation[1:] / radii[1:]
    sigma[0] = sigma[1]
    np.testing.assert_allclose(
        evaluate(driver, balanced, PROPERTIES),
        evaluate(driver, given, PROPERTIES | dict(dpt_a=sigma)),
        rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('disk_type', ['smooth', 'clouds'])
def test_equilibrium_along_z_is_the_dispersion_of_sigma_z(driver, disk_type):
    # With the direction 'ver', the dispersion of the equilibrium
    # is sigma_z alone: the disk is that of traits of sigma_R = sigma_phi
    # = 0 and sigma_z of h V / R at its subnodes
    disk = dict(DISK)
    if disk_type == 'clouds':
        disk |= dict(clouds=dict(cflux=1e-3))
    balanced = galaxy(balanced_by(disk, 'rotation', direction='ver'))
    given = galaxy(unbalanced(disk, [
        dict(type='uniform', direction='rad'),
        dict(type='uniform', direction='tan'),
        dict(type='nw_uniform', sampling='subrings', direction='ver')]))
    radii = np.array(given.constants()['subrnodes'])
    rotation = 200 * 2 / np.pi * np.arctan(radii / 3)
    sigma = np.zeros_like(radii)
    sigma[1:] = 1 * rotation[1:] / radii[1:]
    sigma[0] = sigma[1]
    np.testing.assert_allclose(
        evaluate(driver, balanced, PROPERTIES),
        evaluate(driver, given, PROPERTIES | dict(dpt2_a=0, dpt3_a=sigma)),
        rtol=1e-6, atol=1e-7)


def test_the_equilibrium_alone_is_the_dispersion(driver):
    # A disk whose only dispersion trait is its equilibrium is that of
    # node-wise dispersion traits of the dispersion of the equilibrium
    balanced = galaxy(DISK | dict(
        dptraits=dict(type='equilibrium', gravity='rotation')))
    given = galaxy(unbalanced(
        DISK, dict(type='nw_uniform', sampling='subrings')))
    radii = np.array(given.constants()['subrnodes'])
    rotation = 200 * 2 / np.pi * np.arctan(radii / 3)
    sigma = np.zeros_like(radii)
    sigma[1:] = 1 * rotation[1:] / radii[1:]
    sigma[0] = sigma[1]
    properties = {k: v for k, v in PROPERTIES.items() if k != 'dpt_a'}
    assert 'dpt_a' not in balanced.pdescs()
    np.testing.assert_allclose(
        evaluate(driver, balanced, properties),
        evaluate(driver, given, properties | dict(dpt_a=sigma)),
        rtol=1e-6, atol=1e-7)


def test_sheet_equilibrium_of_its_fluid():
    # sigma^2 = 2 sqrt(3) G Sigma h, with Sigma the surface density of its
    # fluid (an exponential mass disk, M / (2 pi rd^2) exp(-R / rd)) and
    # h its rms height (kpc)
    mass = [dict(
        type='disk', name='gas', surface=dict(type='exponential'),
        vertical=dict(type='thin'))]
    balanced = galaxy(
        balanced_by(DISK, 'sheet') | dict(fluid='gas'),
        mass=mass, distance=dict(type='parameter'))
    params = PROPERTIES | dict(distance=10, mass_gas_log_m=9, mass_gas_rd=0.3)
    extra = Extras()
    states = balanced.plan_stage().evaluate(params, extra).states
    table = extra.get('equilibrium', component='component0')
    R = np.array(balanced.constants()['subrnodes']) * KPC_PER_ARCSEC
    surface = 1e9 / (2 * np.pi * 0.3 ** 2) * np.exp(-R / 0.3)
    np.testing.assert_allclose(table['surface_density'], surface, rtol=1e-12)
    np.testing.assert_allclose(
        table['sigma'],
        np.sqrt(2 * np.sqrt(3) * G * surface * 1 * KPC_PER_ARCSEC),
        rtol=1e-6)
    np.testing.assert_array_equal(
        states[0].equilibrium, table['sigma'])
    dumped = galaxy_parser.dump(balanced)['components'][0]
    assert dumped['dptraits'][-1] == dict(type='equilibrium', gravity='sheet')
    assert dumped['fluid'] == 'gas'


def test_sheet_equilibrium_of_the_mass_disk_that_follows_it():
    # Without a fluid, Sigma of the mass disk that follows the disk:
    # upsilon times its density at its subnodes (exact for its exponential
    # there)
    disk = balanced_by(DISK, 'sheet') | dict(name='gas')
    balanced = galaxy(
        disk, mass=[dict(type='traced_disk', name='gas', component='gas')],
        distance=dict(type='parameter'))
    params = {f'gas_{key}': value for key, value in PROPERTIES.items()} \
        | dict(distance=10, mass_gas_upsilon=1e9)
    extra = Extras()
    balanced.plan_stage().evaluate(params, extra)
    table = extra.get('equilibrium', component='gas')
    surface = 1e9 * np.exp(-np.array(table['radius']) / 4)
    np.testing.assert_allclose(table['surface_density'], surface, rtol=1e-12)
    np.testing.assert_allclose(
        table['sigma'] ** 2,
        2 * np.sqrt(3) * G * surface * 1 * KPC_PER_ARCSEC, rtol=1e-8)


def test_pressure_support_takes_the_dispersion_of_the_equilibrium():
    # The rotation with pressure support of a disk in equilibrium is that
    # of the disk whose node-wise dispersion traits at the subnodes are
    # hypot(sigma_traits, sigma_eq), where the support takes them
    info = dict(
        mass=[dict(type='point', name='bh')],
        distance=dict(type='parameter'))
    supported = balanced_by(DISK, 'galaxy') | dict(
        vptraits=dict(type='mass'), pressure_support=dict(type='jeans'))
    balanced = galaxy(supported, **info)
    given = galaxy(unbalanced(
        supported, dict(type='nw_uniform', sampling='subrings')), **info)
    params = PROPERTIES | dict(distance=10, mass_bh_log_m=10, dpt_a=10)
    extra = Extras()
    rotation = balanced.plan_stage().evaluate(
        params, extra).states[0].rotation['vpt_vt']
    sigma = np.hypot(
        10, extra.get('equilibrium', component='component0')['sigma'])
    extra = Extras()
    expected = given.plan_stage().evaluate(
        params | dict(dpt_a=sigma), extra).states[0].rotation['vpt_vt']
    assert np.max(-extra.get(
        'vpt_vt_rotation', component='component0')['pressure']) > 100
    np.testing.assert_allclose(rotation, expected, rtol=1e-10)


def test_galaxy_equilibrium_in_a_point_mass():
    # sigma^2 = int_0^zmax z rho(z) G M z / (R^2 + z^2)^(3/2) dz /
    # int_0^zmax rho(z) dz (kpc), the dispersion of the column weighted by
    # its density, rho the Gaussian of the disk; at the centre, the next
    # subnode
    balanced = galaxy(
        balanced_by(DISK, 'galaxy'),
        mass=[dict(type='point', name='bh')],
        distance=dict(type='parameter'))
    params = PROPERTIES | dict(distance=10, mass_bh_log_m=10)
    extra = Extras()
    balanced.plan_stage().evaluate(params, extra)
    sigma = extra.get('equilibrium', component='component0')['sigma']
    radii = np.array(balanced.constants()['subrnodes'])

    column = scipy.integrate.quad(
        lambda z: np.exp(-z ** 2 / 2), 0, 10, epsabs=0, epsrel=1e-12)[0]

    def weight(R):
        R = R * KPC_PER_ARCSEC
        return scipy.integrate.quad(
            lambda z: z * np.exp(-z ** 2 / 2) * G * 1e10 * z * KPC_PER_ARCSEC
            / (R ** 2 + (z * KPC_PER_ARCSEC) ** 2) ** 1.5 * KPC_PER_ARCSEC,
            0, 10, epsabs=0, epsrel=1e-12, limit=200)[0]
    np.testing.assert_allclose(
        sigma[1:] ** 2, [weight(R) / column for R in radii[1:]], rtol=1e-8)
    assert sigma[0] == sigma[1]


def test_galaxy_equilibrium_of_a_thin_self_gravitating_layer():
    # In its own gravity alone (a mass disk of its density), a layer thin
    # against its radius is the isothermal sheet: sigma^2 = 2 sqrt(3) G
    # Sigma h, to about h / rd
    disk = balanced_by(DISK, 'galaxy') | dict(
        name='gas', rptraits=dict(type='exponential'),
        rhtraits=dict(type='sech2', zmax=2))
    balanced = galaxy(
        disk, mass=[dict(type='traced_disk', name='gas', component='gas')],
        distance=dict(type='parameter'))
    params = {f'gas_{key}': value for key, value in PROPERTIES.items()} \
        | dict(gas_rht_s=0.05, distance=10, mass_gas_upsilon=1e9)
    extra = Extras()
    balanced.plan_stage().evaluate(params, extra)
    table = extra.get('equilibrium', component='gas')
    radii = np.array(table['radius'])
    inner = (radii > 2) & (radii < 12)
    surface = 1e9 * np.exp(-radii / 4)
    expected = 2 * np.sqrt(3) * G * surface * np.array(table['height']) \
        * KPC_PER_ARCSEC
    np.testing.assert_allclose(
        table['sigma'][inner] ** 2, expected[inner], rtol=2e-2)


POINT = dict(
    mass=[dict(type='point', name='gas')], distance=dict(type='parameter'))


@pytest.mark.parametrize('disk, info, message', [
    (balanced_by(DISK, 'rotation', mass='gas'), {}, "unknown options"),
    (balanced_by(DISK, 'sheet'), {}, "needs the mass of its galaxy"),
    (balanced_by(DISK, 'galaxy'), {}, "needs the mass of its galaxy"),
    (balanced_by(DISK, 'sheet') | dict(fluid='dust'), POINT,
     "'dust' is no mass component"),
    (balanced_by(DISK, 'sheet') | dict(fluid='gas'), POINT,
     "names mass disks; 'gas' is a point"),
    (balanced_by(DISK, 'sheet'), POINT,
     "give it a fluid, or a mass disk that follows it"),
    (DISK, POINT, "equilibrium 'rotation' of the component"),
    (balanced_by(DISK, 'galaxy') | dict(fluid='gas'), POINT,
     "it has neither")])
def test_balance_in_the_mass_is_checked(disk, info, message):
    with pytest.raises((ConfigError, ValueError), match=message):
        galaxy(disk, **info)


def test_galaxy_equilibrium_needs_vertical_forces(tmp_path):
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n    return 100 * R\n")
    with pytest.raises(ValueError, match=r"\['mine'\] do not give"):
        galaxy(
            balanced_by(DISK, 'galaxy'),
            mass=[dict(type='user', name='mine',
                       file=str(tmp_path / 'curve.py'), func='vcirc2')],
            distance=dict(type='parameter'))


@pytest.mark.parametrize('disk, message', [
    (balanced_by(DISK, 'thin'), "'thin'"),
    (balanced_by(DISK, 'rotation', direction='rad'), "'rad'"),
    ({key: value for key, value in DISK.items() if key != 'rhtraits'}
     | dict(type='thin_disk'), "a thin disk has none"),
    (DISK | dict(vptraits=dict(type='arctan', direction='rad')),
     "it has none"),
    (DISK | dict(dptraits=list(reversed(DISK['dptraits']))),
     "the last of its dptraits"),
    (DISK | dict(dptraits=dict(type='equilibrium', gravity='rotation'),
                 dhtraits=dict(type='one')), "takes none"),
    ({key: value for key, value in DISK.items()
      if key not in ('loose', 'tilted')} | dict(type='cone'),
     "equilibrium")])
def test_equilibrium_is_checked(disk, message):
    with pytest.raises((ConfigError, ValueError), match=message):
        galaxy(disk)


# A thin disk that rotates in its mass, of the jeans pressure in the sheet
# of the mass disk that follows it (its thickness not drawn)
THIN = dict(
    type='thin_disk', name='gas', rnodes=list(range(0, 21, 2)), rstep=0.5,
    rptraits=dict(type='exponential'),
    rhtraits=dict(type='equilibrium', gravity='sheet'),
    vptraits=dict(type='mass'), dptraits=dict(type='nw_uniform'),
    pressure_support=dict(type='jeans'), emission=dict(type='lines'))
THIN_MASS = [
    dict(type='traced_disk', name='gas', component='gas',
         vertical=dict(type='thin')),
    dict(type='point', name='bh')]
THIN_PROPERTIES = dict(
    gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60, gas_vsys=0,
    gas_rpt_a=1, gas_rpt_s=4, gas_dpt_a=list(np.linspace(40, 15, 11)),
    distance=10, mass_bh_log_m=10.5, mass_gas_upsilon=1e9)


@pytest.mark.parametrize('fluid', [None, 'gas'])
def test_the_sheet_of_a_thin_disk_is_its_own_gravity(fluid):
    # The density at its midplane, S / (2 z0), z0 = sigma^2 / (pi G
    # Sigma), of Sigma = upsilon S, is ~ S^2 / sigma^2: the pressure is R
    # sigma^2 d(S^2)/dR / S^2 (2 sigma^2 dln S / dln R for an axisymmetric
    # disk: Burkert et al. 2010), whatever its dispersion, at every radius
    # of the rotation (the tail too), of its own S or of the surface
    # density of its fluid
    disk = THIN | ({} if fluid is None else dict(fluid=fluid))
    balanced = galaxy(disk, mass=THIN_MASS, distance=dict(type='parameter'))
    extra = Extras()
    balanced.plan_stage().evaluate(THIN_PROPERTIES, extra)
    table = extra.get('vpt_vt_rotation', component='gas')
    r = np.asarray(table['radius'])
    surface = np.exp(-r / 4)
    sigma = np.interp(r, THIN['rnodes'], THIN_PROPERTIES['gas_dpt_a'])
    np.testing.assert_allclose(
        table['pressure'],
        r * np.gradient(surface ** 2, r, edge_order=2) * sigma ** 2
        / surface ** 2, rtol=1e-8, atol=1e-8)
    # (and its scale heights at the subnodes, z0 = sigma^2 / (pi G Sigma))
    heights = extra.get('equilibrium', component='gas')
    radii = np.asarray(heights['radius'])
    sigma = np.interp(radii, THIN['rnodes'], THIN_PROPERTIES['gas_dpt_a'])
    Sigma = 1e9 * np.exp(-radii / 4)
    np.testing.assert_allclose(heights['sigma'], sigma, rtol=1e-12)
    np.testing.assert_allclose(
        heights['height'],
        sigma ** 2 / (np.pi * G * Sigma) / KPC_PER_ARCSEC, rtol=1e-10)
    dumped = galaxy_parser.dump(balanced)['components'][0]
    assert dumped['rhtraits'] == dict(type='equilibrium', gravity='sheet')


def test_a_thick_disk_draws_the_sheet_of_its_fluid(driver):
    # The disk of the sheet of its fluid (an exponential mass disk) is that
    # of node-wise sech^2 heights at its subnodes of its scale heights,
    # sigma^2 / (pi G Sigma)
    mass = [dict(
        type='disk', name='gas', surface=dict(type='exponential'),
        vertical=dict(type='thin'))]
    info = dict(mass=mass, distance=dict(type='parameter'))
    disk = unbalanced(DISK, dict(type='uniform'))
    balanced = galaxy(disk | dict(
        rhtraits=dict(type='equilibrium', gravity='sheet', zmax=10),
        fluid='gas'), **info)
    given = galaxy(disk | dict(rhtraits=dict(
        type='sech2', zmax=10, nodewise=True, sampling='subrings')), **info)
    properties = {key: value for key, value in PROPERTIES.items()
                  if key != 'rht_s'} | dict(
        dpt_a=20, distance=10, mass_gas_log_m=10, mass_gas_rd=2)
    assert 'rht_s' not in balanced.pdescs()
    extra = Extras()
    balanced.plan_stage().evaluate(properties, extra)
    table = extra.get('equilibrium', component='component0')
    radii = np.asarray(table['radius'])
    Sigma = 1e10 / (2 * np.pi * 2 ** 2) * np.exp(-radii * KPC_PER_ARCSEC / 2)
    np.testing.assert_allclose(
        table['height'],
        20 ** 2 / (np.pi * G * Sigma) / KPC_PER_ARCSEC, rtol=1e-10)
    np.testing.assert_allclose(
        evaluate(driver, balanced, properties),
        evaluate(driver, given, properties | dict(
            rht_s=np.asarray(table['height']))),
        rtol=1e-6, atol=1e-9)


def test_a_thick_disk_draws_the_gaussian_of_its_rotation(driver):
    # Without mass, the disk is that of node-wise Gaussian heights at its
    # subnodes of h = sigma / Omega, Omega = V / R of its rotation V (of
    # its traits); at the centre, Omega of the next subnode
    disk = unbalanced(DISK, dict(type='uniform'))
    balanced = galaxy(disk | dict(rhtraits=dict(
        type='equilibrium', gravity='rotation', zmax=10)))
    given = galaxy(disk | dict(rhtraits=dict(
        type='gauss', zmax=10, nodewise=True, sampling='subrings')))
    radii = np.array(given.constants()['subrnodes'])
    omega = np.zeros_like(radii)
    omega[1:] = 200 * 2 / np.pi * np.arctan(radii[1:] / 3) / radii[1:]
    omega[0] = omega[1]
    properties = {key: value for key, value in PROPERTIES.items()
                  if key != 'rht_s'} | dict(dpt_a=20)
    np.testing.assert_allclose(
        evaluate(driver, balanced, properties),
        evaluate(driver, given, properties | dict(rht_s=20 / omega)),
        rtol=1e-6, atol=1e-9)


FOLLOWED = dict(mass=[dict(type='traced_disk', name='gas', component='gas')],
                distance=dict(type='parameter'))


def test_the_mass_in_the_layer_of_a_thick_disk_takes_its_sheet():
    # A mass disk in the layer of a thick disk in the sheet of that mass
    # (e.g. the gas and its mass) takes the heights of the sheet, z0 =
    # sigma^2 / (pi G Sigma) of Sigma = upsilon S, which do not change
    # Sigma: its density is that of the mass disk of a disk of node-wise
    # sech^2 heights of those scales
    disk = unbalanced(DISK, dict(type='uniform')) | dict(
        name='gas', rhtraits=dict(
            type='equilibrium', gravity='sheet', zmax=10))
    balanced = galaxy(disk, **FOLLOWED)
    given = galaxy(disk | dict(rhtraits=dict(
        type='sech2', zmax=10, nodewise=True, sampling='subrings')),
        **FOLLOWED)
    params = {f'gas_{key}': value for key, value in PROPERTIES.items()
              if key != 'rht_s'} \
        | dict(gas_dpt_a=20, distance=10, mass_gas_upsilon=1e9)
    extra = Extras()
    balanced.plan_stage().evaluate(params, extra)
    scale = np.asarray(extra.get('equilibrium', component='gas')['height'])
    R, z = np.meshgrid(
        np.linspace(0.05, 1, 7), np.linspace(-0.05, 0.05, 5), indexing='ij')

    def density(galaxy_, values):
        mass = galaxy_.plan_stage().mass(values)
        return mass.model.component('gas').density(
            R, z, mass.model.component_params('gas', mass.params),
            mass.context)
    expected = density(given, params | dict(gas_rht_s=scale))
    assert np.max(expected) > 0
    np.testing.assert_allclose(
        density(balanced, params), expected, rtol=1e-10)


@pytest.mark.parametrize('disk, info, message', [
    (DISK | dict(rhtraits=[
        dict(type='equilibrium', gravity='rotation', zmax=10),
        dict(type='gauss', zmax=10)], rptraits=[
        dict(type='exponential'), dict(type='gauss')]), {}, "give it alone"),
    (DISK | dict(rhtraits=dict(
        type='equilibrium', gravity='rotation', zmax=10)), {}, "not both"),
    (unbalanced(DISK, dict(type='uniform')) | dict(rhtraits=dict(
        type='equilibrium', gravity='rotation')), {}, "give it zmax"),
    (unbalanced(DISK, dict(type='uniform')) | dict(
        rhtraits=dict(type='equilibrium', gravity='rotation', zmax=10),
        dhtraits=dict(type='exponential')), {}, "must be 'one'"),
    (THIN | dict(rhtraits=dict(type='equilibrium', gravity='sheet', zmax=2)),
     FOLLOWED, "give it no zmax"),
    ({key: value for key, value in THIN.items()
      if key != 'pressure_support'} | dict(vptraits=dict(
          type='arctan', direction='tan')), {}, "give it pressure_support"),
    (THIN | dict(rhtraits=dict(type='gauss', zmax=2)), {},
     "a thin disk takes one density height trait"),
    (THIN, dict(mass=[dict(type='point', name='bh')],
                distance=dict(type='parameter')),
     "give it a fluid, or a mass disk that follows it"),
    (unbalanced(DISK, dict(type='uniform')) | dict(rhtraits=dict(
        type='equilibrium', gravity='rotation', zmax=10)), POINT,
     "equilibrium 'rotation' of the component"),
    ({key: value for key, value in DISK.items()
      if key not in ('loose', 'tilted', 'dptraits')}
     | dict(type='cone', dptraits=dict(type='uniform'), rhtraits=dict(
         type='equilibrium', gravity='rotation', zmax=10)), {},
     "equilibrium")])
def test_the_balance_of_the_thickness_is_checked(disk, info, message):
    with pytest.raises((ConfigError, ValueError), match=message):
        galaxy(disk, **info)


@pytest.mark.parametrize('disk_type', ['smooth', 'clouds'])
def test_a_thick_disk_draws_its_profile_in_the_force_of_the_mass(
        driver, disk_type):
    # Far from a point mass, its force is nearly harmonic, F_z = -nu^2 z,
    # nu^2 = G M / R^3: the profile exp(-(Phi(z) - Phi(0)) / sigma^2) is
    # nearly the Gaussian of h = sigma / nu, which the disk draws at its
    # rings far from it (here R > 10 arcsec: h / R < 0.04)
    info = dict(mass=[dict(type='point', name='bh')],
                distance=dict(type='parameter'))
    # (of no light within 8 arcsec)
    disk = unbalanced(DISK, dict(type='uniform')) | dict(
        rnodes=list(range(0, 31, 2)), rptraits=dict(type='nw_uniform'))
    if disk_type == 'clouds':
        disk |= dict(clouds=dict(cflux=1e-3))
    balanced = galaxy(disk | dict(rhtraits=dict(
        type='equilibrium', gravity='galaxy', zmax=4)), **info)
    given = galaxy(disk | dict(rhtraits=dict(
        type='gauss', zmax=4, nodewise=True, sampling='subrings')), **info)
    properties = {key: value for key, value in PROPERTIES.items()
                  if key not in ('rht_s', 'rpt_s')} | dict(
        rpt_a=[0] * 5 + [1] * 11, dpt_a=10, distance=10, mass_bh_log_m=11,
        vpt_vt=200)
    radii = np.array(given.constants()['subrnodes'])
    nu = np.zeros_like(radii)
    nu[1:] = np.sqrt(G * 1e11 / (radii[1:] * KPC_PER_ARCSEC) ** 3)
    nu[0] = nu[1]
    extra = Extras()
    balanced.plan_stage().evaluate(properties, extra)
    table = extra.get('equilibrium', component='component0')
    far = radii > 8
    np.testing.assert_allclose(
        np.asarray(table['height'])[far],
        (10 / nu / KPC_PER_ARCSEC)[far], rtol=2e-2)
    assert np.max(table['edge']) < 1e-6
    drawn = evaluate(driver, balanced, properties)
    gaussian = evaluate(
        driver, given, properties | dict(rht_s=10 / nu / KPC_PER_ARCSEC))
    np.testing.assert_allclose(drawn.sum(), gaussian.sum(), rtol=1e-3)
    if disk_type == 'smooth':
        np.testing.assert_allclose(
            drawn, gaussian, atol=3e-2 * gaussian.max())
    else:
        # (clouds drawn by other draws: within the noise of the sampling
        # of the Gaussian clouds, about their smooth disk, in their maps of
        # the sum over the channels)
        smooth = {key: value for key, value in given.dump().items()}
        smooth['components'][0].pop('clouds')
        mean = evaluate(
            driver, galaxy_parser.load(smooth),
            properties | dict(rht_s=10 / nu / KPC_PER_ARCSEC)).sum(axis=0)
        noise = np.abs(gaussian.sum(axis=0) - mean).max()
        assert np.abs(drawn.sum(axis=0) - gaussian.sum(axis=0)).max() \
            < 1.5 * noise


@pytest.mark.parametrize('disk, info, message', [
    (THIN | dict(rhtraits=dict(type='equilibrium', gravity='galaxy')),
     FOLLOWED, "a thin disk takes 'sheet' or 'rotation'"),
    (unbalanced(DISK, dict(type='uniform')) | dict(name='gas', rhtraits=dict(
        type='equilibrium', gravity='galaxy', zmax=4)), FOLLOWED,
     "not supported yet"),
    (unbalanced(DISK, dict(type='uniform')) | dict(rhtraits=dict(
        type='equilibrium', gravity='galaxy', zmax=4)), {},
     "needs the mass of its galaxy")])
def test_the_galaxy_equilibrium_of_the_thickness_is_checked(
        disk, info, message):
    with pytest.raises((ConfigError, ValueError), match=message):
        galaxy(disk, **info)
