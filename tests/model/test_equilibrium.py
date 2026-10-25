"""
Tests of the vertical equilibrium of disks (see parts.Kinematics): the
dispersion that holds a thick disk up in the gravity of its galaxy, in
quadrature with that of its dispersion traits.
"""

import copy

import numpy as np
import pytest
import scipy.integrate

from gbkfit.model import Extras, galaxy_parser
from gbkfit.model.components.disks._disk import EQUILIBRIUM_DISPERSION
from gbkfit.utils.parseutils import ConfigError


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The length of an arcsec (kpc) at 10 Mpc
KPC_PER_ARCSEC = 1e4 * np.pi / 648000

# A thick disk of a Gaussian height profile (its rms height is its s, 1
# arcsec: its cut at zmax is 1e-22 of it), its rotation of arctan traits
DISK = dict(
    type='disk', loose=False, tilted=False, rnodes=list(range(0, 21, 2)),
    rstep=0.5,
    density=dict(
        ptraits=dict(type='exponential'),
        htraits=dict(type='gauss', zmax=10)),
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform'), equilibrium='central'),
    emission=dict(type='lines'))

PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=4, ht_s=1,
    vpt_rt=3, vpt_vt=200, dpt_a=0)


def galaxy(disk, **info):
    """A galaxy of the given disk, and the given options (e.g. mass)."""
    return galaxy_parser.load(
        dict(components=[copy.deepcopy(disk)]) | copy.deepcopy(info))


def with_kinematics(disk, **kinematics):
    """The disk with the given options of its kinematics instead."""
    return disk | dict(kinematics=disk['kinematics'] | kinematics)


def evaluate(driver, galaxy_, properties):
    """Evaluate a spectral cube of a galaxy, along a line of sight grid."""
    from gbkfit.observation import (
        ObservationGroup, ObservationSky, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import ParamSpace
    group = ObservationGroup([galaxy_], [ObservationSky(
        PixelSpectra(size=(32, 32, 51), step=(1, 1, 10)),
        line_of_sight_parser.load(dict(type='grid')))])
    params = ParamSpace(group.pdescs(), properties)
    return group.plan(driver).model_h(
        params.evaluate())[0]['spectra']['d'].copy()


@pytest.mark.parametrize('disk_type', ['smooth', 'clouds'])
def test_central_equilibrium_is_the_dispersion_of_the_rotation(
        driver, disk_type):
    # Without mass, the disk is that of node-wise dispersion traits at its
    # subnodes of h V / R, its rotation V (of its traits) and its rms
    # height h; at the centre, h V / R of the next subnode
    disk = DISK | dict(type='disk')
    if disk_type == 'clouds':
        disk |= dict(clouds=dict(cflux=1e-3))
    balanced = galaxy(disk)
    given = galaxy(with_kinematics(
        disk, dptraits=dict(type='nw_uniform', sampling='subrings'),
        equilibrium=None))
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
    # With equilibrium_direction 'ver', the dispersion of the equilibrium
    # is sigma_z alone: the disk is that of traits of sigma_R = sigma_phi
    # = 0 and sigma_z of h V / R at its subnodes
    disk = DISK | dict(type='disk')
    if disk_type == 'clouds':
        disk |= dict(clouds=dict(cflux=1e-3))
    balanced = galaxy(with_kinematics(disk, equilibrium_direction='ver'))
    given = galaxy(with_kinematics(
        disk, dptraits=[
            dict(type='uniform', direction='rad'),
            dict(type='uniform', direction='tan'),
            dict(type='nw_uniform', sampling='subrings', direction='ver')],
        equilibrium=None))
    radii = np.array(given.constants()['subrnodes'])
    rotation = 200 * 2 / np.pi * np.arctan(radii / 3)
    sigma = np.zeros_like(radii)
    sigma[1:] = 1 * rotation[1:] / radii[1:]
    sigma[0] = sigma[1]
    np.testing.assert_allclose(
        evaluate(driver, balanced, PROPERTIES),
        evaluate(driver, given, PROPERTIES | dict(dpt1_a=0, dpt2_a=sigma)),
        rtol=1e-6, atol=1e-7)


def test_central_equilibrium_in_the_mass_of_the_galaxy():
    # With mass (a point), sigma = h V_c / R, V_c^2 = G M / R (kpc), which
    # the gravity gives the disk and writes in its table
    mass = [dict(type='point', name='bh')]
    balanced = galaxy(
        DISK, mass=mass, distance=dict(type='direct', cosmology='Planck18'))
    params = PROPERTIES | dict(distance=10, mass_bh_log_m=10)
    extra = Extras()
    given = balanced.plan_stage().evaluate(params, extra).given
    table = extra.get('equilibrium', component='component0')
    radii = np.array(balanced.constants()['subrnodes'])
    vcirc = np.sqrt(G * 1e10 / (radii[1:] * KPC_PER_ARCSEC))
    np.testing.assert_allclose(table['radius'], radii)
    np.testing.assert_allclose(table['height'], 1, rtol=1e-9)
    np.testing.assert_allclose(table['vcirc'][1:], vcirc, rtol=1e-6)
    np.testing.assert_allclose(
        table['sigma'][1:], 1 * vcirc / radii[1:], rtol=1e-6)
    assert table['sigma'][0] == table['sigma'][1]
    np.testing.assert_array_equal(
        given[0][EQUILIBRIUM_DISPERSION], table['sigma'])
    dumped = galaxy_parser.dump(balanced)
    assert dumped['components'][0]['kinematics']['equilibrium'] == 'central'


def test_self_gravitating_equilibrium_in_a_mass_disk():
    # sigma^2 = 2 sqrt(3) G Sigma h, with Sigma the surface density of the
    # mass component it names (an exponential disk, M / (2 pi rd^2)
    # exp(-R / rd)) and h its rms height (kpc)
    mass = [dict(
        type='disk', name='gas', surface=dict(type='exponential'),
        vertical=dict(type='thin'))]
    balanced = galaxy(
        with_kinematics(
            DISK, equilibrium='self_gravitating', equilibrium_density='gas'),
        mass=mass, distance=dict(type='direct', cosmology='Planck18'))
    params = PROPERTIES | dict(distance=10, mass_gas_log_m=9, mass_gas_rd=0.3)
    extra = Extras()
    given = balanced.plan_stage().evaluate(params, extra).given
    table = extra.get('equilibrium', component='component0')
    R = np.array(balanced.constants()['subrnodes']) * KPC_PER_ARCSEC
    surface = 1e9 / (2 * np.pi * 0.3 ** 2) * np.exp(-R / 0.3)
    np.testing.assert_allclose(table['surface_density'], surface, rtol=1e-12)
    np.testing.assert_allclose(
        table['sigma'],
        np.sqrt(2 * np.sqrt(3) * G * surface * 1 * KPC_PER_ARCSEC),
        rtol=1e-6)
    np.testing.assert_array_equal(
        given[0][EQUILIBRIUM_DISPERSION], table['sigma'])
    kinematics = galaxy_parser.dump(balanced)['components'][0]['kinematics']
    assert kinematics['equilibrium_density'] == 'gas'


def test_self_gravitating_equilibrium_in_its_own_mass_disk():
    # Sigma of a mass disk that follows the disk: upsilon times its density
    # at its subnodes (exact for its exponential there)
    disk = with_kinematics(
        DISK, equilibrium='self_gravitating', equilibrium_density='gas') \
        | dict(name='gas', geometry='own')
    balanced = galaxy(
        disk, mass=[dict(
            type='disk', name='gas', surface=dict(type='component',
                                                  component='gas'),
            vertical=dict(type='component', component='gas'))],
        distance=dict(type='direct', cosmology='Planck18'))
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
        distance=dict(type='direct', cosmology='Planck18'))
    supported = with_kinematics(
        DISK, vptraits=dict(type='mass', pressure_support='jeans'))
    balanced = galaxy(supported, **info)
    given = galaxy(with_kinematics(
        supported, dptraits=dict(type='nw_uniform', sampling='subrings'),
        equilibrium=None), **info)
    params = PROPERTIES | dict(distance=10, mass_bh_log_m=10, dpt_a=10)
    extra = Extras()
    rotation = balanced.plan_stage().evaluate(params, extra).given[0]['vpt_vt']
    sigma = np.hypot(10, extra.get('equilibrium', component='component0')['sigma'])
    extra = Extras()
    expected = given.plan_stage().evaluate(
        params | dict(dpt_a=sigma), extra).given[0]['vpt_vt']
    assert np.max(-extra.get('vpt_vt_rotation', component='component0')['pressure']) > 100
    np.testing.assert_allclose(rotation, expected, rtol=1e-10)


def test_jeans_equilibrium_in_a_point_mass():
    # sigma^2 = int_0^zmax rho(z) G M z / (R^2 + z^2)^(3/2) dz / rho(0)
    # (kpc), rho the Gaussian of the disk; at the centre, the next subnode
    balanced = galaxy(
        with_kinematics(DISK, equilibrium='jeans'),
        mass=[dict(type='point', name='bh')],
        distance=dict(type='direct', cosmology='Planck18'))
    params = PROPERTIES | dict(distance=10, mass_bh_log_m=10)
    extra = Extras()
    balanced.plan_stage().evaluate(params, extra)
    sigma = extra.get('equilibrium', component='component0')['sigma']
    radii = np.array(balanced.constants()['subrnodes'])

    def weight(R):
        R = R * KPC_PER_ARCSEC
        return scipy.integrate.quad(
            lambda z: np.exp(-z ** 2 / 2) * G * 1e10 * z * KPC_PER_ARCSEC
            / (R ** 2 + (z * KPC_PER_ARCSEC) ** 2) ** 1.5 * KPC_PER_ARCSEC,
            0, 10, epsabs=0, epsrel=1e-12, limit=200)[0]
    np.testing.assert_allclose(
        sigma[1:] ** 2, [weight(R) for R in radii[1:]], rtol=1e-8)
    assert sigma[0] == sigma[1]


def test_jeans_equilibrium_of_a_thin_self_gravitating_layer():
    # In its own gravity alone (a mass disk of its density), a layer thin
    # against its radius is the isothermal slab of the self-gravitating
    # equilibrium: sigma^2 = 2 sqrt(3) G Sigma h, to about h / rd
    disk = with_kinematics(DISK, equilibrium='jeans') | dict(
        name='gas', geometry='own', density=dict(
            ptraits=dict(type='exponential'),
            htraits=dict(type='sech2', zmax=2)))
    balanced = galaxy(
        disk, mass=[dict(
            type='disk', name='gas', surface=dict(type='component',
                                                  component='gas'),
            vertical=dict(type='component', component='gas'))],
        distance=dict(type='direct', cosmology='Planck18'))
    params = {f'gas_{key}': value for key, value in PROPERTIES.items()} \
        | dict(gas_ht_s=0.05, distance=10, mass_gas_upsilon=1e9)
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


@pytest.mark.parametrize('disk, info, message', [
    (with_kinematics(DISK, equilibrium_density='gas'), {},
     "required for it, and only for it"),
    (with_kinematics(DISK, equilibrium='self_gravitating'), {},
     "required for it, and only for it"),
    (with_kinematics(
        DISK, equilibrium='self_gravitating', equilibrium_density='gas'),
     {}, "needs the mass of its galaxy"),
    (with_kinematics(DISK, equilibrium='jeans'), {},
     "needs the mass of its galaxy"),
    (with_kinematics(
        DISK, equilibrium='self_gravitating', equilibrium_density='dust'),
     dict(mass=[dict(type='point', name='gas')],
          distance=dict(type='direct')), "which its galaxy does not have")])
def test_self_gravitating_equilibrium_is_checked(disk, info, message):
    with pytest.raises(ConfigError, match=message):
        galaxy(disk, **info)


def test_jeans_equilibrium_needs_vertical_forces(tmp_path):
    (tmp_path / 'curve.py').write_text(
        "def vcirc2(R, params):\n    return 100 * R\n")
    with pytest.raises(ConfigError, match=r"\['mine'\] do not give"):
        galaxy(
            with_kinematics(DISK, equilibrium='jeans'),
            mass=[dict(type='user', name='mine',
                       file=str(tmp_path / 'curve.py'), func='vcirc2')],
            distance=dict(type='direct'))


@pytest.mark.parametrize('disk, message', [
    (with_kinematics(DISK, equilibrium='thin'), "equilibrium must be one of"),
    (DISK | dict(density=dict(ptraits=dict(type='exponential'))),
     "needs density height traits"),
    (with_kinematics(DISK, vptraits=dict(type='arctan', direction='rad')),
     "it has neither"),
    ({key: value for key, value in DISK.items()
      if key not in ('loose', 'tilted')} | dict(type='cone'),
     "a cone has no vertical equilibrium")])
def test_equilibrium_is_checked(disk, message):
    with pytest.raises(ConfigError, match=message):
        galaxy(disk)
