"""
Tests of the vertical equilibrium of disks (see parts.Kinematics): the
dispersion that holds a thick disk up in the gravity of its galaxy, in
quadrature with that of its dispersion traits.
"""

import copy

import numpy as np
import pytest

from gbkfit.model import galaxy_parser
from gbkfit.model.components.disks._disk import EQUILIBRIUM_DISPERSION
from gbkfit.utils.parseutils import ConfigError


# The gravitational constant (kpc (km/s)^2 / Msun)
G = 4.30091727e-6

# The length of an arcsec (kpc) at 10 Mpc
KPC_PER_ARCSEC = 10 * 4.8481368e-3

# A thick disk of a Gaussian height profile (its rms height is its s, 1
# arcsec: its cut at zmax is 1e-22 of it), its rotation of arctan traits
DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 21, 2)),
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
        Observation, ObservationGroup, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import ParamSpace
    group = ObservationGroup([galaxy_], [Observation(
        PixelSpectra(size=(32, 32, 51), step=(1, 1, 10)),
        line_of_sight_parser.load(dict(type='grid')), driver)])
    params = ParamSpace(group.pdescs(), properties)
    return group.model_h(params.evaluate())[0]['spectra']['d'].copy()


@pytest.mark.parametrize('disk_type', ['smdisk', 'mcdisk'])
def test_central_equilibrium_is_the_dispersion_of_the_rotation(
        driver, disk_type):
    # Without mass, the disk is that of node-wise dispersion traits at its
    # subnodes of h V / R, its rotation V (of its traits) and its rms
    # height h; at the centre, h V / R of the next subnode
    disk = DISK | dict(type=disk_type)
    if disk_type == 'mcdisk':
        disk |= dict(cflux=1e-3)
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


def test_central_equilibrium_in_the_mass_of_the_galaxy():
    # With mass (a point), sigma = h V_c / R, V_c^2 = G M / R (kpc), which
    # the gravity gives the disk and writes in its table
    mass = [dict(type='point', name='bh')]
    balanced = galaxy(
        DISK, mass=mass, distance=dict(type='direct', cosmology='Planck18'))
    params = PROPERTIES | dict(distance=10, mass_bh_log_m=10)
    extra = {}
    given = balanced.plan_gravity().evaluate(params, extra)
    table = extra['component0_equilibrium']
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


@pytest.mark.parametrize('disk, message', [
    (with_kinematics(DISK, equilibrium='thin'), "equilibrium must be one of"),
    (DISK | dict(density=dict(ptraits=dict(type='exponential'))),
     "needs density height traits"),
    (with_kinematics(DISK, vptraits=dict(type='arctan', direction='rad')),
     "it has neither"),
    ({key: value for key, value in DISK.items()
      if key not in ('loose', 'tilted')} | dict(type='smcone'),
     "a cone has no vertical equilibrium")])
def test_equilibrium_is_checked(disk, message):
    with pytest.raises(ConfigError, match=message):
        galaxy(disk)
