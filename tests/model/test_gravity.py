"""
Tests of the gravity of galaxies (see Gravity): how their mass pulls.
"""

import copy

import pytest

from gbkfit.model import GravityNewton, galaxy_parser
from gbkfit.utils.parseutils import ConfigError


# A disk that rotates in the mass of a stellar disk
DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 21, 2)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='mass'), dptraits=dict(type='uniform')),
    emission=dict(type='lines'))
MASS = [dict(
    type='disk', name='stars', surface=dict(type='exponential'),
    vertical=dict(type='thin'))]
DISTANCE = dict(type='direct', cosmology='Planck18')


def galaxy(**info):
    """A galaxy of the disk, and the given options."""
    return galaxy_parser.load(
        dict(components=[copy.deepcopy(DISK)]) | copy.deepcopy(info))


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
