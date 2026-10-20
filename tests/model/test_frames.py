"""
Tests of the frames of observations: the offsets that an observation
fits (see Observation) move what it sees.
"""

import numpy as np
import pytest

from gbkfit.model import galaxy_parser
from gbkfit.observation import (
    Flat, Observation, ObservationGroup, PixelMoments, observation_parser)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


# A disk and a point, which an offset moves together
GALAXY = dict(components=[
    dict(type='smdisk', name='disk', loose=False, tilted=False,
         rnodes=list(range(0, 12)),
         density=dict(ptraits=dict(type='exponential')),
         kinematics=dict(
             vptraits=dict(type='arctan', direction='tan'),
             dptraits=dict(type='uniform')),
         emission=dict(type='lines')),
    dict(type='point', name='core', emission=dict(type='lines'))])

PARAMS = dict(
    disk_vsys=0, disk_xpos=0, disk_ypos=0, disk_posa=30, disk_incl=60,
    disk_pt_a=1, disk_pt_s=4, disk_vpt_rt=2, disk_vpt_vt=40, disk_dpt_a=20,
    core_xpos=0.5, core_ypos=-1, core_vsys=10, core_flux=5, core_disp=30)


def moments(driver, offsets=None, **values):
    """
    The moment maps of observation 'b' of the galaxy, which has the given
    offsets, with observation 'a' as the reference, at PARAMS and the
    given values.
    """
    observations = [
        Observation(
            PixelMoments(size=(32, 32), spec_size=81, spec_step=5,
                         orders=(0, 1)),
            Flat(), driver, name=name, offsets=offsets if name == 'b'
            else None)
        for name in ('a', 'b')]
    group = ObservationGroup([galaxy_parser.load(GALAXY)], observations)
    params = ParamSpace(group.pdescs(), PARAMS | values)
    return group.model_h(params.evaluate())[1]


def test_offsets_move_all_that_an_observation_sees(driver):
    # Seen with offsets, the disk and the point are where they are seen
    # without them, moved by them
    offsets = dict(position=True, velocity=True)
    moved = moments(driver, offsets, b_dx=1.5, b_dy=-0.5, b_dv=20)
    expected = moments(
        driver, disk_xpos=1.5, disk_ypos=-0.5, disk_vsys=20,
        core_xpos=2, core_ypos=-1.5, core_vsys=30)
    for key in ('moment0', 'moment1'):
        np.testing.assert_array_equal(moved[key]['d'], expected[key]['d'])


def test_offsets_are_parameters_of_named_observations():
    observable = PixelMoments(size=(8, 8), spec_size=11, orders=(0,))
    observation = Observation(
        observable, Flat(), name='sinfoni', offsets=dict(position=True))
    assert set(observation.pdescs()) == {'sinfoni_dx', 'sinfoni_dy'}
    loaded = observation_parser.load(observation.dump())
    assert loaded.offset_params() == ('sinfoni_dx', 'sinfoni_dy', None)
    with pytest.raises(ConfigError, match="needs a name"):
        Observation(observable, Flat(), offsets=dict(velocity=True))
    with pytest.raises(ConfigError, match="'position' and 'velocity'"):
        Observation(observable, Flat(), name='a', offsets=dict(x=True))


def test_a_galaxy_needs_an_observation_without_offsets():
    # (offsets alone would move its centre with nothing to place it)
    observations = [
        Observation(
            PixelMoments(size=(8, 8), spec_size=11, orders=(0,)), Flat(),
            name=name, offsets=dict(position=True))
        for name in ('a', 'b')]
    with pytest.raises(ConfigError, match="offsets of position; one needs"):
        ObservationGroup([galaxy_parser.load(GALAXY)], observations)
