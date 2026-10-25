"""
Tests of the frames of observations: the offsets that an observation
fits (see ObservationSky) move what it sees.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from gbkfit.model import galaxy_parser
from gbkfit.observation import (
    Flat, ObservationGroup, ObservationSky, PixelMoments, SlitSpectra,
    observable_parser, observation_parser)
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


# The RA and Dec of the reference pixel of the first observation (deg)
ORIGIN = (49.5, -66.5)


def moments(driver, offsets=None, origin=ORIGIN, **values):
    """
    The moment maps of observation 'b' of the galaxy, which has the given
    offsets and the RA and Dec of its reference pixel, with observation
    'a' (its reference pixel at ORIGIN) first, at PARAMS and the given
    values.
    """
    observations = [
        ObservationSky(
            PixelMoments(size=(32, 32), rval=rval, spec_size=81,
                         spec_step=5, orders=(0, 1),
                         spec_convention='relativistic'),
            Flat(), name=name, offsets=offsets if name == 'b' else None)
        for name, rval in (('a', ORIGIN), ('b', origin))]
    group = ObservationGroup([galaxy_parser.load(GALAXY)], observations)
    params = ParamSpace(group.pdescs(), PARAMS | values)
    return group.plan(driver).model_h(params.evaluate())[1]


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
    observable = PixelMoments(size=(8, 8), spec_size=11, orders=(0,),
                              spec_convention='relativistic')
    observation = ObservationSky(
        observable, Flat(), name='sinfoni', offsets=dict(position=True))
    assert set(observation.pdescs()) == {'sinfoni_dx', 'sinfoni_dy'}
    loaded = observation_parser.load(observation_parser.dump(observation))
    assert loaded.offset_params() == ('sinfoni_dx', 'sinfoni_dy', None)
    with pytest.raises(ConfigError, match="needs a name"):
        ObservationSky(observable, Flat(), offsets=dict(velocity=True))
    with pytest.raises(ConfigError, match="'position' and 'velocity'"):
        ObservationSky(observable, Flat(), name='a', offsets=dict(x=True))


def test_a_galaxy_needs_an_observation_without_offsets():
    # (offsets alone would move its centre with nothing to place it)
    observations = [
        ObservationSky(
            PixelMoments(size=(8, 8), spec_size=11, orders=(0,),
                         spec_convention='relativistic'), Flat(),
            name=name, offsets=dict(position=True))
        for name in ('a', 'b')]
    with pytest.raises(ConfigError, match="offsets of position; one needs"):
        ObservationGroup([galaxy_parser.load(GALAXY)], observations)


def test_observations_are_aligned_by_their_coordinates(driver):
    # A cube whose reference pixel is 2 arcsec west of that of the first
    # sees the galaxy 2 arcsec further east of it: at x - 2 in its frame
    first = SkyCoord(*ORIGIN, unit='deg')
    west = first.spherical_offsets_by(-2 * u.arcsec, 0 * u.arcsec)
    east, north = west.spherical_offsets_to(first)
    dx, dy = -east.to_value('arcsec'), north.to_value('arcsec')
    aligned = moments(driver, origin=(west.ra.deg, west.dec.deg))
    expected = moments(
        driver, disk_xpos=dx, disk_ypos=dy, core_xpos=0.5 + dx,
        core_ypos=-1 + dy)
    assert dx == pytest.approx(-2)
    for key in ('moment0', 'moment1'):
        np.testing.assert_allclose(
            aligned[key]['d'], expected[key]['d'], rtol=1e-6, atol=1e-6)


def test_frames_far_apart_are_warned_of(driver, caplog):
    # (e.g. data without coordinates, at RA and Dec 0, with real ones)
    moments(driver, origin=(0, 0))
    assert "deg apart: check their coordinates" in caplog.text


def test_slits_have_the_coordinates_of_their_reference_point():
    slit = SlitSpectra(size=(10, 11), sky_rval=(49.5, -66.5))
    assert slit.sky_rval() == (49.5, -66.5)
    assert observable_parser.load(slit.dump()).sky_rval() == (49.5, -66.5)
    assert SlitSpectra(size=(10, 11)).sky_rval() == (0, 0)
