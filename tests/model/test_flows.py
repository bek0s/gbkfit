"""
Tests of the flows of the plane of a disk: along a bar (the velocity
traits bar and nw_bar) and in two wedges (wedge), seen on the line of
sight as their formulas give.
"""

import numpy as np
import pytest

from gbkfit.model import component_parser
from gbkfit.utils.parseutils import ConfigError


RNODES = list(range(0, 13))
INCL = 60

# Points of the plane of the disk: radii (arcsec) and azimuths (degrees)
_rng = np.random.default_rng(1)
R = _rng.uniform(0.1, 11.9, 400)
THETA = _rng.uniform(-180, 180, 400)


def disk(vptrait):
    """A disk of the nodes RNODES, of the velocity trait."""
    return component_parser.load(dict(
        type='smdisk', loose=False, tilted=False, rnodes=RNODES,
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(vptraits=vptrait, dptraits=dict(type='uniform')),
        emission=dict(type='lines'))).density_disk()


def seen(driver, vptrait, values):
    """
    The line-of-sight velocity of a velocity trait of a disk at the points
    (R, THETA), at the inclination INCL, for the values of its parameters
    (without their prefix).
    """
    params = dict(
        xpos=0, ypos=0, posa=0, incl=INCL, vsys=0, pt_a=1, pt_s=3,
        dpt_a=20) | {f'vpt_{name}': value for name, value in values.items()}
    plan = disk(vptrait).plan(driver, 0, np.float64)
    return plan.evaluate_polar_traits(params, 'vpt', R, THETA, INCL)[0]


def bar_flow(speed, p, s):
    """
    The line-of-sight velocity of the flow of the speed (at each point)
    along the axis of a bar at the azimuth p, within the strip of the
    width s: v sign(x) sin(p) sin(incl) where |y| < s / 2.
    """
    x = R * np.cos(np.deg2rad(THETA - p))
    y = R * np.sin(np.deg2rad(THETA - p))
    return np.where(
        np.abs(y) < s / 2,
        speed * np.sign(x) * np.sin(np.deg2rad(p)) * np.sin(np.deg2rad(INCL)),
        0)


def test_a_bar_flows_along_its_axis(driver):
    # (inwards, along an axis 30 degrees from the major axis)
    values = dict(v=-90, p=30, s=4)
    np.testing.assert_allclose(
        seen(driver, dict(type='bar'), values),
        bar_flow(-90, 30, 4), atol=1e-9)


def test_the_speed_of_nw_bar_is_that_at_the_distance_along_it(driver):
    # A speed that falls to 0 at 8 arcsec along the bar (its length), at
    # the subnodes and linear between them, as the disk interpolates it
    trait = dict(type='nw_bar')
    speeds = np.clip(80 - 10 * np.array(RNODES, dtype=float), 0, None)
    subrnodes = disk(trait).subrnodes()
    distance = np.abs(R * np.cos(np.deg2rad(THETA - 120)))
    speed = np.interp(
        distance, subrnodes, np.interp(subrnodes, RNODES, speeds))
    np.testing.assert_allclose(
        seen(driver, trait, dict(p=120, s=3, v=speeds)),
        bar_flow(speed, 120, 3), atol=1e-9)


def test_a_wedge_has_its_speed_within_its_two_wedges(driver):
    # A radial flow within 30 degrees of the azimuths 70 and 250 degrees
    within = np.abs(np.cos(np.deg2rad(THETA - 70))) \
        >= np.cos(np.deg2rad(30))
    radial = 50 * np.sin(np.deg2rad(THETA)) * np.sin(np.deg2rad(INCL))
    np.testing.assert_allclose(
        seen(driver, dict(type='wedge', direction='rad'),
             dict(vr=50, p=70, s=60)),
        np.where(within, radial, 0), atol=1e-9)


def test_cones_take_no_flows_of_the_plane_of_a_disk():
    for trait in (dict(type='bar'), dict(type='wedge', direction='tan')):
        with pytest.raises(ConfigError, match="a cone does not take"):
            component_parser.load(dict(
                type='smcone', rnodes=RNODES,
                density=dict(
                    ptraits=dict(type='uniform'),
                    htraits=dict(type='gauss', zmax=1.5)),
                kinematics=dict(
                    vptraits=trait, dptraits=dict(type='uniform')),
                emission=dict(type='lines')))
