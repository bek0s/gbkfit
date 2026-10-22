"""
Tests for the analysis tools (gbkfit.analysis) on velocity fields of
known geometry: an analytic rotating disk, on a map and on bins, a
disturbed one, and a model of GBKFIT.
"""

import numpy as np
import pytest
from modelutils import observation_group

import gbkfit.params
from gbkfit.analysis import (
    asymmetry_maps, kinematic_pa, kinemetry, pattern_speed)


PA, INCL, VSYS, VMAX, RT = 130.0, 55.0, 1200.0, 200.0, 3.0


def disk_field(x, y, centre=(20, 20)):
    """
    The velocity field of a disk of PA (as posa: receding towards (-sin,
    cos) of it) and INCL, rotating with an arctan curve.
    """
    angle, incl = np.radians(PA), np.radians(INCL)
    dx, dy = x - centre[0], y - centre[1]
    along = -dx * np.sin(angle) + dy * np.cos(angle)
    across = -dx * np.cos(angle) - dy * np.sin(angle)
    radius = np.hypot(along, across / np.cos(incl))
    rotation = VMAX * 2 / np.pi * np.arctan(radius / RT)
    cos_theta = np.divide(along, radius, out=np.zeros_like(radius),
                          where=radius > 0)
    return VSYS + rotation * np.sin(incl) * cos_theta, radius


def grid(size=41):
    y, x = np.indices((size, size), dtype=float)
    return x, y


def test_a_rotating_disk_is_symmetric_and_its_geometry_is_found():
    x, y = grid()
    v, radius = disk_field(x, y)
    v[radius > 18] = np.nan
    found = kinematic_pa(v, x, y, (20, 20))
    assert found.pa == pytest.approx(PA, abs=0.5)
    assert found.systemic == pytest.approx(VSYS, abs=0.5)
    # (0 but for the linear interpolation between the pixels, largest at
    # the centre, where the rotation rises steeply)
    asymmetry = asymmetry_maps(v, x, y, (20, 20), PA, systemic=VSYS)
    for name in ('major', 'minor', 'rotation'):
        assert np.nanmax(np.abs(getattr(asymmetry, name))) < 0.02 * VMAX
    assert asymmetry.a_rc < 0.01
    rings = kinemetry(v, x, y, (20, 20), radii=[4, 8, 12])
    np.testing.assert_allclose(rings.pa, PA, atol=1)
    np.testing.assert_allclose(rings.q, np.cos(np.radians(INCL)), atol=0.02)
    np.testing.assert_allclose(rings.a0, VSYS, atol=1)
    k = rings.k()
    np.testing.assert_allclose(
        k[:, 0], VMAX * 2 / np.pi * np.arctan(np.array([4, 8, 12]) / RT)
        * np.sin(np.radians(INCL)), rtol=0.02)
    assert np.all(k[:, 4] / k[:, 0] < 0.02)


def test_a_disturbed_disk_is_asymmetric():
    # A flow of 40 km/s in one quadrant of the disk: its reflections
    # differ there, and A_RC grows
    x, y = grid()
    v, radius = disk_field(x, y)
    v[radius > 18] = np.nan
    disturbed = v + np.where((x > 20) & (y > 20), 40.0, 0.0)
    calm = asymmetry_maps(v, x, y, (20, 20), PA, systemic=VSYS)
    asymmetry = asymmetry_maps(disturbed, x, y, (20, 20), PA, systemic=VSYS)
    assert asymmetry.a_rc > 10 * calm.a_rc
    assert np.nanmax(np.abs(asymmetry.rotation)) == pytest.approx(40, abs=2)


def test_the_pattern_speed_of_a_turning_bar_is_found():
    # A disk with a bar (30 degrees from the line of nodes) turning as a
    # rigid body at omega obeys the continuity equation, and its velocity
    # is omega sin(i) X on the sky: the slits find omega exactly
    omega = 15.0
    x, y = grid()
    angle, incl = np.radians(PA), np.radians(INCL)
    dx, dy = x - 20, y - 20
    along = -dx * np.sin(angle) + dy * np.cos(angle)
    across = (-dx * np.cos(angle) - dy * np.sin(angle)) / np.cos(incl)
    bar = np.radians(30)
    xb = along * np.cos(bar) + across * np.sin(bar)
    yb = -along * np.sin(bar) + across * np.cos(bar)
    flux = np.exp(-np.hypot(along, across) / 6) \
        + 2 * np.exp(-(xb / 8) ** 2 - (yb / 2.5) ** 2)
    velocity = VSYS + omega * np.sin(incl) * along
    found = pattern_speed(
        flux, velocity, x, y, (20, 20), PA, incl=INCL, systemic=VSYS)
    assert found.omega == pytest.approx(omega, rel=1e-9)
    assert found.intercept == pytest.approx(0, abs=1e-9)
    assert len(found.offsets) > 10


def test_bins_are_taken_as_maps_are():
    # Scattered points (as of Voronoi bins) give the same position angle
    rng = np.random.default_rng(5)
    x, y = rng.uniform(2, 38, (2, 1500))
    v, _ = disk_field(x, y)
    assert kinematic_pa(v, x, y, (20, 20)).pa == pytest.approx(PA, abs=1)


def test_the_geometry_of_a_model_is_found(driver):
    # The position angle of the analysis is that of the models (posa), and
    # the axis ratio of a thin disk cos(incl)
    galaxy = dict(components=[dict(
        type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 21)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(vptraits=dict(type='arctan', direction='tan'),
                        dptraits=dict(type='uniform')),
        emission=dict(type='lines'))])
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=galaxy,
        line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_moments', size=[41, 41], spec_size=161,
                        spec_step=5, orders=[1], mask_cutoff=1e-3))])
    values = dict(vsys=0, xpos=0, ypos=0, posa=40, incl=60, pt_a=1, pt_s=4,
                  vpt_rt=2, vpt_vt=150, dpt_a=20)
    params = gbkfit.params.ParamSpace(group.pdescs(), values)
    v = group.model_h(params.evaluate())[0]['moment1']['d']
    x, y = grid()
    assert kinematic_pa(v, x, y, (20, 20)).pa == pytest.approx(40, abs=0.5)
    rings = kinemetry(v, x, y, (20, 20), radii=[6, 10, 14])
    np.testing.assert_allclose(rings.pa, 40, atol=1)
    np.testing.assert_allclose(rings.q, 0.5, atol=0.03)
