"""
Tests of the rays of an observation through lenses
(gbkfit.lensing.TracePlan): the points where the rays of the pixels of
its footprint cross each plane behind lenses, in the pixels of its grid,
moved with the offsets of the observation, evaluated again only when
they change.
"""

import numpy as np
import pytest

from gbkfit.cosmology import DEFAULT
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.grid import make_grid, sky_positions, sky_to_pixel
from gbkfit.lensing import (
    MAX_STRETCH, Lens, LensDeflectionMap, LensSIE, SkyPoints, SourceGrid,
    TracePlan, Tracer, plane_layout, source_step, stretch)
from gbkfit.model import Reference


ORIGIN = (150.0, 2.0)

GRID = make_grid((24, 20), 0.2, rval=ORIGIN)

FOOTPRINT = np.ones((20, 24), dtype=bool)

SIE = dict(theta_e=1.1, q=0.7, posa=30, xpos=0.1, ypos=-0.2)

# (the parameters of the lens: those of its SIE, prefixed by the name of
# the piece, after its type; and their names among those of the scene)
LENS = {f'sie_{name}': value for name, value in SIE.items()}

NAMES = {name: f'lens_{name}' for name in LENS}

PARAMS = {NAMES[name]: value for name, value in LENS.items()} \
    | dict(obs_dx=0.0, obs_dy=0.0)


def tracer_of(lens):
    planes = plane_layout(
        [('deflector', 0.35), ('arc', 2.1)], [(lens.name(), lens.redshift())])
    return Tracer(planes, [lens], DEFAULT)


def sie_tracer():
    return tracer_of(Lens('lens', 0.35, LensSIE(), source_redshift=2.1))


def trace_plan(driver, tracer=None, footprint=FOOTPRINT, offsets=(None, None),
               alignment=(0.0, 0.0), options=SourceGrid(), reference=PARAMS,
               names=NAMES):
    return TracePlan(
        tracer or sie_tracer(), [names], [1], GRID, footprint, ORIGIN,
        alignment, offsets, Reference(reference, {}), options, driver,
        np.float64)


def expected_rays(params, shift=(0.0, 0.0)):
    """The rays of the pixels of the grid, of one SIE quoted for them."""
    x, y = sky_positions(GRID)
    points = SkyPoints(x - shift[0], y - shift[1], ORIGIN)
    ray = sie_tracer().trace(
        points, [{name: params[group] for name, group in NAMES.items()}],
        DriverHost())[1]
    return ray.x + shift[0], ray.y + shift[1]


def on_device(driver, positions):
    return [driver.mem_copy_d2h(array) for array in positions]


def test_the_rays_in_the_pixels_of_the_grid_of_a_plane(driver):
    plan = trace_plan(driver)
    x, y = expected_rays(PARAMS)
    positions = on_device(driver, plan.positions(PARAMS, {})[1])
    matrix, offset = sky_to_pixel(plan.grid(1))
    np.testing.assert_allclose(
        positions[0], matrix[0, 0] * x + matrix[0, 1] * y + offset[0],
        atol=1e-9)
    np.testing.assert_allclose(
        positions[1], matrix[1, 0] * x + matrix[1, 1] * y + offset[1],
        atol=1e-9)
    np.testing.assert_allclose(plan.rays(1), [x, y], atol=1e-12)
    # (the grid holds them, with its margin)
    assert plan.off_grid(1) == 0
    assert positions[0].min() > 0 and positions[0].max() < \
        plan.grid(1).size[0] - 1


def test_the_pixels_off_the_footprint_have_no_rays(driver):
    footprint = FOOTPRINT.copy()
    footprint[:, :5] = False
    plan = trace_plan(driver, footprint=footprint)
    positions = on_device(driver, plan.positions(PARAMS, {})[1])
    assert np.isnan(positions[0][:, :5]).all()
    assert not np.isnan(positions[0][:, 5:]).any()
    assert np.isnan(plan.rays(1)[0][:, :5]).all()


def test_the_lenses_move_with_the_offsets_of_the_observation(driver):
    # An observation of a frame shifted by s (its alignment and offsets)
    # sees the lens at its centre plus s, as it sees the galaxies
    plan = trace_plan(driver, offsets=('obs_dx', 'obs_dy'),
                      alignment=(0.3, -0.1))
    params = PARAMS | dict(obs_dx=0.05, obs_dy=0.2)
    plan.positions(params, {})
    np.testing.assert_allclose(
        plan.rays(1), expected_rays(params, (0.35, 0.1)), atol=1e-12)


def test_the_rays_are_traced_again_only_when_they_change(
        driver, monkeypatch):
    tracer = sie_tracer()
    plan = trace_plan(driver, tracer=tracer, offsets=('obs_dx', 'obs_dy'))
    calls = []
    deflection = tracer.deflection
    monkeypatch.setattr(
        tracer, 'deflection',
        lambda *args: calls.append(1) or deflection(*args))
    plan.positions(PARAMS, {})
    plan.positions(PARAMS | dict(other=3.0), {})
    assert len(calls) == 1
    plan.positions(PARAMS | dict(lens_sie_q=0.8), {})
    plan.positions(PARAMS | dict(lens_sie_q=0.8, obs_dx=0.1), {})
    assert len(calls) == 3


def test_the_magnification_of_an_isothermal_sphere(driver):
    # 1 / (1 - theta_e / r), along the radius 1, of a source at the
    # redshift its Einstein radius is quoted for
    params = PARAMS | dict(
        lens_sie_q=1.0, lens_sie_xpos=0.0, lens_sie_ypos=0.0)
    plan = trace_plan(driver, reference=params)
    plan.positions(params, {})
    x, y = sky_positions(GRID)
    r = np.hypot(x, y)
    inner = (slice(2, -2), slice(2, -2))
    expected = 1 / (1 - 1.1 / r)
    # (by differences between pixels, outside the ring, where they are
    # small against the radius)
    smooth = r[inner] > 1.6
    np.testing.assert_allclose(
        plan.magnification(1)[inner][smooth], expected[inner][smooth],
        rtol=0.05)


def test_the_automatic_step_resolves_the_stretch(driver):
    # The pixels of the image over the stretch of their rays (its high
    # quantile, see source_step)
    tracer = sie_tracer()
    plan = trace_plan(driver, tracer=tracer)
    x, y = sky_positions(GRID)
    jacobian = tracer.jacobian(
        1, SkyPoints(x, y, ORIGIN), [LENS], 0.1, driver)
    expected = source_step(0.2, stretch(jacobian))
    assert 0.2 / MAX_STRETCH <= expected < 0.2
    assert plan.grid(1).coords.step[0] == pytest.approx(expected)
    given = trace_plan(driver, options=SourceGrid(step=0.05))
    assert given.grid(1).coords.step[0] == 0.05


def test_maps_must_cover_the_footprint_at_the_reference_values(driver):
    small = LensDeflectionMap(
        np.zeros((3, 3)), np.zeros((3, 3)), step=0.2, rval=ORIGIN)
    tracer = tracer_of(Lens('lens', 0.35, small))
    with pytest.raises(ValueError, match="reference values"):
        trace_plan(driver, tracer=tracer, names={})


def test_the_planes_and_footprints_of_a_trace_plan(driver):
    tracer = sie_tracer()
    with pytest.raises(ValueError, match="not bent"):
        TracePlan(
            tracer, [NAMES], [0], GRID, FOOTPRINT, ORIGIN, (0, 0),
            (None, None), Reference(PARAMS, {}), SourceGrid(), driver,
            np.float64)
    with pytest.raises(ValueError, match="shape of the grid"):
        trace_plan(driver, footprint=np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="no pixels"):
        trace_plan(driver, footprint=np.zeros((20, 24), dtype=bool))
