"""
Tests of lensed scenes (gbkfit.lensing.LensedScene): each plane drawn on
its grid, and their light composited back to front on the grid of the
image, through the transmission of the planes in front.
"""

import logging

import numpy as np
import pytest

from gbkfit.cosmology import DEFAULT
from gbkfit.grid import make_grid, sky_positions
from gbkfit.lensing import (
    Lens, LensedPlane, LensedScene, LensSheet, LensSIE, SourceGrid,
    TracePlan, Tracer, plane_layout)
from gbkfit.model import (
    AdditiveScenePlan, ContinuumTracer, DepthAxis, Extras, Reference, Scene,
    SceneEvaluation, SceneState, View, galaxy_parser)


ORIGIN = (150.0, 2.0)

STEP = 0.25

IMAGE = make_grid((16, 12), STEP, rval=ORIGIN)

CUBE = make_grid(
    (16, 12, 3), (STEP, STEP, 20.0), rval=ORIGIN + (0.0,), spectral_axis=2)


def tracer_of(lens):
    planes = plane_layout(
        [('front', 0.3), ('arc', 1.5)], [(lens.name(), lens.redshift())])
    return Tracer(planes, [lens], DEFAULT)


# A lens of no deflection, and a grid of the plane behind it of the
# pixels of the image, two more on each side: its rays are at the centres
# of the pixels of the plane
NO_LENS = Lens('lens', 0.3, LensSheet(), source_redshift=1.5)

ALIGNED = SourceGrid(step=STEP, extent=[-2.375, 2.375, -1.875, 1.875])


def trace_plan(driver, grid, tracer=None, names=None, params=None,
               options=ALIGNED, footprint=None):
    if footprint is None:
        footprint = np.ones(grid.size[1::-1], dtype=bool)
    return TracePlan(
        tracer or tracer_of(NO_LENS),
        [names or dict(sheet_kappa='lens_sheet_kappa')], [1], grid,
        footprint, ORIGIN, (0.0, 0.0), (None, None),
        Reference(params or dict(lens_sheet_kappa=0.0), {}), options,
        driver, np.float64)


# Scenes of stand-ins: planes of uniform light and transmission, which
# show what is composited

class StandInScene:
    def __init__(self, behind):
        self._behind = behind

    def behind(self):
        return self._behind

    def evaluation(self):
        return SceneEvaluation.PLANE

    def has_dust(self):
        return False

    def line_opacity(self):
        return False


class StandInPlan:
    """
    The plan of a scene of uniform light on its grid, at each channel,
    and of a transmission at each of its rows, if given.
    """

    def __init__(self, grid, light, transmission, subchannels):
        self.grid = grid
        self.subchannels = subchannels
        self._light = np.asarray(light, dtype=float)
        self._transmission = None if transmission is None else (
            np.asarray(transmission, dtype=float)[:, None, None]
            * np.ones(grid.size[1::-1]))

    def transmission(self):
        return self._transmission

    def evaluate(self, params, galaxies, data, out_extra):
        data += self._light.reshape(-1, 1, 1)


class StandInPlanner:
    """
    A planner of the stand-ins, of each plane in turn (near to far): its
    light and transmission, on the plane of the sky; it keeps the plans
    (made).
    """

    def __init__(self, plans):
        self._plans = plans
        self.made = []

    def depth_axes(self, scene, grid):
        return ()

    def plan(self, scene, grid, axes, subchannels):
        light, transmission = self._plans[len(self.made)]
        self.made.append(StandInPlan(grid, light, transmission, subchannels))
        return self.made[-1]


def stand_in_planner(plans):
    planner = StandInPlanner(plans)
    return planner, planner.made


def composite(scene, planner, plan, grid):
    from gbkfit.driver.drivers.host import DriverHost
    driver = DriverHost()
    lensed = scene.plan(planner, plan(driver, grid), driver, grid, np.float64)
    channels = 1 if grid.spectral_axis is None else grid.size[2]
    cube = np.zeros((channels,) + grid.size[1::-1])
    lensed.evaluate(
        dict(lens_sheet_kappa=0.0), SceneState((None, None), {}), cube, None)
    return cube


def stand_in_scene():
    return LensedScene([
        LensedPlane(StandInScene(True), (0,), None),
        LensedPlane(StandInScene(False), (1,), 1)])


def test_the_light_of_a_plane_behind_adds_to_that_in_front():
    planner, _ = stand_in_planner([([1.0], None), ([2.0], None)])
    cube = composite(stand_in_scene(), planner, trace_plan, IMAGE)
    np.testing.assert_allclose(cube, 3)


def test_the_light_from_behind_passes_the_transmission_in_front():
    planner, _ = stand_in_planner([([1.0], [0.25]), ([2.0], None)])
    cube = composite(stand_in_scene(), planner, trace_plan, IMAGE)
    np.testing.assert_allclose(cube, 1 + 0.25 * 2)


def test_pixels_off_the_footprint_see_no_light_from_behind():
    footprint = np.ones(IMAGE.size[::-1], dtype=bool)
    footprint[:, :4] = False
    planner, _ = stand_in_planner([([1.0], None), ([2.0], None)])
    cube = composite(
        stand_in_scene(), planner,
        lambda driver, grid: trace_plan(driver, grid, footprint=footprint),
        IMAGE)
    np.testing.assert_allclose(cube[:, :, :4], 1)
    np.testing.assert_allclose(cube[:, :, 4:], 3)


def test_planes_behind_line_opacity_are_drawn_on_its_sub_channels():
    # A plane in front of 3 channels whose transmission is of 2
    # sub-channels each: the plane behind is drawn on 6 channels of half
    # the width (its march at one sub-channel of each), and its light
    # passes each sub-channel's transmission, averaged over each channel
    front = ([1.0, 1.0, 1.0], [0.2, 0.6, 1.0, 0.0, 0.5, 0.5])
    behind = ([1.0, 3.0, 2.0, 2.0, 4.0, 0.0], None)
    planner, made = stand_in_planner([front, behind])
    cube = composite(stand_in_scene(), planner, trace_plan, CUBE)
    expected = 1 + np.array([
        (0.2 * 1 + 0.6 * 3) / 2, (1.0 * 2 + 0.0 * 2) / 2,
        (0.5 * 4 + 0.5 * 0) / 2])
    np.testing.assert_allclose(
        cube, expected[:, None, None] * np.ones((12, 16)))
    assert [plan.subchannels for plan in made] == [None, 1]
    fine = made[1].grid
    assert fine.size[2] == 6
    assert fine.coords.step[2] == pytest.approx(10.0)
    # (the sub-channels of each channel about its centre)
    zero = CUBE.zero()[2]
    np.testing.assert_allclose(
        fine.zero()[2] + fine.coords.step[2] * np.arange(6),
        zero + 20 * np.array([-0.25, 0.25, 0.75, 1.25, 1.75, 2.25]))


def test_the_planes_of_a_lensed_scene():
    with pytest.raises(ValueError, match="behind a lens"):
        LensedScene([LensedPlane(StandInScene(False), (0,), None)])
    with pytest.raises(ValueError, match="must have light from behind"):
        LensedScene([
            LensedPlane(StandInScene(False), (0,), None),
            LensedPlane(StandInScene(False), (1,), 1)])
    with pytest.raises(ValueError, match="must not have light"):
        LensedScene([
            LensedPlane(StandInScene(True), (0,), None),
            LensedPlane(StandInScene(True), (1,), 1)])
    with pytest.raises(ValueError, match="first"):
        LensedScene([
            LensedPlane(StandInScene(True), (1,), 1),
            LensedPlane(StandInScene(False), (0,), None)])


# Scenes of galaxies, planned as observations plan them

def disk(name, redshift, light='exponential'):
    return galaxy_parser.load(dict(
        name=name, redshift=redshift, components=[dict(
            type='thin_disk', loose=False, tilted=False,
            rnodes=[0, 0.5, 1, 1.5, 2, 2.5, 3],
            rptraits=dict(type=light), light=dict(type='continuum'))]))


GALAXIES = [disk('front', 0.3), disk('arc', 1.5, 'gauss')]

VALUES = [
    dict(xpos=0.3, ypos=-0.2, posa=40, incl=50, rpt_a=1, rpt_s=0.8),
    dict(xpos=0.0, ypos=0.0, posa=10, incl=30, rpt_a=2, rpt_s=0.3)]

# (and the lens of no deflection)
PARAMS = {
    f'{galaxy.name()}_{name}': value
    for galaxy, values in zip(GALAXIES, VALUES)
    for name, value in values.items()} | dict(lens_sheet_kappa=0.0)


def scene_of(indices, behind):
    galaxies = [GALAXIES[i] for i in indices]
    return Scene(
        galaxies,
        [{name: f'{galaxy.name()}_{name}' for name in galaxy.pdescs()}
         for galaxy in galaxies],
        [galaxy.name() for galaxy in galaxies],
        [(0,)] * len(galaxies), View(False, ContinuumTracer()), behind=behind)


def plan_scene(scene, driver, grid):
    # (thin disks without opacity are drawn on the plane of the sky)
    assert scene.evaluation() is SceneEvaluation.PLANE
    return AdditiveScenePlan(scene, driver, grid, None, np.float64, None)


class Planner:
    """The planner of scenes on the plane of the sky (see plan_scene)."""

    def __init__(self, driver):
        self._driver = driver

    def depth_axes(self, scene, grid):
        return ()

    def plan(self, scene, grid, axes, subchannels):
        return plan_scene(scene, self._driver, grid)


def planner_of(driver):
    return Planner(driver)


def states(driver):
    return SceneState(tuple(
        galaxy.plan_stage(ORIGIN, driver).evaluate(values)
        for galaxy, values in zip(GALAXIES, VALUES)), {})


def evaluate(plan, driver, extras=None, params=PARAMS):
    cube = driver.mem_alloc_d((1,) + IMAGE.size[::-1], np.float64)
    driver.mem_fill(cube, 0)
    plan.evaluate(params, states(driver), cube, extras)
    return driver.mem_copy_d2h(cube)


def test_no_deflection_is_no_lens(driver):
    # The galaxy behind a lens of no deflection, on a grid whose pixels
    # are those of the image, is as drawn on the image
    lensed = LensedScene([
        LensedPlane(scene_of([0], True), (0,), None),
        LensedPlane(scene_of([1], False), (1,), 1)])
    plan = lensed.plan(
        planner_of(driver), trace_plan(driver, IMAGE), driver, IMAGE,
        np.float64)
    direct = plan_scene(scene_of([0, 1], False), driver, IMAGE)
    image = evaluate(plan, driver)
    assert image.max() > 0.1
    np.testing.assert_allclose(
        image, evaluate(direct, driver), rtol=1e-12, atol=1e-14)


def test_an_isothermal_sphere_makes_an_einstein_ring(driver, caplog):
    # A compact source behind the centre of the lens: a ring of radius
    # theta_e, with the extra outputs of its plane
    lens = Lens('lens', 0.3, LensSIE(), source_redshift=1.5)
    names = {name: f'lens_{name}' for name in lens.pdescs()}
    params = dict(
        lens_sie_theta_e=1.2, lens_sie_q=1.0, lens_sie_posa=0.0,
        lens_sie_xpos=0.0, lens_sie_ypos=0.0)
    trace = trace_plan(
        driver, IMAGE, tracer_of(lens), names, params, SourceGrid())
    lensed = LensedScene([LensedPlane(scene_of([1], False), (1,), 1)])
    plan = lensed.plan(planner_of(driver), trace, driver, IMAGE, np.float64)
    extras = Extras()
    with caplog.at_level(logging.WARNING):
        image = evaluate(plan, driver, extras, PARAMS | params)[0]
    x, y = sky_positions(IMAGE)
    r = np.hypot(x, y)
    ring = np.abs(r - 1.2) < STEP
    assert image[ring].mean() > 5 * image[r > 1.9].mean()
    assert image[ring].mean() > 5 * image[r < 0.5].mean()
    names = {(extra.name, extra.ids.get('plane')) for extra in extras}
    for name in ('source_cube', 'ray_x', 'ray_y', 'magnification'):
        assert (name, 'plane1') in names
    source = extras.get('source_cube', plane='plane1')
    assert source.coords == trace.grid(1).coords
    # (the galaxy's own extras, on the grid of its plane)
    assert ('bdata', 'plane1') in names
    assert 'border of its grid' not in caplog.text


def test_light_on_the_border_of_a_grid_is_warned_of_once(driver, caplog):
    # A grid that cuts the galaxy behind: warned of without extra outputs
    # (e.g. in a fit), once
    narrow = SourceGrid(step=STEP, extent=[-0.5, 0.5, -0.5, 0.5])
    lensed = LensedScene([
        LensedPlane(scene_of([0], True), (0,), None),
        LensedPlane(scene_of([1], False), (1,), 1)])
    plan = lensed.plan(
        planner_of(driver), trace_plan(driver, IMAGE, options=narrow),
        driver, IMAGE, np.float64)
    with caplog.at_level(logging.WARNING):
        evaluate(plan, driver)
        evaluate(plan, driver, Extras())
    assert caplog.text.count('border of its grid') == 1


def test_the_memory_of_a_plane_is_estimated_and_warned_of(
        driver, caplog, monkeypatch):
    import gbkfit.lensing.lensed_scene as module
    monkeypatch.setattr(module, 'MEMORY_WARNING', 1000)
    lensed = LensedScene([
        LensedPlane(scene_of([0], True), (0,), None),
        LensedPlane(scene_of([1], False), (1,), 1)])
    with caplog.at_level(logging.INFO):
        lensed.plan(
            planner_of(driver), trace_plan(driver, IMAGE), driver, IMAGE,
            np.float64)
    assert 'plane1: a grid of 20 x 16 pixels and 1 channels' in caplog.text
    assert 'coarser step' in caplog.text


class DepthPlanner(Planner):
    """
    A planner of scenes on a depth axis of 37 planes (drawn on the plane
    of the sky, as thin disks are on any).
    """

    def depth_axes(self, scene, grid):
        return (DepthAxis(37, 0.1, -1.8),)


def test_the_memory_of_a_plane_is_that_of_its_depth_axis(driver, caplog):
    # (the axis planned, and the fields its extra outputs would hold)
    lensed = LensedScene([LensedPlane(scene_of([1], False), (1,), 1)])
    with caplog.at_level(logging.INFO):
        lensed.plan(
            DepthPlanner(driver), trace_plan(driver, IMAGE), driver, IMAGE,
            np.float64)
    fields = scene_of([1], False).extra_fields()
    assert 'plane1: a grid of 20 x 16 pixels' in caplog.text
    assert 'on its depth axis of 37 planes;' in caplog.text
    assert f'hold up to {fields} fields' in caplog.text


def test_the_extra_outputs_of_a_scene_hold_fields_of_each_component():
    from gbkfit.model._component_plans import galaxy_extra_fields
    one = scene_of([1], False).extra_fields()
    assert one == galaxy_extra_fields(1)
    assert scene_of([0, 1], False).extra_fields() == 2 * one
    # (a galaxy of several also has their totals)
    assert galaxy_extra_fields(2) > 2 * one


def test_observations_plan_planes_on_their_depth_axis():
    from gbkfit.observation.depth import Depth, depth_axes
    galaxy = galaxy_parser.load(dict(
        name='arc', redshift=1.5, components=[dict(
            type='thick_disk', loose=False, tilted=False,
            rnodes=[0, 1, 2, 3], rptraits=dict(type='exponential'),
            rhtraits=dict(type='sech2', zmax=1),
            light=dict(type='continuum'))]))
    thick = Scene(
        [galaxy], [{name: f'arc_{name}' for name in galaxy.pdescs()}],
        ['arc'], [(0,)], View(False, ContinuumTracer()))
    (axis,) = depth_axes(thick, Depth(extent=3.2, step=0.02), IMAGE)
    assert (axis.size, axis.step) == (160, 0.02)
    assert depth_axes(scene_of([1], False), Depth(), IMAGE) == ()
