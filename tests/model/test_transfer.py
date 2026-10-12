"""
Tests for radiative transfer along the line of sight (Transfer): the
fields of the light of the emitters, which the march reads.
"""

import numpy as np
import pytest

from gbkfit.model import galaxy_parser
from gbkfit.utils import gridutils


# A spectral axis of velocities (the fields do not depend on it)
SPECTRAL = gridutils.Grid((11,), gridutils.Coords((10,), (5,), (0,), 0), 0)

DISK_PARAMS = dict(
    vsys=0, xpos=0.5, ypos=-1, posa=30, incl=60, pt_a=1, pt_s=3, ht_s=1,
    vpt_rt=2, vpt_vt=150, dpt_a=20)


def disk(type_, thick=True, **options):
    """The configuration of a disk of gas, thick or thin."""
    density = dict(ptraits=dict(type='exponential'))
    if thick:
        density.update(htraits=dict(type='sech2'))
    return dict(
        type=type_, loose=False, tilted=False, rnmin=0, rnmax=10, rnsep=1,
        density=density,
        kinematics=dict(
            vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform')),
        emission=dict(type='lines')) | options


def native_grid(size):
    """A grid of voxels of size 1, centred on 0."""
    return dict(
        spat_size=size, spat_step=(1.0, 1.0, 1.0),
        spat_zero=tuple(-(n / 2 - 0.5) for n in size), spat_rota=0.0,
        spec_size=SPECTRAL.size[0], spec_step=SPECTRAL.coords.step[0],
        spec_zero=SPECTRAL.zero()[0])


def component_plan(driver, component):
    """The plan of the light of a component (of a galaxy of it alone)."""
    galaxy = galaxy_parser.load(dict(components=[component]))
    return galaxy.components()[0].plan(
        driver, SPECTRAL, np.dtype(np.float32), None, None, True, None)


def evaluate_fields(driver, component, params, grid):
    """
    Return the fields of a component on a grid, and its extra outputs.
    """
    plan = component_plan(driver, component)
    fields = driver.mem_alloc_d(
        plan.fields_layout().shape(grid), np.float32)
    extra = {}
    plan.evaluate_fields(params, grid, fields, extra)
    return driver.mem_copy_d2h(fields), extra


@pytest.mark.parametrize('disk_type', ['smdisk', 'mcdisk'])
def test_the_fields_of_a_thick_disk_are_its_light_and_kinematics(
        driver, disk_type):
    options = dict(cflux=1e-3) if disk_type == 'mcdisk' else {}
    (j, v, s, *_), extra = evaluate_fields(
        driver, disk(disk_type, **options), DISK_PARAMS,
        native_grid((24, 24, 24)))
    on = extra['bdata'] != 0
    np.testing.assert_allclose(j, extra['bdata'], rtol=1e-5)
    np.testing.assert_allclose(v[on], extra['vdata'][on], atol=1e-3)
    if disk_type == 'smdisk':
        np.testing.assert_allclose(s[on], extra['ddata'][on], rtol=1e-6)
    else:
        # The line of the clouds of a voxel is as wide as the sum of
        # theirs: the spread of their velocities widens it
        assert np.all(s[on] >= extra['ddata'][on] * (1 - 1e-5))
        assert s[on].mean() > extra['ddata'][on].mean() + 0.1


def test_the_clouds_keep_their_dispersion_at_a_large_systemic_velocity(
        driver):
    # Clouds of one velocity: the line of each voxel is theirs (float32
    # would lose it in the moments of velocities of 3000)
    params = DISK_PARAMS | dict(vsys=3000, vpt_vt=0, dpt_a=10)
    (j, v, s, _), _ = evaluate_fields(
        driver, disk('mcdisk', cflux=1e-3), params, native_grid((24, 24, 24)))
    on = j != 0
    np.testing.assert_allclose(v[on], 3000, rtol=1e-6)
    np.testing.assert_allclose(s[on], 10, rtol=1e-4)


def test_a_thin_disk_is_a_layer_where_its_plane_crosses(driver):
    params = DISK_PARAMS | dict(xpos=0, ypos=0, posa=0)
    grid = native_grid((24, 24, 8))
    component = disk('smdisk', thick=False)
    (j, v, s, position), extra = evaluate_fields(
        driver, component, params, grid)
    image = driver.mem_alloc_d((24, 24), np.float32)
    driver.mem_fill(image, 0)
    component_plan(driver, component).evaluate_emission(
        params, grid, dict(image=image), None)
    np.testing.assert_array_equal(j, driver.mem_copy_d2h(image))
    # The plane of the disk is at the depth |x| tan(incl) from the plane
    # of the sky (x along its minor axis), even beyond the depth axis
    on = j != 0
    x = np.broadcast_to(grid['spat_zero'][0] + np.arange(24), (24, 24))
    depth = grid['spat_zero'][2] + position
    np.testing.assert_allclose(
        np.abs(depth[on]), np.abs(x[on]) * np.tan(np.radians(60)),
        atol=1e-4)
    assert np.abs(depth[on]).max() > 4


def test_a_point_is_a_layer_at_its_depth(driver):
    params = dict(xpos=0.25, ypos=0.5, depth=2, flux=3, vsys=50, disp=-10)
    point = dict(type='point', emission=dict(type='lines'), depth=True)
    grid = native_grid((6, 6, 8))
    (j, v, s, position), _ = evaluate_fields(driver, point, params, grid)
    on = j != 0
    assert on.sum() == 2 and j.sum() == pytest.approx(3)
    np.testing.assert_array_equal(v[on], 50)
    np.testing.assert_array_equal(s[on], 10)
    np.testing.assert_array_equal(position[on], 2 + 3.5)
    np.testing.assert_array_equal(np.stack((v, s, position))[:, ~on], 0)
