"""
Regression tests for the galaxies and their components: every kind of
observation of a galaxy (an image or a spectral cube, with or without a
depth axis), with every type of component, evaluated together with all
its extra outputs (e.g. the velocity field of each component). The cases
keep the names of the model types they replaced: intensity (an image),
kinematics (a spectral cube), 2d (a flat line of sight) and 3d (a grid
with a depth axis).

The references were made by the code itself, so these tests only catch
changes. Regenerate them (pytest --force-regen) after a deliberate
change of the models.
"""

import numpy as np
import pytest
from gbkfit.model import galaxy_parser
from gbkfit.model.components._component_plans import opacity_cubes
from gbkfit.model.components._component_set import IMAGE_SPECTRAL_AXIS
from gbkfit.observation import Depth, Grid
from gbkfit.params import ParamModeOffsets
from gbkfit.utils import gridutils


RNODES = list(range(0, 10, 2))

BAND = dict(type='band')
LINES = dict(type='lines')
DUST = dict(type='dust')


def smdisk(**options):
    return dict(type='disk', rnodes=RNODES, **options)


def mcdisk(seed=0, **options):
    return dict(
        type='disk', rnodes=RNODES, clouds=dict(cflux=1e-3, seed=seed),
        **options)


# A galaxy seen by each kind of observation, and its parameter properties
CASES = dict(
    intensity_2d=(
        dict(
            observable=dict(type='pixel_brightness', size=[20, 16], rota=10),
            line_of_sight=dict(type='flat'),
            galaxy=dict(components=[
                smdisk(
                    loose=True, tilted=True,
                    density=dict(ptraits=[
                        dict(type='exponential'), dict(type='nw_uniform')]),
                    sptraits=dict(type='azrange'),
                    emission=BAND)])),
        dict(
            xpos=[0.5, 0.1, 0.2, 0.3, 0.4], ypos=[-1, -1, -1.2, -1.4, -1.6],
            posa=[30, 32, 34, 36, 38], incl=[50, 52, 54, 56, 58],
            pt_a=1, pt_s=3, pt1_a=[0.5, 0.4, 0.3, 0.2, 0.1],
            spt_p=45, spt_s=270)),
    kinematics_2d=(
        dict(
            observable=dict(
                type='pixel_spectra', size=[20, 16, 11], step=[1, 1, 30]),
            line_of_sight=dict(type='flat'),
            galaxy=dict(components=[
                smdisk(
                    name='disk1', loose=True, tilted=False,
                    density=dict(ptraits=dict(type='gauss')),
                    kinematics=dict(
                        vptraits=[
                            dict(type='arctan', direction='tan'),
                            dict(type='nw_uniform', direction='rad')],
                        dptraits=dict(type='uniform')),
                    emission=LINES),
                smdisk(
                    name='disk2', loose=False, tilted=True,
                    density=dict(ptraits=dict(type='exponential')),
                    kinematics=dict(
                        vptraits=dict(type='tanh', direction='tan'),
                        dptraits=dict(type='nw_uniform')),
                    sptraits=dict(type='nw_azrange'),
                    emission=LINES)])),
        dict(
            disk1_vsys=[10, 5, 5, 5, 5], disk1_xpos=[0, 0.2, 0.4, 0.6, 0.8],
            disk1_ypos=[0] * 5, disk1_posa=30, disk1_incl=60,
            disk1_pt_a=1, disk1_pt_s=3, disk1_vpt_rt=2, disk1_vpt_vt=150,
            disk1_vpt1_vr=[0, 10, 20, 20, 10], disk1_dpt_a=25,
            disk2_vsys=0, disk2_xpos=1, disk2_ypos=-1,
            disk2_posa=[120, 122, 124, 126, 128],
            disk2_incl=[40, 45, 50, 55, 60],
            disk2_pt_a=0.5, disk2_pt_s=4, disk2_vpt_rt=3, disk2_vpt_vt=100,
            disk2_dpt_a=[40, 35, 30, 25, 20],
            disk2_spt_p=[0, 10, 20, 30, 40], disk2_spt_s=[300] * 5)),
    intensity_3d=(
        dict(
            observable=dict(type='pixel_brightness', size=[20, 16]),
            line_of_sight=dict(type='grid'),
            galaxy=dict(components=[
                smdisk(
                    name='smooth', loose=False, tilted=True,
                    density=dict(
                        ptraits=dict(type='exponential'),
                        htraits=dict(type='sech2', zmax=8)),
                    zptraits=dict(type='nw_uniform', zmax=1),
                    emission=BAND),
                mcdisk(
                    name='clouds', loose=False, tilted=False, seed=3,
                    density=dict(
                        ptraits=dict(type='gauss'),
                        htraits=dict(type='exponential', zmax=8)),
                    emission=BAND),
                smdisk(
                    name='dust', loose=False, tilted=False,
                    density=dict(
                        ptraits=dict(type='exponential'),
                        htraits=dict(type='sech2', zmax=8)),
                    opacity=DUST),
                mcdisk(
                    name='dust_clouds', loose=False, tilted=False,
                    density=dict(
                        ptraits=dict(type='gauss'),
                        htraits=dict(type='gauss', zmax=8)),
                    opacity=DUST)])),
        dict(
            smooth_xpos=0, smooth_ypos=0, smooth_posa=[30, 32, 34, 36, 38],
            smooth_incl=[60] * 5, smooth_pt_a=1, smooth_pt_s=3,
            smooth_ht_s=1, smooth_zpt_a=[0, 0.2, 0.4, 0.6, 0.8],
            clouds_xpos=1, clouds_ypos=1, clouds_posa=100, clouds_incl=30,
            clouds_pt_a=2, clouds_pt_s=2, clouds_ht_s=0.5,
            dust_xpos=0, dust_ypos=0, dust_posa=30, dust_incl=60,
            dust_pt_a=0.05, dust_pt_s=4, dust_ht_s=1,
            dust_clouds_xpos=1, dust_clouds_ypos=1, dust_clouds_posa=100,
            dust_clouds_incl=30, dust_clouds_pt_a=0.1, dust_clouds_pt_s=2,
            dust_clouds_ht_s=0.5)),
    kinematics_3d=(
        dict(
            observable=dict(
                type='pixel_spectra', size=[20, 16, 11], step=[1, 1, 30]),
            line_of_sight=dict(
                type='grid', depth=dict(extent=14 * 1.5, step=1.5)),
            galaxy=dict(components=[
                smdisk(
                    name='smooth', loose=True, tilted=True,
                    density=dict(
                        ptraits=dict(type='exponential'),
                        htraits=dict(type='sech2', zmax=8)),
                    kinematics=dict(
                        vptraits=dict(type='arctan', direction='tan'),
                        vhtraits=dict(type='one'),
                        dptraits=dict(type='uniform')),
                    zptraits=dict(type='nw_harmonic', order=1, zmax=1),
                    emission=LINES),
                mcdisk(
                    name='clouds', loose=False, tilted=False,
                    density=dict(
                        ptraits=dict(type='exponential'),
                        htraits=dict(type='gauss', zmax=8)),
                    kinematics=dict(
                        vptraits=dict(type='tanh', direction='tan'),
                        dptraits=dict(type='gauss')),
                    sptraits=dict(type='azrange'),
                    emission=LINES),
                smdisk(
                    name='dust', loose=False, tilted=False,
                    density=dict(
                        ptraits=dict(type='exponential'),
                        htraits=dict(type='sech2', zmax=8)),
                    opacity=DUST)])),
        dict(
            smooth_vsys=[0, 5, 10, 15, 20],
            smooth_xpos=[0, 0.2, 0.4, 0.6, 0.8], smooth_ypos=[0] * 5,
            smooth_posa=[30, 32, 34, 36, 38], smooth_incl=[60, 61, 62, 63, 64],
            smooth_pt_a=1, smooth_pt_s=3, smooth_ht_s=1, smooth_vpt_rt=2,
            smooth_vpt_vt=150, smooth_dpt_a=25,
            smooth_zpt_a=[0, 0.2, 0.4, 0.6, 0.8],
            smooth_zpt_p=[0, 10, 20, 30, 40],
            clouds_vsys=-10, clouds_xpos=1, clouds_ypos=-1, clouds_posa=200,
            clouds_incl=45, clouds_pt_a=0.5, clouds_pt_s=3, clouds_ht_s=1,
            clouds_vpt_rt=3, clouds_vpt_vt=100, clouds_dpt_a=30,
            clouds_dpt_s=5, clouds_spt_p=90, clouds_spt_s=270,
            dust_xpos=0, dust_ypos=0, dust_posa=30, dust_incl=60,
            dust_pt_a=0.05, dust_pt_s=4, dust_ht_s=1)))

# The parameters of the cases given as offsets from their first element
MODES = dict(
    intensity_2d=dict(xpos=ParamModeOffsets()),
    kinematics_2d=dict(disk1_vsys=ParamModeOffsets()),
    intensity_3d=dict(smooth_posa=ParamModeOffsets()))


@pytest.mark.parametrize('name', CASES)
def test_model(driver, name, evaluate_cases, ndarrays_regression):
    case, properties = CASES[name]
    case = dict(case, driver=dict(type=driver.type()))
    data, extra = evaluate_cases([case], properties, MODES.get(name))
    # The extra outputs on a grid are compared by their data, named after
    # their ids and names (e.g. 'observation0.galaxy0.disk1.vdata')
    outputs = {
        f'data_{key}': data_[key]['d']
        for data_ in data for key in data_} | {
        '.'.join([*x.ids.values(), x.name]):
            x.value.data if isinstance(x.value, gridutils.GridData)
            else x.value
        for x in extra}
    # Thick disks on cuda and Monte Carlo disks are not bitwise
    # reproducible, because the threads add to the outputs in a different
    # order; their float32 sums of many terms differ by about 1e-5
    ndarrays_regression.check(outputs, tolerances={
        key: dict(rtol=1e-4, atol=1e-6 * np.nanmax(np.abs(value)))
        for key, value in outputs.items()})


@pytest.mark.parametrize('name', CASES)
def test_galaxy_dump_and_load(driver, name, evaluate_cases):
    # A dumped galaxy loads back to the same galaxy
    case, properties = CASES[name]
    case = dict(case, driver=dict(type=driver.type()))
    info = galaxy_parser.dump(galaxy_parser.load(case['galaxy']))
    assert galaxy_parser.dump(galaxy_parser.load(info)) == info
    modes = MODES.get(name)
    data, _ = evaluate_cases([case], properties, modes)
    data_loaded, _ = evaluate_cases(
        [dict(case, galaxy=info)], properties, modes)
    # Thick disks on cuda differ by float32 rounding between runs
    for key, value in data[0].items():
        np.testing.assert_allclose(
            data_loaded[0][key]['d'], value['d'],
            rtol=1e-4, atol=1e-6 * np.abs(value['d']).max())


THICK_DISK = dict(components=smdisk(
    loose=False, tilted=False,
    density=dict(
        ptraits=dict(type='exponential'),
        htraits=dict(type='sech2', zmax=8)),
    emission=BAND))


def image_plan(driver, galaxy, size):
    """
    The plan of an image of a galaxy (its first component) on a grid of
    the given size, with the depth axis of the legacy extent: that of the
    grid.
    """
    rpix = tuple(n / 2 - 0.5 for n in size)
    grid = gridutils.Grid(size, gridutils.Coords((1, 1), rpix, (0, 0), 0), None)
    return galaxy.plan(
        driver, grid, IMAGE_SPECTRAL_AXIS,
        Grid(Depth(extent='legacy')).depth_axis(grid, 0),
        np.float32, (0,),
        opacity_cubes([galaxy], [(0,)], None, None, np.float32)[0],
        None, None, None, None)


def evaluate_image(driver, plan, size, params):
    image = driver.mem_alloc_d(size[::-1], np.float32)
    driver.mem_fill(image, 0)
    plan.evaluate(params, {}, dict(image=image), None)
    return driver.mem_copy_d2h(image)


def test_the_depth_axis_follows_each_grid(driver):
    # Without a configured extent, the depth axis is that of the x and y
    # axes of the grid. An edge-on thick disk is cut off by a depth axis
    # that is too short, so a galaxy evaluated on a small grid and then on
    # a large one must give the same as one evaluated on the large one
    # only.
    params = dict(xpos=0, ypos=0, posa=0, incl=90, pt_a=1, pt_s=4, ht_s=3)
    galaxy = galaxy_parser.load(THICK_DISK)
    evaluate_image(driver, image_plan(driver, galaxy, (6, 6)), (6, 6), params)
    np.testing.assert_allclose(
        evaluate_image(
            driver, image_plan(driver, galaxy, (32, 32)), (32, 32), params),
        evaluate_image(
            driver, image_plan(driver, galaxy_parser.load(THICK_DISK),
                               (32, 32)), (32, 32), params),
        rtol=1e-5)


def test_one_galaxy_serves_several_grids(driver):
    # A galaxy is a description: each plan owns the memory of its own
    # grid, so plans of one galaxy on different grids, evaluated in turn,
    # give what separate galaxies give
    params = dict(xpos=0, ypos=0, posa=30, incl=60, pt_a=1, pt_s=4, ht_s=1)
    shared = galaxy_parser.load(THICK_DISK)
    small = image_plan(driver, shared, (12, 12))
    large = image_plan(driver, shared, (32, 32))
    results = [
        evaluate_image(driver, small, (12, 12), params),
        evaluate_image(driver, large, (32, 32), params),
        evaluate_image(driver, small, (12, 12), params)]
    alone = [
        evaluate_image(
            driver, image_plan(driver, galaxy_parser.load(THICK_DISK), size),
            size, params)
        for size in ((12, 12), (32, 32))]
    np.testing.assert_allclose(results[0], alone[0], rtol=1e-5)
    np.testing.assert_allclose(results[1], alone[1], rtol=1e-5)
    np.testing.assert_allclose(results[2], results[0], rtol=1e-5)
