"""
Regression tests for the models and their components: every type of
model, with every type of component, evaluated together with all its
extra outputs (e.g. the velocity field of each component).

The references were made by the code itself, so these tests only catch
changes. Regenerate them (pytest --force-regen) after a deliberate
change of the models.
"""

import numpy as np
import pytest
import gbkfit.model
from gbkfit.model import model_parser
from gbkfit.params import ParamModeOffsets
from gbkfit.utils import gridutils
from modelutils import WeightComponent


RNODES = list(range(0, 10, 2))


def smdisk(**options):
    return dict(type='smdisk', rnodes=RNODES, **options)


def mcdisk(**options):
    return dict(type='mcdisk', rnodes=RNODES, cflux=1e-3, **options)


# A model of each type of model, and its parameter properties
CASES = dict(
    intensity_2d=(
        dict(
            observable=dict(type='pixel_brightness', size=[20, 16], rota=10),
            model=dict(type='intensity_2d', components=[
                smdisk(
                    loose=True, tilted=True,
                    bptraits=[
                        dict(type='exponential'),
                        dict(type='nw_uniform')],
                    sptraits=dict(type='azrange'))])),
        dict(
            xpos=[0.5, 0.1, 0.2, 0.3, 0.4], ypos=[-1, -1, -1.2, -1.4, -1.6],
            posa=[30, 32, 34, 36, 38], incl=[50, 52, 54, 56, 58],
            bpt_a=1, bpt_s=3, bpt1_a=[0.5, 0.4, 0.3, 0.2, 0.1],
            spt_p=45, spt_s=270)),
    kinematics_2d=(
        dict(
            observable=dict(
                type='pixel_spectra', size=[20, 16, 11], step=[1, 1, 30]),
            model=dict(type='kinematics_2d', components=[
                smdisk(
                    loose=True, tilted=False,
                    bptraits=dict(type='gauss'),
                    vptraits=[
                        dict(type='tan_arctan'),
                        dict(type='nw_rad_uniform')],
                    dptraits=dict(type='uniform')),
                smdisk(
                    loose=False, tilted=True,
                    bptraits=dict(type='exponential'),
                    vptraits=dict(type='tan_tanh'),
                    dptraits=dict(type='nw_uniform'),
                    sptraits=dict(type='nw_azrange'))])),
        dict(
            vsys=[10, 5, 5, 5, 5], xpos=[0, 0.2, 0.4, 0.6, 0.8], ypos=[0] * 5,
            posa=30, incl=60,
            bpt_a=1, bpt_s=3, vpt_rt=2, vpt_vt=150,
            vpt1_vr=[0, 10, 20, 20, 10], dpt_a=25,
            cmp1_vsys=0, cmp1_xpos=1, cmp1_ypos=-1,
            cmp1_posa=[120, 122, 124, 126, 128],
            cmp1_incl=[40, 45, 50, 55, 60],
            cmp1_bpt_a=0.5, cmp1_bpt_s=4, cmp1_vpt_rt=3, cmp1_vpt_vt=100,
            cmp1_dpt_a=[40, 35, 30, 25, 20],
            cmp1_spt_p=[0, 10, 20, 30, 40], cmp1_spt_s=[300] * 5)),
    intensity_3d=(
        dict(
            observable=dict(type='pixel_brightness', size=[20, 16]),
            model=dict(
                type='intensity_3d',
                components=[
                    smdisk(
                        loose=False, tilted=True,
                        bptraits=dict(type='exponential'),
                        bhtraits=dict(type='sech2'),
                        zptraits=dict(type='nw_uniform')),
                    mcdisk(
                        loose=False, tilted=False, seed=3,
                        bptraits=dict(type='gauss'),
                        bhtraits=dict(type='exponential'))],
                opacity_components=[
                    smdisk(
                        loose=False, tilted=False,
                        optraits=dict(type='exponential'),
                        ohtraits=dict(type='sech2')),
                    mcdisk(
                        loose=False, tilted=False,
                        optraits=dict(type='gauss'),
                        ohtraits=dict(type='gauss'))])),
        dict(
            xpos=0, ypos=0, posa=[30, 32, 34, 36, 38], incl=[60] * 5,
            bpt_a=1, bpt_s=3, bht_s=1, zpt_a=[0, 0.2, 0.4, 0.6, 0.8],
            cmp1_xpos=1, cmp1_ypos=1, cmp1_posa=100, cmp1_incl=30,
            cmp1_bpt_a=2, cmp1_bpt_s=2, cmp1_bht_s=0.5,
            ocmp_xpos=0, ocmp_ypos=0, ocmp_posa=30, ocmp_incl=60,
            ocmp_opt_a=0.05, ocmp_opt_s=4, ocmp_oht_s=1,
            ocmp1_xpos=1, ocmp1_ypos=1, ocmp1_posa=100, ocmp1_incl=30,
            ocmp1_opt_a=0.1, ocmp1_opt_s=2, ocmp1_oht_s=0.5)),
    kinematics_3d=(
        dict(
            observable=dict(
                type='pixel_spectra', size=[20, 16, 11], step=[1, 1, 30]),
            model=dict(
                type='kinematics_3d', size_z=14, step_z=1.5, zero_z=-9,
                components=[
                    smdisk(
                        loose=True, tilted=True,
                        bptraits=dict(type='exponential'),
                        bhtraits=dict(type='sech2'),
                        vptraits=dict(type='tan_arctan'),
                        vhtraits=dict(type='one'),
                        dptraits=dict(type='uniform'),
                        zptraits=dict(type='nw_harmonic', order=1)),
                    mcdisk(
                        loose=False, tilted=False,
                        bptraits=dict(type='exponential'),
                        bhtraits=dict(type='gauss'),
                        vptraits=dict(type='tan_tanh'),
                        dptraits=dict(type='gauss'),
                        sptraits=dict(type='azrange'))],
                opacity_components=[
                    smdisk(
                        loose=False, tilted=False,
                        optraits=dict(type='exponential'),
                        ohtraits=dict(type='sech2'))])),
        dict(
            vsys=[0, 5, 10, 15, 20], xpos=[0, 0.2, 0.4, 0.6, 0.8],
            ypos=[0] * 5, posa=[30, 32, 34, 36, 38],
            incl=[60, 61, 62, 63, 64],
            bpt_a=1, bpt_s=3, bht_s=1, vpt_rt=2, vpt_vt=150, dpt_a=25,
            zpt_a=[0, 0.2, 0.4, 0.6, 0.8], zpt_p=[0, 10, 20, 30, 40],
            cmp1_vsys=-10, cmp1_xpos=1, cmp1_ypos=-1, cmp1_posa=200,
            cmp1_incl=45, cmp1_bpt_a=0.5, cmp1_bpt_s=3, cmp1_bht_s=1,
            cmp1_vpt_rt=3, cmp1_vpt_vt=100, cmp1_dpt_a=30, cmp1_dpt_s=5,
            cmp1_spt_p=90, cmp1_spt_s=270,
            ocmp_xpos=0, ocmp_ypos=0, ocmp_posa=30, ocmp_incl=60,
            ocmp_opt_a=0.05, ocmp_opt_s=4, ocmp_oht_s=1)))

# The parameters of the cases given as offsets from their first element
MODES = dict(
    intensity_2d=dict(xpos=ParamModeOffsets()),
    kinematics_2d=dict(vsys=ParamModeOffsets()),
    intensity_3d=dict(posa=ParamModeOffsets()))


@pytest.mark.parametrize('name', CASES)
def test_model(driver, name, evaluate_cases, ndarrays_regression):
    case, properties = CASES[name]
    case = dict(case, driver=dict(type=driver.type()))
    data, extra = evaluate_cases([case], properties, MODES.get(name))
    # The extra outputs on a grid are compared by their data
    outputs = {
        f'data_{key}': data_[key]['d']
        for data_ in data for key in data_} | {
        key: value.data if isinstance(value, gridutils.GridData) else value
        for key, value in extra.items()}
    # Thick disks on cuda and Monte Carlo disks are not bitwise
    # reproducible, because the threads add to the outputs in a different
    # order; their float32 sums of many terms differ by about 1e-5
    ndarrays_regression.check(outputs, tolerances={
        key: dict(rtol=1e-4, atol=1e-6 * np.nanmax(np.abs(value)))
        for key, value in outputs.items()})


@pytest.mark.parametrize('name', CASES)
def test_model_dump_and_load(driver, name, evaluate_cases):
    # A dumped model loads back to the same model
    from gbkfit.model import model_parser
    case, properties = CASES[name]
    case = dict(case, driver=dict(type=driver.type()))
    info = model_parser.dump(model_parser.load(case['model']))
    assert model_parser.dump(model_parser.load(info)) == info
    modes = MODES.get(name)
    data, _ = evaluate_cases([case], properties, modes)
    data_loaded, _ = evaluate_cases(
        [dict(case, model=info)], properties, modes)
    # Thick disks on cuda differ by float32 rounding between runs
    for key, value in data[0].items():
        np.testing.assert_allclose(
            data_loaded[0][key]['d'], value['d'],
            rtol=1e-4, atol=1e-6 * np.abs(value['d']).max())


# The 2d models, the size of their data and its spectral axis, and its
# shape (an image is a cube with one channel)
MODELS_2D = [
    (gbkfit.model.ModelIntensity2D, (20, 16), None, (1, 16, 20)),
    (gbkfit.model.ModelKinematics2D, (20, 16, 11), 2, (11, 16, 20))]


@pytest.mark.parametrize(
    'model_type, size, spectral_axis, shape', MODELS_2D)
def test_2d_model_weights_the_data(
        driver, model_type, size, spectral_axis, shape):
    # The components of a 2d model write their weights to its spatial
    # weights, which then become the weights of the data: 0 where they
    # are 0, and 1 elsewhere (normalised to their maximum along z)
    model = model_type([WeightComponent()])
    data = driver.mem_alloc_d(shape, np.float32)
    weights = driver.mem_alloc_d(shape, np.float32)
    driver.mem_fill(data, 0)
    driver.mem_fill(weights, 1)
    ndim = len(size)
    grid = gridutils.Grid(size, gridutils.Coords(
        (1,) * ndim, (0,) * ndim, (0,) * ndim, 0), spectral_axis)
    model.plan(driver, grid, True, np.float32).evaluate(
        {}, data, weights, None)
    desired = np.ones(shape)
    desired[:, 0, :] = 0
    np.testing.assert_array_equal(driver.mem_copy_d2h(weights), desired)


def test_3d_model_picks_the_z_axis_of_each_grid(driver):
    # Without a configured z axis, a 3d model picks one from the x and y
    # axes of the data. An edge-on thick disk is cut off by a z axis that
    # is too short, so a model evaluated on a small grid and then on a
    # large one must give the same as one evaluated on the large one only.
    info = dict(type='intensity_3d', components=smdisk(
        loose=False, tilted=False,
        bptraits=dict(type='exponential'), bhtraits=dict(type='sech2')))
    params = dict(
        xpos=0, ypos=0, posa=0, incl=90, bpt_a=1, bpt_s=4, bht_s=3)

    def evaluate(model, size):
        image = driver.mem_alloc_d(size[::-1], np.float32)
        driver.mem_fill(image, 0)
        rpix = tuple(n / 2 - 0.5 for n in size)
        grid = gridutils.Grid(
            size, gridutils.Coords((1, 1), rpix, (0, 0), 0), None)
        model.plan(driver, grid, False, np.float32).evaluate(
            params, image, None, None)
        return driver.mem_copy_d2h(image)

    model = model_parser.load(info)
    evaluate(model, (6, 6))
    np.testing.assert_allclose(
        evaluate(model, (32, 32)),
        evaluate(model_parser.load(info), (32, 32)), rtol=1e-5)



def test_one_model_serves_several_grids(driver):
    # A model is a description: each plan owns the memory of its own
    # grid, so plans of one model on different grids, evaluated in turn,
    # give what separate models give
    info = dict(type='intensity_3d', components=smdisk(
        loose=False, tilted=False,
        bptraits=dict(type='exponential'), bhtraits=dict(type='sech2')))
    params = dict(
        xpos=0, ypos=0, posa=30, incl=60, bpt_a=1, bpt_s=4, bht_s=1)

    def plan(model, size):
        rpix = tuple(n / 2 - 0.5 for n in size)
        grid = gridutils.Grid(
            size, gridutils.Coords((1, 1), rpix, (0, 0), 0), None)
        return model.plan(driver, grid, False, np.float32)

    def evaluate(plan_, size):
        image = driver.mem_alloc_d(size[::-1], np.float32)
        driver.mem_fill(image, 0)
        plan_.evaluate(params, image, None, None)
        return driver.mem_copy_d2h(image)

    shared = model_parser.load(info)
    small, large = plan(shared, (12, 12)), plan(shared, (32, 32))
    results = [evaluate(small, (12, 12)), evaluate(large, (32, 32)),
               evaluate(small, (12, 12))]
    np.testing.assert_allclose(
        results[0], evaluate(plan(model_parser.load(info), (12, 12)),
                             (12, 12)), rtol=1e-5)
    np.testing.assert_allclose(
        results[1], evaluate(plan(model_parser.load(info), (32, 32)),
                             (32, 32)), rtol=1e-5)
    np.testing.assert_allclose(results[2], results[0], rtol=1e-5)
