"""
Regression tests for the gmodels and their components: every type of
gmodel, with every type of component, evaluated together with all its
extra outputs (e.g. the velocity field of each component).

The references were made by the code itself, so these tests only catch
changes. Regenerate them (pytest --force-regen) after a deliberate
change of the models.
"""

import numpy as np
import pytest
from gbkfit.model import gmodel_parser, gmodels
from gbkfit.model.gmodels.core import Component


RNODES = list(range(0, 10, 2))

NWMODE = dict(type='relative1', origin=0)


def smdisk(**options):
    return dict(type='smdisk', rnodes=RNODES, **options)


def mcdisk(**options):
    return dict(type='mcdisk', rnodes=RNODES, cflux=1e-3, **options)


# A model of each type of gmodel, and its parameter properties
MODELS = dict(
    intensity_2d=(
        dict(
            dmodel=dict(type='image', size=[20, 16], rota=10),
            gmodel=dict(type='intensity_2d', components=[
                smdisk(
                    loose=True, tilted=True, xpos_nwmode=NWMODE,
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
            dmodel=dict(type='scube', size=[20, 16, 11], step=[1, 1, 30]),
            gmodel=dict(type='kinematics_2d', components=[
                smdisk(
                    loose=True, tilted=False, vsys_nwmode=NWMODE,
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
            dmodel=dict(type='image', size=[20, 16]),
            gmodel=dict(
                type='intensity_3d',
                components=[
                    smdisk(
                        loose=False, tilted=True, posa_nwmode=NWMODE,
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
            ocmp_opt_a=0.05, ocmp_opt_s=4, ocmp_oht_a=1, ocmp_oht_s=1,
            ocmp1_xpos=1, ocmp1_ypos=1, ocmp1_posa=100, ocmp1_incl=30,
            ocmp1_opt_a=0.1, ocmp1_opt_s=2, ocmp1_oht_a=1, ocmp1_oht_s=0.5)),
    kinematics_3d=(
        dict(
            dmodel=dict(type='scube', size=[20, 16, 11], step=[1, 1, 30]),
            gmodel=dict(
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
            ocmp_opt_a=0.05, ocmp_opt_s=4, ocmp_oht_a=1, ocmp_oht_s=1)))


# The velocity and dispersion of each voxel of a Monte Carlo disk are
# those of the last cloud added to it, which depends on the order of the
# threads, so they are not reproducible
NOT_REPRODUCIBLE = (
    'model0_gmodel_component1_vdata',
    'model0_gmodel_component1_ddata')


@pytest.mark.parametrize('name', MODELS)
def test_gmodel(driver, name, evaluate_models, ndarrays_regression):
    model, properties = MODELS[name]
    model = dict(model, driver=dict(type=driver.type()))
    data, extra = evaluate_models([model], properties)
    outputs = {
        f'data_{key}': data_[key]['d']
        for data_ in data for key in data_} | extra
    if name == 'kinematics_3d':
        for key in NOT_REPRODUCIBLE:
            del outputs[key]
    # Thick disks and Monte Carlo disks are not bitwise reproducible,
    # because the threads add to the outputs in a different order
    ndarrays_regression.check(outputs, tolerances={
        key: dict(rtol=1e-5, atol=1e-6 * np.nanmax(np.abs(value)))
        for key, value in outputs.items()})


@pytest.mark.parametrize('name', MODELS)
def test_gmodel_dump_and_load(driver, name, evaluate_models):
    # A dumped gmodel loads back to the same gmodel
    from gbkfit.model import gmodel_parser
    model, properties = MODELS[name]
    model = dict(model, driver=dict(type=driver.type()))
    info = gmodel_parser.dump(gmodel_parser.load(model['gmodel']))
    assert gmodel_parser.dump(gmodel_parser.load(info)) == info
    data, _ = evaluate_models([model], properties)
    data_loaded, _ = evaluate_models([dict(model, gmodel=info)], properties)
    for key, value in data[0].items():
        np.testing.assert_allclose(
            data_loaded[0][key]['d'], value['d'],
            rtol=1e-5, atol=1e-6 * np.abs(value['d']).max())


class WeightComponent(Component):
    """
    A component that sets the spatial weights to 2, except those of the
    first row of pixels, which it sets to 0.
    """

    @staticmethod
    def type():
        return 'weights'

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return {}

    def pdescs(self):
        return {}

    def has_weights(self):
        return True

    def evaluate(self, driver, params, grid, outputs, dtype, out_extra):
        wdata = np.full(outputs['wdata'].shape, 2, dtype)
        wdata[..., 0, :] = 0
        driver.mem_copy_h2d(wdata, outputs['wdata'])


# The 2d gmodels, the method that evaluates them, and the size of their
# data and its shape (an image is a cube with one channel)
GMODELS_2D = [
    (gmodels.GModelIntensity2D, 'evaluate_image', (20, 16), (1, 16, 20)),
    (gmodels.GModelKinematics2D, 'evaluate_scube', (20, 16, 11), (11, 16, 20))]


@pytest.mark.parametrize('gmodel_type, method, size, shape', GMODELS_2D)
def test_2d_gmodel_weights_the_data(driver, gmodel_type, method, size, shape):
    # The components of a 2d gmodel write their weights to its spatial
    # weights, which then become the weights of the data: 0 where they
    # are 0, and 1 elsewhere (normalised to their maximum along z)
    gmodel = gmodel_type([WeightComponent()])
    data = driver.mem_alloc_d(shape, np.float32)
    weights = driver.mem_alloc_d(shape, np.float32)
    driver.mem_fill(data, 0)
    driver.mem_fill(weights, 1)
    getattr(gmodel, method)(
        driver, {}, data, weights,
        size, (1,) * len(size), (0,) * len(size), 0, np.float32, None)
    desired = np.ones(shape)
    desired[:, 0, :] = 0
    np.testing.assert_array_equal(driver.mem_copy_d2h(weights), desired)


def test_3d_gmodel_picks_the_z_axis_of_each_grid(driver):
    # Without a configured z axis, a 3d gmodel picks one from the x and y
    # axes of the data. An edge-on thick disk is cut off by a z axis that
    # is too short, so a gmodel evaluated on a small grid and then on a
    # large one must give the same as one evaluated on the large one only.
    info = dict(type='intensity_3d', components=smdisk(
        loose=False, tilted=False,
        bptraits=dict(type='exponential'), bhtraits=dict(type='sech2')))
    params = dict(
        xpos=0, ypos=0, posa=0, incl=90, bpt_a=1, bpt_s=4, bht_s=3)

    def evaluate(gmodel, size):
        image = driver.mem_alloc_d(size[::-1], np.float32)
        driver.mem_fill(image, 0)
        zero = tuple(-(n / 2 - 0.5) for n in size)
        gmodel.evaluate_image(
            driver, params, image, None, size, (1, 1), zero, 0,
            np.float32, None)
        return driver.mem_copy_d2h(image)

    gmodel = gmodel_parser.load(info)
    evaluate(gmodel, (6, 6))
    np.testing.assert_allclose(
        evaluate(gmodel, (32, 32)),
        evaluate(gmodel_parser.load(info), (32, 32)), rtol=1e-5)

