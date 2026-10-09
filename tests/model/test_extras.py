"""
Tests for the extra outputs of the models: those on a grid have its world
coordinates, in the layout of the data of their dmodel.
"""

import numpy as np
import pytest

from gbkfit.utils import fitsutils


DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 8)),
    bptraits=dict(type='exponential'))
KINEMATICS = dict(
    vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform'))
PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=45, bpt_a=1, bpt_s=3, vsys=0,
    vpt_rt=2, vpt_vt=100, dpt_a=20)

# A model of each dmodel with a PSF (so that the high-res grid is padded):
# the dmodel, its gmodel, and the shape and spectral axis of its low-res
# extra outputs
MODELS = dict(
    image=(
        dict(type='image', size=[16, 12], rval=[150, 2], scale=[2, 2],
             psf=dict(type='gauss', sigma=1)),
        dict(type='intensity_2d', components=[DISK]),
        (12, 16), None),
    scube=(
        dict(type='scube', size=[16, 12, 20], step=[1, 1, 10],
             rval=[150, 2, 0], psf=dict(type='gauss', sigma=1)),
        dict(type='kinematics_2d', components=[DISK | KINEMATICS]),
        (20, 12, 16), 2),
    lslit=(
        dict(type='lslit', size=[16, 20], step=[1, 10],
             psf=dict(type='gauss', sigma=1)),
        dict(type='kinematics_2d', components=[DISK | KINEMATICS]),
        (20, 16), 1))


@pytest.mark.parametrize('name', MODELS)
def test_dcube_extras_have_the_layout_of_the_data(evaluate_models, name):
    dmodel, gmodel, shape, spectral_axis = MODELS[name]
    model = dict(driver=dict(type='host'), dmodel=dmodel, gmodel=gmodel)
    properties = {
        k: v for k, v in PROPERTIES.items()
        if name != 'image' or not k.startswith(('v', 'd'))}
    _, extra = evaluate_models([model], properties)
    dcube_lo = extra['model0_dmodel_dcube_lo']
    assert isinstance(dcube_lo, fitsutils.GridData)
    assert dcube_lo.data.shape == shape
    assert dcube_lo.spectral_axis == spectral_axis
    assert len(dcube_lo.coords.step) == len(shape)
    # The high-res grid is on the sky, except across the slit
    dcube_hi = extra['model0_dmodel_dcube_hi']
    if name == 'lslit':
        assert isinstance(dcube_hi, np.ndarray)
    else:
        assert isinstance(dcube_hi, fitsutils.GridData)
        assert dcube_hi.spectral_axis == spectral_axis
    # The PSF is an array of offsets, not on the grid
    assert isinstance(extra['model0_dmodel_psf_hi'], np.ndarray)


def test_dcube_extras_have_the_coordinates_of_the_dmodel(evaluate_models):
    dmodel, gmodel, _, _ = MODELS['scube']
    model = dict(driver=dict(type='host'), dmodel=dmodel, gmodel=gmodel)
    _, extra = evaluate_models([model], PROPERTIES)
    coords = extra['model0_dmodel_dcube_lo'].coords
    assert coords.step == (1, 1, 10)
    assert coords.rpix == (7.5, 5.5, 9.5)
    assert coords.rval == (150, 2, 0)


@pytest.mark.parametrize('name', MODELS)
def test_gmodel_extras_are_on_the_sky(evaluate_models, name):
    # The extra outputs of a 2d gmodel are images on the high-res grid,
    # except for a long slit, which has no position on the sky
    dmodel, gmodel, _, _ = MODELS[name]
    model = dict(driver=dict(type='host'), dmodel=dmodel, gmodel=gmodel)
    properties = {
        k: v for k, v in PROPERTIES.items()
        if name != 'image' or not k.startswith(('v', 'd'))}
    _, extra = evaluate_models([model], properties)
    bdata = extra['model0_gmodel_component0_bdata']
    if name == 'lslit':
        assert isinstance(bdata, np.ndarray)
        return
    dcube_hi = extra['model0_dmodel_dcube_hi']
    assert isinstance(bdata, fitsutils.GridData)
    assert bdata.data.ndim == 2 and bdata.spectral_axis is None
    assert bdata.coords == dcube_hi.coords.axes(0, 1)


def test_3d_gmodel_extras_have_a_line_of_sight_axis(evaluate_models):
    # The z axis of a 3d gmodel is along the line of sight, centred on 0
    dmodel, _, _, _ = MODELS['scube']
    gmodel = dict(
        type='kinematics_3d', size_z=10, step_z=0.5, components=[
            DISK | KINEMATICS | dict(bhtraits=dict(type='sech2'))])
    model = dict(driver=dict(type='host'), dmodel=dmodel, gmodel=gmodel)
    _, extra = evaluate_models([model], PROPERTIES | dict(bht_s=1))
    bdata = extra['model0_gmodel_component0_bdata']
    assert bdata.data.shape[0] == 10 and bdata.spectral_axis is None
    sky = extra['model0_dmodel_dcube_hi'].coords.axes(0, 1)
    assert bdata.coords.axes(0, 1) == sky
    assert bdata.coords.axes(2) == fitsutils.Coords((0.5,), (4.5,), (0.0,), 0)
