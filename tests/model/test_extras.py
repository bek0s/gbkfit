"""
Tests for the extra outputs of the models: those on a grid have its world
coordinates, in the layout of the data of their observable.
"""

import numpy as np
import pytest

from gbkfit.utils import gridutils


DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 8)),
    bptraits=dict(type='exponential'))
KINEMATICS = dict(
    vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform'))
PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=45, bpt_a=1, bpt_s=3, vsys=0,
    vpt_rt=2, vpt_vt=100, dpt_a=20)

# A case of each observable with a PSF (so that the high-res grid is padded):
# the observable, its model, and the shape and spectral axis of its low-res
# extra outputs
CASES = dict(
    pixel_brightness=(
        dict(type='pixel_brightness', size=[16, 12], rval=[150, 2],
             scale=[2, 2], psf=dict(type='gauss', sigma=1)),
        dict(type='intensity_2d', components=[DISK]),
        (12, 16), None),
    pixel_spectra=(
        dict(type='pixel_spectra', size=[16, 12, 20], step=[1, 1, 10],
             rval=[150, 2, 0], psf=dict(type='gauss', sigma=1)),
        dict(type='kinematics_2d', components=[DISK | KINEMATICS]),
        (20, 12, 16), 2),
    slit_spectra=(
        dict(type='slit_spectra', size=[16, 20], step=[1, 10],
             psf=dict(type='gauss', sigma=1)),
        dict(type='kinematics_2d', components=[DISK | KINEMATICS]),
        (20, 16), 1))


@pytest.mark.parametrize('name', CASES)
def test_dcube_extras_have_the_layout_of_the_data(evaluate_cases, name):
    observable, model, shape, spectral_axis = CASES[name]
    case = dict(driver=dict(type='host'), observable=observable, model=model)
    properties = {
        k: v for k, v in PROPERTIES.items()
        if name != 'pixel_brightness' or not k.startswith(('v', 'd'))}
    _, extra = evaluate_cases([case], properties)
    dcube_lo = extra['observation0_dcube_lo']
    assert isinstance(dcube_lo, gridutils.GridData)
    assert dcube_lo.data.shape == shape
    assert dcube_lo.spectral_axis == spectral_axis
    assert len(dcube_lo.coords.step) == len(shape)
    # The high-res grid is on the sky, except across the slit
    dcube_hi = extra['observation0_dcube_hi']
    if name == 'slit_spectra':
        assert isinstance(dcube_hi, np.ndarray)
    else:
        assert isinstance(dcube_hi, gridutils.GridData)
        assert dcube_hi.spectral_axis == spectral_axis
    # The PSF is an array of offsets, not on the grid
    assert isinstance(extra['observation0_psf_hi'], np.ndarray)


def test_dcube_extras_have_the_coordinates_of_the_observable(evaluate_cases):
    observable, model, _, _ = CASES['pixel_spectra']
    case = dict(driver=dict(type='host'), observable=observable, model=model)
    _, extra = evaluate_cases([case], PROPERTIES)
    coords = extra['observation0_dcube_lo'].coords
    assert coords.step == (1, 1, 10)
    assert coords.rpix == (7.5, 5.5, 9.5)
    assert coords.rval == (150, 2, 0)


@pytest.mark.parametrize('name', CASES)
def test_model_extras_are_on_the_sky(evaluate_cases, name):
    # The extra outputs of a 2d model are images on the high-res grid,
    # except for a long slit, which has no position on the sky
    observable, model, _, _ = CASES[name]
    case = dict(driver=dict(type='host'), observable=observable, model=model)
    properties = {
        k: v for k, v in PROPERTIES.items()
        if name != 'pixel_brightness' or not k.startswith(('v', 'd'))}
    _, extra = evaluate_cases([case], properties)
    bdata = extra['observation0_model_component0_bdata']
    if name == 'slit_spectra':
        assert isinstance(bdata, np.ndarray)
        return
    dcube_hi = extra['observation0_dcube_hi']
    assert isinstance(bdata, gridutils.GridData)
    assert bdata.data.ndim == 2 and bdata.spectral_axis is None
    assert bdata.coords == dcube_hi.coords.axes(0, 1)


def test_3d_model_extras_have_a_line_of_sight_axis(evaluate_cases):
    # The z axis of a 3d model is along the line of sight, centred on 0
    observable, _, _, _ = CASES['pixel_spectra']
    model = dict(
        type='kinematics_3d', size_z=10, step_z=0.5, components=[
            DISK | KINEMATICS | dict(bhtraits=dict(type='sech2'))])
    case = dict(driver=dict(type='host'), observable=observable, model=model)
    _, extra = evaluate_cases([case], PROPERTIES | dict(bht_s=1))
    bdata = extra['observation0_model_component0_bdata']
    assert bdata.data.shape[0] == 10 and bdata.spectral_axis is None
    sky = extra['observation0_dcube_hi'].coords.axes(0, 1)
    assert bdata.coords.axes(0, 1) == sky
    assert bdata.coords.axes(2) == gridutils.Coords((0.5,), (4.5,), (0.0,), 0)
