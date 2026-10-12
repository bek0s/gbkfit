"""
Tests for the extra outputs of the observations and their galaxies: those
on a grid have its world coordinates, in the layout of the data of their
observable.
"""

import numpy as np
import pytest

from gbkfit.utils import gridutils


DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 8)),
    density=dict(ptraits=dict(type='exponential')),
    emission=dict(type='band'))
LINE_DISK = DISK | dict(
    kinematics=dict(
        vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform')),
    emission=dict(type='lines'))
PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=45, pt_a=1, pt_s=3, vsys=0,
    vpt_rt=2, vpt_vt=100, dpt_a=20)

# A case of each observable with a PSF (so that the high-res grid is padded):
# the observable, its galaxy, and the shape and spectral axis of its low-res
# extra outputs
CASES = dict(
    pixel_brightness=(
        dict(type='pixel_brightness', size=[16, 12], rval=[150, 2],
             scale=[2, 2], psf=dict(type='gauss', sigma=1)),
        dict(components=[DISK]),
        (12, 16), None),
    pixel_spectra=(
        dict(type='pixel_spectra', size=[16, 12, 20], step=[1, 1, 10],
             rval=[150, 2, 0], psf=dict(type='gauss', sigma=1)),
        dict(components=[LINE_DISK]),
        (20, 12, 16), 2),
    slit_spectra=(
        dict(type='slit_spectra', size=[16, 20], step=[1, 10],
             psf=dict(type='gauss', sigma=1)),
        dict(components=[LINE_DISK]),
        (20, 16), 1))


def flat_case(observable, galaxy):
    """A case of a galaxy seen through a flat line of sight."""
    return dict(
        driver=dict(type='host'), observable=observable,
        line_of_sight=dict(type='flat'), galaxy=galaxy)


@pytest.mark.parametrize('name', CASES)
def test_dcube_extras_have_the_layout_of_the_data(evaluate_cases, name):
    observable, galaxy, shape, spectral_axis = CASES[name]
    case = flat_case(observable, galaxy)
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
    observable, galaxy, _, _ = CASES['pixel_spectra']
    case = flat_case(observable, galaxy)
    _, extra = evaluate_cases([case], PROPERTIES)
    coords = extra['observation0_dcube_lo'].coords
    assert coords.step == (1, 1, 10)
    assert coords.rpix == (7.5, 5.5, 9.5)
    assert coords.rval == (150, 2, 0)


@pytest.mark.parametrize('name', CASES)
def test_model_extras_are_on_the_sky(evaluate_cases, name):
    # The extra outputs of a galaxy seen through a flat line of sight are
    # images on the high-res grid, except for a long slit, which has no
    # position on the sky
    observable, galaxy, _, _ = CASES[name]
    case = flat_case(observable, galaxy)
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


def test_grid_extras_have_a_line_of_sight_axis(evaluate_cases):
    # The depth axis of a grid is along the line of sight, centred on 0
    observable, _, _, _ = CASES['pixel_spectra']
    thick_disk = LINE_DISK | dict(density=dict(
        ptraits=dict(type='exponential'), htraits=dict(type='sech2')))
    case = dict(
        driver=dict(type='host'), observable=observable,
        line_of_sight=dict(type='grid', depth=dict(extent=5.0, step=0.5)),
        galaxy=dict(components=[thick_disk]))
    _, extra = evaluate_cases([case], PROPERTIES | dict(ht_s=1))
    bdata = extra['observation0_model_component0_bdata']
    assert bdata.data.shape[0] == 10 and bdata.spectral_axis is None
    sky = extra['observation0_dcube_hi'].coords.axes(0, 1)
    assert bdata.coords.axes(0, 1) == sky
    assert bdata.coords.axes(2) == gridutils.Coords((0.5,), (4.5,), (0.0,), 0)
