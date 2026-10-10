"""
Tests for the models in double precision (float64).
"""

import numpy as np
import pytest

import gbkfit.params
from gbkfit.model import model_parser
from gbkfit.observation import ObservationGroup, observation_parser


DISK = dict(
    loose=False, tilted=False, rnodes=list(range(0, 12)),
    bptraits=dict(type='exponential'), vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))
PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.2, posa=30, incl=60, bpt_a=1, bpt_s=2.5,
    vpt_rt=2, vpt_vt=150, dpt_a=20)
GAUSS_PSF = dict(type='gauss', sigma=1.3)
GAUSS_LSF = dict(type='gauss', sigma=17)
VARYING_PSF = dict(type='gauss', sigma=dict(
    velocity=[-500, 500], values=[1, 2]))
CASES = {
    '2d spectra': (
        dict(type='kinematics_2d', components=[dict(type='smdisk', **DISK)]),
        dict(type='pixel_spectra', size=[33, 32, 40], step=[1, 1, 12]),
        dict(psf=GAUSS_PSF, lsf=GAUSS_LSF)),
    '3d spectra': (
        dict(type='kinematics_3d', components=[dict(
            type='smdisk', bhtraits=dict(type='sech2'), **DISK)]),
        dict(type='pixel_spectra', size=[33, 32, 40], step=[1, 1, 12]),
        dict(psf=GAUSS_PSF, lsf=GAUSS_LSF)),
    'moments': (
        dict(type='kinematics_2d', components=[dict(type='smdisk', **DISK)]),
        dict(type='pixel_moments', size=[33, 32], step=[1, 1]),
        dict(psf=GAUSS_PSF, lsf=GAUSS_LSF)),
    'varying psf': (
        dict(type='kinematics_2d', components=[dict(type='smdisk', **DISK)]),
        dict(type='pixel_spectra', size=[33, 32, 40], step=[1, 1, 12]),
        dict(psf=VARYING_PSF, lsf=GAUSS_LSF))}


def evaluate(driver, case, dtype):
    gmodel, observable, instrument = CASES[case]
    group = ObservationGroup([model_parser.load(gmodel)], [
        observation_parser.load(dict(
            driver=dict(type=driver.type()), observable=observable,
            instrument=instrument, dtype=dtype))])
    # (the 3D disks have a thickness)
    properties = PROPERTIES | (dict(bht_s=0.5) if '3d' in case else {})
    params = gbkfit.params.EvaluationParams(
        group.pdescs(), properties).evaluate()
    return {key: value['d'].copy()
            for key, value in group.model_h(params)[0].items()}


@pytest.mark.parametrize('case', CASES)
def test_float64_models_agree_with_float32_models(driver, case):
    # The moments of the faint wings of a convolved model (below about
    # 1e-4 of its peak) are beyond the precision of float32, so they are
    # compared where there is flux
    single = evaluate(driver, case, 'float32')
    double = evaluate(driver, case, 'float64')
    flux = double.get('moment0')
    for key, value in double.items():
        assert value.dtype == np.float64
        reference = single[key]
        compared = np.isfinite(reference)
        np.testing.assert_array_equal(np.isfinite(value), compared)
        if flux is not None:
            compared &= flux > 0.01 * np.nanmax(flux)
        np.testing.assert_allclose(
            value[compared], reference[compared], rtol=1e-4,
            atol=1e-5 * np.abs(reference[compared]).max())
