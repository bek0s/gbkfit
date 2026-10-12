"""
Tests for the evaluation of galaxies in double precision (float64).
"""

import numpy as np
import pytest

import gbkfit.params
from gbkfit.model import galaxy_parser
from gbkfit.observation import ObservationGroup, observation_parser


KINEMATICS = dict(
    vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform'))
THIN_DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 12)),
    density=dict(ptraits=dict(type='exponential')), kinematics=KINEMATICS,
    emission=dict(type='lines'))
THICK_DISK = THIN_DISK | dict(density=dict(
    ptraits=dict(type='exponential'), htraits=dict(type='sech2')))
PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.2, posa=30, incl=60, pt_a=1, pt_s=2.5,
    vpt_rt=2, vpt_vt=150, dpt_a=20)
GAUSS_PSF = dict(type='gauss', sigma=1.3)
GAUSS_LSF = dict(type='gauss', sigma=17)
VARYING_PSF = dict(type='gauss', sigma=dict(
    velocity=[-500, 500], values=[1, 2]))
FLAT = dict(type='flat')
GRID = dict(type='grid')
# A galaxy, and an observable seen along a line of sight through an
# instrument
CASES = {
    '2d spectra': (
        dict(components=[THIN_DISK]),
        dict(type='pixel_spectra', size=[33, 32, 40], step=[1, 1, 12]),
        FLAT, dict(psf=GAUSS_PSF, lsf=GAUSS_LSF)),
    '3d spectra': (
        dict(components=[THICK_DISK]),
        dict(type='pixel_spectra', size=[33, 32, 40], step=[1, 1, 12]),
        GRID, dict(psf=GAUSS_PSF, lsf=GAUSS_LSF)),
    'moments': (
        dict(components=[THIN_DISK]),
        dict(type='pixel_moments', size=[33, 32], step=[1, 1]),
        FLAT, dict(psf=GAUSS_PSF, lsf=GAUSS_LSF)),
    'varying psf': (
        dict(components=[THIN_DISK]),
        dict(type='pixel_spectra', size=[33, 32, 40], step=[1, 1, 12]),
        FLAT, dict(psf=VARYING_PSF, lsf=GAUSS_LSF))}


def evaluate(driver, case, dtype):
    galaxy, observable, line_of_sight, instrument = CASES[case]
    group = ObservationGroup([galaxy_parser.load(galaxy)], [
        observation_parser.load(dict(
            observable=observable, line_of_sight=line_of_sight,
            instrument=instrument, dtype=dtype),
            drivers={None: driver})])
    # (the disk on a grid is thick: it has a scale height)
    properties = PROPERTIES | (dict(ht_s=0.5) if '3d' in case else {})
    params = gbkfit.params.ParamSpace(
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
