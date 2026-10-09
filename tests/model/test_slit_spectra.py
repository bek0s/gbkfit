"""
Tests for the long-slit data model (lslit): a position-velocity image
along a slit, which must match a row of a spectral cube.
"""

import copy
import gbkfit.model
import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group


GMODEL = dict(type='kinematics_2d', components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    bptraits=dict(type='exponential'),
    vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))])

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=20)


def evaluate(driver, dmodel):
    model_group = observation_group([
        dict(driver=dict(type=driver.type()), dmodel=dmodel, gmodel=GMODEL)])
    params = gbkfit.params.EvaluationParams(model_group.pdescs(), PROPERTIES)
    data = model_group.model_h(params.evaluate())[0]
    return next(iter(data.values()))['d'].copy()


@pytest.mark.parametrize('rota', [0, 30])
@pytest.mark.parametrize('psf', [None, dict(type='gauss', sigma=1.5)])
def test_slit_spectra_matches_the_central_row_of_pixel_spectra(driver, psf, rota):
    # The cube has an odd number of rows, so its central row is
    # centred on the slit, which passes through the reference position
    common = dict(rota=rota, psf=psf, lsf=dict(type='gauss', sigma=15))
    scube = evaluate(driver, dict(
        type='pixel_spectra', size=[32, 41, 51], step=[1, 1, 10], **common))
    lslit = evaluate(driver, dict(
        type='slit_spectra', size=[32, 51], step=[1, 10], **common))
    assert lslit.shape == (51, 32)
    np.testing.assert_allclose(
        lslit, scube[:, 20, :], rtol=1e-4, atol=1e-5 * scube.max())


def test_slit_spectra_width_is_the_mean_of_its_rows(driver):
    # A slit of 3 pixels is the mean of the 3 central rows of a cube
    scube = evaluate(driver, dict(
        type='pixel_spectra', size=[32, 41, 51], step=[1, 1, 10]))
    lslit = evaluate(driver, dict(
        type='slit_spectra', size=[32, 51], step=[1, 10], slit_width=3))
    np.testing.assert_allclose(
        lslit, scube[:, 19:22, :].mean(axis=1), rtol=1e-4,
        atol=1e-5 * scube.max())

def test_slit_spectra_objective_residual(driver):
    # Data equal to the model plus 1, with errors of 2: every residual
    # (model - data) / error is -0.5
    from gbkfit.dataset import Data
    from gbkfit.dataset.datasets import DatasetSlitSpectra
    from gbkfit.objective import Objective
    dmodel = dict(type='slit_spectra', size=[32, 51], step=[1, 10], rota=30)
    model = evaluate(driver, dmodel)
    dataset = DatasetSlitSpectra(
        Data(model + 1, error=np.full_like(model, 2)), step=(1, 10),
        rota=30)
    from gbkfit.model import gmodel_parser
    from gbkfit.observation import (
        Observation, ObservationGroup, observable_parser)
    observation = Observation(
        driver, observable_parser.load(copy.deepcopy(dmodel)), data=dataset)
    model_group = ObservationGroup(
        [gmodel_parser.load(copy.deepcopy(GMODEL))], [observation])
    objective = Objective(model_group)
    params = gbkfit.params.EvaluationParams(model_group.pdescs(), PROPERTIES)
    residual = objective.residual_nddata_h(params.evaluate(), False)
    np.testing.assert_allclose(residual[0]['spectra'], -0.5, rtol=1e-5)
    # The chi-squared is the sum of their squares, and the log-likelihood
    # -1/2 of it
    chi2 = 0.25 * model.size
    assert objective.residual_scalar(params.evaluate(), squared=True)[0] \
        == pytest.approx(chi2, rel=1e-5)
    assert objective.log_likelihood(params.evaluate())[0] == pytest.approx(
        -0.5 * chi2, rel=1e-5)


def test_hanning_smoothing_of_the_channels(driver):
    # A Gaussian LSF convolved with the Hanning smoothing of the channels
    # is the Gaussian LSF and then 1/4, 1/2 and 1/4 of each channel in
    # the channel before, itself and the one after
    dmodel = dict(type='slit_spectra', size=[32, 61], step=[1, 10])
    gauss = dict(type='gauss', sigma=15)
    plain = evaluate(driver, dmodel | dict(lsf=gauss))
    smoothed = evaluate(driver, dmodel | dict(lsf=dict(
        type='convolution', lsfs=[gauss, dict(type='hanning', width=10)])))
    expected = 0.5 * plain
    expected[1:] += 0.25 * plain[:-1]
    expected[:-1] += 0.25 * plain[1:]
    np.testing.assert_allclose(
        smoothed, expected, rtol=1e-4, atol=1e-6 * plain.max())
