"""
Tests of the free amplitudes of spectra (LikelihoodGaussian(amplitudes=
'per_spectrum')), which the objective solves.
"""

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelSpectra
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    LikelihoodGaussian, Observation, ObservationGroup, PixelMoments,
    PixelSpectra, line_of_sight_parser)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


SIZE = (12, 12, 31)
STEP = (1, 1, 20)
PARAMS = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=3, vpt_rt=2,
    vpt_vt=150, dpt_a=30)
GALAXY = dict(components=dict(
    type='smdisk', rnodes=list(range(0, 11)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform')),
    emission=dict(type='lines')))


def objective(driver, data, error, amplitudes='per_spectrum', weights=1.0):
    """
    The objective of a disk seen in data with the given errors, under a
    likelihood of the given amplitudes and weights.
    """
    dataset = DatasetPixelSpectra(
        Data(data, error=np.full_like(data, error)), step=STEP)
    group = ObservationGroup(
        [galaxy_parser.load(GALAXY)], [Observation(
            PixelSpectra.from_data(dataset),
            line_of_sight_parser.load(dict(type='flat')), driver,
            data=dataset, likelihood=LikelihoodGaussian(
                weights=weights, amplitudes=amplitudes))])
    return Objective(group), ParamSpace(group.pdescs(), PARAMS).evaluate()


def model(driver):
    """The spectra of the disk, as data of no amplitudes see them."""
    blank = np.zeros(SIZE[::-1])
    obj, values = objective(driver, blank, 1.0, None)
    return obj.group().model_h(values)[0]['spectra']['d'].astype(float)


def test_amplitudes_take_the_brightness_of_each_spectrum(driver):
    # Data that are the model times a scale for each spectrum: free
    # amplitudes fit them exactly, and are those scales
    spectra = model(driver)
    rng = np.random.default_rng(1)
    scales = rng.uniform(0.5, 2, SIZE[1::-1])
    error = 1e-3 * spectra.max()
    obj, values = objective(driver, spectra * scales, error)
    extra = {}
    assert obj.chi2(values, extra) == pytest.approx(0, abs=1e-6)
    lit = spectra.sum(axis=0) > 1e-3 * spectra.sum(axis=0).max()
    np.testing.assert_allclose(
        extra['observation0_amplitudes'][lit], scales[lit], rtol=1e-4)
    np.testing.assert_array_equal(
        obj.amplitudes()[0], extra['observation0_amplitudes'])
    plain, _ = objective(driver, spectra * scales, error, None)
    assert plain.chi2(values) > 1e3
    assert obj.nnuisance() == SIZE[0] * SIZE[1]


def test_the_log_likelihood_integrates_the_amplitudes(driver):
    # Noisy data, some spectra of which the model fits best with no light
    # (a = 0): the log-likelihood is the sum of the logs of the integrals
    # of exp(-chi2(a) / 2) of each spectrum over its flux, a F (F the flux
    # of the model there), computed here on a fine grid of a
    spectra = model(driver)
    rng = np.random.default_rng(2)
    error = 0.05 * spectra.max()
    data = 0.7 * spectra + rng.normal(0, error, spectra.shape)
    obj, values = objective(driver, data, error)
    _, log_likelihood = obj.chi2_and_log_likelihood(values)
    expected = 0.0
    a = np.linspace(0, 20, 40001)[:, None]
    for m, d in zip(*(x.reshape(SIZE[2], -1).T for x in (spectra, data))):
        chi2_a = (((a * m - d) / error) ** 2).sum(axis=1)
        least = chi2_a.min()
        if m.sum() > 0:
            least -= 2 * np.log(np.trapezoid(
                np.exp(-(chi2_a - least) / 2), a[:, 0] * m.sum()))
        expected -= least / 2
    assert log_likelihood == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize('observable, calibration, message', [
    ('moments', False, "has none"),
    ('spectra', True, "has no calibration")])
def test_amplitudes_scale_spectra_only(observable, calibration, message):
    likelihood = LikelihoodGaussian(amplitudes='per_spectrum')
    if observable == 'moments':
        from gbkfit.dataset import DatasetPixelMoments
        data = DatasetPixelMoments({1: Data(np.zeros((4, 4)))})
        observable = PixelMoments.from_data(data)
    else:
        data = DatasetPixelSpectra(Data(np.zeros((5, 4, 4))))
        observable = PixelSpectra.from_data(data)
    with pytest.raises(ConfigError, match=message):
        Observation(
            observable, line_of_sight_parser.load(dict(type='flat')),
            data=data, likelihood=likelihood, calibration=calibration,
            name='obs')


def test_amplitudes_have_one_kind():
    with pytest.raises(ConfigError, match="must be 'per_spectrum'"):
        LikelihoodGaussian(amplitudes='per_pixel')


def test_a_likelihood_of_no_weight_has_no_amplitude_terms(driver):
    # (an item of weight 0 counts nothing, not NaN)
    spectra = model(driver)
    obj, values = objective(
        driver, spectra, 1e-3 * spectra.max(), weights=0.0)
    assert obj.chi2_and_log_likelihood(values) == (0.0, 0.0)
