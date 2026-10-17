"""
Tests of the amplitudes of spectra (LikelihoodGaussian(amplitudes=
'per_spectrum')), which the objective solves under their prior.
"""

import numpy as np
import pytest
import scipy.stats

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
        vptraits=dict(type='arctan', direction='tan'), dptraits=dict(type='uniform')),
    emission=dict(type='lines')))


# The standard deviation of the prior of the amplitudes
SIGMA = 0.5


def objective(
        driver, data, error, amplitudes='per_spectrum', weights=1.0,
        noise=None, **values):
    """
    The objective of a disk seen in data with the given errors, under a
    likelihood of the given amplitudes (of the prior SIGMA), weights and
    fitted noise (of an observation named 'obs'), and the values of the
    parameters (PARAMS and the given values).
    """
    dataset = DatasetPixelSpectra(
        Data(data, error=np.full_like(data, error)), step=STEP)
    group = ObservationGroup(
        [galaxy_parser.load(GALAXY)], [Observation(
            PixelSpectra.from_data(dataset),
            line_of_sight_parser.load(dict(type='flat')), driver,
            data=dataset, likelihood=LikelihoodGaussian(
                weights=weights, amplitudes=amplitudes,
                amplitude_sigma=None if amplitudes is None else SIGMA,
                noise=noise),
            name=None if noise is None else 'obs')])
    return Objective(group), ParamSpace(
        group.pdescs(), PARAMS | values).evaluate()


def model(driver):
    """The spectra of the disk, as data of no amplitudes see them."""
    blank = np.zeros(SIZE[::-1])
    obj, values = objective(driver, blank, 1.0, None)
    return obj.group().model_h(values)[0]['spectra']['d'].astype(float)


def best_amplitudes(spectra, data, error):
    """
    The amplitudes of the spectra (along the first axis) of the given
    data, under their prior: (sum m d / e^2 + 1/s^2) / (sum m^2 / e^2 +
    1/s^2), at least 0.
    """
    a = (spectra ** 2).sum(axis=0) / error ** 2 + 1 / SIGMA ** 2
    b = (spectra * data).sum(axis=0) / error ** 2 + 1 / SIGMA ** 2
    return np.maximum(b, 0) / a


def test_amplitudes_take_the_brightness_of_each_spectrum(driver):
    # Data that are the model times a scale for each spectrum, of small
    # errors: the amplitudes are those scales (but for the pull of the
    # prior, small), and the chi-square mostly that of their prior,
    # ((a - 1) / s)^2
    spectra = model(driver)
    rng = np.random.default_rng(1)
    scales = rng.uniform(0.5, 2, SIZE[1::-1])
    error = 1e-3 * spectra.max()
    obj, values = objective(driver, spectra * scales, error)
    extra = {}
    chi2 = obj.chi2(values, extra)
    amplitudes = extra['observation0_amplitudes']
    np.testing.assert_allclose(
        amplitudes, best_amplitudes(spectra, spectra * scales, error),
        rtol=1e-5)
    lit = spectra.sum(axis=0) > 1e-3 * spectra.sum(axis=0).max()
    np.testing.assert_allclose(amplitudes[lit], scales[lit], rtol=1e-3)
    assert chi2 == pytest.approx(
        np.sum(((amplitudes - 1) / SIGMA) ** 2), rel=1e-3)
    np.testing.assert_array_equal(
        obj.amplitudes()[0], extra['observation0_amplitudes'])
    plain, _ = objective(driver, spectra * scales, error, None)
    assert plain.chi2(values) > 1e3
    assert obj.nnuisance() == SIZE[0] * SIZE[1]


def test_amplitudes_weigh_the_values_by_their_fitted_noise(driver):
    # With a fitted jitter j, the noise of each value, sqrt(e^2 + j^2),
    # weighs it in place of its error e, as in its residual
    spectra = model(driver)
    rng = np.random.default_rng(2)
    error, jitter = 0.01 * spectra.max(), 0.03 * spectra.max()
    data = spectra * rng.uniform(0.5, 2, SIZE[1::-1]) \
        + rng.normal(0, jitter, spectra.shape)
    obj, values = objective(
        driver, data, error, noise=dict(spectra=dict(jitter=True)),
        obs_noise_jitter_spectra=jitter)
    obj.chi2(values)
    np.testing.assert_allclose(
        obj.amplitudes()[0],
        best_amplitudes(spectra, data, np.hypot(error, jitter)), rtol=1e-5)


def test_the_log_likelihood_integrates_the_amplitudes(driver):
    # Noisy data, some spectra of which the model fits best with no light
    # (a = 0): the log-likelihood is the sum of the logs of the integrals
    # of exp(-chi2(a) / 2) of each spectrum times the prior of a (Gaussian
    # around 1, cut at 0), computed here on a fine grid of a
    spectra = model(driver)
    rng = np.random.default_rng(2)
    error = 0.05 * spectra.max()
    data = 0.7 * spectra + rng.normal(0, error, spectra.shape)
    obj, values = objective(driver, data, error)
    _, log_likelihood = obj.chi2_and_log_likelihood(values)
    expected = 0.0
    a = np.linspace(0, 20, 40001)[:, None]
    prior = scipy.stats.truncnorm.pdf(
        a[:, 0], -1 / SIGMA, np.inf, loc=1, scale=SIGMA)
    for m, d in zip(*(x.reshape(SIZE[2], -1).T for x in (spectra, data))):
        chi2_a = (((a * m - d) / error) ** 2).sum(axis=1)
        least = chi2_a.min()
        expected += np.log(np.trapezoid(
            np.exp(-(chi2_a - least) / 2) * prior, a[:, 0])) - least / 2
    assert log_likelihood == pytest.approx(expected, abs=1e-2)


@pytest.mark.parametrize('observable, calibration, message', [
    ('moments', False, "has none"),
    ('spectra', True, "has no calibration")])
def test_amplitudes_scale_spectra_only(observable, calibration, message):
    likelihood = LikelihoodGaussian(
        amplitudes='per_spectrum', amplitude_sigma=SIGMA)
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


@pytest.mark.parametrize('options, message', [
    (dict(amplitudes='per_pixel', amplitude_sigma=SIGMA),
     "must be 'per_spectrum'"),
    (dict(amplitudes='per_spectrum'), "go together"),
    (dict(amplitude_sigma=SIGMA), "go together"),
    (dict(amplitudes='per_spectrum', amplitude_sigma=0), "positive")])
def test_amplitudes_need_their_prior(options, message):
    with pytest.raises(ConfigError, match=message):
        LikelihoodGaussian(**options)


def test_a_likelihood_of_no_weight_has_no_amplitude_terms(driver):
    # (an item of weight 0 counts nothing, not NaN)
    spectra = model(driver)
    obj, values = objective(
        driver, spectra, 1e-3 * spectra.max(), weights=0.0)
    assert obj.chi2_and_log_likelihood(values) == (0.0, 0.0)


def test_the_outputs_of_a_fit_have_the_amplitudes(driver):
    # The model of the spectra in the outputs is that which the residuals
    # compare (times the amplitudes), beside the model without them
    from gbkfit.tasks._detail import evaluation_outputs
    spectra = model(driver)
    error = 1e-3 * spectra.max()
    obj, values = objective(driver, 2 * spectra, error)
    outputs = evaluation_outputs(obj.group(), obj, values, True, True)
    np.testing.assert_allclose(
        outputs['model_0_spectra_d'].data,
        spectra * best_amplitudes(spectra, 2 * spectra, error), rtol=1e-5,
        atol=1e-6 * spectra.max())
    np.testing.assert_allclose(
        outputs['model_0_spectra_unscaled_d'].data, spectra, rtol=1e-6)
    lit = spectra.sum(axis=0) > 1e-3 * spectra.sum(axis=0).max()
    np.testing.assert_allclose(
        outputs['residual_extra_observation0_amplitudes'][lit], 2,
        rtol=1e-3)
