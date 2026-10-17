"""
Tests for the objective: the totals that the fitters use agree with the
residuals, and the residuals are new arrays.
"""

import concurrent.futures

import numpy as np
import pytest
import scipy.special

from gbkfit.dataset import Data, DatasetPixelBrightness, DatasetPixelMoments
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    Flat, LikelihoodGaussian, Observation, ObservationGroup, PixelBrightness,
    PixelMoments, observable_parser)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


GALAXY = dict(components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    density=dict(ptraits=dict(type='exponential')),
    emission=dict(type='band'))])

PROPERTIES = dict(xpos=0.0, ypos=0.0, posa=50, incl=60, pt_a=1, pt_s=2)


def make_objective(driver, data, likelihood=None, name=None, **values):
    """
    The objective of an observation of the given name, and the values of
    the parameters (PROPERTIES and the given values).
    """
    dataset = DatasetPixelBrightness(data)
    observation = Observation(
        PixelBrightness.from_data(dataset), Flat(), driver, data=dataset,
        likelihood=likelihood, name=name)
    group = ObservationGroup(galaxy_parser.load(GALAXY), observation)
    params = ParamSpace(group.pdescs(), PROPERTIES | values).evaluate()
    return Objective(group), params


def test_totals_agree_with_the_residuals(driver):
    rng = np.random.default_rng(1)
    data = Data(rng.normal(0.1, 1, (24, 32)), error=np.full((24, 32), 0.5))
    objective, params = make_objective(
        driver, data, LikelihoodGaussian(weights=2.0))
    residuals = objective.residuals(params)
    assert residuals.dtype == np.float64
    assert objective.chi2(params) == pytest.approx(
        np.sum(residuals ** 2), rel=1e-5)
    assert objective.log_likelihood(params) == pytest.approx(
        -0.5 * objective.chi2(params))


def test_residuals_are_a_new_array_on_every_call(driver):
    # (scipy's least_squares keeps the arrays of earlier calls)
    data = Data(np.ones((24, 32)))
    objective, params = make_objective(driver, data)
    first = objective.residuals(params)
    second = objective.residuals(params)
    assert first is not second
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, second)


def test_objectives_evaluate_in_several_threads_at_once(driver):
    # (as copies of an objective on several devices do; here they share
    # one)
    data = Data(np.random.default_rng(2).normal(0.1, 1, (24, 32)))
    (objective1, params), (objective2, _) = \
        make_objective(driver, data), make_objective(driver, data)
    points = [params | dict(incl=incl) for incl in (30, 45, 60, 75)]
    expected = [objective1.residuals(point) for point in points]

    def evaluate(objective):
        return [objective.residuals(point) for point in points]

    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        results = list(executor.map(evaluate, [objective1, objective2]))
    for residuals in results:
        for got, want in zip(residuals, expected):
            np.testing.assert_array_equal(got, want)


def test_a_copy_on_another_driver_gives_the_same_values(driver):
    data = Data(np.random.default_rng(3).normal(0.1, 1, (24, 32)))
    objective, params = make_objective(DriverHost(), data)
    copy = objective.with_driver(driver)
    assert copy.group().observations()[0].driver() is driver
    assert copy.datasets()[0] is objective.datasets()[0]
    assert copy.chi2(params) == pytest.approx(
        objective.chi2(params), rel=1e-5)


def test_masked_values_do_not_count(driver):
    mask = np.ones((24, 32))
    mask[:5] = 0
    data = Data(np.ones((24, 32)), mask=mask)
    objective, params = make_objective(driver, data)
    assert objective.nresiduals() == 24 * 32 - 5 * 32
    residuals = objective.residuals(params).reshape(24, 32)
    np.testing.assert_array_equal(residuals[:5], 0)


def test_masked_values_with_an_upper_limit_are_non_detections(driver):
    # Below the limit L, of noise e: -2 log Phi((L - model) / e) each, and
    # they count as residuals
    mask = np.ones((24, 32))
    mask[:, :8] = 0
    data = Data(
        np.full((24, 32), 0.1), mask=mask, error=np.full((24, 32), 0.5))
    objective, params = make_objective(
        driver, data, LikelihoodGaussian(
            upper_limits=dict(brightness=dict(limit=0.01, error=0.2))))
    model = objective.group().model_h(params)[0]['brightness']['d']
    residuals = objective.residuals(params).reshape(24, 32)
    z = (0.01 - model[:, :8].astype(float)) / 0.2
    np.testing.assert_allclose(
        residuals[:, :8], np.sqrt(-2 * scipy.special.log_ndtr(z)),
        rtol=1e-5, atol=1e-7)
    assert objective.nresiduals() == 24 * 32


def test_a_fitted_noise_is_exact_to_a_constant(driver):
    # -2 log L = sum ((m - d) / s)^2 + sum log s^2, s = sqrt((f e)^2 +
    # j^2): the residuals of the data are over s, and the noise residual,
    # last, holds the rest, to a constant (the same at every f and j)
    rng = np.random.default_rng(4)
    error = rng.uniform(0.2, 0.6, (24, 32))
    data = Data(rng.normal(0.1, 1, (24, 32)), error=error)
    noise = dict(brightness=dict(scale=True, scale_minimum=0.1, jitter=True))
    objective, params = make_objective(
        driver, data, LikelihoodGaussian(noise=noise), 'obs',
        obs_noise_scale_brightness=1.0, obs_noise_jitter_brightness=0.0)
    model = objective.group().model_h(params)[0]['brightness']['d']
    offsets = []
    for f, j in [(1.0, 0.0), (1.7, 0.3), (0.6, 1.2)]:
        point = params | dict(
            obs_noise_scale_brightness=f, obs_noise_jitter_brightness=j)
        sigma = np.hypot(f * error, j)
        residuals = objective.residuals(point)
        np.testing.assert_allclose(
            residuals[:-1].reshape(24, 32), (model - data.data()) / sigma,
            rtol=1e-5, atol=1e-6)
        chi2 = objective.chi2(point)
        assert chi2 == pytest.approx(np.sum(residuals ** 2), rel=1e-6)
        assert objective.noise_chi2() == pytest.approx(residuals[-1] ** 2)
        offsets.append(chi2 - np.sum(
            ((model - data.data()) / sigma) ** 2 + np.log(sigma ** 2)))
    np.testing.assert_allclose(offsets, offsets[0], atol=1e-3)


@pytest.mark.parametrize('noise, name, error, message', [
    (dict(brightness=dict(scale=True, scale_minimum=0.1)), None, None,
     "needs a name"),
    (dict(moment1=dict(jitter=True)), 'obs', None, "does not have"),
    (dict(brightness=dict(scale=True, scale_minimum=0.1, jitter=True)),
     'obs', 0.5, "cannot both be fitted"),
    (dict(brightness=dict(scale=True)), 'obs', None, "scale_minimum"),
    (dict(brightness=dict(jitter=True, scale_minimum=0.1)), 'obs', None,
     "scale_minimum"),
    (dict(brightness=dict(scale=False)), 'obs', None, "fit its scale"),
])
def test_fitted_noise_options_are_checked(noise, name, error, message):
    errors = None if error is None else np.full((24, 32), error)
    data = Data(np.ones((24, 32)), error=errors)
    with pytest.raises(ConfigError, match=message):
        make_objective(
            DriverHost(), data, LikelihoodGaussian(noise=noise), name)


def test_a_fitted_noise_has_parameters_of_the_observation():
    noise = dict(brightness=dict(scale=True, scale_minimum=0.1, jitter=True))
    likelihood = LikelihoodGaussian(noise=noise)
    data = Data(np.ones((24, 32)), error=np.linspace(0.1, 1, 768).reshape(
        24, 32))
    objective, _ = make_objective(
        DriverHost(), data, likelihood, 'obs',
        obs_noise_scale_brightness=1.0, obs_noise_jitter_brightness=0.0)
    pdescs = objective.pdescs()
    assert pdescs['obs_noise_scale_brightness'].minimum() == 0.1
    assert pdescs['obs_noise_jitter_brightness'].minimum() == 0
    assert likelihood.dump()['noise'] == noise
    assert LikelihoodGaussian(**{
        k: v for k, v in likelihood.dump().items() if k != 'type'}).dump() \
        == likelihood.dump()


def test_an_observation_without_data_is_an_error(driver):
    dataset = DatasetPixelBrightness(Data(np.ones((24, 32))))
    observation = Observation(
        PixelBrightness.from_data(dataset), Flat(), driver)
    group = ObservationGroup(galaxy_parser.load(GALAXY), observation)
    with pytest.raises(ConfigError, match="needs data"):
        Objective(group)


KINEMATICS = dict(components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 30)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))])

DISK = dict(
    vsys=0, xpos=0, ypos=0, posa=40, incl=60, pt_a=1, pt_s=4, vpt_rt=2,
    vpt_vt=150, dpt_a=20)


def test_a_model_without_values_where_the_data_have_some_pays(driver):
    # A model can no longer lower the chi-square by having no flux (and so
    # no velocity) where the data have some: its moment 0 there is 0, and
    # its moments 1 and 2 count as misses of 3 errors (they counted 0)
    def group(observable, dataset=None):
        likelihood = None if dataset is None \
            else LikelihoodGaussian(missing_residual=3)
        observation = Observation(
            observable, Flat(), driver, data=dataset, likelihood=likelihood)
        return ObservationGroup(galaxy_parser.load(KINEMATICS), observation)
    image = group(observable_parser.load(dict(
        type='pixel_moments', size=[24, 24], orders=[0, 1, 2])))
    model = image.model_h(ParamSpace(image.pdescs(), DISK).evaluate())[0]
    errors = dict(moment0=0.01, moment1=5.0, moment2=5.0)
    dataset = DatasetPixelMoments({
        order: Data(model[f'moment{order}']['d'], error=np.full(
            (24, 24), errors[f'moment{order}']))
        for order in (0, 1, 2)})
    objective = Objective(group(PixelMoments.from_data(dataset), dataset))
    truth = ParamSpace(objective.pdescs(), DISK).evaluate()
    compact = ParamSpace(
        objective.pdescs(), DISK | dict(pt_s=0.1)).evaluate()
    compact_moment1 = objective.group().model_h(compact)[0]['moment1']['d']
    without = np.isnan(compact_moment1)
    has_data = dataset['moment1'].mask() != 0
    residuals = objective.residual_nddata_h(compact, False)[0]
    assert np.any(without & has_data)
    np.testing.assert_allclose(
        residuals['moment1'][without & has_data], 3)
    np.testing.assert_allclose(
        residuals['moment0'][without & has_data],
        (0 - dataset['moment0'].data()[without & has_data]) / 0.01,
        rtol=1e-4, atol=1e-3)
    assert objective.chi2(compact) > objective.chi2(truth)
