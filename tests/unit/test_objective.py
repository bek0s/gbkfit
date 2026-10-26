"""
Tests for the objective: the totals that the fitters use agree with the
residuals, and the residuals are new arrays.
"""

import concurrent.futures

import numpy as np
import pytest
import scipy.optimize
import scipy.special

from gbkfit.dataset import Data, DatasetPixelBrightness, DatasetPixelMoments
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    Flat, LikelihoodGaussian, ObservationGroup, ObservationSky,
    PixelBrightness, PixelMoments, observable_parser)
from gbkfit.params import (
    ParamSpace, SolveGaussian, solve_together, solve_together_rows)
from gbkfit.utils.parseutils import ConfigError


GALAXY = dict(components=[dict(
    type='disk', loose=False, tilted=False,
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
    observation = ObservationSky(
        PixelBrightness.from_data(dataset), Flat(), data=dataset,
        likelihood=likelihood, name=name)
    group = ObservationGroup(galaxy_parser.load(GALAXY), observation)
    params = ParamSpace(group.pdescs(), PROPERTIES | values).evaluate()
    return Objective(group.plan(driver)), params


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


def test_the_log_likelihoods_of_the_points_sum_to_it(driver):
    # (and the samples of a fit give them, one evaluation each; not with
    # a fitted noise, whose terms are sums over the points)
    from gbkfit.analysis import pointwise_log_likelihoods
    rng = np.random.default_rng(1)
    data = Data(rng.normal(0.1, 1, (24, 32)), error=np.full((24, 32), 0.5))
    objective, params = make_objective(
        driver, data, LikelihoodGaussian(weights=2.0))
    pointwise = objective.pointwise_log_likelihood(params)
    assert pointwise.sum() == pytest.approx(
        objective.log_likelihood(params), rel=1e-5)
    space = ParamSpace(objective.pdescs(), PROPERTIES | dict(
        pt_a=dict(min=0, max=2)))
    lls = pointwise_log_likelihoods(objective, space, [[1.0], [0.5]])
    assert lls.shape == (2, len(pointwise))
    np.testing.assert_allclose(lls[0], pointwise, rtol=1e-6)
    noise = dict(brightness=dict(scale=True, scale_minimum=0.1))
    objective, params = make_objective(
        driver, data, LikelihoodGaussian(noise=noise), 'obs',
        obs_noise_scale_brightness=1.0)
    with pytest.raises(NotImplementedError, match="fitted noise"):
        objective.pointwise_log_likelihood(params)


def test_prior_predictive_checks_tell_priors_that_reach_the_data(driver):
    # Data of the model at pt_a = 1 with noise: priors about it predict
    # them (most within the 5-95 band); priors far from it do not (the
    # faint pixels, about 0 in every model, stay within the band)
    from gbkfit.analysis import prior_predictive
    rng = np.random.default_rng(5)
    empty, values = make_objective(driver, Data(np.zeros((24, 32))))
    model = empty.group_plan().model_h(values)[0]['brightness']['d']
    error = np.full(model.shape, 0.01 * model.max())
    data = Data(model + rng.normal(0, 1, model.shape) * error, error=error)
    objective, _ = make_objective(driver, data)
    near = ParamSpace(objective.pdescs(), PROPERTIES | dict(
        pt_a=dict(min=0.98, max=1.02)))
    far = ParamSpace(objective.pdescs(), PROPERTIES | dict(
        pt_a=dict(min=3, max=4)))
    good = prior_predictive(objective, near, ndraws=40, seed=1)
    bad = prior_predictive(objective, far, ndraws=40, seed=1)
    assert good.coverage[0]['brightness'] > 0.85
    assert bad.coverage[0]['brightness'] \
        < good.coverage[0]['brightness'] - 0.2
    assert np.median(bad.chi2) > 100 * np.median(good.chi2)


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


def test_copies_with_other_data_evaluate_in_two_threads_at_once(driver):
    # (they share the plans of the models and their memory, and so their
    # lock: each evaluation, from the models to the residuals, is whole)
    rng = np.random.default_rng(4)
    objective, params = make_objective(
        driver, Data(rng.normal(0.1, 1, (24, 32))))
    other = objective.datasets()[0].with_data(
        dict(brightness=rng.normal(0.1, 1, (24, 32))))
    copy = objective.with_data([other])
    assert copy.group_plan().lock() is objective.group_plan().lock()
    # (without it, a few of 20 points mix the models of the two)
    points = [params | dict(incl=incl) for incl in (30, 45, 60, 75)] * 5

    def evaluate(objective_):
        return [objective_.residuals(point) for point in points]
    expected = [evaluate(objective), evaluate(copy)]
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        results = list(executor.map(evaluate, [objective, copy]))
    for residuals, wanted in zip(results, expected):
        for got, want in zip(residuals, wanted):
            np.testing.assert_array_equal(got, want)


def test_a_copy_on_another_driver_gives_the_same_values(driver):
    data = Data(np.random.default_rng(3).normal(0.1, 1, (24, 32)))
    objective, params = make_objective(DriverHost(), data)
    copy = objective.with_driver(driver)
    assert copy.group_plan().observation_drivers() == (driver,)
    assert copy.datasets()[0] is objective.datasets()[0]
    assert copy.chi2(params) == pytest.approx(
        objective.chi2(params), rel=1e-5)


def test_masked_values_do_not_count(driver):
    mask = np.ones((24, 32))
    mask[:5] = 0
    data = Data(np.ones((24, 32)), mask=mask)
    objective, params = make_objective(driver, data)
    assert objective.nresiduals() == 24 * 32 - 5 * 32
    assert objective.nindependent() == objective.nresiduals()
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
    model = objective.group_plan().model_h(params)[0]['brightness']['d']
    residuals = objective.residuals(params).reshape(24, 32)
    z = (0.01 - model[:, :8].astype(float)) / 0.2
    np.testing.assert_allclose(
        residuals[:, :8], np.sqrt(-2 * scipy.special.log_ndtr(z)),
        rtol=1e-5, atol=1e-7)
    assert objective.nresiduals() == 24 * 32


def test_upper_limits_count_within_their_footprint(driver, tmp_path):
    # The masked values outside it (e.g. the corners of the field of an
    # IFU, never observed) count nothing; data there are refused
    from gbkfit.utils import fitsutils, gridutils
    mask = np.ones((24, 32))
    mask[:, :8] = 0
    footprint = np.ones((24, 32))
    footprint[:, :4] = 0
    file = str(tmp_path / 'field.fits')
    fitsutils.write_data(
        file, footprint, gridutils.Coords((1, 1), (15.5, 11.5), (0, 0), 0))
    limits = dict(brightness=dict(limit=0.01, error=0.2, footprint=file))
    data = Data(
        np.full((24, 32), 0.1), mask=mask, error=np.full((24, 32), 0.5))
    objective, params = make_objective(
        driver, data, LikelihoodGaussian(upper_limits=limits))
    residuals = objective.residuals(params).reshape(24, 32)
    np.testing.assert_array_equal(residuals[:, :4], 0)
    assert np.all(residuals[:, 4:8] > 0)
    assert objective.nresiduals() == 24 * 28
    with pytest.raises(ConfigError, match="96 values outside the footprint"):
        make_objective(
            driver, Data(np.full((24, 32), 0.1), mask=mask[:, ::-1]),
            LikelihoodGaussian(upper_limits=limits))


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
    model = objective.group_plan().model_h(params)[0]['brightness']['d']
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
    observation = ObservationSky(PixelBrightness.from_data(dataset), Flat())
    group = ObservationGroup(galaxy_parser.load(GALAXY), observation)
    with pytest.raises(ConfigError, match="needs data"):
        Objective(group.plan(driver))


KINEMATICS = dict(components=[dict(
    type='disk', loose=False, tilted=False,
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
        observation = ObservationSky(
            observable, Flat(), data=dataset, likelihood=likelihood)
        return ObservationGroup(
            galaxy_parser.load(KINEMATICS), observation).plan(driver)
    image = group(observable_parser.load(dict(
        type='pixel_moments', size=[24, 24], orders=[0, 1, 2],
        spec_convention='relativistic')))
    model = image.model_h(
        ParamSpace(image.group().pdescs(), DISK).evaluate())[0]
    errors = dict(moment0=0.01, moment1=5.0, moment2=5.0)
    dataset = DatasetPixelMoments({
        order: Data(model[f'moment{order}']['d'], error=np.full(
            (24, 24), errors[f'moment{order}']))
        for order in (0, 1, 2)})
    observable = PixelMoments.from_data(
        dataset, spec_convention='relativistic')
    objective = Objective(group(observable, dataset))
    truth = ParamSpace(objective.pdescs(), DISK).evaluate()
    compact = ParamSpace(
        objective.pdescs(), DISK | dict(pt_s=0.1)).evaluate()
    compact_moment1 = \
        objective.group_plan().model_h(compact)[0]['moment1']['d']
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


def test_values_the_model_lacks_take_no_noise(driver):
    # Their residual, missing_residual, does not take the fitted noise:
    # they add nothing to its normalisation (the noise residual), sum
    # log((f e)^2 / (f_min e)^2) over the values the model has
    def group(observable, dataset=None):
        likelihood = None if dataset is None else LikelihoodGaussian(
            missing_residual=3, noise=dict(
                moment1=dict(scale=True, scale_minimum=0.1)))
        observation = ObservationSky(
            observable, Flat(), data=dataset, likelihood=likelihood,
            name='obs')
        return ObservationGroup(
            galaxy_parser.load(KINEMATICS), observation).plan(driver)
    image = group(observable_parser.load(dict(
        type='pixel_moments', size=[24, 24], orders=[0, 1],
        spec_convention='relativistic')))
    model = image.model_h(
        ParamSpace(image.group().pdescs(), DISK).evaluate())[0]
    dataset = DatasetPixelMoments({
        order: Data(model[f'moment{order}']['d'], error=np.full((24, 24), 5.0))
        for order in (0, 1)})
    observable = PixelMoments.from_data(
        dataset, spec_convention='relativistic')
    objective = Objective(group(observable, dataset))
    compact = ParamSpace(objective.pdescs(), DISK | dict(
        pt_s=0.1, obs_noise_scale_moment1=2.0)).evaluate()
    moment1 = objective.group_plan().model_h(compact)[0]['moment1']['d']
    has = ~np.isnan(moment1) & (dataset['moment1'].mask() != 0)
    assert np.any(np.isnan(moment1) & (dataset['moment1'].mask() != 0))
    objective.chi2(compact)
    assert objective.noise_chi2() == pytest.approx(
        np.count_nonzero(has) * np.log((2 / 0.1) ** 2), rel=1e-6)


def correlated_data(rng, correlation):
    """
    An image of noise whose errors differ, with some pixels masked, and
    the correlation of its noise.
    """
    mask = np.ones((24, 32))
    mask[:3] = 0
    mask[rng.integers(0, 24, 20), rng.integers(0, 32, 20)] = 0
    return Data(
        rng.normal(0.1, 1, (24, 32)), mask=mask,
        error=rng.uniform(0.2, 0.6, (24, 32)),
        spatial_correlation=correlation)


def test_correlated_noise_has_the_chi2_of_its_covariance(driver):
    # chi2 = r^T C^-1 r over the kept pixels: r = w (m - d) / s, s =
    # sqrt((f e)^2 + j^2), and C = (1 - floor) K + floor I, K_ab the
    # correlation at the separation of pixels a and b (an elongated beam,
    # which a swap of x and y would turn)
    from gbkfit.instrument import PSFGauss
    data = correlated_data(np.random.default_rng(5), PSFGauss(2, 0.5, 20))
    noise = dict(brightness=dict(scale=True, scale_minimum=0.1, jitter=True))
    objective, params = make_objective(
        driver, data, LikelihoodGaussian(
            weights=2.0, noise=noise, white_floor=0.05), 'obs',
        obs_noise_scale_brightness=1.3, obs_noise_jitter_brightness=0.2)
    model = objective.group_plan().model_h(params)[0]['brightness']['d']
    sigma = np.hypot(1.3 * data.error(), 0.2)
    kept = data.mask() != 0
    r = (2 * (model - data.data()) / sigma)[kept]
    y, x = np.nonzero(kept)
    shape = objective.datasets()[0].noise_correlation('brightness').spatial
    k = shape[23 + y[:, None] - y, 31 + x[:, None] - x]
    c = 0.95 * k + 0.05 * np.eye(len(k))
    chi2 = objective.chi2(params) - objective.noise_chi2()
    assert chi2 == pytest.approx(r @ np.linalg.solve(c, r), rel=1e-4)
    # Noise of exactly K gives the chi-square tr(K C^-1) on average
    assert objective.nindependent() == pytest.approx(
        np.trace(np.linalg.solve(c, k)))
    # The noise residual counts the values as the independent ones they
    # are worth: c^2 log(s^2 / (f_min e)^2) of each, times n_ind / n
    share = objective.nindependent() / np.count_nonzero(kept)
    assert objective.noise_chi2() == pytest.approx(share * np.sum(
        4 * np.log((sigma / (0.1 * data.error()))[kept] ** 2)), rel=1e-6)
    residuals = objective.residuals(params)[:-1]
    assert np.sum(residuals ** 2) == pytest.approx(chi2, rel=1e-5)
    # The residuals of each value, unweighted, are not whitened
    unweighted = objective.residual_nddata_h(params, False)[0]['brightness']
    np.testing.assert_allclose(
        unweighted[kept], r / 2, rtol=1e-5, atol=1e-6)


def test_correlated_noise_is_in_float64_by_default():
    # (the whitening magnifies the rounding of float32 models)
    from gbkfit.instrument import PSFGauss
    data = correlated_data(np.random.default_rng(6), PSFGauss(2))
    objective, _ = make_objective(DriverHost(), data)
    assert objective.group().observations()[0].dtype() == np.float64
    objective, _ = make_objective(DriverHost(), Data(np.ones((24, 32))))
    assert objective.group().observations()[0].dtype() == np.float32


def test_correlated_noise_goes_without_upper_limits_and_a_floor_of_0():
    from gbkfit.instrument import PSFGauss
    data = correlated_data(np.random.default_rng(6), PSFGauss(4))
    with pytest.raises(ConfigError, match="does not go with upper limits"):
        make_objective(DriverHost(), data, LikelihoodGaussian(
            upper_limits=dict(brightness=dict(limit=0.01, error=0.2))))
    # (a Gaussian beam of sigma 4 pixels alone is singular on them: its
    # power at the shortest wavelength, of 2 pixels, is exp(-8 pi^2))
    objective, params = make_objective(
        DriverHost(), data, LikelihoodGaussian(white_floor=0))
    with pytest.raises(ConfigError, match="singular"):
        objective.chi2(params)


@pytest.mark.parametrize('mean, sigma, minimum', [
    (1.0, 0.5, 0.0), (1.3, 0.4, 0.2), (-0.5, 2.0, None)])
def test_solved_values_are_the_best_and_integrate_out(mean, sigma, minimum):
    # (sums of a value that the data pull below its minimum, one near its
    # mean, and one whose values do not count)
    a, b = np.array([4.0, 9.0, 0.0]), np.array([-3.0, 11.0, 0.0])
    solution = SolveGaussian(mean, sigma, minimum).solve(a, b)
    lower = -np.inf if minimum is None else minimum

    def chi2(x, k):
        return a[k] * x ** 2 - 2 * b[k] * x + ((x - mean) / sigma) ** 2

    def prior(x):
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi)
            * scipy.special.ndtr((mean - lower) / sigma))

    x = np.linspace(max(lower, mean - 40 * sigma), mean + 40 * sigma, 400001)
    log_terms = 0.0
    for k in range(3):
        best = x[np.argmin(chi2(x, k))]
        assert solution.values[k] == pytest.approx(best, abs=1e-3)
        integral = np.trapezoid(
            np.exp(-0.5 * (a[k] * x ** 2 - 2 * b[k] * x)) * prior(x), x)
        # (the log of the integral, beyond -chi2 / 2 at the best value)
        log_terms += np.log(integral) + 0.5 * chi2(solution.values[k], k)
    assert solution.log_term == pytest.approx(log_terms, rel=1e-8)
    np.testing.assert_allclose(
        solution.residuals, (solution.values - mean) / sigma)
    # (the effective number of each value: A / (A + 1/sigma^2), none at
    # the minimum; profiled, the same values without the integral)
    free = (b + mean / sigma ** 2) / (a + 1 / sigma ** 2) > lower
    assert solution.m_eff == pytest.approx(
        np.sum(np.where(free, a / (a + 1 / sigma ** 2), 0)))
    profiled = SolveGaussian(mean, sigma, minimum, 'profiled').solve(a, b)
    np.testing.assert_allclose(profiled.values, solution.values)
    assert profiled.log_term == 0


def least_squares_problem(seed):
    """
    The weighted model matrix and data of a linear problem of 3 values on
    40 data values (pieces that overlap, so the values are coupled), and
    its sums N and b (see SolveGaussian.solve_coupled).
    """
    rng = np.random.default_rng(seed)
    pieces = np.abs(rng.normal(size=(40, 3))) + 0.2
    data = pieces @ np.array([1.0, -0.4, 2.0]) + rng.normal(size=40)
    return pieces, data, pieces.T @ pieces, pieces.T @ data


def effective_number(solve, pieces, data, eps=1e-6):
    """
    The trace of the hat matrix of a solve by finite differences: how much
    the fitted model at each data value moves with that data value.
    """
    def fitted(d):
        return pieces @ solve.solve_coupled(
            pieces.T @ pieces, pieces.T @ d).values
    base = fitted(data)
    return sum(
        (fitted(data + eps * np.eye(len(data))[i]) - base)[i] / eps
        for i in range(len(data)))


def test_values_solved_together_are_the_best_and_integrate_out():
    # Without a minimum: the best values of the chi-square of the data and
    # the prior (numpy's least squares of the data and the prior rows), its
    # integral over the values (a grid in 3-D) and their effective number
    # (the trace of the hat matrix, by finite differences)
    pieces, data, normal, rhs = least_squares_problem(1)
    mean, sigma = 0.5, 1.5
    solve = SolveGaussian(mean, sigma)
    solution = solve.solve_coupled(normal, rhs)
    rows = np.vstack((pieces, np.eye(3) / sigma))
    target = np.concatenate((data, np.full(3, mean / sigma)))
    np.testing.assert_allclose(
        solution.values, np.linalg.lstsq(rows, target)[0], rtol=1e-10)

    def chi2(u):
        return np.sum((data - u @ pieces.T) ** 2, axis=-1) \
            + np.sum(((u - mean) / sigma) ** 2, axis=-1)
    axes = [np.linspace(v - 1.5, v + 1.5, 121) for v in solution.values]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    best = chi2(solution.values)
    density = np.exp(-0.5 * (chi2(grid) - best)) \
        / (sigma * np.sqrt(2 * np.pi)) ** 3
    integral = np.trapezoid(np.trapezoid(np.trapezoid(
        density, axes[2]), axes[1]), axes[0])
    assert solution.log_term == pytest.approx(np.log(integral), abs=1e-4)
    assert solution.m_eff == pytest.approx(
        effective_number(solve, pieces, data), rel=1e-5)


def test_values_solved_together_above_a_minimum_are_profiled():
    # The best values above the minimum (scipy's bounded least squares of
    # the data and the prior rows), and their effective number over those
    # above it; their integral has no exact formula, so 'profiled' only
    pieces, data, normal, rhs = least_squares_problem(2)
    mean, sigma, minimum = 0.0, 3.0, 0.0
    with pytest.raises(ConfigError, match="'profiled'"):
        SolveGaussian(mean, sigma, minimum).solve_coupled(normal, rhs)
    solve = SolveGaussian(mean, sigma, minimum, 'profiled')
    solution = solve.solve_coupled(normal, rhs)
    rows = np.vstack((pieces, np.eye(3) / sigma))
    target = np.concatenate((data, np.full(3, mean / sigma)))
    expected = scipy.optimize.lsq_linear(
        rows, target, bounds=(minimum, np.inf)).x
    np.testing.assert_allclose(solution.values, expected, atol=1e-8)
    assert solution.values[1] == minimum
    assert solution.log_term == 0
    assert solution.m_eff == pytest.approx(
        effective_number(solve, pieces, data), rel=1e-5)


@pytest.mark.parametrize('minimum, likelihood', [
    (None, 'integrated'), (0.0, 'profiled')])
def test_values_solved_on_rows_are_those_of_their_sums(minimum, likelihood):
    # Of two solves, with and without a minimum: their values, the
    # integral of the likelihood and the effective number, as those of the
    # sums of the rows (solve_together)
    rng = np.random.default_rng(4)
    rows = np.abs(rng.normal(size=(30, 12)))
    target = rows @ rng.normal(1, 1, 12) + rng.normal(size=30)
    solves = [SolveGaussian(0.5, 1.5, minimum, likelihood),
              SolveGaussian(1.0, 0.3, minimum, likelihood)]
    expected = solve_together(
        solves, [5, 7], rows.T @ rows, rows.T @ target)
    solution = solve_together_rows(solves, [5, 7], rows, target)
    np.testing.assert_allclose(
        solution.values, expected.values, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        solution.residuals, expected.residuals, rtol=1e-9, atol=1e-12)
    assert solution.log_term == pytest.approx(expected.log_term)
    assert solution.m_eff == pytest.approx(expected.m_eff)


def test_many_values_solved_on_few_rows_are_the_bounded_least_squares():
    # 1000 values (e.g. the weights of orbits) above 0 of 100 rows, of a
    # prior of the mean 1: those of scipy's bounded least squares of the
    # rows and the prior rows, many at their minimum
    rng = np.random.default_rng(5)
    rows = rng.random((100, 1000)) * (rng.random((100, 1000)) < 0.05)
    truth = rng.exponential(1, 1000) * (rng.random(1000) < 0.3)
    target = rows @ truth + 0.1 * rng.normal(size=100)
    sigma = 2.0
    solution = solve_together_rows(
        [SolveGaussian(1.0, sigma, 0.0, 'profiled')], [1000], rows, target)
    stacked = np.vstack((rows, np.eye(1000) / sigma))
    expected = scipy.optimize.lsq_linear(
        stacked, np.concatenate((target, np.full(1000, 1 / sigma))),
        bounds=(0, np.inf), tol=1e-12).x
    np.testing.assert_allclose(solution.values, expected, atol=1e-7)
    assert np.mean(solution.values == 0) > 0.05


def test_a_sigma_chosen_by_aicp_has_the_least_aicp():
    # Of 300 values above 0 on 40 rows (e.g. the weights of orbits): the
    # values of the sigma of the least AICp = chi2 of the data + 2 m_eff,
    # within its range, as low as that of the best of a fine grid of
    # sigmas, and inside the range (the data favour neither end)
    rng = np.random.default_rng(6)
    rows = rng.random((40, 300)) * (rng.random((40, 300)) < 0.2)
    target = rows @ rng.exponential(1, 300) + rng.normal(size=40)

    def aicp(solution):
        residuals = rows @ solution.values - target
        return residuals @ residuals + 2 * solution.m_eff
    chosen = solve_together_rows(
        [SolveGaussian(1.0, minimum=0.0, likelihood='profiled',
                       sigma_aicp=[0.01, 100])], [300], rows, target)
    grid = [aicp(solve_together_rows(
        [SolveGaussian(1.0, sigma, 0.0, 'profiled')], [300], rows, target))
        for sigma in np.geomspace(0.01, 100, 81)]
    assert aicp(chosen) <= min(grid) + 1e-3
    assert np.argmin(grid) not in (0, len(grid) - 1)


@pytest.mark.parametrize('options, message', [
    (dict(sigma=1.0, sigma_aicp=[0.1, 10]), "not both"),
    (dict(), "not both"),
    (dict(sigma_aicp=[10, 0.1], likelihood='profiled'), "increasing"),
    (dict(sigma_aicp=[0.1, 10]), "'profiled'"),
])
def test_invalid_sigmas_chosen_by_aicp_are_refused(options, message):
    with pytest.raises(ConfigError, match=message):
        SolveGaussian(1.0, **options)


def test_a_closed_objective_releases_its_memory_and_evaluates_again(driver):
    data = Data(np.random.default_rng(10).normal(0.1, 1, (24, 32)))
    objective, params = make_objective(driver, data)
    chi2 = objective.chi2(params)
    with objective:
        assert objective.chi2(params) == pytest.approx(chi2)
    assert objective._comparisons[0].residual_vector_d() is None
    assert objective.chi2(params) == pytest.approx(chi2)


def test_copies_with_other_data_share_the_whitening_of_their_masks(driver):
    # (a bootstrap works it out once; a copy whose mask differs works out
    # its own)
    from gbkfit.instrument import PSFGauss
    rng = np.random.default_rng(11)
    data = correlated_data(rng, PSFGauss(2))
    objective, params = make_objective(driver, data)
    objective.prepare()
    dataset = objective.datasets()[0]
    same = dataset.with_data(dict(brightness=rng.normal(0.1, 1, (24, 32))))
    other = same.with_data(dict(brightness=np.where(
        np.arange(32) < 4, np.nan, same['brightness'].data())))
    whitener = objective._comparisons[0]._whiteners['brightness']
    for copy_data, shared in ((same, True), (other, False)):
        with objective.with_data([copy_data]) as copy:
            fresh, _ = make_objective(driver, copy_data['brightness'])
            assert copy.chi2(params) == pytest.approx(
                fresh.chi2(params), rel=1e-6)
            assert (copy._comparisons[0]._whiteners['brightness']._spatial
                    is whitener._spatial) == shared


def test_copies_with_other_data_share_the_plans_of_the_models(driver):
    # (the models do not depend on the values of the data: a bootstrap
    # makes them once)
    rng = np.random.default_rng(12)
    data = Data(rng.normal(0, 1, (24, 32)))
    objective, params = make_objective(driver, data)
    group_plan = objective.group_plan()
    other = objective.datasets()[0].with_data(
        dict(brightness=rng.normal(0, 1, (24, 32))))
    copy = group_plan.with_data([other])
    assert copy._plans[0] is group_plan._plans[0]
    np.testing.assert_array_equal(
        copy.model_h(params)[0]['brightness']['d'],
        group_plan.model_h(params)[0]['brightness']['d'])


USER_LIKELIHOODS = '''
import numpy as np


def gaussian_residuals(item, data, model, error, mask, params):
    return (model - data) / error


def gaussian_log_likelihood(item, data, model, error, mask, params):
    return -0.5 * np.sum(((model - data) / error)[mask != 0] ** 2)


def cauchy_residuals(item, data, model, error, mask, params):
    x = (model - data) / params['s']
    return np.sign(x) * np.sqrt(2 * np.log1p(x ** 2))
'''


def user_data():
    # (a third of the pixels masked)
    rng = np.random.default_rng(1)
    mask = np.ones((24, 32))
    mask[:8] = 0
    return Data(rng.normal(0.1, 1, (24, 32)), error=np.full((24, 32), 0.5),
                mask=mask)


def test_user_likelihoods_of_gaussian_noise_are_the_gaussian(
        driver, tmp_path):
    from gbkfit.observation import LikelihoodUser
    file = tmp_path / 'likelihoods.py'
    file.write_text(USER_LIKELIHOODS)
    gaussian, params = make_objective(driver, user_data())
    residuals, _ = make_objective(driver, user_data(), LikelihoodUser(
        str(file), 'gaussian_residuals', 'residuals'))
    log_likelihood, _ = make_objective(driver, user_data(), LikelihoodUser(
        str(file), 'gaussian_log_likelihood', 'log_likelihood'))
    chi2 = gaussian.chi2(params)
    np.testing.assert_allclose(
        residuals.residuals(params), gaussian.residuals(params), rtol=1e-5,
        atol=1e-6)
    assert residuals.chi2(params) == pytest.approx(chi2, rel=1e-5)
    assert log_likelihood.chi2(params) == pytest.approx(chi2, rel=1e-5)
    assert log_likelihood.log_likelihood(params) == pytest.approx(
        -chi2 / 2, rel=1e-5)
    with pytest.raises(ConfigError, match="has no residuals"):
        log_likelihood.residuals(params)
    from gbkfit.observation import likelihood_parser
    info = dict(type='user', file=str(file), func='gaussian_log_likelihood',
                form='log_likelihood', params=[])
    assert likelihood_parser.dump(likelihood_parser.load(info)) == info


def test_user_likelihoods_take_parameters_of_their_observation(
        driver, tmp_path):
    # (Cauchy noise of the scale s: -2 log L = sum 2 log(1 + x^2))
    from gbkfit.observation import LikelihoodUser
    file = tmp_path / 'likelihoods.py'
    file.write_text(USER_LIKELIHOODS)
    cauchy = LikelihoodUser(str(file), 'cauchy_residuals', 'residuals', ['s'])
    objective, params = make_objective(
        driver, user_data(), cauchy, name='hst', hst_s=0.7)
    assert 'hst_s' in objective.pdescs()
    gaussian, _ = make_objective(driver, user_data())
    x = gaussian.residual_nddata_h(params, False)[0]['brightness'] * 0.5 / 0.7
    kept = user_data().mask() != 0
    assert objective.chi2(params) == pytest.approx(
        np.sum(2 * np.log1p(x[kept] ** 2)), rel=1e-5)
    with pytest.raises(ConfigError, match="needs a name"):
        make_objective(driver, user_data(), cauchy)
