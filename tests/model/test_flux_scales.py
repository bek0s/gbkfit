"""
Tests of the flux scales of observations (see ObservationSky): of one
value (e.g. a calibration), per spectrum or per row; fixed, free, or
solved by the objective under their prior (see SolveGaussian).
"""

import pathlib

import numpy as np
import pytest
import scipy.stats
from modelutils import config_group_plan

from gbkfit.dataset import Data, DatasetPixelMoments, DatasetPixelSpectra
from gbkfit.fitting import FitProblem
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    LikelihoodGaussian, LikelihoodUser, ObservationGroup, ObservationSky,
    PixelMoments, PixelSpectra, line_of_sight_parser, noise_parser)
from gbkfit.params import ParamSpace, SolveGaussian
from gbkfit.utils.parseutils import ConfigError


SIZE = (12, 12, 31)
STEP = (1, 1, 20)
PARAMS = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=3, vpt_rt=2,
    vpt_vt=150, dpt_a=30)
GALAXY = dict(components=dict(
    type='disk', rnodes=list(range(0, 11)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        type='traits', vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines')))

# The standard deviation of the prior of the scales, and their solve
SIGMA = 0.5
SOLVE = dict(type='gaussian', mean=1, sigma=SIGMA, minimum=0)

GOLDEN = pathlib.Path(__file__).parents[1] / 'data' / 'golden'


def group_plan(driver, data, error, likelihood=None):
    """
    The plan of the observations of a disk seen in data with the given
    errors, named 'obs', with a flux scale per spectrum, under a
    likelihood (Gaussian by default).
    """
    dataset = DatasetPixelSpectra(
        Data(data, error=np.full_like(data, error)), step=STEP)
    return ObservationGroup(
        [galaxy_parser.load(GALAXY)], [ObservationSky(
            PixelSpectra.from_data(dataset),
            line_of_sight_parser.load(dict(type='flat')), data=dataset,
            likelihood=likelihood, flux_scale='per_spectrum', name='obs')]
    ).plan(driver)


def objective(driver, data, error, scale=None, likelihood=None, **values):
    """
    The objective of a disk seen in data (see group_plan), and its
    parameters (PARAMS and the given values), of the given flux scale
    (solved by SOLVE by default).
    """
    plan = group_plan(driver, data, error, likelihood)
    scale = dict(solve=SOLVE) if scale is None else scale
    params = ParamSpace(
        plan.group().pdescs(), PARAMS | dict(obs_flux_scale=scale) | values)
    return Objective(plan, params.solves()), params


def model(driver):
    """The spectra of the disk, of a flux scale of 1."""
    obj, params = objective(driver, np.zeros(SIZE[::-1]), 1.0, scale=1)
    return obj.group_plan().model_h(params.evaluate())[0]['spectra']['d'] \
        .astype(float)


def solved(obj, params):
    """The solved scales of the spectra at the values, an image."""
    return np.reshape(
        list(obj.evaluate(params.evaluate()).solved.values()), SIZE[1::-1])


def best_scales(spectra, data, error):
    """
    The best scales of the spectra (along the first axis) of the given
    data, under their prior: (sum m d / e^2 + 1/s^2) / (sum m^2 / e^2 +
    1/s^2), at least 0.
    """
    a = (spectra ** 2).sum(axis=0) / error ** 2 + 1 / SIGMA ** 2
    b = (spectra * data).sum(axis=0) / error ** 2 + 1 / SIGMA ** 2
    return np.maximum(b, 0) / a


def config(driver, observables, flux_scale):
    """
    A configuration of the disk seen as the given observables, by
    observations named 'first' and 'second'; the first with the given
    flux scale.
    """
    observations = [
        dict(type='sky', observable=observable,
             line_of_sight=dict(type='flat'), name=name)
        for observable, name in zip(observables, ['first', 'second'])]
    observations[0]['flux_scale'] = flux_scale
    return dict(
        driver=dict(type=driver.type()), galaxies=[GALAXY],
        observations=observations)


def test_a_flux_scale_of_one_value_scales_the_flux(driver):
    # Moment 0 of a scaled observation is times its scale; the velocities
    # (moment 1) and the other observations are not
    observables = [
        dict(type='pixel_moments', size=[24, 24], orders=[0, 1],
             spec_convention='relativistic'),
        dict(type='pixel_spectra', size=[24, 24, 21], step=[1, 1, 20])]

    def evaluate(flux_scale):
        plan = config_group_plan(config(driver, observables, flux_scale))
        scale = dict(first_flux_scale=2) if flux_scale else {}
        data = plan.model_h(
            ParamSpace(plan.group().pdescs(), PARAMS | scale).evaluate())
        return (
            data[0]['moment0']['d'].copy(), data[0]['moment1']['d'].copy(),
            data[1]['spectra']['d'].copy())
    moment0, moment1, spectra = evaluate(None)
    scaled = evaluate('one')
    np.testing.assert_allclose(scaled[0], 2 * moment0, rtol=1e-6)
    np.testing.assert_allclose(scaled[1], moment1, rtol=1e-5, atol=1e-3)
    np.testing.assert_array_equal(scaled[2], spectra)


def test_a_solved_flux_scale_of_one_value_is_that_of_the_data(driver):
    # Data of moment 0 twice the model, and of its moment 1: the solved
    # scale is 2 (but for the pull of its prior), of moment 0 only
    def moments(moment0, moment1):
        dataset = DatasetPixelMoments({
            0: Data(moment0, error=np.full_like(moment0, 1e-3)),
            1: Data(moment1, error=np.full_like(moment1, 1.0))})
        return ObservationGroup(
            [galaxy_parser.load(GALAXY)], [ObservationSky(
                PixelMoments.from_data(
                    dataset, spec_size=21, spec_step=20,
                    spec_convention='relativistic'),
                line_of_sight_parser.load(dict(type='flat')), data=dataset,
                flux_scale='one', name='obs',
                likelihood=LikelihoodGaussian(missing_residual=3))]
        ).plan(driver)
    blank = moments(np.zeros(SIZE[1::-1]), np.zeros(SIZE[1::-1]))
    model = blank.model_h(ParamSpace(
        blank.group().pdescs(), PARAMS | dict(obs_flux_scale=1)
    ).evaluate())[0]
    scaled = moments(
        2 * model['moment0']['d'], np.nan_to_num(model['moment1']['d']))
    params = ParamSpace(
        scaled.group().pdescs(),
        PARAMS | dict(obs_flux_scale=dict(solve=SOLVE)))
    obj = Objective(scaled, params.solves())
    assert obj.evaluate(params.evaluate()).solved['obs_flux_scale'] \
        == pytest.approx(2, rel=1e-6)


def test_solved_scales_take_the_brightness_of_each_spectrum(driver):
    # Data that are the model times a scale for each spectrum, of small
    # errors: the solved scales are those scales (but for the pull of the
    # prior, small), and the chi-square mostly that of their prior,
    # ((a - 1) / s)^2; the scales count in the degrees of freedom by how
    # much their data decide them
    spectra = model(driver)
    rng = np.random.default_rng(1)
    scales = rng.uniform(0.5, 2, SIZE[1::-1])
    error = 1e-3 * spectra.max()
    obj, params = objective(driver, spectra * scales, error)
    chi2 = obj.chi2(params.evaluate())
    found = solved(obj, params)
    np.testing.assert_allclose(
        found, best_scales(spectra, spectra * scales, error), rtol=1e-5)
    lit = spectra.sum(axis=0) > 1e-3 * spectra.sum(axis=0).max()
    np.testing.assert_allclose(found[lit], scales[lit], rtol=1e-3)
    assert chi2 == pytest.approx(
        np.sum(((found - 1) / SIGMA) ** 2), rel=1e-3)
    # (each scale counts by how much its data decide it, a / (a + 1/s^2),
    # a = sum m^2 / e^2: about 1 where the spectrum has light, 0 where not)
    a = (spectra ** 2).sum(axis=0) / error ** 2
    assert obj.evaluate(params.evaluate()).nnuisance == pytest.approx(
        np.sum(a / (a + 1 / SIGMA ** 2)), rel=1e-4)
    fixed, params = objective(driver, spectra * scales, error, scale=1)
    assert fixed.chi2(params.evaluate()) > 1e3
    # (fitted, the same scales give the same model as solved)
    values = params.evaluate() | dict(obs_flux_scale=found.ravel())
    np.testing.assert_allclose(
        fixed.group_plan().model_h(values)[0]['spectra']['d'],
        spectra * found, rtol=1e-6, atol=1e-9 * spectra.max())


def test_the_aicp_counts_the_data_and_the_solved_scales(driver):
    # The AICp: the chi-square of the data alone (not that of the prior
    # of the scales), plus twice the effective number of the scales
    spectra = model(driver)
    rng = np.random.default_rng(3)
    error = 0.05 * spectra.max()
    data = spectra * rng.uniform(0.5, 2, SIZE[1::-1]) \
        + rng.normal(0, error, spectra.shape)
    obj, params = objective(driver, data, error)
    evaluation = obj.evaluate(params.evaluate())
    data_chi2 = np.sum(((data - spectra * solved(obj, params)) / error) ** 2)
    assert evaluation.merits.aicp == pytest.approx(
        data_chi2 + 2 * evaluation.nnuisance, rel=1e-5)


def test_solved_scales_weigh_the_values_by_their_fitted_noise(driver):
    # With a fitted jitter j, the noise of each value, sqrt(e^2 + j^2),
    # weighs it in place of its error e, as in its residual
    spectra = model(driver)
    rng = np.random.default_rng(2)
    error, jitter = 0.01 * spectra.max(), 0.03 * spectra.max()
    data = spectra * rng.uniform(0.5, 2, SIZE[1::-1]) \
        + rng.normal(0, jitter, spectra.shape)
    obj, params = objective(
        driver, data, error,
        likelihood=LikelihoodGaussian(noise=dict(spectra=dict(jitter=True))),
        obs_noise_jitter_spectra=jitter)
    np.testing.assert_allclose(
        solved(obj, params),
        best_scales(spectra, data, np.hypot(error, jitter)), rtol=1e-5)


def test_the_log_likelihood_integrates_the_solved_scales(driver):
    # Noisy data, some spectra of which the model fits best with no light
    # (a = 0): the log-likelihood is the sum of the logs of the integrals
    # of exp(-chi2(a) / 2) of each spectrum times the prior of a (Gaussian
    # around 1, cut at 0), computed here on a fine grid of a
    spectra = model(driver)
    rng = np.random.default_rng(2)
    error = 0.05 * spectra.max()
    data = 0.7 * spectra + rng.normal(0, error, spectra.shape)
    obj, params = objective(driver, data, error)
    log_likelihood = obj.log_likelihood(params.evaluate())
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


def test_a_likelihood_of_no_weight_has_no_terms_of_solved_scales(driver):
    # (an item of weight 0 counts nothing, not NaN)
    spectra = model(driver)
    obj, params = objective(
        driver, spectra, 1e-3 * spectra.max(),
        likelihood=LikelihoodGaussian(weights=0.0))
    assert obj.merits(params.evaluate())[:2] == (0.0, 0.0)


def test_flux_scales_need_light_and_a_name():
    data = DatasetPixelMoments({1: Data(np.zeros((4, 4)))})
    observable = PixelMoments.from_data(data, spec_convention='relativistic')
    with pytest.raises(ConfigError, match=r"has the flux scales \[\]"):
        ObservationSky(
            observable, line_of_sight_parser.load(dict(type='flat')),
            data=data, flux_scale='one', name='obs')
    with pytest.raises(ConfigError, match="needs a name"):
        ObservationSky(
            PixelSpectra.from_data(DatasetPixelSpectra(
                Data(np.zeros((5, 4, 4))))),
            line_of_sight_parser.load(dict(type='flat')),
            flux_scale='per_spectrum')


def test_only_linear_parameters_of_gaussian_likelihoods_are_solved(
        driver, tmp_path):
    spectra = model(driver)
    plan = group_plan(driver, spectra, 1.0)
    with pytest.raises(ConfigError, match="vsys cannot be solved"):
        Objective(plan, dict(vsys=SolveGaussian(0, 10)))
    (tmp_path / 'like.py').write_text(
        "def like(key, data, model, error, mask, params):\n"
        "    return model - data\n")
    user = LikelihoodUser(
        str(tmp_path / 'like.py'), 'like', 'residuals')
    with pytest.raises(ConfigError, match="not known to be linear"):
        objective(driver, spectra, 1.0, likelihood=user)
    # (and a fit needs an objective that solves those of its parameters)
    obj, params = objective(driver, spectra, 1.0, vsys=dict(value=0))
    with pytest.raises(ValueError, match="must solve the solved elements"):
        FitProblem(Objective(obj.group_plan()), params)


def test_mocks_take_flux_scales_per_spectrum_of_1_only(driver):
    obj, params = objective(driver, np.zeros(SIZE[::-1]), 1.0, scale=1)
    noise = noise_parser.load(dict(type='gauss', rms=1.0))
    obj.group_plan().mock(params.evaluate(), noise, nsamples=2, seed=1)
    values = params.evaluate() | dict(
        obs_flux_scale=np.full(SIZE[0] * SIZE[1], 2.0))
    with pytest.raises(ConfigError, match="of 1 only"):
        obj.group_plan().mock(values, noise, nsamples=2, seed=1)


def test_the_outputs_have_the_solved_scales(driver):
    # The model in the outputs is that which the residuals compare (times
    # the solved scales), and the scales are an image
    from gbkfit import Setup
    spectra = model(driver)
    error = 1e-3 * spectra.max()
    obj, params = objective(driver, 2 * spectra, error)
    setup = Setup(obj.group(), params, driver=driver)
    outputs = {
        x.kind: x.value for x in setup.outputs(setup.evaluate()).values()}
    scales = best_scales(spectra, 2 * spectra, error)
    np.testing.assert_allclose(
        outputs['model'].data, spectra * scales, rtol=1e-5,
        atol=1e-6 * spectra.max())
    np.testing.assert_allclose(outputs['flux_scale'], scales, rtol=1e-6)
    lit = spectra.sum(axis=0) > 1e-3 * spectra.sum(axis=0).max()
    np.testing.assert_allclose(outputs['flux_scale'][lit], 2, rtol=1e-3)


def test_the_scales_of_slitless_rows_scale_their_rows():
    # Slitless spectra dispersed along x (the golden case slitless_rows: 24
    # rows of 32 pixels) have a solved scale for each row, which scales its
    # row in the model of the outputs
    import gbkfit
    setup = gbkfit.load(str(GOLDEN / 'slitless_rows' / 'config.yaml'))
    unscaled = setup.group_plan.model_h(
        setup.params.evaluate())[0]['spectra']['d'].copy()
    outputs = {
        x.kind: x.value for x in setup.outputs(setup.evaluate()).values()}
    scales = outputs['flux_scale']
    assert scales.shape == (24,)
    np.testing.assert_allclose(
        outputs['model'].data, unscaled * scales[:, None], rtol=1e-12)
