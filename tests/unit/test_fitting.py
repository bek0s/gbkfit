"""
Tests for the fitting core (priors, the options of the free elements, the
uncertainty) and the fitters, on a mock image with known parameters.
"""

import math

import numpy as np
import pytest
import scipy.integrate

from gbkfit.dataset import Data, DatasetPixelBrightness
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.fitting import (
    FitParamPrior, PriorGauss, PriorGaussTrunc, PriorLogUniform,
    PriorUniform)
from gbkfit.instrument import Instrument, PSFGauss
from gbkfit.model import model_parser
from gbkfit.objective import Objective
from gbkfit.observation import Observation, ObservationGroup, PixelBrightness
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


# A disk that extends beyond the image: a disk that ends inside the data
# makes the chi-square jump as its edge crosses pixels
MODEL = dict(type='intensity_2d', components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(np.arange(0.5, 30)),
    bptraits=dict(type='exponential'))])

TRUE = dict(xpos=0.7, ypos=-0.4, posa=40.0, incl=55.0, bpt_a=1.0, bpt_s=4.0)

START = dict(
    xpos=(0.0, -4, 4), ypos=(0.0, -4, 4), posa=(30.0, 0, 90),
    incl=(45.0, 20, 80), bpt_a=(0.7, 0.1, 5), bpt_s=(3.0, 1, 10))


def mock_objective(driver, seed=1):
    """
    An objective of a mock image, seen through a PSF (without one, the
    cusp of the exponential disk makes the chi-square kink at the centre
    of a pixel), with noise of 1% of its peak.
    """
    def group(dataset):
        observation = Observation(
            PixelBrightness.from_data(dataset), driver,
            instrument=Instrument(psf=PSFGauss(1.5)), data=dataset)
        return ObservationGroup(model_parser.load(dict(MODEL)), observation)
    shape = (32, 32)
    image = group(DatasetPixelBrightness(Data(np.zeros(shape))))
    model = image.model_h(ParamSpace(image.pdescs(), TRUE).evaluate())[0]
    model = model['brightness']['d']
    sigma = 0.01 * model.max()
    noise = np.random.default_rng(seed).normal(0, sigma, shape)
    dataset = DatasetPixelBrightness(
        Data(model + noise, error=np.full(shape, sigma)))
    return Objective(group(dataset))


def bounded_properties():
    return {
        name: dict(value=value, min=lo, max=hi)
        for name, (value, lo, hi) in START.items()}


@pytest.mark.parametrize('prior', [
    PriorUniform(-1, 3), PriorLogUniform(0.1, 10), PriorGauss(1, 2),
    PriorGaussTrunc(1, 2, 0, 2)])
def test_priors_are_normalised_and_ppf_inverts_their_cdf(prior):
    lo = max(prior.minimum(), -30)
    hi = min(prior.maximum(), 30)
    total, _ = scipy.integrate.quad(
        lambda x: math.exp(prior.log_pdf(x)), lo, hi, points=[1])
    assert total == pytest.approx(1, rel=1e-6)
    # The probability below ppf(u) is u
    for u in (0.1, 0.5, 0.9):
        below, _ = scipy.integrate.quad(
            lambda x: math.exp(prior.log_pdf(x)), lo, float(prior.ppf(u)))
        assert below == pytest.approx(u, rel=1e-6)
    assert prior.log_pdf(prior.minimum() - 1) == -np.inf or \
        prior.minimum() == -np.inf


def test_min_and_max_without_a_prior_are_a_uniform_prior():
    param = FitParamPrior.load(dict(min=-2, max=5, value=1))
    assert isinstance(param.prior(), PriorUniform)
    assert (param.minimum(), param.maximum()) == (-2, 5)
    with pytest.raises(ConfigError, match="not both"):
        FitParamPrior.load(dict(
            prior=dict(type='gauss', mean=0, std=1), min=-2, max=5))


def test_lmfit_recovers_the_parameters_and_their_errors(driver):
    pytest.importorskip('lmfit')
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(driver)
    params = ParamSpace(objective.pdescs(), bounded_properties())
    result = FitterLMFitLeastSquares().fit(objective, params)
    best = result.best()
    assert result.success
    assert best.reduced_chi2 == pytest.approx(1, abs=0.15)
    for name, value in TRUE.items():
        assert abs(best.free[name] - value) < 4 * best.covariance.std[name]
    # The covariance predicts how the chi-square grows away from the best
    # fit: by (dx)^2 / var along each free element, at fixed others
    x = np.array(list(best.free.values()))
    hessian = np.linalg.inv(best.covariance.matrix)
    i = list(best.free).index('posa')
    x[i] += 0.1
    growth = objective.chi2(params.evaluate(x)) - best.chi2
    assert growth == pytest.approx(0.01 * hessian[i, i], rel=0.05)


def test_cmaes_finds_the_minimum_of_lmfit(driver):
    pytest.importorskip('cma')
    pytest.importorskip('lmfit')
    from gbkfit.fitting.fitters.cmaes import FitterCMAES
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(driver)
    params = ParamSpace(objective.pdescs(), bounded_properties())
    local = FitterLMFitLeastSquares().fit(objective, params).best()
    best = FitterCMAES().fit(objective, params).best()
    assert best.chi2 == pytest.approx(local.chi2, rel=1e-6)




def test_errors_propagate_to_tied_and_fixed_elements():
    pytest.importorskip('lmfit')
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(DriverHost())
    properties = bounded_properties() | dict(ypos='xpos - 1.1', bpt_s=4.0)
    params = ParamSpace(objective.pdescs(), properties)
    best = FitterLMFitLeastSquares().fit(objective, params).best()
    std = best.covariance.element_std
    assert std['ypos'] == pytest.approx(std['xpos'], rel=1e-6)
    assert std['bpt_s'] == 0
    assert best.covariance.at_bounds == ()
    laplace = best.laplace.summary()
    assert laplace['ypos'][1] == pytest.approx(laplace['xpos'][1] - 1.1)

@pytest.mark.parametrize('fitter', [
    dict(type='scipy.least_squares', method='trf'),
    dict(type='scipy.least_squares', method='lm'),
    dict(type='lmfit.least_squares', method='least_squares')])
def test_least_squares_methods_find_the_minimum_of_lmfit(driver, fitter):
    # xpos and ypos start at 0, where scipy's own finite differences take
    # a step for float64 (1.5e-8), too small for float32 models: they
    # never moved. ('lm' takes no bounds: the fitter maps them, as lmfit
    # does.)
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(driver)
    params = ParamSpace(objective.pdescs(), bounded_properties())
    local = FitterLMFitLeastSquares().fit(objective, params).best()
    result = fitter_parser.load(fitter).fit(objective, params)
    assert result.success
    assert result.best().chi2 == pytest.approx(local.chi2, rel=1e-6)


def test_multistart_finds_the_four_minima_of_a_thin_disk():
    # The image of a thin disk is the same at posa and posa + 180, and at
    # incl and 180 - incl: in these bounds there are four minima
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    properties = bounded_properties() | dict(
        posa=dict(value=100, min=0, max=360),
        incl=dict(value=90.5, min=10, max=170))
    params = ParamSpace(objective.pdescs(), properties)
    fitter = fitter_parser.load(dict(
        type='multistart', local=dict(type='lmfit.least_squares'),
        nstarts=16))
    result = fitter.fit(objective, params)
    best = result.best().chi2
    minima = {
        (round(s.free['posa']), round(s.free['incl']))
        for s in result.solutions if s.chi2 < best * (1 + 1e-6)}
    assert minima == {(40, 55), (220, 55), (40, 125), (220, 125)}


@pytest.mark.parametrize('fitter', [
    dict(type='lmfit.least_squares'),
    dict(type='scipy.least_squares', method='lm'),
    dict(type='scipy.least_squares', method='trf')])
def test_least_squares_move_values_from_awkward_starts(fitter):
    # xpos starts near 0, where MINPACK's own step (sqrt(epsfcn) times
    # the value) is below the precision of float32 models; bpt_s starts
    # on its bound, where the map of bounded values of lmfit and 'lm' is
    # flat. Neither moved.
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    local = FitterLMFitLeastSquares().fit(objective, params).best()
    properties = bounded_properties()
    properties['xpos']['value'] = 1e-3
    properties['bpt_s'] |= dict(value=1, min=1)
    params = ParamSpace(objective.pdescs(), properties)
    result = fitter_parser.load(fitter).fit(objective, params)
    assert result.best().chi2 == pytest.approx(local.chi2, rel=1e-6)


@pytest.mark.parametrize('fitter', [
    dict(type='scipy.least_squares', method='trf'),
    dict(type='lmfit.least_squares', method='least_squares')])
def test_trf_moves_a_value_that_starts_on_its_bound(fitter):
    # When every free value starts on a bound, TRF's first steps were
    # tiny and it stopped there, reporting success
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    properties = dict(TRUE) | dict(bpt_s=dict(value=1, min=1, max=10))
    params = ParamSpace(objective.pdescs(), properties)
    best = fitter_parser.load(fitter).fit(objective, params).best()
    assert best.free['bpt_s'] == pytest.approx(
        TRUE['bpt_s'], abs=5 * best.covariance.std['bpt_s'])


@pytest.mark.filterwarnings('ignore:invalid value:RuntimeWarning')
def test_multistart_skips_starts_the_parameters_reject():
    # bpt_s is nan for xpos < 0.6: the starts there fail, and are skipped
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    properties = bounded_properties() | dict(
        xpos=dict(value=0.65, min=-4, max=4),
        bpt_s='np.sqrt(xpos - 0.6) * 0 + 4')
    params = ParamSpace(objective.pdescs(), properties)
    result = fitter_parser.load(dict(
        type='multistart', local=dict(type='scipy.least_squares'),
        nstarts=8)).fit(objective, params)
    starts = result.diagnostics['starts']
    assert any('error' in start for start in starts)
    assert result.best().free['xpos'] == pytest.approx(TRUE['xpos'], abs=0.05)

def prior_properties(nsigma):
    # Uniform priors of +-nsigma errors around the truth (see the lmfit
    # test), so that the samplers need few evaluations
    errors = dict(
        xpos=0.01, ypos=0.01, posa=0.25, incl=0.25, bpt_a=0.01, bpt_s=0.03)
    return {
        name: dict(min=TRUE[name] - nsigma * error,
                   max=TRUE[name] + nsigma * error)
        for name, error in errors.items()}


def check_posterior(result):
    # The posterior agrees with the Laplace approximation at the best fit:
    # the medians (a sampler's best is its most likely sample, a fraction
    # of an error from the optimum), and the widths (each roughly, and on
    # average to 7%; emcee's DESnookerMove made them all 12% too narrow)
    posterior = result.posterior.summary()
    laplace = result.best().laplace.summary()
    std = result.best().covariance.std
    ratios = []
    for name in TRUE:
        lo, median, hi = posterior[name]
        assert median == pytest.approx(laplace[name][1], abs=std[name])
        ratios.append((hi - lo) / 2 / std[name])
    assert ratios == pytest.approx(np.ones(len(TRUE)), abs=0.25)
    assert np.mean(ratios) == pytest.approx(1, abs=0.07)


# The samplers run on the host only: they test the fitters, not the
# drivers

def test_emcee_posterior_agrees_with_the_covariance():
    pytest.importorskip('emcee')
    from gbkfit.fitting.fitters.emcee import FitterEmcee
    objective = mock_objective(DriverHost())
    properties = prior_properties(10)
    for name in properties:
        properties[name] |= dict(value=TRUE[name], width=1e-3)
    params = ParamSpace(objective.pdescs(), properties)
    result = FitterEmcee(nwalkers=24, nsteps=1000, burnin=300).fit(
        objective, params)
    check_posterior(result)


def test_nautilus_posterior_agrees_with_the_covariance():
    pytest.importorskip('nautilus')
    from gbkfit.fitting.fitters.nautilus import FitterNautilus
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    result = FitterNautilus(n_live=300, n_eff=2000).fit(objective, params)
    check_posterior(result)
    assert math.isfinite(result.evidence.log_z)


def test_dynesty_posterior_agrees_with_the_covariance():
    pytest.importorskip('dynesty')
    from gbkfit.fitting.fitters.dynesty import FitterDynesty
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    result = FitterDynesty(nlive=300).fit(objective, params)
    check_posterior(result)
    assert result.evidence.log_z_error > 0


@pytest.mark.filterwarnings('ignore:invalid value:RuntimeWarning')
def test_samplers_leave_out_points_the_parameters_reject():
    # bpt_s is nan for xpos < 0.6: there the likelihood is -inf, and the
    # points nautilus keeps (with no weight) crashed the posterior
    pytest.importorskip('nautilus')
    from gbkfit.fitting.fitters.nautilus import FitterNautilus
    objective = mock_objective(DriverHost())
    properties = dict(TRUE) | dict(
        xpos=dict(min=0.5, max=0.9), posa=dict(min=35, max=45),
        bpt_s='np.sqrt(xpos - 0.6) * 0 + 4')
    params = ParamSpace(objective.pdescs(), properties)
    result = FitterNautilus(n_live=200, n_eff=500).fit(objective, params)
    assert np.all(result.posterior.samples[:, 0] >= 0.6)
    assert result.posterior.summary()['xpos'][1] == pytest.approx(
        TRUE['xpos'], abs=0.05)
