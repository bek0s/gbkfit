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
    # The posterior agrees with the Laplace approximation at the best fit
    posterior = result.posterior.summary()
    laplace = result.best().laplace.summary()
    std = result.best().covariance.std
    for name in TRUE:
        lo, median, hi = posterior[name]
        assert median == pytest.approx(laplace[name][1], abs=0.3 * std[name])
        assert (hi - lo) / 2 == pytest.approx(std[name], rel=0.25)


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
