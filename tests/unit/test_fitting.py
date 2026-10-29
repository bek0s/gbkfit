"""
Tests for the fitting core (priors, the options of the free elements, the
uncertainty) and the fitters, on a mock image with known parameters.
"""

import math
import threading
import time

import numpy as np
from astropy.table import Table
import pytest
import scipy.integrate
import scipy.stats

from gbkfit.dataset import Data, DatasetPixelBrightness
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.fitting import (
    EvaluationLog, FitControl, FitParamBounded, FitParamPrior, FitProblem,
    FitStopped, SubFitLog)
from gbkfit.instrument import Instrument, PSFGauss
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.parallel import WorkersProcesses, WorkersThreads
from gbkfit.observation import (
    FittedNoise, Flat, LikelihoodGaussian, LikelihoodUser, ObservationGroup,
    ObservationSky, PixelBrightness)
from gbkfit.params import (
    FreeElement, ParamEvaluationError, ParamScalarDesc, ParamSpace,
    PriorGauss, PriorGaussTrunc,
    PriorHalfCauchy, PriorHalfGauss, PriorLogUniform, PriorScipy, PriorSine,
    PriorTruncated, PriorUniform)
from gbkfit.utils.parseutils import ConfigError


# A disk that extends beyond the image: a disk that ends inside the data
# makes the chi-square jump as its edge crosses pixels
GALAXY = dict(components=[dict(
    type='disk', loose=False, tilted=False,
    rnodes=list(np.arange(0.5, 30)),
    density=dict(ptraits=dict(type='exponential')),
    emission=dict(type='band'))])

TRUE = dict(xpos=0.7, ypos=-0.4, posa=40.0, incl=55.0, pt_a=1.0, pt_s=4.0)

START = dict(
    xpos=(0.0, -4, 4), ypos=(0.0, -4, 4), posa=(30.0, 0, 90),
    incl=(45.0, 20, 80), pt_a=(0.7, 0.1, 5), pt_s=(3.0, 1, 10))


def mock_objective(
        driver, seed=1, error_scale=1.0, likelihood=None, name=None,
        flux_scale=None):
    """
    An objective of a mock image, seen through a PSF (without one, the
    cusp of the exponential disk makes the chi-square kink at the centre
    of a pixel), with noise of 1% of its peak, and errors of error_scale
    times it, under a likelihood, of an observation of a name and a flux
    scale (with nothing solved).
    """
    def group(dataset, likelihood=None, flux_scale=None):
        observation = ObservationSky(
            PixelBrightness.from_data(dataset), Flat(),
            instrument=Instrument(psf=PSFGauss(1.5)), data=dataset,
            likelihood=likelihood, flux_scale=flux_scale, name=name)
        return ObservationGroup(
            galaxy_parser.load(GALAXY), observation).plan(driver)
    shape = (32, 32)
    image = group(DatasetPixelBrightness(Data(np.zeros(shape))))
    model = image.model_h(
        ParamSpace(image.group().pdescs(), TRUE).evaluate())[0]
    model = model['brightness']['d']
    sigma = 0.01 * model.max()
    noise = np.random.default_rng(seed).normal(0, sigma, shape)
    dataset = DatasetPixelBrightness(
        Data(model + noise, error=np.full(shape, error_scale * sigma)))
    return Objective(group(dataset, likelihood, flux_scale))


def bounded_properties():
    return {
        name: dict(value=value, min=lo, max=hi)
        for name, (value, lo, hi) in START.items()}



def test_fitter_configs_round_trip():
    # (the fitters whose libraries are installed)
    from gbkfit.fitting import fitter_parser
    for type_ in fitter_parser.registered_classes():
        info = dict(type=type_)
        if type_ == 'multistart':
            info['local'] = dict(type='scipy.least_squares')
        dumped = fitter_parser.dump(fitter_parser.load(info))
        assert fitter_parser.dump(fitter_parser.load(dumped)) == dumped

@pytest.mark.parametrize('prior', [
    PriorUniform(-1, 3), PriorLogUniform(0.1, 10), PriorGauss(1, 2),
    PriorGaussTrunc(1, 2, 0, 2), PriorSine(20, 85), PriorHalfGauss(2),
    PriorHalfCauchy(0.5), PriorScipy('lognorm', dict(s=0.5, scale=2)),
    PriorScipy('beta', dict(a=2, b=5)),
    PriorTruncated(PriorGauss(1, 2), 0, math.inf),
    PriorTruncated(PriorSine(0, 90), 30, 60)])
def test_priors_are_normalised_and_ppf_inverts_their_cdf(prior):
    # (over the whole support: the half-Cauchy has 1% of its mass beyond
    # 60 scales)
    lo, hi = prior.support()
    total, _ = scipy.integrate.quad(
        lambda x: math.exp(prior.log_pdf(x)), lo, hi)
    assert total == pytest.approx(1, rel=1e-6)
    # The probability below ppf(u) is u
    for u in (0.1, 0.5, 0.9):
        below, _ = scipy.integrate.quad(
            lambda x: math.exp(prior.log_pdf(x)), lo, float(prior.ppf(u)))
        assert below == pytest.approx(u, rel=1e-6)
        assert prior.cdf(prior.ppf(u)) == pytest.approx(u, rel=1e-9)
    assert prior.log_pdf(lo - 1) == -np.inf or lo == -np.inf


def test_the_residual_of_the_sine_prior_is_smooth_through_edge_on():
    # (its square is the chi-square, and it changes sign at 90 degrees
    # without a corner, as least squares need)
    prior = PriorSine(0, 180)
    x = np.array([30, 89.9, 90, 90.1, 150])
    np.testing.assert_allclose(prior.residual(x) ** 2, prior.chi2(x))
    slopes = np.diff(prior.residual([89.8, 89.9, 90, 90.1, 90.2])) / 0.1
    np.testing.assert_allclose(slopes, np.radians(1), rtol=1e-4)

@pytest.mark.parametrize('prior, slope', [
    (PriorHalfGauss(2), 0.5), (PriorHalfCauchy(0.5), math.sqrt(2) / 0.5)])
def test_the_residuals_of_half_priors_have_a_slope_at_0(prior, slope):
    # (their squares are the chi-square, and they leave 0 with a finite
    # slope, as least squares need; that of a log-uniform prior would be
    # infinite)
    x = np.array([0, 0.3, 1, 5, 50])
    np.testing.assert_allclose(prior.residual(x) ** 2, prior.chi2(x))
    assert prior.residual(1e-6) / 1e-6 == pytest.approx(slope, rel=1e-6)
    assert prior.chi2(-1) == np.inf


def test_priors_load_from_their_configs_and_round_trip():
    element = FreeElement.load(dict(
        prior=dict(type='gauss_trunc', mean=0, std=1, min=-1, max=2),
        value=0.5))
    assert isinstance(element.prior(), PriorGaussTrunc)
    assert (element.minimum(), element.maximum(), element.value()) == (
        -1, 2, 0.5)
    from gbkfit.params import prior_parser
    for info in (dict(type='gauss_trunc', mean=0, std=1, min=-1, max=2),
                 dict(type='half_gauss', std=2.0),
                 dict(type='half_cauchy', scale=0.5)):
        assert prior_parser.dump(prior_parser.load(info)) == info


def test_priors_of_scipy_distributions():
    # A log-normal prior on a mass of 10^10.5 Msun, to 0.2 dex: its
    # chi-square is 0 at its most probable value, scale exp(-s^2)
    from gbkfit.params import prior_parser
    s, scale = 0.2 * math.log(10), 10 ** 10.5
    prior = PriorScipy('lognorm', dict(s=s, scale=scale))
    x = np.array([1e10, 10 ** 10.5, 1e11])
    np.testing.assert_allclose(
        prior.log_pdf(x), scipy.stats.lognorm(s, scale=scale).logpdf(x))
    assert prior.chi2(scale * math.exp(-s ** 2)) == pytest.approx(
        0, abs=1e-9)
    assert prior.support() == (0, np.inf)
    assert not prior.has_residual()
    info = dict(type='scipy', distribution='lognorm',
                params=dict(s=s, scale=scale))
    assert prior_parser.dump(prior_parser.load(info)) == info
    with pytest.raises(ConfigError, match="continuous distribution"):
        PriorScipy('poisson', dict(mu=3))
    with pytest.raises(ConfigError, match="are not valid"):
        PriorScipy('lognorm', dict(s=-1))
    with pytest.raises(ConfigError, match="are not valid"):
        PriorScipy('lognorm', dict(shape=1))


def test_free_elements_mean_the_same_for_every_fitter():
    # Bounds without a prior: flat for optimizers, uniform for samplers
    bounded = FreeElement.load(dict(min=-2, max=5, value=1))
    assert FitParamBounded.of(bounded, {}).prior() is None
    prior = FitParamPrior.of(bounded, {}).prior()
    assert isinstance(prior, PriorUniform)
    assert prior.support() == (-2, 5)
    # A prior with bounds is cut to them, for both
    cut = FreeElement.load(dict(
        prior=dict(type='gauss', mean=0, std=1), min=-2, max=5, value=1))
    for param in (FitParamBounded.of(cut, {}), FitParamPrior.of(cut, {})):
        assert isinstance(param.prior(), PriorTruncated)
        assert (param.minimum(), param.maximum()) == (-2, 5)
    with pytest.raises(ConfigError, match="needs a prior, or both"):
        FitParamPrior.of(FreeElement.load(dict(min=0, value=1)), {})
    with pytest.raises(ConfigError, match="needs option 'value'"):
        FitParamBounded.of(FreeElement.load(dict(min=0, max=1)), {})


def test_the_tuning_of_an_element_or_else_of_its_fitter():
    # (each fitter reads its own; the others ignore it)
    from gbkfit.fitting.fitters.cmaes import FitParamCMAES
    from gbkfit.fitting.fitters.grid import FitParamGrid
    element = FreeElement.load(dict(value=1, min=0, max=4, sigma=0.3))
    assert FitParamCMAES.of(element, dict(sigma=1)).sigma() == 0.3
    assert FitParamGrid.of(element, dict(n=5)).values().size == 5
    with pytest.raises(ConfigError, match="a grid needs option 'n'"):
        FitParamGrid.of(element, {})
    # (a property of another fitter is known; a typo is not)
    with pytest.raises(
            ConfigError, match=r"'sigmaa' \(did you mean 'sigma'\?\)"):
        ParamSpace(
            dict(a=ParamScalarDesc()), dict(a=dict(value=1, n=5, sigmaa=1)))


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
    # fit: by (dx)^2 times the diagonal of the inverse covariance, along
    # each free element at fixed others
    x = np.array(list(best.free.values()))
    hessian = np.linalg.inv(best.covariance.matrix)
    i = list(best.free).index('posa')
    x[i] += 0.1
    growth = objective.chi2(params.evaluate(x)) - best.chi2
    assert growth == pytest.approx(0.01 * hessian[i, i], rel=0.05)


def test_least_squares_fit_the_scale_of_the_errors(driver):
    # Errors of half the noise: the fitted scale is about 2 (to 1/sqrt(2N)
    # of it, of N values), and the reduced chi-square, of the data alone,
    # N / dof (at the best scale the data have a chi-square of N; to the
    # tolerance of the fit), not about 7 (with the noise residual, N log
    # (2 / 0.1)^2)
    from gbkfit.fitting.fitters.scipy import FitterScipyLeastSquares
    likelihood = LikelihoodGaussian(
        noise=dict(brightness=FittedNoise(scale=True, scale_min=0.1)))
    objective = mock_objective(
        driver, error_scale=0.5, likelihood=likelihood, name='mock')
    params = ParamSpace(objective.pdescs(), bounded_properties() | dict(
        mock_noise_scale_brightness=dict(value=1.0, min=0.1, max=10)))
    best = FitterScipyLeastSquares().fit(objective, params).best()
    n = 32 * 32
    assert best.free['mock_noise_scale_brightness'] == pytest.approx(
        2, abs=4 * 2 / np.sqrt(2 * n))
    assert best.reduced_chi2 == pytest.approx(n / best.dof, rel=1e-2)


def test_a_point_where_the_model_fails_does_not_end_the_fit(monkeypatch):
    # (e.g. a solve that does not converge there): it is a point that
    # cannot be evaluated, counted, and the fit goes on
    from gbkfit.fitting import fitter_parser
    from gbkfit.params import ModelEvaluationError
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    merits = objective.merits

    def failing(values):
        if values['incl'] > 60:
            raise ModelEvaluationError("no model here")
        return merits(values)
    monkeypatch.setattr(objective, 'merits', failing)
    result = fitter_parser.load(dict(
        type='cmaes', maxfevals=300, seed=1,
        uncertainty=dict(covariance=False))).fit(objective, params)
    assert result.diagnostics['failed_nevals'] > 0
    assert result.best().free['incl'] <= 60


def test_a_prior_chosen_by_the_aicp_needs_fits_of_the_aicp():
    # (a fit of the chi-square would minimise a mix of both)
    from gbkfit.fitting import fitter_parser
    plan = mock_objective(
        DriverHost(), name='obs', flux_scale='one').group_plan()
    params = ParamSpace(plan.group().pdescs(), bounded_properties() | dict(
        obs_flux_scale=dict(solve=dict(
            type='gaussian', mean=1, sigma_aicp=[0.1, 10], min=0,
            likelihood='profiled'))))
    objective = Objective(plan, params.solves())
    with pytest.raises(ConfigError, match="minimises the aicp"):
        fitter_parser.load(dict(type='cmaes')).fit(objective, params)
    result = fitter_parser.load(dict(
        type='cmaes', merit='aicp', maxfevals=100)).fit(objective, params)
    assert result.best().aicp < np.inf


def test_the_reduced_chi2_is_of_the_data_alone():
    # A solved flux scale under a prior far from what the data say (the
    # brightness fixed at its truth, so the scale is 1): the residual of
    # its prior is in the chi-square, but not of the data, and the degrees
    # of freedom leave its value out (nnuisance)
    from gbkfit.fitting.fitters.scipy import FitterScipyLeastSquares
    plan = mock_objective(
        DriverHost(), name='obs', flux_scale='one').group_plan()
    params = ParamSpace(plan.group().pdescs(), bounded_properties() | dict(
        pt_a=TRUE['pt_a'], obs_flux_scale=dict(solve=dict(
            type='gaussian', mean=1.5, sigma=0.05, min=0))))
    objective = Objective(plan, params.solves())
    best = FitterScipyLeastSquares().fit(objective, params).best()
    residuals = objective.residuals(params.evaluate(list(best.free.values())))
    data_chi2 = float(np.sum(residuals[:objective.nresiduals()] ** 2))
    assert best.chi2 - data_chi2 > 50
    assert best.reduced_chi2 == pytest.approx(data_chi2 / best.dof, rel=1e-6)


def test_lmfit_says_why_leastsq_stopped():
    # lmfit reported every MINPACK convergence as 'Fit succeeded.' (also
    # when no step was accepted), and a run stopped by max_nfev as
    # 'Tolerance seems to be too small.'
    pytest.importorskip('lmfit')
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    converged = FitterLMFitLeastSquares().fit(objective, params)
    assert converged.success
    assert 'relative' in converged.message or 'cosine' in converged.message
    stopped = FitterLMFitLeastSquares(max_nfev=3).fit(objective, params)
    assert not stopped.success
    assert stopped.message == \
        'Fit aborted: number of function evaluations > 3'


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


@pytest.mark.parametrize('fitter', [
    dict(type='lmfit.least_squares'), dict(type='cmaes')])
def test_optimizers_with_a_gaussian_prior_find_the_maximum_a_posteriori(
        fitter):
    # The data give incl = x +- s; a Gaussian prior as wide, two errors
    # above, moves it halfway, with an error of s / sqrt(2). The chi2 of
    # the solution is that of the data.
    pytest.importorskip('cma')
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    alone = FitterLMFitLeastSquares().fit(objective, params).best()
    x, s = alone.free['incl'], alone.covariance.std['incl']
    properties = bounded_properties()
    properties['incl']['prior'] = dict(type='gauss', mean=x + 2 * s, std=s)
    best = fitter_parser.load(fitter).fit(
        objective, ParamSpace(objective.pdescs(), properties)).best()
    assert best.free['incl'] == pytest.approx(x + s, abs=0.1 * s)
    assert best.covariance.std['incl'] == pytest.approx(
        s / np.sqrt(2), rel=0.05)
    assert best.chi2 > alone.chi2 and best.log_prior is not None


@pytest.mark.parametrize('expr, derived', [
    ('incl', None), ('i', dict(i='incl'))])
@pytest.mark.parametrize('fitter', [
    dict(type='lmfit.least_squares'), dict(type='cmaes')])
def test_a_prior_on_an_expression_acts_as_the_prior_of_an_element(
        fitter, expr, derived):
    # (a Gaussian prior on the expression 'incl', as the prior of incl
    # in the test above: it moves incl halfway, with an error s / sqrt(2);
    # the same on a derived value of it)
    pytest.importorskip('cma')
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    alone = FitterLMFitLeastSquares().fit(objective, params).best()
    x, s = alone.free['incl'], alone.covariance.std['incl']
    params = ParamSpace(
        objective.pdescs(), bounded_properties(),
        priors=[dict(type='gauss', expr=expr, mean=x + 2 * s, std=s)],
        derived=derived)
    best = fitter_parser.load(fitter).fit(objective, params).best()
    assert best.free['incl'] == pytest.approx(x + s, abs=0.1 * s)
    assert best.covariance.std['incl'] == pytest.approx(
        s / np.sqrt(2), rel=0.05)

def test_points_where_a_prior_cannot_be_evaluated_are_rejected():
    # Where the derived value that a prior reads is NaN (xpos < 0), the
    # cost is inf, and the log-prior and what nested samplers integrate
    # are -inf; least squares get the error
    objective = mock_objective(DriverHost())
    params = ParamSpace(
        objective.pdescs(), bounded_properties(),
        priors=[dict(type='gauss', expr='w', mean=1, std=0.5)],
        derived=dict(w='xpos if xpos > 0 else np.nan'))
    problem = FitProblem(objective, params, [
        PriorUniform(lo, hi) for _, lo, hi in START.values()])
    inside = np.array([TRUE[name] for name in START])
    outside = inside.copy()
    outside[list(START).index('xpos')] = -0.5
    assert problem.joint_prior_log_density(inside) == pytest.approx(
        -0.5 * ((TRUE['xpos'] - 1) / 0.5) ** 2)
    assert math.isfinite(problem.cost(inside))
    assert problem.cost(outside) == math.inf
    assert problem.log_prior(outside) == -math.inf
    assert problem.nested_log_likelihood(outside) == -math.inf
    assert problem.joint_prior_log_density(outside) == -math.inf
    with pytest.raises(ParamEvaluationError, match="'w' is nan"):
        problem.residuals(outside)


class StopAfter(FitControl):
    """A control that stops a fit after n evaluations."""

    def __init__(self, n):
        super().__init__()
        self._n = n

    def stopped(self):
        self._n -= 1
        return self._n < 0


@pytest.mark.parametrize('fitter', [
    dict(type='lmfit.least_squares'), dict(type='cmaes')])
def test_a_stopped_fit_ends_with_its_best_point(fitter):
    # Evaluated once, without its covariance
    pytest.importorskip('cma')
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    result = fitter_parser.load(fitter).fit(
        objective, params, control=StopAfter(30))
    assert not result.success and 'stopped' in result.message
    assert result.nevals == 30
    assert result.best().covariance is None
    assert result.diagnostics['uncertainty_nevals'] == 1


def test_a_fit_stopped_in_its_uncertainty_ends_without_it():
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    control = FitControl()

    def observer(progress):
        if progress.stage == 'uncertainty':
            control.stop()
    result = fitter_parser.load(dict(type='lmfit.least_squares')).fit(
        objective, params, [observer], control=control)
    assert result.success
    assert result.best().covariance is None
    assert result.diagnostics['uncertainty_nevals'] == 1
    assert result.diagnostics['uncertainty_errors'] == [
        "no covariance: the fit was asked to stop"]


def test_a_fit_resumed_from_its_log_is_the_fit_that_did_not_stop(tmp_path):
    pytest.importorskip('cma')
    from gbkfit.fitting.fitters.cmaes import FitterCMAES
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    whole = FitterCMAES().fit(objective, params).best()
    filename = str(tmp_path / 'evaluations.bin')
    FitterCMAES().fit(
        objective, params, control=StopAfter(500),
        log=EvaluationLog(filename))
    log = EvaluationLog.resume(filename)
    resumed = FitterCMAES().fit(objective, params, log=log).best()
    assert log.nreplayed() == 500
    assert resumed.free == whole.free
    assert resumed.chi2 == pytest.approx(whole.chi2, rel=1e-12)


def two_workers(kind, driver):
    """Return two workers of a kind on a driver."""
    return WorkersThreads([driver, driver]) if kind == 'threads' \
        else WorkersProcesses(drivers=[driver, driver])


@pytest.mark.parametrize('kind', ['threads', 'processes'])
def test_workers_evaluate_batches_as_the_objective_does(driver, kind):
    # (copies on one device: the chi-squares differ at rounding, their
    # sums being in the order of the threads of each copy)
    objective = mock_objective(driver)
    params = ParamSpace(objective.pdescs(), bounded_properties())
    xs = np.random.default_rng(2).uniform(
        [-1, -1, 30, 45, 0.8, 3], [1, 1, 50, 65, 1.2, 5], (5, 6))
    alone = FitProblem(objective, params)
    with two_workers(kind, driver).start(objective) as pool:
        pooled = FitProblem(objective, params, pool=pool)
        np.testing.assert_allclose(
            pooled.chi2_batch(xs), alone.chi2_batch(xs), rtol=1e-12)
        for got, want in zip(
                pooled.residuals_batch(xs), alone.residuals_batch(xs)):
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert pooled.nevals() == alone.nevals() == 10
    assert pooled.best()[1] == pytest.approx(alone.best()[1], rel=1e-12)


def test_workers_take_placements_of_their_copies(driver):
    # (one of other drivers of its observation, by its name, gives the
    # same values)
    objective = mock_objective(driver, name='image')
    params = ParamSpace(objective.pdescs(), bounded_properties())
    xs = np.random.default_rng(3).uniform(
        [-1, -1, 30, 45, 0.8, 3], [1, 1, 50, 65, 1.2, 5], (4, 6))
    workers = WorkersThreads([(driver, {}), (driver, dict(image=DriverHost()))])
    with workers.start(objective) as pool:
        np.testing.assert_allclose(
            FitProblem(objective, params, pool=pool).chi2_batch(xs),
            FitProblem(objective, params).chi2_batch(xs), rtol=1e-5)
    with pytest.raises(ConfigError, match="unknown galaxies or observations"):
        WorkersThreads([(driver, dict(hi=driver))]).start(objective)


@pytest.mark.parametrize('kind', ['threads', 'processes'])
@pytest.mark.parametrize('fitter', [
    dict(type='lmfit.least_squares'), dict(type='cmaes')])
def test_a_fit_with_workers_is_the_fit_without(driver, fitter, kind):
    pytest.importorskip('cma')
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(driver)
    params = ParamSpace(objective.pdescs(), bounded_properties())
    alone = fitter_parser.load(fitter).fit(objective, params)
    with two_workers(kind, driver).start(objective) as pool:
        pooled = fitter_parser.load(fitter).fit(objective, params, pool=pool)
    assert pooled.best().free == pytest.approx(alone.best().free, rel=1e-12)
    assert pooled.best().covariance.std == pytest.approx(
        alone.best().covariance.std, rel=1e-12)
    assert pooled.nevals == alone.nevals


def test_a_fit_with_workers_logs_its_evaluations_in_order(tmp_path):
    # (a fit that resumes from its log without workers replays all of it)
    pytest.importorskip('cma')
    from gbkfit.fitting.fitters.cmaes import FitterCMAES
    driver = DriverHost()
    objective = mock_objective(driver)
    params = ParamSpace(objective.pdescs(), bounded_properties())
    filename = str(tmp_path / 'evaluations.bin')
    with WorkersThreads([driver, driver]).start(objective) as pool:
        pooled = FitterCMAES().fit(
            objective, params, log=EvaluationLog(filename), pool=pool)
    log = EvaluationLog.resume(filename)
    resumed = FitterCMAES().fit(objective, params, log=log)
    assert log.nreplayed() == pooled.nevals
    assert resumed.best().free == pooled.best().free


@pytest.mark.parametrize('sampler', ['emcee', 'zeus'])
def test_a_finished_chain_discards_all_its_burn_in(sampler):
    # A burn-in of more than half the chain (15 of 20 steps of 16
    # walkers): the posterior is of the 5 steps after it
    pytest.importorskip(sampler)
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    fitter = fitter_parser.load(
        dict(type=sampler, nwalkers=16, nsteps=20, burnin=15))
    result = fitter.fit(objective, params)
    assert len(result.posterior.samples) == 16 * 5


def test_evaluations_that_are_not_numbers_are_outside(tmp_path, caplog):
    # A chi-square that is not a number (of a likelihood that is NaN where
    # cut > 0) is that of parameters that cannot be evaluated, for every
    # fitter (emcee refused it): counted, and the first logged
    pytest.importorskip('emcee')
    from gbkfit.fitting import fitter_parser
    (tmp_path / 'like.py').write_text(
        "import numpy as np\n"
        "def log_like(item, data, model, error, mask, params):\n"
        "    if params['cut'] > 0:\n"
        "        return np.nan\n"
        "    return -0.5 * np.sum(((model - data) / error) ** 2)\n")
    likelihood = LikelihoodUser.from_file(
        str(tmp_path / 'like.py'), 'log_like', 'log_likelihood', ['cut'])
    objective = mock_objective(
        DriverHost(), likelihood=likelihood, name='mock')
    params = ParamSpace(objective.pdescs(), prior_properties(10) | dict(
        mock_cut=dict(min=-1, max=1)))
    problem = FitProblem(objective, params)
    xs = [[TRUE[name] for name in problem.names()[:-1]] + [cut]
          for cut in (-0.5, 0.5)]
    assert np.isfinite(problem.cost_batch(xs)[0])
    assert problem.cost_batch(xs)[1] == np.inf
    assert problem.log_likelihood_batch(xs)[1] == -np.inf
    assert problem.nan_nevals() == 3
    # (without residuals, a fit has no covariance by default, and one
    # asked for is refused before the fit, not after it)
    fitter = dict(type='emcee', nwalkers=16, nsteps=20, burnin=10)
    with pytest.raises(ConfigError, match="has no residuals"):
        fitter_parser.load(fitter | dict(
            uncertainty=dict(covariance=True))).fit(objective, params)
    result = fitter_parser.load(fitter).fit(objective, params)
    assert result.best().covariance is None
    assert result.diagnostics['nan_nevals'] > 0
    assert result.best().free['mock_cut'] <= 0
    logged = [record for record in caplog.records
              if 'not a number' in record.message]
    assert len(logged) == 2


def test_a_stopped_emcee_keeps_its_chain_so_far(tmp_path):
    # The steps go to the log as the chain runs (16 walkers, 6 elements
    # and the log-probability)
    pytest.importorskip('emcee')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    fitter = fitter_parser.load(dict(type='emcee', nwalkers=16, nsteps=100))
    result = fitter.fit(
        objective, params, control=StopAfter(16 * 21),
        log=EvaluationLog(str(tmp_path / 'evaluations.bin')))
    nsteps = result.diagnostics['nsteps']
    assert not result.success and 0 < nsteps < 100
    assert result.posterior is not None
    with open(tmp_path / 'evaluations-chain.bin', 'rb') as file:
        file.readline()
        file.readline()
        chain = np.frombuffer(file.read()).reshape(-1, 16, 6 + 1)
    assert len(chain) == nsteps


def test_a_stopped_zeus_keeps_its_chain_so_far(tmp_path):
    # The steps go to the log as the chain runs (16 walkers, 6 elements
    # and the log-probability)
    pytest.importorskip('zeus')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    fitter = fitter_parser.load(dict(type='zeus', nwalkers=16, nsteps=100))
    result = fitter.fit(
        objective, params, control=StopAfter(16 * 40),
        log=EvaluationLog(str(tmp_path / 'evaluations.bin')))
    nsteps = result.diagnostics['nsteps']
    assert not result.success and 0 < nsteps < 100
    assert result.posterior is not None
    with open(tmp_path / 'evaluations-chain.bin', 'rb') as file:
        file.readline()
        file.readline()
        chain = np.frombuffer(file.read()).reshape(-1, 16, 6 + 1)
    assert len(chain) == nsteps


def test_zeus_is_seeded_and_leaves_the_global_random_numbers_alone():
    # (zeus draws the global random numbers of NumPy and of random)
    pytest.importorskip('zeus')
    import random
    from gbkfit.fitting.fitters.zeus import FitterZeus
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    np.random.seed(7)
    random.seed(7)
    numpy_state = np.random.get_state()[1].copy()
    random_state = random.getstate()
    chains = [
        FitterZeus(nwalkers=16, nsteps=6, seed=3).fit(
            objective, params).posterior.samples
        for _ in range(2)]
    np.testing.assert_array_equal(chains[0], chains[1])
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
    assert random.getstate() == random_state
    with pytest.raises(ConfigError, match="must be even"):
        FitterZeus(nwalkers=15)
    with pytest.raises(ConfigError, match="start after a burn-in"):
        FitterZeus(nsteps=10, burnin=0, moves='global')


def test_a_grid_maps_the_chi2_of_two_free_elements():
    # (the others fixed at the truth, which is on the grid: its lowest
    # point, to the noise)
    from gbkfit.fitting.fitters.grid import FitterGrid
    objective = mock_objective(DriverHost())
    properties = TRUE | dict(
        posa=dict(min=36, max=44, n=9), incl=dict(min=50, max=60, n=11))
    params = ParamSpace(objective.pdescs(), properties)
    result = FitterGrid(batch=7).fit(objective, params)
    grid = result.diagnostics['grid']
    chi2 = np.array(grid['cost'])
    assert grid['names'] == ['posa', 'incl'] and chi2.shape == (9, 11)
    assert result.success and result.nevals == 9 * 11
    assert result.best().free == dict(posa=40, incl=55)
    i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    assert (grid['axes'][0][i], grid['axes'][1][j]) == (40, 55)
    assert chi2[2, 3] == pytest.approx(objective.chi2(params.evaluate(
        [grid['axes'][0][2], grid['axes'][1][3]])))


def test_a_grid_takes_the_priors_on_several_parameters():
    # A tight prior on posa at 36 moves the best point of the grid there
    from gbkfit.fitting.fitters.grid import FitterGrid
    objective = mock_objective(DriverHost())
    properties = TRUE | dict(
        posa=dict(min=36, max=44, n=9), incl=dict(min=50, max=60, n=11))
    params = ParamSpace(objective.pdescs(), properties, priors=[
        dict(type='gauss', expr='posa', mean=36, std=0.01)])
    result = FitterGrid().fit(objective, params)
    assert result.best().free['posa'] == 36


def test_a_stopped_grid_keeps_the_points_it_evaluated():
    from gbkfit.fitting.fitters.grid import FitterGrid
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), TRUE | dict(
        posa=dict(min=36, max=44, n=9), incl=dict(min=50, max=60, n=6)))
    result = FitterGrid(batch=10).fit(
        objective, params, control=StopAfter(25))
    chi2 = np.array(result.diagnostics['grid']['cost'])
    assert not result.success and result.message.startswith('stopped')
    assert np.count_nonzero(np.isfinite(chi2)) == 20
    with pytest.raises(ConfigError, match="more than max_points"):
        FitterGrid(max_points=50).fit(objective, params)
    from gbkfit.fitting.fitters.grid import FitParamGrid
    with pytest.raises(ConfigError, match="at least 2"):
        FitParamGrid(0, 1, 1)


def test_optimizers_of_the_aicp_minimise_it():
    # A flux scale solved under a prior: the AICp is not the chi-square.
    # A grid of merit 'aicp' maps it; its best solution reports it with
    # twice the free element, and has no covariance (the AICp is not a
    # likelihood)
    from gbkfit.fitting.fitters.cmaes import FitterCMAES
    from gbkfit.fitting.fitters.grid import FitterGrid
    from gbkfit.fitting.fitters.multistart import FitterMultiStart
    from gbkfit.fitting.uncertainty import UncertaintyOptions
    plan = mock_objective(
        DriverHost(), name='obs', flux_scale='one').group_plan()
    params = ParamSpace(plan.group().pdescs(), TRUE | dict(
        obs_flux_scale=dict(solve=dict(
            type='gaussian', mean=1, sigma=0.1, min=0)),
        posa=dict(min=36, max=44, n=5)))
    objective = Objective(plan, params.solves())
    result = FitterGrid(merit='aicp').fit(objective, params)
    grid = result.diagnostics['grid']
    merits = objective.merits(params.evaluate([grid['axes'][0][1]]))
    assert merits.aicp > merits.chi2 + 1
    assert grid['cost'][1] == pytest.approx(merits.aicp)
    best = result.best()
    assert best.aicp == pytest.approx(
        objective.aicp(params.evaluate(list(best.free.values()))) + 2)
    assert best.covariance is None and best.laplace is None
    assert FitterMultiStart(FitterCMAES(merit='aicp')).merit() == 'aicp'
    with pytest.raises(ConfigError, match="not a likelihood"):
        FitterCMAES(merit='aicp', uncertainty=UncertaintyOptions())
    with pytest.raises(ConfigError, match="option 'merit'"):
        FitterGrid(merit='bic')


def test_least_squares_take_priors_with_residuals_only():
    from gbkfit.fitting.fitparams import FitParamLeastSquares
    element = FreeElement.load(dict(
        value=1.0, min=0, max=3, prior=dict(type='gauss', mean=1, std=0.5)))
    assert FitParamLeastSquares.of(element, {}).prior().has_residual()
    with pytest.raises(ConfigError, match="least squares cannot take"):
        FitParamLeastSquares.of(FreeElement.load(dict(
            value=2, prior=dict(type='log_uniform', min=1, max=10))), {})




def test_errors_propagate_to_tied_and_fixed_elements():
    pytest.importorskip('lmfit')
    from gbkfit.fitting.fitters.lmfit import FitterLMFitLeastSquares
    objective = mock_objective(DriverHost())
    properties = bounded_properties() | dict(ypos='xpos - 1.1', pt_s=4.0)
    params = ParamSpace(objective.pdescs(), properties)
    best = FitterLMFitLeastSquares().fit(objective, params).best()
    std = best.covariance.element_std
    assert std['ypos'] == pytest.approx(std['xpos'], rel=1e-6)
    assert std['pt_s'] == 0
    assert best.covariance.at_bounds == ()
    laplace = best.laplace.summary()
    assert laplace['ypos'][1] == pytest.approx(laplace['xpos'][1] - 1.1)


def test_unconstrained_elements_have_infinite_errors(tmp_path):
    # A parameter that changes nothing (of a likelihood that does not use
    # it): its error, and that of a value of it, is infinite (not 0), and
    # there are no Laplace samples
    from gbkfit.fitting.fitters.scipy import FitterScipyLeastSquares
    (tmp_path / 'like.py').write_text(
        "def residuals(item, data, model, error, mask, params):\n"
        "    return (model - data) / error\n")
    likelihood = LikelihoodUser.from_file(
        str(tmp_path / 'like.py'), 'residuals', 'residuals', ['unused'])
    objective = mock_objective(
        DriverHost(), likelihood=likelihood, name='mock')
    params = ParamSpace(
        objective.pdescs(), bounded_properties() | dict(
            mock_unused=dict(value=0.5, min=0, max=1)),
        derived=dict(d='mock_unused + xpos'))
    result = FitterScipyLeastSquares().fit(objective, params)
    cov = result.best().covariance
    assert cov.unconstrained == ('mock_unused',)
    assert cov.std['mock_unused'] == np.inf
    assert np.isfinite(cov.std['xpos'])
    assert cov.derived_std['d'] == np.inf
    assert result.best().laplace is None
    assert any('no Laplace samples' in message
               for message in result.diagnostics['uncertainty_errors'])


def test_derived_values_get_errors_and_samples():
    # The error of a derived value is propagated from the covariance, as
    # that of a tied element (here of xpos + 2 ypos, with their
    # covariance), and its Laplace samples are its values at theirs
    from gbkfit import outputs
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(
        objective.pdescs(), bounded_properties(),
        derived=dict(d='xpos + 2 * ypos', vs='np.array([pt_a, pt_s])'))
    result = fitter_parser.load(dict(type='scipy.least_squares')).fit(
        objective, params)
    best = result.best()
    free = best.free
    assert best.derived == pytest.approx({
        'd': free['xpos'] + 2 * free['ypos'], 'vs[0]': free['pt_a'],
        'vs[1]': free['pt_s']})
    cov = best.covariance
    i, j = cov.names.index('xpos'), cov.names.index('ypos')
    m = cov.matrix
    assert cov.derived_std['d'] == pytest.approx(
        np.sqrt(m[i, i] + 4 * m[j, j] + 4 * m[i, j]), rel=1e-6)
    assert cov.derived_std['vs[1]'] == pytest.approx(cov.std['pt_s'], rel=1e-6)
    names = best.laplace.names
    samples = best.laplace.samples
    np.testing.assert_allclose(
        samples[:, names.index('d')],
        samples[:, names.index('xpos')] + 2 * samples[:, names.index('ypos')],
        rtol=1e-12)
    assert any(line.startswith('vs[1] ') for line in result.summary().split(
        '\n'))
    with outputs.create('run', 'fit') as run:
        result.write(run)
    run = outputs.open('run')
    written = run.read(run.product('fit_result'))['solutions']
    assert written[0]['derived'] == best.derived
    assert written[0]['covariance']['derived_std'] == cov.derived_std
    laplace = run.read(run.product('laplace', solution='0'))
    assert laplace.colnames[-3:] == ['d', 'vs[0]', 'vs[1]']


def test_derived_values_that_fail_are_reported():
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(
        objective.pdescs(), bounded_properties(),
        derived=dict(bad='np.take(np.array([xpos]), 5)'))
    result = fitter_parser.load(dict(type='scipy.least_squares')).fit(
        objective, params)
    assert result.best().derived == {}
    assert result.best().covariance.derived_std == {}
    assert "derived value 'bad'" in result.diagnostics['derived_errors'][0]

def test_a_bootstrap_refits_the_model_plus_its_residuals_redrawn():
    # Refits of the residuals redrawn in blocks: of white noise, they
    # scatter about as the covariance says; the derived values are those
    # of each refit
    import dataclasses
    from gbkfit import outputs
    from gbkfit.fitting import (
        BootstrapBlocks, BootstrapOptions, bootstrap, fitter_parser)
    objective = mock_objective(DriverHost())
    params = ParamSpace(
        objective.pdescs(), bounded_properties(),
        derived=dict(d='xpos + 2 * ypos'))
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    result = fitter.fit(objective, params)
    boot = bootstrap(
        objective, params, result.best().elements, fitter,
        BootstrapOptions(10, BootstrapBlocks([4, 4])))
    assert boot.samples.shape == (10, 7) and not boot.failures
    x, y, d = (boot.samples[:, boot.names.index(name)]
               for name in ('xpos', 'ypos', 'd'))
    np.testing.assert_allclose(d, x + 2 * y, rtol=1e-12)
    std = result.best().covariance.std
    for name in TRUE:
        assert 0.5 < boot.std()[name] / std[name] < 2
    result = dataclasses.replace(result, bootstrap=boot)
    assert 'bootstrap 16/50/84' in result.summary()
    with outputs.create('run', 'fit') as run:
        result.write(run)
    run = outputs.open('run')
    written = run.read(run.product('fit_result'))['bootstrap']
    assert (written['nsamples'], written['nfailed']) == (10, 0)
    assert written['options']['scheme'] == dict(type='blocks', size=[4, 4])
    assert run.read(run.product('bootstrap')).colnames \
        == list(boot.names) + ['converged']


@pytest.mark.parametrize('kind', ['threads', 'processes'])
def test_a_bootstrap_on_workers_is_the_bootstrap_without(kind):
    # (each refit draws its residuals with a stream of its own: on workers,
    # in any order, it refits the same data)
    from gbkfit.fitting import (
        BootstrapBlocks, BootstrapOptions, bootstrap, fitter_parser)
    driver = DriverHost()
    objective = mock_objective(driver)
    params = ParamSpace(
        objective.pdescs(), bounded_properties(),
        derived=dict(d='xpos + 2 * ypos'))
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    start = fitter.fit(objective, params).best().elements
    options = BootstrapOptions(5, BootstrapBlocks([4, 4]))
    alone = bootstrap(objective, params, start, fitter, options)
    with two_workers(kind, driver).start(objective) as pool:
        pooled = bootstrap(objective, params, start, fitter, options, pool)
    assert pooled.names == alone.names
    np.testing.assert_allclose(pooled.samples, alone.samples, rtol=1e-9)
    assert list(pooled.converged) == list(alone.converged)


class StopAfterOutcomes(SubFitLog):
    """A log of sub-fits that stops a fit once it has n outcomes."""

    def __init__(self, filename, control, n):
        super().__init__(filename)
        self._control = control
        self._n = n

    def append(self, index, outcome):
        super().append(index, outcome)
        if len(self.outcomes()) >= self._n:
            self._control.stop()


def test_a_stopped_bootstrap_resumes_from_its_log(tmp_path):
    # (stopped after two refits, it resumes with them, and ends as the
    # bootstrap that did not stop)
    from gbkfit.fitting import (
        BootstrapGaussian, BootstrapOptions, bootstrap, fitter_parser)
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    start = fitter.fit(objective, params).best().elements
    options = BootstrapOptions(4, BootstrapGaussian())
    whole = bootstrap(objective, params, start, fitter, options)
    filename = str(tmp_path / 'bootstrap.jsonl')
    control = FitControl()
    log = StopAfterOutcomes(filename, control, 2)
    with pytest.raises(FitStopped):
        bootstrap(objective, params, start, fitter, options, None, control,
                  log)
    log.close()
    assert sorted(SubFitLog.resume(filename).outcomes()) == [0, 1]
    resumed = bootstrap(
        objective, params, start, fitter, options,
        log=SubFitLog.resume(filename))
    np.testing.assert_array_equal(resumed.samples, whole.samples)


def run_until_stopped(objective, item, control):
    """A sub-fit that runs until its control stops it."""
    while not control.stopped():
        time.sleep(0.01)
    raise FitStopped("stopped")


@pytest.mark.parametrize('kind', ['threads', 'processes'])
def test_workers_stop_their_sub_fits(kind):
    driver = DriverHost()
    control = FitControl()
    threading.Timer(0.5, control.stop).start()
    with two_workers(kind, driver).start(mock_objective(driver)) as pool:
        begin = time.perf_counter()
        with pytest.raises(FitStopped):
            pool.map_lazily(
                run_until_stopped, lambda i: i, 10, control=control)
        # (processes take seconds to start; the sub-fits would never end)
        assert time.perf_counter() - begin < 30


def test_a_gaussian_bootstrap_draws_the_noise_anew():
    # Refits of the model plus Gaussian noise of the errors: of white
    # noise, they scatter about as the covariance says
    from gbkfit.fitting import (
        BootstrapGaussian, BootstrapOptions, bootstrap, fitter_parser)
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    result = fitter.fit(objective, params)
    boot = bootstrap(
        objective, params, result.best().elements, fitter,
        BootstrapOptions(10, BootstrapGaussian()))
    assert boot.samples.shape == (10, 6) and not boot.failures
    std = result.best().covariance.std
    for name in TRUE:
        assert 0.5 < boot.std()[name] / std[name] < 2


def test_a_bootstrap_keeps_refits_that_do_not_converge():
    # Refits stopped by their budget of evaluations are kept, and flagged:
    # leaving them out would bias the samples to the refits easy to fit
    from gbkfit.fitting import (
        BootstrapGaussian, BootstrapOptions, bootstrap, fitter_parser)
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    result = fitter_parser.load(dict(type='scipy.least_squares')).fit(
        objective, params)
    boot = bootstrap(
        objective, params, result.best().elements,
        fitter_parser.load(dict(type='scipy.least_squares', max_nfev=2)),
        BootstrapOptions(3, BootstrapGaussian()))
    assert len(boot.samples) == 3 and not boot.failures
    assert not boot.converged.any()
    assert boot.info()['nunconverged'] == 3


def test_a_bootstrap_refits_fits_of_solved_values():
    # (the refits start at copies of the parameters, whose solves are equal
    # to those of the objective)
    from gbkfit.fitting import (
        BootstrapGaussian, BootstrapOptions, bootstrap, fitter_parser)
    plan = mock_objective(
        DriverHost(), name='obs', flux_scale='one').group_plan()
    params = ParamSpace(plan.group().pdescs(), bounded_properties() | dict(
        obs_flux_scale=dict(solve=dict(
            type='gaussian', mean=1, sigma=0.1, min=0))))
    objective = Objective(plan, params.solves())
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    result = fitter.fit(objective, params)
    boot = bootstrap(
        objective, params, result.best().elements, fitter,
        BootstrapOptions(3, BootstrapGaussian()))
    assert len(boot.samples) == 3 and not boot.failures


@pytest.mark.parametrize('scheme, message', [
    ([dict(type='blocks', size=[4, 4])] * 2, 'one for each of the 1 obs'),
    (dict(type='blocks', size=[4, 4, 2]), 'for each of x and y'),
    (dict(type='blocks', size=[4, 0]), 'of positive numbers'),
])
def test_bootstrap_options_that_do_not_suit_the_fit_refused(scheme, message):
    from gbkfit.fitting import BootstrapOptions, bootstrap, fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    with pytest.raises(ConfigError, match=message):
        bootstrap(objective, params, TRUE, fitter, BootstrapOptions.load(
            dict(nsamples=5, scheme=scheme)))


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


def test_cmaes_gives_the_minimum_of_its_own_run():
    # Two runs on one problem (as multistart makes them): the second, in
    # bounds of posa without a minimum (the minima are at 40 and 220),
    # gives its own best point, within them, not the minimum of the first
    pytest.importorskip('cma')
    from gbkfit.fitting.fitters.cmaes import FitterCMAES
    from gbkfit.fitting.problem import FitProblem
    objective = mock_objective(DriverHost())
    fitter = FitterCMAES(maxfevals=300)
    rng = np.random.default_rng(1)
    params = ParamSpace(objective.pdescs(), bounded_properties())
    problem = FitProblem(objective, params)
    fitter._run(problem, list(fitter.load_fit_params(params).values()), rng)
    posa = list(problem.names()).index('posa')
    boxed = ParamSpace(objective.pdescs(), bounded_properties() | dict(
        posa=dict(value=120, min=100, max=150)))
    run = fitter._run(
        problem, list(fitter.load_fit_params(boxed).values()), rng)
    assert 100 <= run.solutions[0][posa] <= 150


@pytest.mark.parametrize('fitter', [
    dict(type='lmfit.least_squares'),
    dict(type='scipy.least_squares', method='lm'),
    dict(type='scipy.least_squares', method='trf')])
def test_least_squares_move_values_from_awkward_starts(fitter):
    # xpos starts near 0, where MINPACK's own step (sqrt(epsfcn) times
    # the value) is below the precision of float32 models; pt_s starts
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
    properties['pt_s'] |= dict(value=1, min=1)
    params = ParamSpace(objective.pdescs(), properties)
    result = fitter_parser.load(fitter).fit(objective, params)
    # (with float64 FFTs, scipy's 'lm' stops 1.4e-5 above the others, where
    # its xtol is met with the default step)
    assert result.best().chi2 == pytest.approx(local.chi2, rel=1e-4)


@pytest.mark.parametrize('fitter', [
    dict(type='scipy.least_squares', method='trf'),
    dict(type='lmfit.least_squares', method='least_squares')])
def test_trf_moves_a_value_that_starts_on_its_bound(fitter):
    # When every free value starts on a bound, TRF's first steps were
    # tiny and it stopped there, reporting success
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    properties = dict(TRUE) | dict(pt_s=dict(value=1, min=1, max=10))
    params = ParamSpace(objective.pdescs(), properties)
    best = fitter_parser.load(fitter).fit(objective, params).best()
    assert best.free['pt_s'] == pytest.approx(
        TRUE['pt_s'], abs=5 * best.covariance.std['pt_s'])


@pytest.mark.filterwarnings('ignore:invalid value:RuntimeWarning')
def test_a_fit_beyond_the_valid_range_of_a_parameter_fails():
    # The bounds of pt_s reach below its valid range: the fit fails before
    # it starts, instead of trying values where the model is not defined
    from gbkfit.fitting import fitter_parser
    from gbkfit.params import ParamScalarDesc
    objective = mock_objective(DriverHost())
    pdescs = objective.pdescs() | dict(pt_s=ParamScalarDesc(min=2))
    params = ParamSpace(pdescs, bounded_properties())
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    with pytest.raises(ConfigError, match=r"'pt_s' \[1, 10\] \(valid: "):
        fitter.fit(objective, params)


@pytest.mark.parametrize('kind', ['threads', 'processes'])
def test_multistart_on_workers_is_multistart_without(kind):
    # (each start is a fit of its own, with its own random stream)
    pytest.importorskip('cma')
    from gbkfit.fitting import fitter_parser
    driver = DriverHost()
    objective = mock_objective(driver)
    params = ParamSpace(objective.pdescs(), bounded_properties())
    fitter = dict(type='multistart', local=dict(type='cmaes'), nstarts=4)
    alone = fitter_parser.load(fitter).fit(objective, params)
    with two_workers(kind, driver).start(objective) as pool:
        pooled = fitter_parser.load(fitter).fit(objective, params, pool=pool)
    assert pooled.best().free == pytest.approx(alone.best().free, rel=1e-9)
    assert pooled.nevals == alone.nevals


class LogStoppingAfterStarts(EvaluationLog):
    """A log of a fit whose log of sub-fits stops it after n outcomes."""

    def __init__(self, filename, control, n):
        super().__init__(filename)
        self._stopping = StopAfterOutcomes(
            super().subfits().filename(), control, n)

    def subfits(self):
        return self._stopping


def test_a_stopped_multistart_resumes_from_the_starts_of_its_log(tmp_path):
    # (stopped after two starts, it resumes with them, and runs the others)
    pytest.importorskip('cma')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    fitter = dict(type='multistart', local=dict(type='cmaes'), nstarts=4)
    whole = fitter_parser.load(fitter).fit(objective, params)
    filename = str(tmp_path / 'evaluations.bin')
    control = FitControl()
    stopped = fitter_parser.load(fitter).fit(
        objective, params, control=control,
        log=LogStoppingAfterStarts(filename, control, 2))
    assert not stopped.success
    log = EvaluationLog.resume(filename)
    assert sorted(log.subfits().outcomes()) == [0, 1]
    resumed = fitter_parser.load(fitter).fit(objective, params, log=log)
    assert resumed.best().free == whole.best().free
    assert resumed.nevals == whole.nevals


def test_multistart_skips_starts_the_parameters_reject():
    # pt_s is nan for xpos < 0.6: the starts there fail, and are skipped
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    properties = bounded_properties() | dict(
        xpos=dict(value=0.65, min=-4, max=4),
        pt_s='np.sqrt(xpos - 0.6) * 0 + 4')
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
        xpos=0.01, ypos=0.01, posa=0.25, incl=0.25, pt_a=0.01, pt_s=0.03)
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

@pytest.mark.slow
def test_bootstrap_errors_take_the_noise_the_beam_correlates():
    # Noise correlated by the PSF (as the beam correlates that of radio
    # data), of data that do not say so: the covariance, which takes it
    # as independent, gives errors about 3.5 times smaller than the
    # scatter of the fits over independent mocks; the bootstrap, in blocks
    # of about 2 FWHM of the PSF, about that scatter (0.85 of it on
    # average over 8 mocks, 1/2 in blocks of 2 pixels)
    from gbkfit.fitting import (
        BootstrapBlocks, BootstrapOptions, bootstrap, fitter_parser)
    from gbkfit.observation import NoiseGauss

    def group(dataset):
        observation = ObservationSky(
            PixelBrightness.from_data(dataset), Flat(),
            instrument=Instrument(psf=PSFGauss(1.5)), data=dataset)
        return ObservationGroup(
            galaxy_parser.load(GALAXY), observation).plan(DriverHost())
    image = group(DatasetPixelBrightness(Data(np.zeros((48, 48)))))
    values = ParamSpace(image.group().pdescs(), TRUE).evaluate()
    peak = image.model_h(values)[0]['brightness']['d'].max()
    noise = NoiseGauss(rms=0.01 * peak, spatial_correlation='instrument')
    fitter = fitter_parser.load(dict(type='scipy.least_squares'))
    fits = []
    for mock in image.mock(values, noise, nmocks=41, seed=5)[0]:
        # (without the correlation that mocks declare)
        dataset = DatasetPixelBrightness(Data(
            mock['brightness'].data(), error=mock['brightness'].error()))
        objective = Objective(group(dataset))
        params = ParamSpace(objective.pdescs(), bounded_properties())
        fits.append((objective, params, fitter.fit(objective, params)))
    scatter = np.std(
        [[result.best().free[name] for name in TRUE] for *_, result in fits],
        axis=0, ddof=1)
    ratios = []
    for objective, params, result in fits[:4]:
        covariance = np.array(
            [result.best().covariance.std[name] for name in TRUE])
        assert np.all(covariance < 0.5 * scatter)
        boot = bootstrap(
            objective, params, result.best().elements, fitter,
            BootstrapOptions(40, BootstrapBlocks([8, 8])))
        ratios.append(np.array([boot.std()[name] for name in TRUE]) / scatter)
    assert 0.7 < np.mean(ratios) < 1.3


@pytest.mark.slow
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


@pytest.mark.slow
def test_zeus_posterior_agrees_with_the_covariance():
    pytest.importorskip('zeus')
    from gbkfit.fitting.fitters.zeus import FitterZeus
    objective = mock_objective(DriverHost())
    properties = prior_properties(10)
    for name in properties:
        properties[name] |= dict(value=TRUE[name], width=1e-3)
    params = ParamSpace(objective.pdescs(), properties)
    result = FitterZeus(nwalkers=24, nsteps=400, burnin=100).fit(
        objective, params)
    check_posterior(result)


@pytest.mark.slow
def test_nautilus_posterior_agrees_with_the_covariance():
    pytest.importorskip('nautilus')
    from gbkfit.fitting.fitters.nautilus import FitterNautilus
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    result = FitterNautilus(n_live=300, n_eff=2000).fit(objective, params)
    check_posterior(result)
    assert math.isfinite(result.evidence.log_z)


@pytest.mark.slow
def test_dnest4_posterior_agrees_with_the_covariance():
    from gbkfit.fitting.fitters.dnest4 import FitterDNest4
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    result = FitterDNest4(
        new_level_interval=1000, save_interval=100, max_saves=2000
    ).fit(objective, params)
    check_posterior(result)
    assert math.isfinite(result.evidence.log_z)


@pytest.mark.slow
def test_dynesty_posterior_agrees_with_the_covariance():
    pytest.importorskip('dynesty')
    from gbkfit.fitting.fitters.dynesty import FitterDynestyStatic
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    result = FitterDynestyStatic(nlive=300).fit(objective, params)
    check_posterior(result)
    assert result.evidence.log_z_error > 0


def test_a_nested_evidence_of_profiled_solves_says_it_is_not_one(caplog):
    # (the likelihood takes their best values, not their integral)
    pytest.importorskip('dynesty')
    from gbkfit.fitting.fitters.dynesty import FitterDynestyStatic
    plan = mock_objective(
        DriverHost(), name='obs', flux_scale='one').group_plan()
    params = ParamSpace(plan.group().pdescs(), TRUE | dict(
        incl=dict(prior=dict(type='uniform', min=50, max=60)),
        obs_flux_scale=dict(solve=dict(
            type='gaussian', mean=1, sigma=0.5, min=0,
            likelihood='profiled'))))
    objective = Objective(plan, params.solves())
    result = FitterDynestyStatic(nlive=20, dlogz=10).fit(objective, params)
    assert "['obs_flux_scale']" in result.diagnostics['evidence_note']
    assert any('is not one' in r.message for r in caplog.records)


@pytest.mark.slow
def test_nested_evidence_with_a_prior_on_an_expression_is_corrected():
    # A uniform prior on incl times a Gaussian prior on the expression
    # 'incl' is a truncated Gaussian prior on incl: the same evidence,
    # once that of the run with the joint prior is normalised
    pytest.importorskip('dynesty')
    from gbkfit.fitting.fitters.dynesty import FitterDynestyStatic
    objective = mock_objective(DriverHost())
    properties = prior_properties(10)
    lo, hi = properties['incl']['min'], properties['incl']['max']
    mean, std = TRUE['incl'] + 0.5, 0.5
    joint = ParamSpace(
        objective.pdescs(), properties,
        priors=[dict(type='gauss', expr='incl', mean=mean, std=std)])
    properties['incl'] = dict(prior=dict(
        type='gauss_trunc', mean=mean, std=std, min=lo, max=hi))
    alone = ParamSpace(objective.pdescs(), properties)
    z_joint = FitterDynestyStatic(nlive=300).fit(objective, joint).evidence
    z_alone = FitterDynestyStatic(nlive=300).fit(objective, alone).evidence
    error = math.hypot(z_joint.log_z_error, z_alone.log_z_error)
    assert z_joint.log_z == pytest.approx(z_alone.log_z, abs=3 * error)

@pytest.mark.slow
@pytest.mark.filterwarnings('ignore:invalid value:RuntimeWarning')
def test_samplers_leave_out_points_the_parameters_reject():
    # pt_s is nan for xpos < 0.6: there the likelihood is -inf, and the
    # points nautilus keeps (with no weight) crashed the posterior
    pytest.importorskip('nautilus')
    from gbkfit.fitting.fitters.nautilus import FitterNautilus
    objective = mock_objective(DriverHost())
    properties = dict(TRUE) | dict(
        xpos=dict(min=0.5, max=0.9), posa=dict(min=35, max=45),
        pt_s='np.sqrt(xpos - 0.6) * 0 + 4')
    params = ParamSpace(objective.pdescs(), properties)
    result = FitterNautilus(n_live=200, n_eff=500).fit(objective, params)
    assert np.all(result.posterior.samples[:, 0] >= 0.6)
    assert result.posterior.summary()['xpos'][1] == pytest.approx(
        TRUE['xpos'], abs=0.05)


@pytest.mark.slow
def test_ultranest_posterior_agrees_with_the_covariance():
    pytest.importorskip('ultranest')
    from gbkfit.fitting.fitters.ultranest import FitterUltraNest
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), prior_properties(10))
    result = FitterUltraNest().fit(objective, params)
    check_posterior(result)
    assert result.evidence.log_z_error > 0


def test_multistart_runs_any_optimizer_with_its_options():
    # Each start keeps the options of the optimizer (here the sigma of
    # CMA-ES) at another initial value; samplers have no initial value
    pytest.importorskip('cma')
    from gbkfit.fitting import fitter_parser
    from gbkfit.fitting.fitters.cmaes.core import FitParamCMAES
    multistart = fitter_parser.load(dict(
        type='multistart', local=dict(type='cmaes')))
    assert multistart.fit_param_class() is FitParamCMAES
    start = FitParamCMAES(1, 0, 4, sigma=0.3).with_value(2.5)
    assert (start.value(), start.sigma(), start.maximum()) == (2.5, 0.3, 4)
    pytest.importorskip('dynesty')
    with pytest.raises(ConfigError, match="must be an optimizer"):
        fitter_parser.load(dict(
            type='multistart', local=dict(type='dynesty.static')))


def test_fits_report_their_progress():
    # Every stage (as the starts of multistart finish, then the
    # uncertainty), at the end a final report; the evaluations grow, the
    # best never worsens
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    reports = []
    result = fitter_parser.load(dict(
        type='multistart', local=dict(type='scipy.least_squares'),
        nstarts=2)).fit(objective, params, [reports.append])
    stages = list(dict.fromkeys(report.stage for report in reports))
    assert stages == [
        '0 of 2 starts done', '1 of 2 starts done', '2 of 2 starts done',
        'uncertainty']
    assert reports[-1].final and not any(r.final for r in reports[:-1])
    assert reports[-1].nevals == result.nevals
    nevals = [report.nevals for report in reports]
    best = [report.best_chi2 for report in reports]
    assert nevals == sorted(nevals) and best == sorted(best, reverse=True)
    assert set(reports[-1].best) == set(result.best().free)


def test_observers_that_fail_are_dropped(caplog):
    # The fit goes on, and the other observers receive every report
    pytest.importorskip('lmfit')
    from gbkfit.fitting import fitter_parser
    objective = mock_objective(DriverHost())
    params = ParamSpace(objective.pdescs(), bounded_properties())
    reports = []

    def failing(progress):
        raise OSError("no space left on device")
    result = fitter_parser.load(dict(type='scipy.least_squares')).fit(
        objective, params, [failing, reports.append])
    assert result.best().chi2 < np.inf
    assert reports[-1].final
    failures = [record for record in caplog.records
                if 'progress observer failed' in record.message]
    assert len(failures) == 1


def test_summaries_show_the_free_elements_decoded():
    # Of a parameter with a mode (offsets: v[1] = v[0] + 5), the best
    # value of v[1] and its error, as the quantiles beside them, not those
    # of its coded offset
    from gbkfit.fitting.result import Covariance, FitResult, FitSolution
    covariance = Covariance(
        names=('v[0]', 'v[1]'), matrix=np.diag([0.25, 1.0]),
        std={'v[0]': 0.5, 'v[1]': 1.0}, correlation=np.eye(2),
        element_std={'v[0]': 0.5, 'v[1]': 1.25}, derived_std={}, rank=2,
        unconstrained=(), at_bounds=(), scaled=False)
    solution = FitSolution(
        free={'v[0]': 100.0, 'v[1]': 5.0},
        values={'v': np.array([100.0, 105.0])},
        elements={'v[0]': 100.0, 'v[1]': 105.0}, derived={}, chi2=10.0,
        nresiduals=12, nfitted=2, dof=10, reduced_chi2=1.0,
        log_likelihood=-5.0, log_prior=None, aic=14.0, bic=15.0, aicp=14.0,
        covariance=covariance, laplace=None)
    result = FitResult(
        fitter=dict(type='grid'), params=None, versions={},
        solutions=(solution,), posterior=None, evidence=None, nevals=1,
        time=1.0, success=True, message='', diagnostics={},
        units=dict(v='km / s'))
    line = next(line for line in result.summary().split('\n')
                if line.startswith('v[1] '))
    assert line.split()[1:] == ['105', '1.25', 'km', '/', 's']
    # (the tables of samples have the units of their columns)
    table = result._with_units(Table({'v[0]': [1.0]}))
    assert table['v[0]'].unit == 'km / s'


def test_the_information_criteria_take_solved_values_by_their_likelihood():
    # A flux scale solved under a prior. Integrated: the log-likelihood
    # holds it (its Occam factor) and it does not count (the marginal AIC
    # and BIC). Profiled: it counts by its effective number, and the
    # residual of its prior is left out (the conditional AIC: the AICp)
    from gbkfit.fitting.fitters.grid import FitterGrid
    best = {}
    for likelihood in ('integrated', 'profiled'):
        plan = mock_objective(
            DriverHost(), name='obs', flux_scale='one').group_plan()
        params = ParamSpace(plan.group().pdescs(), TRUE | dict(
            obs_flux_scale=dict(solve=dict(
                type='gaussian', mean=1, sigma=0.1, min=0,
                likelihood=likelihood)),
            posa=dict(min=36, max=44, n=3)))
        best[likelihood] = FitterGrid().fit(
            Objective(plan, params.solves()), params).best()
    integrated, profiled = best['integrated'], best['profiled']
    ln_n = np.log(integrated.nresiduals)
    assert integrated.aic == pytest.approx(-2 * integrated.log_likelihood + 2)
    assert integrated.bic == pytest.approx(
        -2 * integrated.log_likelihood + ln_n)
    assert profiled.aic == pytest.approx(profiled.aicp)
    assert profiled.bic == pytest.approx(
        profiled.aicp + profiled.nfitted * (ln_n - 2))
