"""
Tests for mock data (draw_mocks): the noise of each observation (its
own, or that of the errors of its data) goes in where its instrument
records the data, and the errors of the data are those of the noise, or
the scatter of the measured data.
"""

import astropy.units as u
import gbkfit.params
import numpy as np
import pytest

from gbkfit.config import FORMAT_VERSION, ConfigError
from gbkfit.model import Extras
from gbkfit.observation import (
    MockOptions, NoiseGauss, NoisePoisson, draw_mocks)
from modelutils import group_plan


def disk(emission):
    return dict(
        type='thin_disk', loose=False, tilted=False, rnodes=list(range(0, 10)),
        rptraits=dict(type='exponential'),
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform'),
        emission=dict(type=emission))


PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60, rpt_a=1, rpt_s=3,
    vpt_rt=2, vpt_vt=150, dpt_a=20)

# The recorded data, of noise of rms 0.01 (pixel_brightness and
# slitless_spectra record an image)
RECORDED = dict(
    pixel_spectra=dict(size=[16, 17], spectral=dict(
        size=31, step=20, convention='relativistic')),
    pixel_brightness=dict(size=[16, 17]),
    slit_spectra=dict(size=[16], spectral=dict(
        size=31, step=20, convention='relativistic')),
    slitless_spectra=dict(
        size=[16, 12], spectral=dict(size=21, step=30, rest='4 um'),
        trace=dict(type='polynomial', wave='4 um',
                   dx=[0, 2.99792458e5 / 120])))


def plan_of(driver, observable):
    emission = 'band' if observable['type'] == 'pixel_brightness' \
        else 'lines'
    return group_plan([dict(
        driver=dict(type=driver.type()), galaxy=dict(
            components=[disk(emission)]), observation=observable)])


def evaluate(plan):
    """The model data of the observation and its parameters."""
    params = gbkfit.params.ParamSpace.from_properties(
        plan.group().pdescs(), PROPERTIES).evaluate()
    extra = Extras()
    data = plan.model_h(params, extra)[0]
    return {k: v.data.copy() for k, v in data.items()}, extra, params


def mock_data(plan, params, **options):
    """
    The mock datasets of the observation, of its noise (see draw_mocks).
    """
    return [
        dataset for dataset, in draw_mocks(
            plan, params, MockOptions(**options))]


@pytest.mark.parametrize('name', ['pixel_brightness', 'pixel_spectra'])
def test_poisson_mocks_have_the_errors_of_their_signal(driver, name):
    # The errors of the mock data are the dispersion of the counts of
    # the model, sqrt(model gain) / gain; where it has none (0), the data
    # are missing (an error of 0 cannot be fitted)
    plan = plan_of(
        driver, dict(type=name, noise=NoisePoisson(gain=1e4)) | RECORDED[name])
    model, _, params = evaluate(plan)
    [mock] = mock_data(plan, params, seed=1)
    for key, value in model.items():
        bright = value > 0
        np.testing.assert_allclose(
            mock[key].error()[bright], np.sqrt(value[bright] * 1e4) / 1e4,
            rtol=1e-5)
        assert np.isnan(mock[key].error()[~bright]).all()


@pytest.mark.parametrize('name', ['pixel_brightness', 'pixel_spectra'])
def test_poisson_mocks_over_a_background_keep_the_empty_sky(driver, name):
    # Their errors are the dispersion of the counts of the model and the
    # background, sqrt((model + b) gain) / gain, at every value (the empty
    # sky was missing without one: audit 2026-11-11 d)
    noise = NoisePoisson(gain=1e4, background=0.01)
    plan = plan_of(driver, dict(type=name, noise=noise) | RECORDED[name])
    model, _, params = evaluate(plan)
    [mock] = mock_data(plan, params, seed=1)
    for key, value in model.items():
        np.testing.assert_allclose(
            mock[key].error(), np.sqrt((value + 0.01) * 1e4) / 1e4,
            rtol=1e-5)


@pytest.mark.parametrize('name', list(RECORDED))
def test_recorded_data_have_the_noise_as_their_error(driver, name):
    plan = plan_of(
        driver, dict(type=name, noise=NoiseGauss(rms=0.01)) | RECORDED[name])
    model, _, params = evaluate(plan)
    mocks = mock_data(plan, params, nmocks=2, seed=1)
    for key, value in model.items():
        residual = mocks[0][key].data() - value
        assert residual.std() == pytest.approx(0.01, rel=0.05)
        np.testing.assert_allclose(mocks[0][key].error(), 0.01)
        assert not np.array_equal(mocks[0][key].data(), mocks[1][key].data())
    again = mock_data(plan, params, nmocks=1, seed=1)
    for key in model:
        np.testing.assert_array_equal(
            again[0][key].data(), mocks[0][key].data())


def test_setups_mock_observations_for_fits_under_their_likelihoods(
        driver):
    # (the likelihoods of a setup without data are those of the fits of
    # its mocks, as a configuration of them takes them)
    from gbkfit import Setup
    from gbkfit.objective import LikelihoodGauss, Likelihoods
    from gbkfit.observation import ObservationGroup
    plan = plan_of(driver, dict(
        type='pixel_brightness', noise=NoiseGauss(rms=0.01))
        | RECORDED['pixel_brightness'])
    group = plan.group()
    params = gbkfit.params.ParamSpace.from_properties(
        group.pdescs(), PROPERTIES)
    likelihoods = Likelihoods(
        group, LikelihoodGauss(white_floor=0.2), mocked=True)
    options = MockOptions(nmocks=2, seed=1)
    with Setup(
            group, params, likelihoods, driver=driver,
            mock_options=options) as setup:
        mocks = setup.mock(params.evaluate())
    assert [len(mock) for mock in mocks] == [1, 1]
    for [mock] in mocks:
        assert mock.dataset() is not None
        fitted = likelihoods.with_group(
            ObservationGroup(group.galaxies(), [mock]))
        assert fitted.observations() == (0,)
        assert fitted.of(0).white_floor() == 0.2


def test_mocks_have_the_unit_beam_and_correlation_of_their_data(driver):
    # A mock of an observation of data in Jy/beam with a beam, of noise
    # correlated by the PSF of the instrument, has that unit and beam and
    # declares the correlation: a fit of it whitens its residuals (the
    # observation with the data keeps its noise, rather than that of the
    # errors of the data, which declare none)
    from gbkfit.dataset import Data
    from gbkfit.objective import Objective
    plan = plan_of(driver, dict(
        type='pixel_spectra', psf=dict(type='gauss', sigma=1.5),
        noise=NoiseGauss(rms=0.01, spatial_correlation='instrument'))
        | RECORDED['pixel_spectra'])
    model, _, params = evaluate(plan)
    observation = plan.group().observations()[0]
    data = observation.make_dataset({'spectra': Data(
        model['spectra'], error=np.full_like(model['spectra'], 0.01),
        unit='Jy/beam', beam=dict(bmaj=4, bmin=3, bpa=10))})
    plan = plan.with_datasets({0: data})
    [mock] = mock_data(plan, params, seed=1)
    assert mock['spectra'].unit() == data['spectra'].unit()
    assert mock['spectra'].beam() == data['spectra'].beam()
    assert mock['spectra'].spatial_correlation() \
        == observation.instrument().psf()
    with Objective(plan.with_datasets({0: mock})) as objective:
        assert objective.nindependent() < objective.nresiduals()


@pytest.mark.parametrize('psf', [
    dict(type='gauss', sigma=1.5), dict(type='moffat', alpha=2, beta=3)])
def test_mocks_without_data_have_the_unit_and_beam_of_their_model(
        driver, psf):
    # Without data, the model of an observation in Jy/beam (its option
    # unit) is per the beam of its PSF: its mocks have that unit and beam
    # (of a PSF that is not a Gaussian, the round Gaussian of its solid
    # angle), and the observation of them has the model they were drawn
    # from (they had neither, and could not be fitted: audit 2026-11-11)
    from gbkfit.observation import ObservationGroup
    hi = disk('lines') | dict(emission=dict(
        type='lines', lines=['hi21'], unit='Msun/pc2'))
    plan = group_plan([dict(
        driver=dict(type=driver.type()), galaxy=dict(components=[hi]),
        observation=dict(
            type='pixel_spectra', psf=psf, noise=NoiseGauss(rms=0.01))
        | RECORDED['pixel_spectra'] | dict(spectral=dict(
            size=31, step=20, rest='1420.405751768 MHz')),
        unit='Jy/beam')])
    model, _, params = evaluate(plan)
    [mock] = mock_data(plan, params, seed=1)
    observation = plan.group().observations()[0]
    psf = observation.instrument().psf()
    beam = mock['spectra'].beam()
    assert mock['spectra'].unit() == u.Jy / u.beam
    assert beam.area() == pytest.approx(psf.beam_area(), rel=1e-12)
    if psf.type() == 'gauss':
        assert beam == psf.beam()
    else:
        assert beam.bmaj == beam.bmin
    mocked = ObservationGroup(
        plan.group().galaxies(), [observation.with_dataset(mock)])
    np.testing.assert_allclose(
        evaluate(mocked.plan(driver))[0]['spectra'], model['spectra'],
        rtol=1e-6)


def test_mocks_of_correlated_noise_have_float64_models(driver):
    # As other data of correlated noise (see ObservationSky: dtype): an
    # observation without data, of a float32 model, has a float64 model of
    # such mocks, with a plan of its own (it kept float32: audit
    # 2026-11-11)
    plan = plan_of(driver, dict(
        type='pixel_spectra', psf=dict(type='gauss', sigma=1.5),
        noise=NoiseGauss(rms=0.01, spatial_correlation='instrument'))
        | RECORDED['pixel_spectra'])
    model, _, params = evaluate(plan)
    assert model['spectra'].dtype == np.float32
    [mock] = mock_data(plan, params, seed=1)
    mocked = evaluate(plan.with_datasets({0: mock}))[0]['spectra']
    assert mocked.dtype == np.float64
    np.testing.assert_allclose(
        mocked, model['spectra'], rtol=1e-5,
        atol=1e-6 * model['spectra'].max())


def test_mocks_refuse_noise_of_a_varying_correlation(driver):
    # Noise correlated by a PSF that varies along the spectral axis can be
    # drawn, but the errors of data take one correlation for all the
    # channels: refused before any draw
    plan = plan_of(driver, dict(
        type='pixel_spectra', psf=dict(type='gauss', sigma=dict(
            points=dict(value=[-500, 500], unit='km/s'), values=[1, 2])),
        noise=NoiseGauss(rms=0.01, spatial_correlation='instrument'))
        | RECORDED['pixel_spectra'])
    _, _, params = evaluate(plan)
    with pytest.raises(ValueError, match="varies along the spectral axis"):
        mock_data(plan, params, seed=1)


@pytest.mark.parametrize('method', ['moments', 'gauss_fit'])
def test_the_errors_of_moments_are_their_scatter(driver, method):
    # The moments of the spectra of the cube plus noise (a twentieth of
    # its peak): their errors, the scatter of 100 other draws, are the
    # scatter of the mock moments, where the model is bright (the moments
    # of noisy spectra are biased, so their mean is not the moments of
    # the model)
    moments = dict(
        type='pixel_moments', size=[16, 17],
        spectral=dict(size=31, step=20, convention='relativistic'),
        flux_floor=1e-4, method=method)
    model, extra, params = evaluate(plan_of(driver, moments))
    peak = extra.get('dcube_lo').data.max()
    plan = plan_of(driver, moments | dict(noise=NoiseGauss(rms=0.05 * peak)))
    mocks = mock_data(plan, params, nmocks=100, seed=2)
    bright = model['moment0'] > 0.2 * model['moment0'].max()
    assert bright.sum() > 30
    for key in ('moment0', 'moment1'):
        # (the error of a value is missing where it is; the moments of
        # orders 1 and above of a noisy spectrum often are, where its
        # variance is negative)
        values, errors = (
            np.stack([getattr(mock[key], kind)()[bright] for mock in mocks])
            for kind in ('data', 'error'))
        scatter = np.nanstd(values, axis=0, ddof=1)
        ratio = scatter / np.nanmax(errors, axis=0)
        assert np.nanmedian(ratio) == pytest.approx(1, abs=0.1)


@pytest.mark.parametrize('method', ['moments', 'gauss_fit'])
def test_mocks_of_moments_miss_the_spectra_the_noise_blanks(driver, method):
    # An error map that is not finite somewhere blanks the spectra there
    # on purpose (see NoiseGauss): their moments are missing from the
    # mock data, and the others are drawn; a model whose flux is not a
    # number fails, as it does to evaluate
    moments = dict(
        type='pixel_moments', size=[16, 17],
        spectral=dict(size=31, step=20, convention='relativistic'),
        method=method)
    _, extra, params = evaluate(plan_of(driver, moments))
    error = np.full((17, 16), 0.05 * extra.get('dcube_lo').data.max())
    error[:, :5] = np.nan
    plan = plan_of(driver, moments | dict(noise=NoiseGauss(error=error)))
    [mock] = mock_data(plan, params, seed=1)
    for key in ('moment0', 'moment1'):
        item = mock[key]
        assert np.all(item.mask()[:, :5] == 0)
        assert np.all(np.isnan(item.data()[:, :5]))
        assert np.all(np.isnan(item.error()[:, :5]))
        assert item.mask()[:, 5:].sum() > 50
    with pytest.raises(gbkfit.params.ModelEvaluationError, match="no flux"):
        mock_data(plan, dict(params, vpt_vt=np.nan), seed=1)


@pytest.mark.parametrize('options, message', [
    (dict(nmocks='2'), "option 'nmocks' must be of type int; it is '2'"),
    (dict(nmocks=0), "option 'nmocks' must be at least 1; it is 0"),
    (dict(nsamples=1), "option 'nsamples' must be at least 2; it is 1"),
    (dict(likelihood=dict(type='gauss')),
     "unknown options: 'likelihood'"),
    (dict(noise=dict(type='gauss', rms=1)), "unknown options: 'noise'")])
def test_mock_options_are_checked_as_they_load(options, message):
    # (they were checked only by the task eval, after it evaluated the
    # model: audit 2026-10-31 #42; the fits of mocks take the likelihoods
    # of the setup, and the noise of each observation is its own)
    with pytest.raises((ConfigError, ValueError)) as error:
        MockOptions.load(options)
    assert str(error.value).startswith(message)


def test_each_observation_has_its_own_noise(driver):
    # (they were given in the options of mock data, one for all or by
    # observation name)
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    image = dict(type='pixel_brightness') | RECORDED['pixel_brightness']
    group = ObservationGroup(
        galaxy_parser.load(dict(components=[disk('band')])),
        observation_parser.load([
            image | dict(noise=dict(type='gauss', rms=rms))
            for rms in (0.01, 0.02)]))
    params = gbkfit.params.ParamSpace.from_properties(
        group.pdescs(), PROPERTIES).evaluate()
    [mocks] = draw_mocks(group.plan(driver), params, MockOptions(seed=1))
    for mock, rms in zip(mocks, (0.01, 0.02), strict=True):
        np.testing.assert_allclose(mock['brightness'].error(), rms)


@pytest.mark.parametrize('name', ['pixel_brightness', 'pixel_spectra'])
def test_the_noise_of_recorded_data_is_that_of_their_errors(driver, name):
    # Without a noise of its own, an observation of what its instrument
    # records (an image, a cube) has Gaussian noise of the errors of its
    # data, with their correlation
    from gbkfit.dataset import Data
    from gbkfit.instrument import PSFGauss
    plan = plan_of(driver, dict(type=name) | RECORDED[name])
    model, _, params = evaluate(plan)
    [(key, value)] = model.items()
    error = np.linspace(0.01, 0.02, value.size).reshape(value.shape)
    data = plan.group().observations()[0].make_dataset({key: Data(
        value, error=error, spatial_correlation=PSFGauss(1.5))})
    [mock] = mock_data(plan.with_datasets({0: data}), params, seed=1)
    np.testing.assert_allclose(mock[key].error(), error, rtol=1e-6)
    assert mock[key].spatial_correlation() == data[key].spatial_correlation()
    assert ((mock[key].data() - value) / error).std() \
        == pytest.approx(1, rel=0.1)


@pytest.mark.parametrize('observable, errors, message', [
    (RECORDED['pixel_brightness'] | dict(type='pixel_brightness'), None,
     "it has no data, whose errors would be its noise"),
    (RECORDED['pixel_brightness'] | dict(type='pixel_brightness'), False,
     "its data have no errors"),
    (dict(type='pixel_moments', size=[16, 17], spectral=dict(
        size=31, step=20, convention='relativistic')), True,
     "its data are measured from what its instrument records")])
def test_mocks_need_a_noise_the_data_cannot_give(
        driver, observable, errors, message):
    # An observation without a noise of its own and without data, or of
    # data without errors, or of data measured from what its instrument
    # records (moments of a cube, whose errors are not those of the noise
    # of the cube)
    from gbkfit.dataset import Data
    plan = plan_of(driver, observable)
    model, _, params = evaluate(plan)
    if errors is not None:
        # (its model as its data, with errors of 1 or without errors)
        observation = plan.group().observations()[0]
        data = observation.make_dataset({
            key: Data(value, error=np.ones_like(value) if errors else None)
            for key, value in model.items()})
        plan = plan.with_datasets({0: data})
    with pytest.raises(
            ValueError, match=r"the observation has no noise for its mock "
            r"data \(its option noise\): " + message):
        mock_data(plan, params, seed=1)


def test_the_noise_of_an_observation_dumps_and_loads(tmp_path):
    # Its error map in a file of its own, named after the observation,
    # which loads again (as the noise of a profile, without files)
    from gbkfit.observation import observation_parser
    error = np.arange(1.0, 17.0).reshape(4, 4)
    image = observation_parser.load(dict(
        type='pixel_brightness', size=[4, 4],
        noise=NoiseGauss(error=error, spatial_correlation='instrument')))
    prefix = str(tmp_path / 'img_')
    info = observation_parser.dump(image, prefix)
    assert info['noise'] == dict(
        type='gauss', error=f'{prefix}noise_error.fits',
        spatial_correlation='instrument')
    noise = observation_parser.load(info).noise()
    np.testing.assert_array_equal(noise.error(), error)
    assert noise.spatial_correlation() == 'instrument'
    profile = dict(
        type='profile', functions=['vcirc'],
        radii=dict(value=[1.0, 3.0], unit='kpc'),
        noise=dict(type='gauss', rms=5.0))
    assert observation_parser.dump(observation_parser.load(profile))[
        'noise'] == profile['noise']


@pytest.mark.parametrize('name', ['pixel_brightness', 'pixel_spectra'])
def test_mocks_of_an_observation_without_data_share_its_plan(driver, name):
    # (its plan takes nothing of mock data without a unit: the light of
    # the components as it is, as without data; see Observation.same_plan)
    plan = plan_of(
        driver, dict(type=name, noise=NoiseGauss(rms=0.01)) | RECORDED[name])
    model, _, params = evaluate(plan)
    [dataset] = mock_data(plan, params)
    copy = plan.with_datasets({0: dataset})
    assert copy._plans[0] is plan._plans[0]
    np.testing.assert_array_equal(
        next(iter(copy.model_h(params)[0].values())).data,
        next(iter(model.values())))


def test_data_without_a_unit_take_no_light_of_a_unit(driver):
    # (checked with the observation, also of a copy that shares its plan)
    image = dict(type='pixel_brightness') | RECORDED['pixel_brightness']
    plan = plan_of(driver, image | dict(noise=NoiseGauss(rms=0.01)))
    _, _, params = evaluate(plan)
    [dataset] = mock_data(plan, params)
    lit = group_plan([dict(
        driver=dict(type=driver.type()), observation=image,
        galaxy=dict(components=[disk('band') | dict(
            emission=dict(type='band', unit='MJy/sr'))]),
        unit='MJy/sr')])
    with pytest.raises(ValueError, match="the data have no unit"):
        lit.with_datasets({0: dataset})


def mock_setup_config(observable, likelihoods=None):
    """
    A configuration of mock data of an observation 'img' of a disk, of
    noise of rms 0.01, with the given section likelihoods, if any.
    """
    emission = 'band' if observable['type'] == 'pixel_brightness' \
        else 'lines'
    return dict(
        version=FORMAT_VERSION, driver=dict(type='host'),
        galaxies=[dict(components=[disk(emission)])],
        observations=[observable | dict(
            name='img', noise=dict(type='gauss', rms=0.01))],
        params=dict(properties=dict(PROPERTIES)),
        mock=dict(seed=1)) | (
        {} if likelihoods is None else dict(likelihoods=likelihoods))


def test_fits_of_mocks_have_the_parameters_of_their_likelihoods():
    # A fitted noise of an observation without data, in a setup of mock
    # data: its parameter is the setup's, and that of the fit of a mock
    import gbkfit
    from gbkfit import Setup
    from gbkfit.observation import ObservationGroup
    image = dict(type='pixel_brightness') | RECORDED['pixel_brightness']
    config = mock_setup_config(image, dict(
        type='gauss',
        fitted_noise=dict(brightness=dict(scale=True, scale_min=0.1))))
    config['params']['properties'] |= dict(
        img_noise_scale_brightness=dict(value=1, min=0.5, max=2))
    with gbkfit.load(config) as setup:
        assert 'img_noise_scale_brightness' in setup.params().pdescs()
        [[mock]] = setup.mock(setup.evaluate().values)
        group = ObservationGroup(setup.group().galaxies(), [mock])
        with Setup(group, setup.params(),
                   setup.likelihoods().with_group(group)) as fit:
            evaluation = fit.evaluate()
    assert np.isfinite(evaluation.objective.merits.chi2)


def test_a_setup_of_mocks_checks_the_likelihoods_of_their_fits():
    # The Gaussian of moment maps, by default, needs a missing_residual:
    # the setup of mock data says so, rather than the fit of its mocks
    import gbkfit
    moments = dict(type='pixel_moments', size=[16, 17], spectral=dict(
        size=31, step=20, convention='relativistic'))
    with pytest.raises(ValueError, match="missing_residual"):
        gbkfit.load(mock_setup_config(moments))
    with gbkfit.load(mock_setup_config(
            moments, dict(type='gauss', missing_residual=3))) as setup:
        assert setup.likelihoods().of(0).takes_missing_values()
