"""
Tests for mock data (ObservationGroup.mock): the noise of each observable
goes in where its instrument records the data, and the errors of the
data are those of the noise, or the scatter of the measured data.
"""

import astropy.units as u
import gbkfit.params
import numpy as np
import pytest

from gbkfit.config import ConfigError
from gbkfit.model import Extras
from gbkfit.observation import NoiseGauss, NoisePoisson
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
            components=[disk(emission)]),
        line_of_sight=dict(type='flat'), observable=observable)])


def evaluate(plan):
    """The model data of the observation and its parameters."""
    params = gbkfit.params.ParamSpace.from_properties(
        plan.group().pdescs(), PROPERTIES).evaluate()
    extra = Extras()
    data = plan.model_h(params, extra)[0]
    return {k: v['d'].copy() for k, v in data.items()}, extra, params


@pytest.mark.parametrize('name', ['pixel_brightness', 'pixel_spectra'])
def test_poisson_mocks_have_the_errors_of_their_signal(driver, name):
    # The errors of the mock data are the dispersion of the counts of
    # the model, sqrt(model gain) / gain; where it has none (0), the data
    # are missing (an error of 0 cannot be fitted)
    plan = plan_of(driver, dict(type=name) | RECORDED[name])
    model, _, params = evaluate(plan)
    [mock] = plan.mock(params, NoisePoisson(gain=1e4), seed=1)[0]
    for key, value in model.items():
        bright = value > 0
        np.testing.assert_allclose(
            mock[key].error()[bright], np.sqrt(value[bright] * 1e4) / 1e4,
            rtol=1e-5)
        assert np.isnan(mock[key].error()[~bright]).all()


@pytest.mark.parametrize('name', list(RECORDED))
def test_recorded_data_have_the_noise_as_their_error(driver, name):
    plan = plan_of(driver, dict(type=name) | RECORDED[name])
    model, _, params = evaluate(plan)
    mocks = plan.mock(params, NoiseGauss(rms=0.01), nmocks=2, seed=1)[0]
    for key, value in model.items():
        residual = mocks[0][key].data() - value
        assert residual.std() == pytest.approx(0.01, rel=0.05)
        np.testing.assert_allclose(mocks[0][key].error(), 0.01)
        assert not np.array_equal(mocks[0][key].data(), mocks[1][key].data())
    again = plan.mock(params, NoiseGauss(rms=0.01), nmocks=1, seed=1)[0]
    for key in model:
        np.testing.assert_array_equal(again[0][key].data(), mocks[0][key].data())


def test_mocks_have_the_unit_beam_and_correlation_of_their_data(driver):
    # A mock of an observation of data in Jy/beam with a beam, of noise
    # correlated by the PSF of the instrument, has that unit and beam and
    # declares the correlation: a fit of it whitens its residuals
    from gbkfit.dataset import Data
    from gbkfit.objective import Objective
    plan = plan_of(driver, dict(
        type='pixel_spectra', psf=dict(type='gauss', sigma=1.5))
        | RECORDED['pixel_spectra'])
    model, _, params = evaluate(plan)
    observation = plan.group().observations()[0]
    data = observation.make_dataset({'spectra': Data(
        model['spectra'], error=np.full_like(model['spectra'], 0.01),
        unit='Jy/beam', beam=dict(bmaj=4, bmin=3, bpa=10))})
    plan = plan.with_data([data])
    [mock] = plan.mock(
        params, NoiseGauss(rms=0.01, spatial_correlation='instrument'),
        seed=1)[0]
    assert mock['spectra'].unit() == data['spectra'].unit()
    assert mock['spectra'].beam() == data['spectra'].beam()
    assert mock['spectra'].spatial_correlation() \
        == observation.instrument().psf()
    with Objective(plan.with_data([mock])) as objective:
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
        line_of_sight=dict(type='flat'), observable=dict(
            type='pixel_spectra', psf=psf) | RECORDED['pixel_spectra']
        | dict(spectral=dict(
            size=31, step=20, rest='1420.405751768 MHz')),
        unit='Jy/beam')])
    model, _, params = evaluate(plan)
    [mock] = plan.mock(params, NoiseGauss(rms=0.01), seed=1)[0]
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
        plan.group().galaxies(), [observation.with_data(mock)])
    np.testing.assert_allclose(
        evaluate(mocked.plan(driver))[0]['spectra'], model['spectra'],
        rtol=1e-6)


def test_mocks_of_correlated_noise_have_float64_models(driver):
    # As other data of correlated noise (see ObservationSky: dtype): an
    # observation without data, of a float32 model, has a float64 model of
    # such mocks, with a plan of its own (it kept float32: audit
    # 2026-11-11)
    plan = plan_of(driver, dict(
        type='pixel_spectra', psf=dict(type='gauss', sigma=1.5))
        | RECORDED['pixel_spectra'])
    model, _, params = evaluate(plan)
    assert model['spectra'].dtype == np.float32
    [mock] = plan.mock(
        params, NoiseGauss(rms=0.01, spatial_correlation='instrument'),
        seed=1)[0]
    mocked = evaluate(plan.with_data([mock]))[0]['spectra']
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
            points=dict(value=[-500, 500], unit='km/s'), values=[1, 2])))
        | RECORDED['pixel_spectra'])
    _, _, params = evaluate(plan)
    with pytest.raises(ValueError, match="varies along the spectral axis"):
        plan.mock(params, NoiseGauss(
            rms=0.01, spatial_correlation='instrument'), seed=1)


@pytest.mark.parametrize('method', ['moments', 'gaussian_fit'])
def test_the_errors_of_moments_are_their_scatter(driver, method):
    # The moments of the spectra of the cube plus noise (a twentieth of
    # its peak): their errors, the scatter of 100 other draws, are the
    # scatter of the mock moments, where the model is bright (the moments
    # of noisy spectra are biased, so their mean is not the moments of
    # the model)
    plan = plan_of(driver, dict(
        type='pixel_moments', size=[16, 17],
        spectral=dict(size=31, step=20, convention='relativistic'),
        flux_floor=1e-4, method=method))
    model, extra, params = evaluate(plan)
    peak = extra.get('dcube_lo').data.max()
    mocks = plan.mock(
        params, NoiseGauss(rms=0.05 * peak), nmocks=100, seed=2)[0]
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


@pytest.mark.parametrize('options, message', [
    (dict(nmocks='2'), "option 'nmocks' must be of type int; it is '2'"),
    (dict(nmocks=0), "option 'nmocks' must be at least 1; it is 0"),
    (dict(nsamples=1), "option 'nsamples' must be at least 2; it is 1"),
    (dict(likelihood=[None, None]),
     "option 'likelihood' is one, or a list of one for each of the 3 "
     "observations; it is a list of 2")])
def test_mock_options_are_checked_as_they_load(options, message):
    # (they were checked only by the task eval, after it evaluated the
    # model: audit 2026-10-31 #42)
    from gbkfit.observation import MockOptions
    with pytest.raises((ConfigError, ValueError)) as error:
        MockOptions.load(
            dict(noise=dict(type='gauss', rms=1)) | options).per_observation(3)
    assert str(error.value) == message


def test_mock_options_dump_the_noise_of_each_observation(tmp_path):
    # Its error map in a file of its own, named after the position of its
    # observation, which loads again (they were all noise_error.fits, in
    # the working directory, and the second was refused: audit 2026-11-11)
    from gbkfit.observation import MockOptions
    errors = [np.ones((4, 4)), np.full((4, 4), 2.0)]
    prefix = str(tmp_path / 'mock_')
    info = MockOptions([NoiseGauss(error=error) for error in errors]).dump(
        prefix)
    assert [noise['error'] for noise in info['noise']] == [
        f'{prefix}observation{i}_noise_error.fits' for i in range(2)]
    noises, _ = MockOptions.load(info).per_observation(2)
    for noise, error in zip(noises, errors):
        np.testing.assert_array_equal(noise.error(), error)
