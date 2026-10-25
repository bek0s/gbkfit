"""
Tests for mock data (ObservationGroup.mock): the noise of each observable
goes in where its instrument records the data, and the errors of the
data are those of the noise, or the scatter of the measured data.
"""

import gbkfit.params
import numpy as np
import pytest

from gbkfit.model import Extras
from gbkfit.observation import NoiseGauss, NoisePoisson
from modelutils import group_plan


def disk(emission):
    return dict(
        type='disk', loose=False, tilted=False, rnodes=list(range(0, 10)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform')),
        emission=dict(type=emission))


PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60, pt_a=1, pt_s=3,
    vpt_rt=2, vpt_vt=150, dpt_a=20)

# The recorded data, of noise of rms 0.01 (pixel_brightness and
# slitless_spectra record an image)
RECORDED = dict(
    pixel_spectra=dict(size=[16, 17, 31], step=[1, 1, 20]),
    pixel_brightness=dict(size=[16, 17]),
    slit_spectra=dict(size=[16, 31], step=[1, 20]),
    slitless_spectra=dict(
        size=[16, 12], spec_size=21, spec_step=30, spec_rest='4 um',
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
    params = gbkfit.params.ParamSpace(
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
    data = observation.dataset({'spectra': Data(
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


@pytest.mark.parametrize('method', ['moments', 'gaussian_fit'])
def test_the_errors_of_moments_are_their_scatter(driver, method):
    # The moments of the spectra of the cube plus noise (a twentieth of
    # its peak): their errors, the scatter of 100 other draws, are the
    # scatter of the mock moments, where the model is bright (the moments
    # of noisy spectra are biased, so their mean is not the moments of
    # the model)
    plan = plan_of(driver, dict(
        type='pixel_moments', size=[16, 17], spec_size=31, spec_step=20,
        flux_floor=1e-4, method=method, spec_convention='relativistic'))
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
