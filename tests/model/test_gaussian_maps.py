"""
Tests for moment maps measured by Gaussian fits (method gaussian_fit of
mmaps and bmaps), as the maps of data often are.
"""

import copy

import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group

from gbkfit.observation.observables._moments import MomentsPlan


GALAXY = dict(components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))])

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    pt_a=1, pt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=20)


def evaluate(driver, observable):
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=GALAXY,
        line_of_sight=dict(type='flat'),
        observable=copy.deepcopy(observable))])
    params = gbkfit.params.ParamSpace(group.pdescs(), PROPERTIES)
    data = group.model_h(params.evaluate())[0]
    return {key: value['d'].copy() for key, value in data.items()}


def test_gaussian_fit_recovers_gaussians(driver):
    # Spectra that are Gaussians sampled at the centres of the channels:
    # the fit finds their flux, centre and dispersion
    rng = np.random.default_rng(2)
    nz, ny, nx = 81, 3, 4
    step, zero = 5.0, -200.0
    flux = rng.uniform(1, 10, (ny, nx))
    centre = rng.uniform(-100, 100, (ny, nx))
    sigma = rng.uniform(10, 40, (ny, nx))
    velocity = zero + np.arange(nz)[:, None, None] * step
    cube = flux / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * ((velocity - centre) / sigma) ** 2)
    # A spectrum below the cutoff is masked
    cube[:, 0, 0] = 0
    plan = MomentsPlan(
        driver, (nx, ny), (0, 1, 2), 1e-6, 'gaussian_fit', np.float32)
    maps = plan.evaluate(
        (1, 1, step), (0, 0, zero),
        driver.mem_copy_h2d(cube.astype(np.float32)), None)
    maps = {key: driver.mem_copy_d2h(value['d']) for key, value in
            maps.items()} | dict(mask=driver.mem_copy_d2h(maps['moment0']['m']))
    assert maps['mask'][0, 0] == 0 and np.isnan(maps['moment1'][0, 0])
    good = maps['mask'] == 1
    assert good.sum() == nx * ny - 1
    np.testing.assert_allclose(maps['moment0'][good], flux[good], rtol=1e-3)
    np.testing.assert_allclose(maps['moment1'][good], centre[good], atol=0.05)
    np.testing.assert_allclose(maps['moment2'][good], sigma[good], rtol=1e-3)


@pytest.mark.parametrize('step', [5.0, -5.0])
def test_gauss_hermite_fit_recovers_the_series(driver, step):
    # Spectra that are Gauss-Hermite series (van der Marel & Franx 1993) of
    # the velocity: the fit finds their flux, centre, dispersion, h3 and
    # h4, whichever way the velocities run along the channels
    rng = np.random.default_rng(4)
    nz, ny, nx = 81, 3, 4
    zero = -200.0 if step > 0 else 200.0
    flux = rng.uniform(1, 10, (ny, nx))
    centre = rng.uniform(-80, 80, (ny, nx))
    sigma = rng.uniform(10, 30, (ny, nx))
    h3 = rng.uniform(-0.15, 0.15, (ny, nx))
    h4 = rng.uniform(-0.1, 0.1, (ny, nx))
    w = (zero + np.arange(nz)[:, None, None] * step - centre) / sigma
    hermite3 = (2 * w ** 3 - 3 * w) / np.sqrt(3)
    hermite4 = (4 * w ** 4 - 12 * w ** 2 + 3) / np.sqrt(24)
    amplitude = flux / (sigma * np.sqrt(2 * np.pi) * (1 + h4 * np.sqrt(6) / 4))
    cube = amplitude * np.exp(-0.5 * w ** 2) * (1 + h3 * hermite3 + h4 * hermite4)
    plan = MomentsPlan(
        driver, (nx, ny), (0, 1, 2, 3, 4), 1e-6, 'gauss_hermite', np.float32)
    maps = plan.evaluate(
        (1, 1, step), (0, 0, zero),
        driver.mem_copy_h2d(cube.astype(np.float32)), None)
    maps = {key: driver.mem_copy_d2h(value['d']) for key, value in maps.items()}
    np.testing.assert_allclose(maps['moment0'], flux * np.sign(step), rtol=1e-3)
    np.testing.assert_allclose(maps['moment1'], centre, atol=0.05)
    np.testing.assert_allclose(np.abs(maps['moment2']), sigma, rtol=1e-3)
    np.testing.assert_allclose(maps['moment3'], h3, atol=2e-3)
    np.testing.assert_allclose(maps['moment4'], h4, atol=2e-3)


def test_gaussian_fit_of_noisy_spectra_stays_on_the_line(driver):
    # A line of dispersion 1.5 channels at S/N 10 per channel, in 4000
    # spectra: every fit is of the line (from the moments of the whole
    # spectrum alone, some fell on a spike of noise or on the wing of a
    # Gaussian beyond the spectrum, of flux up to 1e13)
    rng = np.random.default_rng(3)
    nz, ny, nx = 31, 40, 100
    channels = np.arange(nz)[:, None, None]
    line = np.exp(-0.5 * ((channels - 15) / 1.5) ** 2)
    cube = line + rng.standard_normal((nz, ny, nx)) / 10
    plan = MomentsPlan(
        driver, (nx, ny), (0, 1, 2), 1e-6, 'gaussian_fit', np.float32)
    maps = plan.evaluate(
        (1, 1, 1), (0, 0, 0),
        driver.mem_copy_h2d(cube.astype(np.float32)), None)
    flux, centre, sigma = (
        driver.mem_copy_d2h(maps[f'moment{order}']['d']) for order in range(3))
    assert (driver.mem_copy_d2h(maps['moment0']['m']) == 1).all()
    np.testing.assert_allclose(flux, 1.5 * np.sqrt(2 * np.pi), rtol=0.5)
    np.testing.assert_allclose(centre, 15, atol=1.5)
    np.testing.assert_allclose(sigma, 1.5, rtol=0.6)


@pytest.mark.parametrize('method', ['moments', 'gaussian_fit'])
def test_the_flux_floor_is_relative_to_the_brightest_spectrum(
        driver, method):
    # The spectra of a moment 0 not above the floor times the largest are
    # masked, whatever the unit (here of 1e-20: the floor of an absolute
    # 1e-6 masked every spectrum)
    nz, ny, nx = 41, 2, 3
    velocity = -100 + 5.0 * np.arange(nz)[:, None, None]
    flux = np.array([[1.0, 0.5, 0.2], [2e-3, 5e-4, 1e-4]])
    line = np.exp(-0.5 * (velocity / 20) ** 2) / (20 * np.sqrt(2 * np.pi))
    plan = MomentsPlan(driver, (nx, ny), (0, 1), 1e-3, method, np.float64)
    for unit in (1.0, 1e-20):
        maps = plan.evaluate(
            (1, 1, 5.0), (0, 0, -100.0),
            driver.mem_copy_h2d(flux * line * unit), None)
        mask = driver.mem_copy_d2h(maps['moment0']['m'])
        np.testing.assert_array_equal(mask, flux > 1e-3)


def test_gaussian_fit_without_beam_smearing_is_the_moments(driver):
    # Without a psf, each spectrum of a thin disk is one line, whose fit
    # is its moments (up to the channel integration of the model)
    mmaps = dict(type='pixel_moments', size=[32, 41], spec_size=161, spec_step=5,
                 orders=[0, 1, 2], flux_floor=1e-4)
    moments = evaluate(driver, mmaps)
    fitted = evaluate(driver, mmaps | dict(method='gaussian_fit'))
    good = np.isfinite(moments['moment1']) & np.isfinite(fitted['moment1'])
    # Both mask the same spectra (none of the fits fails)
    assert good.sum() == np.isfinite(moments['moment1']).sum() > 200
    np.testing.assert_allclose(
        fitted['moment1'][good], moments['moment1'][good], atol=0.1)
    np.testing.assert_allclose(
        fitted['moment2'][good], moments['moment2'][good], rtol=1e-2)
    np.testing.assert_allclose(
        fitted['moment0'][good], moments['moment0'][good], rtol=1e-2)


def test_gaussian_fit_differs_from_the_moments_with_beam_smearing(driver):
    # A psf mixes the lines of a rotating disk into skewed spectra, whose
    # Gaussian fits and moments differ (e.g. near the centre)
    mmaps = dict(type='pixel_moments', size=[32, 41], spec_size=161, spec_step=5,
                 orders=[0, 1, 2], flux_floor=1e-4,
                 psf=dict(type='gauss', sigma=2))
    moments = evaluate(driver, mmaps)
    fitted = evaluate(driver, mmaps | dict(method='gaussian_fit'))
    good = np.isfinite(moments['moment2']) & np.isfinite(fitted['moment2'])
    assert np.nanmax(np.abs(fitted['moment2'] - moments['moment2'])[good]) > 2


def test_gaussian_fit_of_bins(driver):
    # The fits of single-pixel bins are those of the maps
    from gbkfit.utils import fitsutils, gridutils
    fitsutils.write_data(
        'bins.fits', np.arange(32 * 41).reshape(41, 32),
        gridutils.Coords((1, 1), (15.5, 20), (0, 0), 0))
    options = dict(spec_size=161, spec_step=5, orders=[1, 2],
                   flux_floor=1e-4, method='gaussian_fit')
    mmaps = evaluate(driver, dict(type='pixel_moments', size=[32, 41]) | options)
    bmaps = evaluate(driver, dict(
        type='region_moments', regions=dict(type='bins', file='bins.fits')) | options)
    for key in ('moment1', 'moment2'):
        np.testing.assert_allclose(
            bmaps[key], mmaps[key].ravel(), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('options, message', [
    (dict(method='gauss'), "method must be one of"),
    (dict(method='gaussian_fit', orders=[1, 3]), "between 0 and 2"),
    (dict(method='gauss_hermite', orders=[1, 5]), "between 0 and 4")])
def test_method_options_are_checked(options, message):
    from gbkfit.observation import PixelMoments
    with pytest.raises(RuntimeError, match=message):
        PixelMoments(size=(8, 8), **options)


def test_method_round_trip():
    from gbkfit.observation import observable_parser
    info = dict(type='pixel_moments', size=[8, 8], method='gaussian_fit')
    dumped = observable_parser.dump(observable_parser.load(dict(info)))
    assert dumped['method'] == 'gaussian_fit'
