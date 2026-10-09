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


GMODEL = dict(type='kinematics_2d', components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    bptraits=dict(type='exponential'),
    vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))])

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=20)


def evaluate(driver, dmodel):
    group = observation_group([dict(
        driver=dict(type=driver.type()), dmodel=copy.deepcopy(dmodel),
        gmodel=GMODEL)])
    params = gbkfit.params.EvaluationParams(group.pdescs(), PROPERTIES)
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
            maps.items()} | dict(mask=driver.mem_copy_d2h(maps['mmap0']['m']))
    assert maps['mask'][0, 0] == 0 and np.isnan(maps['mmap1'][0, 0])
    good = maps['mask'] == 1
    assert good.sum() == nx * ny - 1
    np.testing.assert_allclose(maps['mmap0'][good], flux[good], rtol=1e-3)
    np.testing.assert_allclose(maps['mmap1'][good], centre[good], atol=0.05)
    np.testing.assert_allclose(maps['mmap2'][good], sigma[good], rtol=1e-3)


def test_gaussian_fit_without_beam_smearing_is_the_moments(driver):
    # Without a psf, each spectrum of a thin disk is one line, whose fit
    # is its moments (up to the channel integration of the model)
    mmaps = dict(type='mmaps', size=[32, 41], spec_size=161, spec_step=5,
                 orders=[0, 1, 2], mask_cutoff=1e-3)
    moments = evaluate(driver, mmaps)
    fitted = evaluate(driver, mmaps | dict(method='gaussian_fit'))
    good = np.isfinite(moments['mmap1']) & np.isfinite(fitted['mmap1'])
    # Both mask the same spectra (none of the fits fails)
    assert good.sum() == np.isfinite(moments['mmap1']).sum() > 200
    np.testing.assert_allclose(
        fitted['mmap1'][good], moments['mmap1'][good], atol=0.1)
    np.testing.assert_allclose(
        fitted['mmap2'][good], moments['mmap2'][good], rtol=1e-2)
    np.testing.assert_allclose(
        fitted['mmap0'][good], moments['mmap0'][good], rtol=1e-2)


def test_gaussian_fit_differs_from_the_moments_with_beam_smearing(driver):
    # A psf mixes the lines of a rotating disk into skewed spectra, whose
    # Gaussian fits and moments differ (e.g. near the centre)
    mmaps = dict(type='mmaps', size=[32, 41], spec_size=161, spec_step=5,
                 orders=[0, 1, 2], mask_cutoff=1e-3,
                 psf=dict(type='gauss', sigma=2))
    moments = evaluate(driver, mmaps)
    fitted = evaluate(driver, mmaps | dict(method='gaussian_fit'))
    good = np.isfinite(moments['mmap2']) & np.isfinite(fitted['mmap2'])
    assert np.nanmax(np.abs(fitted['mmap2'] - moments['mmap2'])[good]) > 2


def test_gaussian_fit_of_bins(driver):
    # The fits of single-pixel bins are those of the maps
    from gbkfit.utils import fitsutils
    fitsutils.write_data(
        'bins.fits', np.arange(32 * 41).reshape(41, 32),
        fitsutils.Coords((1, 1), (15.5, 20), (0, 0), 0))
    options = dict(spec_size=161, spec_step=5, orders=[1, 2],
                   mask_cutoff=1e-3, method='gaussian_fit')
    mmaps = evaluate(driver, dict(type='mmaps', size=[32, 41]) | options)
    bmaps = evaluate(driver, dict(
        type='bmaps', regions=dict(type='bins', file='bins.fits')) | options)
    for key in ('mmap1', 'mmap2'):
        np.testing.assert_allclose(
            bmaps[key], mmaps[key].ravel(), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('options, message', [
    (dict(method='gauss'), "method of"),
    (dict(method='gaussian_fit', orders=[1, 3]), "between 0 and 2")])
def test_method_options_are_checked(options, message):
    from gbkfit.observation import MMaps
    with pytest.raises(RuntimeError, match=message):
        MMaps(size=(8, 8), **options)


def test_method_round_trip():
    from gbkfit.observation import observable_parser
    info = dict(type='mmaps', size=[8, 8], method='gaussian_fit')
    dumped = observable_parser.dump(observable_parser.load(dict(info)))
    assert dumped['method'] == 'gaussian_fit'
