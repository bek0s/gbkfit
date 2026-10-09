"""
Tests for the primary beam of the instrument, which attenuates the light
of the model before the PSF.
"""

import copy

import gbkfit.params
import numpy as np
import pytest
import scipy.signal
from modelutils import observation_group

from gbkfit.observation import (
    Instrument, PrimaryBeamAiry, PrimaryBeamGauss, instrument_parser)
from gbkfit.utils import fitsutils


GMODEL = dict(type='intensity_2d', components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    bptraits=dict(type='exponential'))])

PROPERTIES = dict(xpos=0.3, ypos=-0.6, posa=50, incl=60, bpt_a=1, bpt_s=4)


def evaluate(driver, dmodel):
    group = observation_group([dict(
        driver=dict(type=driver.type()), dmodel=copy.deepcopy(dmodel),
        gmodel=GMODEL)])
    params = gbkfit.params.EvaluationParams(group.pdescs(), PROPERTIES)
    return group.model_h(params.evaluate())[0]['image']['d'].copy()


@pytest.mark.parametrize('beam_class', [PrimaryBeamGauss, PrimaryBeamAiry])
def test_primary_beam_is_half_at_half_its_fwhm(beam_class):
    grid = fitsutils.make_grid((5, 1), (1.5, 1), rpix=(0, 0))
    response = beam_class(fwhm=6).response(grid)[0]
    np.testing.assert_allclose(response[0], 1)
    np.testing.assert_allclose(response[2], 0.5, rtol=1e-9)
    assert np.all(np.diff(response) < 0)


def test_airy_primary_beam_has_its_first_null_where_expected():
    # The first zero of J1 is at 3.8317, which is 1.1853 fwhm
    grid = fitsutils.make_grid((1, 1), rpix=(0, 0))
    beam = PrimaryBeamAiry(fwhm=1, x=1.18530, y=0)
    assert beam.response(grid)[0, 0] < 1e-9


@pytest.mark.parametrize('rota', [0, 40])
def test_primary_beam_attenuates_the_model(driver, rota):
    # Without a psf and oversampling, the model is that without the beam
    # times the beam at the centres of the pixels
    image = dict(type='image', size=[32, 41], step=[0.5, 0.5], rota=rota)
    beam = dict(type='gauss', fwhm=6, x=2, y=-1)
    plain = evaluate(driver, image)
    attenuated = evaluate(driver, image | dict(primary_beam=beam))
    grid = fitsutils.make_grid((32, 41), 0.5, rota=rota)
    expected = plain * PrimaryBeamGauss(6, 2, -1).response(grid)
    np.testing.assert_allclose(
        attenuated, expected, rtol=1e-5, atol=1e-6 * plain.max())


def test_primary_beam_comes_before_the_psf(driver):
    # The model is the psf convolved with the attenuated light
    image = dict(type='image', size=[64, 64], step=[0.5, 0.5])
    beam = dict(type='gauss', fwhm=5, x=1, y=0)
    psf = dict(type='gauss', sigma=1.0)
    attenuated = evaluate(driver, image | dict(primary_beam=beam))
    observed = evaluate(driver, image | dict(primary_beam=beam, psf=psf))
    kernel = instrument_parser.load(dict(psf=psf)).psf().asarray((0.5, 0.5))
    expected = scipy.signal.fftconvolve(attenuated, kernel, mode='same')
    np.testing.assert_allclose(
        observed, expected, rtol=1e-4, atol=1e-5 * observed.max())
    # The beam after the psf would give another model
    plain = evaluate(driver, image | dict(psf=psf))
    grid = fitsutils.make_grid((64, 64), 0.5)
    after = plain * PrimaryBeamGauss(5, 1, 0).response(grid)
    assert not np.allclose(observed, after, rtol=1e-2)


def test_instrument_round_trip():
    info = dict(
        primary_beam=dict(type='airy', fwhm=30, x=1, y=2),
        psf=dict(type='gauss', sigma=1.5), lsf=None)
    instrument = instrument_parser.load(copy.deepcopy(info))
    assert isinstance(instrument, Instrument)
    dumped = instrument_parser.dump(instrument)
    assert dumped['primary_beam'] == info['primary_beam']
    assert instrument_parser.dump(instrument_parser.load(dumped)) == dumped


def test_primary_beam_needs_a_positive_fwhm():
    with pytest.raises(RuntimeError, match="fwhm"):
        PrimaryBeamGauss(fwhm=0)
