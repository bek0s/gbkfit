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

from gbkfit.instrument import (
    Instrument, PrimaryBeamAiry, PrimaryBeamGauss, instrument_parser)
from gbkfit.utils import fitsutils, gridutils


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
    return group.model_h(params.evaluate())[0]['brightness']['d'].copy()


@pytest.mark.parametrize('beam_class', [PrimaryBeamGauss, PrimaryBeamAiry])
def test_primary_beam_is_half_at_half_its_fwhm(beam_class):
    grid = gridutils.make_grid((5, 1), (1.5, 1), rpix=(0, 0))
    response = beam_class(fwhm=6).response(grid)[0]
    np.testing.assert_allclose(response[0], 1)
    np.testing.assert_allclose(response[2], 0.5, rtol=1e-9)
    assert np.all(np.diff(response) < 0)


def test_airy_primary_beam_has_its_first_null_where_expected():
    # The first zero of J1 is at 3.8317, which is 1.1853 fwhm
    grid = gridutils.make_grid((1, 1), rpix=(0, 0))
    beam = PrimaryBeamAiry(fwhm=1, x=1.18530, y=0)
    assert beam.response(grid)[0, 0] < 1e-9


@pytest.mark.parametrize('rota', [0, 40])
def test_primary_beam_attenuates_the_model(driver, rota):
    # Without a psf and oversampling, the model is that without the beam
    # times the beam at the centres of the pixels
    image = dict(type='pixel_brightness', size=[32, 41], step=[0.5, 0.5], rota=rota)
    beam = dict(type='gauss', fwhm=6, x=2, y=-1)
    plain = evaluate(driver, image)
    attenuated = evaluate(driver, image | dict(primary_beam=beam))
    grid = gridutils.make_grid((32, 41), 0.5, rota=rota)
    expected = plain * PrimaryBeamGauss(6, 2, -1).response(grid)
    np.testing.assert_allclose(
        attenuated, expected, rtol=1e-5, atol=1e-6 * plain.max())


def test_primary_beam_comes_before_the_psf(driver):
    # The model is the psf convolved with the attenuated light
    image = dict(type='pixel_brightness', size=[64, 64], step=[0.5, 0.5])
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
    grid = gridutils.make_grid((64, 64), 0.5)
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


def test_primary_beam_points_at_its_position():
    # Its peak is at x = 3, y = -2 (pixels 10 + 3 and 10 - 2)
    grid = gridutils.make_grid((21, 21), 1)
    response = PrimaryBeamGauss(6, x=3, y=-2).response(grid)
    j, i = np.unravel_index(np.argmax(response), response.shape)
    assert (i, j) == (13, 8)

def test_primary_beam_from_an_image(tmp_path):
    # An image of a Gaussian beam, on a grid rotated on the sky, gives the
    # response of the Gaussian beam, and 0 beyond the image
    from gbkfit.instrument import PrimaryBeamImage, primary_beam_parser
    beam_grid = gridutils.make_grid((81, 81), 0.25, rota=25)
    gauss = PrimaryBeamGauss(6, 1, -0.5)
    fitsutils.write_data(
        str(tmp_path / 'pb.fits'), gauss.response(beam_grid),
        beam_grid.coords)
    image = primary_beam_parser.load(dict(
        type='image', file=str(tmp_path / 'pb.fits')))
    assert isinstance(image, PrimaryBeamImage)
    grid = gridutils.make_grid((20, 20), 0.5)
    # (bilinear interpolation of pixels of 0.25 arcsec)
    np.testing.assert_allclose(
        image.response(grid), gauss.response(grid), atol=5e-3)
    # (pixels at the centre, and beyond the image)
    beyond = gridutils.make_grid((2, 1), 30, rpix=(0, 0))
    response = image.response(beyond)
    assert response[0, 0] > 0 and response[0, 1] == 0
    far = gridutils.make_grid((2, 2), 1, rpix=(-30, -30))
    with pytest.raises(RuntimeError, match="does not cover"):
        image.response(far)
    # Round trip through the configuration
    dumped = primary_beam_parser.dump(image, prefix=str(tmp_path / 'd_'))
    again = primary_beam_parser.load(dict(dumped))
    np.testing.assert_allclose(
        again.response(grid), image.response(grid), atol=1e-6)
    # The world coordinates given replace those of the file
    stretched = primary_beam_parser.load(dict(
        type='image', file=str(tmp_path / 'pb.fits'), step=[0.5, 0.5],
        rota=0))
    assert stretched._grid.coords.step == (0.5, 0.5)
    assert stretched._grid.coords.rota == 0


def test_primary_beam_image_is_placed_by_its_ra_and_dec():
    # An image of a beam that peaks at its reference pixel, which is
    # 3 arcsec east and 2 arcsec north of that of the grid (with 1 arcsec
    # pixels; east is to the left)
    from gbkfit.instrument import PrimaryBeamImage
    beam_grid = gridutils.make_grid((41, 41), 0.5)
    beam = PrimaryBeamImage(
        PrimaryBeamGauss(4).response(beam_grid), 0.5,
        rval=(10 + 3 / 3600, 2 / 3600))
    grid = gridutils.make_grid((21, 21), 1, rval=(10, 0))
    response = beam.response(grid)
    j, i = np.unravel_index(np.argmax(response), response.shape)
    assert (i, j) == (10 - 3, 10 + 2)


def test_observation_dumps_the_image_of_its_primary_beam(tmp_path):
    # The image of the primary beam of an observation goes to a file named
    # with the prefix of the dump, which can be dumped again (overwrite)
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.instrument import PrimaryBeamImage
    from gbkfit.observation import Observation, observation_parser
    from gbkfit.observation.observables import PixelBrightness
    beam = PrimaryBeamImage(np.ones((8, 8)))
    observation = Observation(
        PixelBrightness(size=(8, 8)), DriverHost(),
        instrument=Instrument(primary_beam=beam))
    prefix = str(tmp_path / 'out_')
    for _ in range(2):
        dumped = observation_parser.dump(
            observation, prefix=prefix, overwrite=True)
    assert dumped['instrument']['primary_beam']['file'] == (
        f'{prefix}primary_beam.fits')


def test_primary_beam_image_options_are_checked(tmp_path):
    # (an option of a wrong type is an error with its path when loaded)
    from gbkfit.instrument import primary_beam_parser
    from gbkfit.utils.parseutils import ConfigError
    fitsutils.write_data(
        str(tmp_path / 'pb.fits'), np.ones((4, 4)),
        gridutils.make_grid((4, 4)).coords)
    with pytest.raises(ConfigError, match="image: option 'step' must be"):
        primary_beam_parser.load(dict(
            type='image', file=str(tmp_path / 'pb.fits'), step='abc'))
