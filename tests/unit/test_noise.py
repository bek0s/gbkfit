"""
Tests for the noise of mock observations: its dispersion, its correlation
on the sky and along the spectral axis, and its configuration.
"""

import numpy as np
import pytest

from gbkfit.instrument import LSFConvolution, LSFHanning, PSFGauss
from gbkfit.observation.noise import NoiseGauss, noise_parser
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError


# A cube of 128 by 128 pixels of 5 arcsec and 64 channels of 10 km/s
GRID = gridutils.make_grid((128, 128, 64), (5, 5, 10), None, (0, 0, 0), 0, 2)

# A beam of FWHM 30 arcsec
BEAM = PSFGauss(sigma=30 / np.sqrt(8 * np.log(2)))


def correlation(noise, shift, axis):
    """The correlation of the noise of values shift apart along an axis."""
    head = np.take(noise, range(shift, noise.shape[axis]), axis)
    tail = np.take(noise, range(noise.shape[axis] - shift), axis)
    return np.mean(head * tail) / np.mean(noise ** 2)


def test_white_noise_has_the_dispersion_of_each_value():
    # One rms everywhere, or an error map of one value per pixel for all
    # the channels
    rng = np.random.default_rng(1)
    noise = NoiseGauss(rms=2).draw(rng, GRID, None)
    assert noise.shape == (64, 128, 128)
    assert noise.std() == pytest.approx(2, rel=5e-3)
    assert abs(correlation(noise, 1, 2)) < 5e-3
    error = np.tile(np.linspace(1, 3, 128), (128, 1))
    noise = NoiseGauss(error=error).draw(rng, GRID, None)
    np.testing.assert_allclose(noise.std(axis=(0, 1)) / error[0], 1, rtol=0.03)


def test_the_beam_is_the_correlation_of_the_noise():
    # The noise of a radio cube: the correlation of two pixels half a
    # FWHM apart is half, as the beam there (not that of the beam
    # convolved by itself, 0.71)
    rng = np.random.default_rng(2)
    noise = NoiseGauss(rms=2, spatial_correlation='instrument').draw(
        rng, GRID, BEAM)
    assert noise.std() == pytest.approx(2, rel=0.02)
    assert correlation(noise, 3, 2) == pytest.approx(0.5, abs=0.02)
    assert correlation(noise, 3, 1) == pytest.approx(0.5, abs=0.02)
    assert abs(correlation(noise, 1, 0)) < 0.02


def test_the_noise_of_hanning_smoothed_spectra():
    # White noise smoothed by the Hanning kernel (1/4, 1/2, 1/4) has the
    # correlation of the kernel convolved by itself: (1, 4, 6, 4, 1) / 16,
    # 4/6 for neighbouring channels and 1/6 for channels two apart
    rng = np.random.default_rng(3)
    hanning = LSFHanning(width=10)
    noise = NoiseGauss(
        rms=2, spectral_correlation=LSFConvolution([hanning, hanning])).draw(
            rng, GRID, None)
    assert noise.std() == pytest.approx(2, rel=5e-3)
    assert correlation(noise, 1, 0) == pytest.approx(4 / 6, abs=5e-3)
    assert correlation(noise, 2, 0) == pytest.approx(1 / 6, abs=5e-3)
    assert abs(correlation(noise, 3, 0)) < 5e-3


def test_the_configuration_of_the_noise(tmp_path):
    error = np.tile(np.linspace(1, 3, 128), (128, 1))
    noise = NoiseGauss(
        error=error, spatial_correlation='instrument',
        spectral_correlation=LSFHanning(width=10))
    info = noise_parser.dump(noise, prefix=f'{tmp_path}/')
    loaded = noise_parser.load(info)
    np.testing.assert_array_equal(loaded.error(), error)
    assert loaded.spatial_correlation() == 'instrument'
    assert noise_parser.dump(loaded, prefix=f'{tmp_path}/', overwrite=True) \
        == info


@pytest.mark.parametrize('options, psf, message', [
    (dict(), BEAM, "give one of rms and error"),
    (dict(rms=1, error=np.ones(3)), BEAM, "give one of rms and error"),
    (dict(error=np.ones((2, 2))), BEAM, "does not broadcast to the data"),
    (dict(rms=1, spatial_correlation='instrument'), None,
     "the instrument has no PSF")])
def test_what_the_noise_refuses(options, psf, message):
    with pytest.raises(ConfigError, match=message):
        NoiseGauss(**options).draw(np.random.default_rng(), GRID, psf)
