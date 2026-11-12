"""
Tests for the noise of mock observations: its dispersion, its correlation
on the sky and along the spectral axis, and its configuration.
"""

import numpy as np
import pytest

from gbkfit.grid import make_grid
from gbkfit.instrument import LSFConvolution, LSFHanning, PSFGauss
from gbkfit.observation.noise import (
    NoiseGauss, NoisePoisson, NoiseStudentT, noise_parser)


# A cube of 128 by 128 pixels of 5 arcsec and 64 channels of 10 km/s
GRID = make_grid((128, 128, 64), (5, 5, 10), None, (0, 0, 0), 0, 2)

# The data of GRID without noise (of no account to signal-free noise)
SIGNAL = np.zeros((64, 128, 128))

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
    noise = NoiseGauss(rms=2).draw(rng, GRID, None, SIGNAL)
    assert noise.shape == (64, 128, 128)
    assert noise.std() == pytest.approx(2, rel=5e-3)
    assert abs(correlation(noise, 1, 2)) < 5e-3
    error = np.tile(np.linspace(1, 3, 128), (128, 1))
    noise = NoiseGauss(error=error).draw(rng, GRID, None, SIGNAL)
    np.testing.assert_allclose(noise.std(axis=(0, 1)) / error[0], 1, rtol=0.03)


def test_student_t_noise_has_the_dispersion_and_heavier_tails():
    # Of rms 2 at dof 5: its dispersion 2, and its kurtosis 3 + 6 / (dof -
    # 4) = 9, beyond the 3 of Gaussian noise
    rng = np.random.default_rng(2)
    noise = NoiseStudentT(5, rms=2).draw(rng, GRID, None, SIGNAL)
    assert noise.std() == pytest.approx(2, rel=0.01)
    kurtosis = np.mean(noise ** 4) / np.mean(noise ** 2) ** 2
    assert kurtosis > 6
    info = dict(type='student_t', dof=5, rms=2)
    assert noise_parser.dump(noise_parser.load(dict(info))) == info
    with pytest.raises(ValueError, match="must be greater than 2"):
        NoiseStudentT(2, rms=1)


def test_student_t_noise_of_an_error_map(tmp_path):
    # Its dispersion is that of Gaussian noise: rms or an error map
    error = np.tile(np.linspace(1, 3, 128), (128, 1))
    noise = NoiseStudentT(5, error=error)
    assert noise.rms() is None
    loaded = noise_parser.load(
        noise_parser.dump(noise, prefix=f'{tmp_path}/'))
    np.testing.assert_array_equal(loaded.error(), error)
    np.testing.assert_array_equal(
        loaded.sigma(GRID, SIGNAL), np.broadcast_to(error, SIGNAL.shape))
    assert NoiseStudentT(5, rms=2).rms() == 2


def test_poisson_noise_counts_the_signal():
    # Of a gain of 4 counts per unit: the data are whole counts / 4, of
    # the dispersion sqrt(4 s) / 4; a signal below 0 counts as 0
    rng = np.random.default_rng(4)
    signal = np.broadcast_to(np.linspace(-1, 50, 128), (64, 128, 128))
    noise = NoisePoisson(gain=4)
    data = signal + noise.draw(rng, GRID, None, signal)
    np.testing.assert_allclose(data * 4, np.round(data * 4), atol=1e-9)
    sigma = noise.sigma(GRID, signal)
    np.testing.assert_allclose(sigma[0, 0], np.sqrt(4 * np.maximum(
        signal[0, 0], 0)) / 4)
    bright = signal[0, 0] > 10
    np.testing.assert_allclose(
        data.std(axis=(0, 1))[bright], sigma[0, 0][bright], rtol=0.03)
    assert (data[..., signal[0, 0] < 0] == 0).all()
    info = dict(type='poisson', gain=4)
    assert noise_parser.dump(noise_parser.load(dict(info))) == info


def test_poisson_noise_over_a_background():
    # The background b (e.g. of the sky) is counted with the signal: the
    # data are Poisson draws of (s + b) 4, divided by 4, less b, of the
    # dispersion sqrt(4 (s + b)) / 4, so that the data without signal have
    # errors too (they had none, and were left out of fits: audit
    # 2026-11-11 d)
    rng = np.random.default_rng(5)
    signal = np.broadcast_to(np.linspace(0, 50, 128), (64, 128, 128))
    noise = NoisePoisson(gain=4, background=10)
    data = signal + noise.draw(rng, GRID, None, signal)
    counts = (data + 10) * 4
    np.testing.assert_allclose(counts, np.round(counts), atol=1e-9)
    sigma = noise.sigma(GRID, signal)
    np.testing.assert_allclose(
        sigma[0, 0], np.sqrt(4 * (signal[0, 0] + 10)) / 4)
    assert sigma.min() == pytest.approx(np.sqrt(40) / 4)
    np.testing.assert_allclose(data.std(axis=(0, 1)), sigma[0, 0], rtol=0.03)
    # (within 5 standard errors of the mean of the draws of each value)
    np.testing.assert_allclose(
        data.mean(axis=(0, 1)), signal[0, 0], rtol=0, atol=0.25)
    info = dict(type='poisson', gain=4, background=10)
    assert noise_parser.dump(noise_parser.load(dict(info))) == info
    assert NoisePoisson().background() == 0
    with pytest.raises(ValueError, match="background"):
        NoisePoisson(background=-1)


# The sky of GRID, of an error map of one value per pixel
SKY = GRID.spatial()


def write_error(path, error, coords):
    """Write an error map with world coordinates; return its filename."""
    from gbkfit.io import write_data
    filename = str(path / 'error.fits')
    write_data(filename, error, coords, 2 if error.ndim == 3 else None)
    return filename


@pytest.mark.parametrize('kind', [NoiseGauss, NoiseStudentT])
def test_an_error_map_is_drawn_on_the_pixels_of_its_file(tmp_path, kind):
    # Its world coordinates must put its pixels on those of the data on
    # the sky (an error map of another field of the same shape was drawn
    # there: audit 2026-11-11 d#10), and do so again once dumped
    error = np.tile(np.linspace(1, 3, 128), (128, 1))
    filename = write_error(tmp_path, error, SKY.coords)
    noise = kind.from_file(error=filename) if kind is NoiseGauss \
        else kind.from_file(5, filename)
    np.testing.assert_array_equal(
        noise.sigma(GRID, SIGNAL), np.broadcast_to(error, SIGNAL.shape))
    loaded = noise_parser.load(
        noise_parser.dump(noise, prefix=f'{tmp_path}/d_'))
    for other in (
            GRID._replace(coords=GRID.coords._replace(rval=(0.01, 0, 0))),
            GRID._replace(coords=GRID.coords._replace(step=(4, 4, 10))),
            GRID._replace(coords=GRID.coords._replace(rota=10))):
        for each in (noise, loaded):
            with pytest.raises(ValueError, match="not on the pixels"):
                each.sigma(other, SIGNAL)
    # (on the sky alone: not on visibilities, whatever their shape)
    rows = make_grid((128, 128), (1, 10), (0, 0), (0, 0), 0, 1)
    with pytest.raises(ValueError, match="the data are not on the sky"):
        noise.sigma(rows, SIGNAL[0])


def test_an_error_cube_is_drawn_on_the_pixels_of_its_file(tmp_path):
    # (its spectral axis kept, once dumped)
    error = np.broadcast_to(np.linspace(1, 3, 128), (64, 128, 128))
    noise = NoiseGauss.from_file(write_error(tmp_path, error, GRID.coords))
    np.testing.assert_array_equal(noise.sigma(GRID, SIGNAL), error)
    loaded = noise_parser.load(
        noise_parser.dump(noise, prefix=f'{tmp_path}/d_'))
    np.testing.assert_array_equal(loaded.sigma(GRID, SIGNAL), error)
    moved = GRID._replace(coords=GRID.coords._replace(rpix=(64, 63.5, 31.5)))
    with pytest.raises(ValueError, match="not on the pixels"):
        loaded.sigma(moved, SIGNAL)


def test_the_beam_is_the_correlation_of_the_noise():
    # The noise of a radio cube: the correlation of two pixels half a
    # FWHM apart is half, as the beam there (not that of the beam
    # convolved by itself, 0.71)
    rng = np.random.default_rng(2)
    noise = NoiseGauss(rms=2, spatial_correlation='instrument').draw(
        rng, GRID, BEAM, SIGNAL)
    assert noise.std() == pytest.approx(2, rel=0.02)
    assert correlation(noise, 3, 2) == pytest.approx(0.5, abs=0.02)
    assert correlation(noise, 3, 1) == pytest.approx(0.5, abs=0.02)
    assert abs(correlation(noise, 1, 0)) < 0.02


def test_the_noise_of_hanning_smoothed_spectra():
    # White noise smoothed by the Hanning kernel (1/4, 1/2, 1/4) has the
    # correlation of the kernel convolved by itself: (1, 4, 6, 4, 1) / 16,
    # 4/6 for neighbouring channels and 1/6 for channels two apart
    rng = np.random.default_rng(3)
    hanning = LSFHanning()
    noise = NoiseGauss(
        rms=2, spectral_correlation=LSFConvolution([hanning, hanning])).draw(
            rng, GRID, None, SIGNAL)
    assert noise.std() == pytest.approx(2, rel=5e-3)
    assert correlation(noise, 1, 0) == pytest.approx(4 / 6, abs=5e-3)
    assert correlation(noise, 2, 0) == pytest.approx(1 / 6, abs=5e-3)
    assert abs(correlation(noise, 3, 0)) < 5e-3


def test_the_configuration_of_the_noise(tmp_path):
    error = np.tile(np.linspace(1, 3, 128), (128, 1))
    noise = NoiseGauss(
        error=error, spatial_correlation='instrument',
        spectral_correlation=LSFHanning())
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
    with pytest.raises(ValueError, match=message):
        NoiseGauss(**options).draw(
            np.random.default_rng(), GRID, psf, SIGNAL)


def test_a_spectral_correlation_is_drawn_on_the_channels_of_the_data():
    # (a Hanning smoothing of half a channel is not: it needs channels of
    # the data divided by 2, as of a model)
    with pytest.raises(RuntimeError, match="the spectral correlation of the "
                                           "noise is drawn on the channels"):
        NoiseGauss(rms=1, spectral_correlation=LSFHanning(0.5)).draw(
            np.random.default_rng(), GRID, None, SIGNAL)


def test_a_varying_correlation_covers_the_channels():
    # Beyond the points of its table it would be that of the nearest, as
    # the PSF of a model may not be
    import astropy.units as u
    from gbkfit.instrument import SpectralTable
    psf = PSFGauss.varying(
        sigma=SpectralTable([-100, 100] * u.km / u.s, [10.0, 20.0]))
    with pytest.raises(ValueError, match="the spatial correlation of the "
                                         "noise is given from -100 to 100"):
        NoiseGauss(rms=1, spatial_correlation=psf).draw(
            np.random.default_rng(), GRID, None, SIGNAL)
