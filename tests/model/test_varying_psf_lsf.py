"""
Tests for the observations with a PSF or an LSF that varies along the
spectral axis.
"""

import numpy as np
import pytest
import scipy.signal

import gbkfit.params
from gbkfit.instrument import LSFGauss, PSFGauss
from gbkfit.model import model_parser
from gbkfit.observation import ObservationGroup, observation_parser
from gbkfit.utils.parseutils import ConfigError


GMODEL = dict(type='kinematics_2d', components=[dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 12)),
    bptraits=dict(type='exponential'), vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))])
PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.2, posa=30, incl=60, bpt_a=1, bpt_s=2.5,
    vpt_rt=2, vpt_vt=150, dpt_a=20)
# 40 channels of 12 km/s, centred on 0
SIZE, STEP = (33, 32, 40), (1, 1, 12)
VELOCITIES = (np.arange(SIZE[2]) - 19.5) * STEP[2]


def spectra(driver, psf=None, lsf=None, scale=(1, 1, 1)):
    """The spectra of the gmodel seen with the PSF and the LSF."""
    group = ObservationGroup([model_parser.load(GMODEL)], [
        observation_parser.load(dict(
            driver=dict(type=driver.type()), scale=list(scale),
            instrument=dict(psf=psf, lsf=lsf),
            observable=dict(type='pixel_spectra', size=list(SIZE),
                            step=list(STEP))))])
    params = gbkfit.params.EvaluationParams(
        group.pdescs(), PROPERTIES).evaluate()
    return group.model_h(params)[0]['spectra']['d'].copy()


def table(low, high):
    """A table of a value from low at -500 km/s to high at 500 km/s."""
    return dict(velocity=[-500, 500], values=[low, high])


@pytest.mark.parametrize('psf, lsf', [
    (table(1.3, 1.3), table(17, 17)), (table(1.3, 1.3), 17),
    (1.3, table(17, 17)), (table(1.3, 1.3), None), (None, table(17, 17))])
@pytest.mark.parametrize('scale', [(1, 1, 1), (2, 2, 3)])
def test_tables_of_one_value_are_the_constant_psf_and_lsf(
        driver, psf, lsf, scale):
    # The convolutions along y and x, and along z, give what the 3D FFT
    # of a constant PSF and LSF gives
    def kernels(psf_sigma, lsf_sigma):
        return dict(
            psf=None if psf_sigma is None
            else dict(type='gauss', sigma=psf_sigma),
            lsf=None if lsf_sigma is None
            else dict(type='gauss', sigma=lsf_sigma))
    varying = spectra(driver, **kernels(psf, lsf), scale=scale)
    constant = spectra(driver, **kernels(
        None if psf is None else 1.3, None if lsf is None else 17),
        scale=scale)
    np.testing.assert_allclose(
        varying, constant, atol=2e-6 * np.abs(constant).max())


def test_each_channel_has_its_psf(driver):
    sigma = np.interp(VELOCITIES, [-500, 500], [1, 2])
    varying = spectra(driver, psf=dict(type='gauss', sigma=table(1, 2)))
    plain = spectra(driver)
    expected = [
        scipy.signal.fftconvolve(
            image, PSFGauss(s).asarray((1, 1)), mode='same')
        for image, s in zip(plain, sigma)]
    np.testing.assert_allclose(
        varying, expected, atol=2e-6 * np.abs(varying).max())


def test_each_channel_has_its_lsf(driver):
    # The light of each channel spreads with the LSF of that channel
    sigma = np.interp(VELOCITIES, [-500, 500], [10, 30])
    varying = spectra(driver, lsf=dict(type='gauss', sigma=table(10, 30)))
    plain = spectra(driver)
    expected = np.zeros((SIZE[2] + 200,) + plain.shape[1:])
    for z, s in enumerate(sigma):
        kernel = LSFGauss(s).asarray(STEP[2], 201)
        expected[z:z + 201] += kernel[:, None, None] * plain[z]
    np.testing.assert_allclose(
        varying, expected[100:100 + SIZE[2]],
        atol=2e-6 * np.abs(varying).max())


def test_the_psf_must_be_known_at_the_channels_of_the_data(driver):
    psf = dict(type='gauss', sigma=dict(velocity=[-100, 500], values=[1, 2]))
    with pytest.raises(ConfigError, match="from -100 to 500 km/s"):
        spectra(driver, psf=psf)


def test_psf_images_of_one_image_are_that_image(driver, tmp_path):
    import astropy.io.fits
    image = PSFGauss(1.5).asarray((0.5, 0.5))
    astropy.io.fits.writeto(tmp_path / 'image.fits', image)
    astropy.io.fits.writeto(tmp_path / 'cube.fits', np.stack([image, image]))
    cube = spectra(driver, psf=dict(
        type='images', file=str(tmp_path / 'cube.fits'), step=[0.5, 0.5],
        velocity=[-500, 500]))
    single = spectra(driver, psf=dict(
        type='image', file=str(tmp_path / 'image.fits'), step=[0.5, 0.5]))
    np.testing.assert_allclose(cube, single, atol=2e-6 * np.abs(single).max())


def test_lsf_images_of_one_profile_are_that_profile(driver, tmp_path):
    import astropy.io.fits
    profile = LSFGauss(17).asarray(4)
    astropy.io.fits.writeto(tmp_path / 'profile.fits', profile)
    astropy.io.fits.writeto(
        tmp_path / 'profiles.fits', np.stack([profile, profile]))
    images = spectra(driver, lsf=dict(
        type='images', file=str(tmp_path / 'profiles.fits'), step=4,
        velocity=[-500, 500]))
    single = spectra(driver, lsf=dict(
        type='image', file=str(tmp_path / 'profile.fits'), step=4))
    np.testing.assert_allclose(
        images, single, atol=2e-6 * np.abs(single).max())
