
import json
import math

import numpy as np
import pytest
import scipy.integrate
import scipy.special

import gbkfit.math
from gbkfit.instrument import *


@pytest.mark.parametrize(
    "psf_type, psf_class, psf_params", [
        ('point', PSFPoint, dict()),
        ('gauss', PSFGauss, dict(
            sigma=5.0, ratio=1.0, posa=0.0)),
        ('ggauss', PSFGGauss, dict(
            alpha=5.0, beta=1.0, ratio=1.0, posa=0.0)),
        ('moffat', PSFMoffat, dict(
            alpha=5.0, beta=2.5, ratio=1.0, posa=0.0))
    ]
)
def test_psf_analytic(psf_type, psf_class, psf_params):
    step = (0.5, 0.2)
    # Creation tests
    psf = psf_class(**psf_params)
    psf_arr = psf.asarray(step)
    psf_size = psf.size(step)
    arr_max_index = np.unravel_index(np.argmax(psf_arr), psf_arr.shape)
    arr_max_index = tuple(i.item() for i in arr_max_index)
    arr_max_index = arr_max_index[::-1]
    assert psf_arr.shape[::-1] == psf_size
    assert all(gbkfit.math.is_odd(n) for n in psf_size)
    assert math.isclose(np.sum(psf_arr), 1.0, abs_tol=1e-9)
    assert arr_max_index == (psf_size[0] // 2, psf_size[1] // 2)
    # Dump tests
    dumped_psf_info = psf.dump()
    psf_info = dict(
        type=psf_type,
        **psf_params)
    assert dumped_psf_info == psf_info
    # Load tests
    loaded_psf = psf_parser.load(dumped_psf_info)
    assert vars(loaded_psf) == vars(psf)


@pytest.mark.parametrize(
    "lsf_type, lsf_class, lsf_params", [
        ('point', LSFPoint, dict()),
        ('gauss', LSFGauss, dict(
            sigma=5.0)),
        ('ggauss', LSFGGauss, dict(
            alpha=5.0, beta=1.0)),
        ('lorentz', LSFLorentz, dict(
            gamma=5.0)),
        ('moffat', LSFMoffat, dict(
            alpha=5.0, beta=1.0))
    ]
)
def test_lsf_analytic(lsf_type, lsf_class, lsf_params):
    step = 0.5
    # Creation tests
    lsf = lsf_class(**lsf_params)
    lsf_arr = lsf.asarray(step)
    lsf_size = lsf.size(step)
    arr_max_index = np.unravel_index(np.argmax(lsf_arr), lsf_arr.shape)
    arr_max_index = tuple(i.item() for i in arr_max_index)
    arr_max_index = arr_max_index[::-1][0]
    assert lsf_arr.shape[0] == lsf_size
    assert gbkfit.math.is_odd(lsf_size)
    assert math.isclose(np.sum(lsf_arr), 1.0, abs_tol=1e-9)
    assert arr_max_index == lsf_size // 2
    # Dump tests
    dumped_lsf_info = lsf.dump()
    lsf_info = dict(
        type=lsf_type,
        **lsf_params)
    assert dumped_lsf_info == lsf_info
    # Load tests
    loaded_lsf = lsf_parser.load(dumped_lsf_info)
    assert vars(loaded_lsf) == vars(lsf)



def assert_profile(array, profile):
    """
    The array is the profile (unnormalised) at the centres of its pixels,
    normalised over those it keeps (it is 0 beyond its extent).
    """
    kept = array > 0
    expected = np.where(kept, profile, 0)
    np.testing.assert_allclose(
        array, expected / expected.sum(), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize('lsf, profile', [
    (LSFGauss(3.0), lambda x: np.exp(-0.5 * (x / 3) ** 2)),
    (LSFGGauss(3.0, 1.5), lambda x: np.exp(-(np.abs(x) / 3) ** 1.5)),
    (LSFLorentz(3.0), lambda x: 1 / (1 + (x / 3) ** 2)),
    (LSFMoffat(3.0, 2.5), lambda x: (1 + (x / 3) ** 2) ** -2.5)])
def test_lsf_profiles(lsf, profile):
    step = 0.7
    array = lsf.asarray(step)
    x = (np.arange(array.size) - array.size // 2) * step
    assert_profile(array, profile(x))


@pytest.mark.parametrize('psf, profile', [
    # Of the distance r along the major axis (at the position angle posa,
    # north through east) and the minor axis (shorter by the ratio)
    (PSFGauss(2.0, 0.5, 30), lambda r: np.exp(-0.5 * (r / 2) ** 2)),
    (PSFGGauss(2.0, 1.5, 0.6, 120), lambda r: np.exp(-(r / 2) ** 1.5)),
    (PSFMoffat(2.0, 2.5, 0.7, -40), lambda r: (1 + (r / 2) ** 2) ** -2.5)])
def test_psf_profiles(psf, profile):
    step = (0.5, 0.4)
    array = psf.asarray(step)
    posa = np.radians(psf.dump()['posa'])
    ratio = psf.dump()['ratio']
    ny, nx = array.shape
    y, x = np.indices(array.shape)
    x = (x - nx // 2) * step[0]
    y = (y - ny // 2) * step[1]
    major = -x * np.sin(posa) + y * np.cos(posa)
    minor = x * np.cos(posa) + y * np.sin(posa)
    assert_profile(array, profile(np.hypot(major, minor / ratio)))


def test_psf_image_round_trip():
    image = np.exp(-np.add.outer(np.arange(-4, 5) ** 2, np.arange(-3, 4) ** 2))
    psf = PSFImage(image, step=(0.2, 0.3))
    # (through json, as configurations are: the step becomes a list)
    loaded = psf_parser.load(json.loads(json.dumps(psf_parser.dump(psf))))
    np.testing.assert_allclose(loaded._step, (0.2, 0.3), rtol=1e-12)
    np.testing.assert_array_equal(loaded._data, psf._data)



def test_images_in_sums_and_convolutions_have_their_own_files():
    # The images of the terms of a sum or a convolution are written to
    # files named with the prefix and the index of the term
    from gbkfit.instrument import LSFConvolution, LSFSum
    from gbkfit.instrument import PSFConvolution, PSFSum
    psf_a = PSFImage(np.outer([1, 2, 1], [1, 3, 1]))
    psf_b = PSFImage(np.outer([1, 1, 1], [1, 1, 1]))
    lsf_a = LSFImage(np.array([1, 2, 1]), step=2)
    lsf_b = LSFImage(np.array([1, 1, 1]), step=2)
    psf_sum = psf_parser.dump(
        PSFSum([psf_a, PSFConvolution([psf_b])], [1, 1]), prefix='out_')
    assert [psf_sum['psfs'][0]['file'],
            psf_sum['psfs'][1]['psfs'][0]['file']] == [
        'out_term0_psf.fits', 'out_term1_term0_psf.fits']
    lsf_sum = lsf_parser.dump(
        LSFSum([lsf_a, LSFConvolution([lsf_b])], [1, 1]), prefix='out_')
    assert [lsf_sum['lsfs'][0]['file'],
            lsf_sum['lsfs'][1]['lsfs'][0]['file']] == [
        'out_term0_lsf.fits', 'out_term1_term0_lsf.fits']
    loaded = psf_parser.load(psf_sum)
    np.testing.assert_array_equal(loaded._psfs[0]._data, psf_a._data)
    np.testing.assert_array_equal(
        loaded._psfs[1]._psfs[0]._data, psf_b._data)


def test_kernels_of_too_heavy_wings_are_an_error():
    # No finite array holds 99% of the flux of a Moffat LSF of beta 0.6
    # (its extent divides by 0)
    with (pytest.raises(RuntimeError, match="wings are too heavy"),
          np.errstate(divide='ignore')):
        LSFMoffat(1, 0.6).size(1)

def test_image_kernels_keep_their_edges():
    # Resampling an image samples it up to half a pixel beyond the
    # centres of its edge pixels (always, for an even size): there it
    # tends to the 0 beyond, instead of being 0 (a 2 x 2 image became a
    # delta, and the width of a narrow kernel shrank)
    x = np.arange(6) - 2.5
    gauss = np.exp(-0.5 * x ** 2)
    lsf = LSFImage(gauss, 1).asarray(1)
    centre = np.arange(lsf.size) - lsf.size // 2
    width = np.sqrt(np.sum(lsf * centre ** 2))
    assert width == pytest.approx(np.sqrt(np.sum(gauss * x ** 2) / gauss.sum()),
                                  rel=0.02)
    psf = PSFImage(np.ones((2, 2))).asarray((1, 1))
    assert (psf > 0).all()

def test_psf_image_pixel_scale_is_in_arcsec():
    from astropy.io import fits
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CUNIT1='deg', CDELT1=-0.1 / 3600, CRVAL1=150.0,
        CTYPE2='DEC--TAN', CUNIT2='deg', CDELT2=0.1 / 3600, CRVAL2=2.0))
    fits.writeto('psf.fits', np.ones((9, 9)), header)
    psf = psf_parser.load(dict(type='image', file='psf.fits'))
    np.testing.assert_allclose(psf._step, (0.1, 0.1), rtol=1e-12)


def test_lsf_image_round_trip_and_channel_width_in_km_s():
    from astropy.io import fits
    lsf = LSFImage(np.exp(-np.arange(-5, 6) ** 2 / 4), step=2.5)
    loaded = lsf_parser.load(lsf_parser.dump(lsf))
    np.testing.assert_allclose(loaded._step, 2.5, rtol=1e-12)
    np.testing.assert_array_equal(loaded._data, lsf._data)
    header = fits.Header(dict(CTYPE1='VRAD', CUNIT1='m/s', CDELT1=2500.0))
    fits.writeto('lsf_m_s.fits', np.ones(11), header)
    loaded = lsf_parser.load(dict(type='image', file='lsf_m_s.fits'))
    np.testing.assert_allclose(loaded._step, 2.5, rtol=1e-12)


def major_axis_position_angle(image):
    """
    The position angle (degrees, from +y towards -x, i.e. north through
    east) of the major axis of an image, from its second moments.
    """
    y, x = np.indices(image.shape)
    x = x - (image.shape[1] - 1) / 2
    y = y - (image.shape[0] - 1) / 2
    w = image / image.sum()
    covariance = [
        [(w * x * x).sum(), (w * x * y).sum()],
        [(w * x * y).sum(), (w * y * y).sum()]]
    _, vectors = np.linalg.eigh(covariance)
    vx, vy = vectors[:, 1]
    return np.degrees(np.arctan2(-vx, vy)) % 180


@pytest.mark.parametrize('psf_type', [PSFGauss, PSFGGauss, PSFMoffat])
@pytest.mark.parametrize('posa, rota', [(0, 0), (30, 0), (120, 0), (50, 20)])
def test_psf_position_angle_is_that_of_its_major_axis(psf_type, posa, rota):
    # Like the disks: posa is the position angle of the major axis, from
    # north through east, and on a grid rotated by rota it is drawn at
    # posa - rota
    options = dict(ratio=0.5, posa=posa)
    psf = (psf_type(2, **options) if psf_type is PSFGauss
           else psf_type(2, 2, **options))
    image = psf.asarray((1, 1), (41, 41), rota=rota)
    # The second moments of profiles with wide wings depend on the pixels
    # at their edge, so they measure the angle to a few hundredths of a
    # degree
    np.testing.assert_allclose(
        major_axis_position_angle(image), (posa - rota) % 180, atol=0.1)


def centroid(array, axis):
    """The centroid of an array along an axis, in pixels."""
    index = np.indices(array.shape)[axis]
    return (index * array).sum() / array.sum()


@pytest.mark.parametrize('size, offset', [(41, 0), (64, -1)])
def test_image_psf_and_lsf_are_centred_like_the_analytic_ones(size, offset):
    # The FFT convolution expects the centre at size // 2 + offset (with
    # offset -1 for even sizes), as the analytic PSFs and LSFs put it
    psf = PSFGauss(2)
    psf_image = PSFImage(psf.asarray((1, 1), (21, 21)))
    expected = psf.asarray((1, 1), (size, size), (offset, offset))
    actual = psf_image.asarray((1, 1), (size, size), (offset, offset))
    for axis in (0, 1):
        np.testing.assert_allclose(
            centroid(actual, axis), centroid(expected, axis), atol=1e-6)
    lsf = LSFGauss(2)
    lsf_image = LSFImage(lsf.asarray(1, 21))
    np.testing.assert_allclose(
        centroid(lsf_image.asarray(1, size, offset), 0),
        centroid(lsf.asarray(1, size, offset), 0), atol=1e-6)


def test_image_psf_on_a_non_square_grid():
    # An array of size (x, y) has shape (y, x), like the analytic PSFs
    psf = PSFGauss(2, ratio=0.5, posa=30)
    psf_image = PSFImage(psf.asarray((1, 1), (31, 31)))
    expected = psf.asarray((1, 1), (65, 33))
    actual = psf_image.asarray((1, 1), (65, 33))
    assert actual.shape == expected.shape == (33, 65)
    np.testing.assert_allclose(actual, expected, atol=1e-4 * expected.max())


@pytest.mark.parametrize('lsf', [
    LSFGauss(10), LSFGGauss(10, 0.5), LSFLorentz(10), LSFMoffat(10, 1.5)])
def test_analytic_lsfs_are_symmetric(lsf):
    # Cut at the same distance on both sides, also when the array is much
    # wider than the cut (as in the FFT convolution)
    data = lsf.asarray(5, 257)
    np.testing.assert_allclose(data, data[::-1], rtol=1e-12)
    np.testing.assert_allclose(centroid(lsf.asarray(5, 256, -1), 0), 127)


@pytest.mark.parametrize('profile, size', [
    (PSFGauss(2), (33, 33)), (PSFGGauss(2, 2), (33, 33)),
    (PSFMoffat(2, 4.765), (33, 33)), (LSFGauss(2), 33)])
def test_compact_profiles_extend_to_the_minimum_extent(profile, size):
    # 2 x MIN_EXTENT scale lengths of 2, made odd
    step = (1, 1) if isinstance(profile, PSF) else 1
    assert profile.size(step) == size


def moffat_1d_flux(z1, z2, beta=0.75):
    """The flux of a 1D Moffat profile with alpha 1 from z1 to z2."""
    return scipy.integrate.quad(lambda z: (1 + z * z) ** -beta, z1, z2)[0]


@pytest.mark.parametrize('profile, wing_flux', [
    (PSFGGauss(1, 0.5),
     lambda r: scipy.special.gammaincc(4, np.sqrt(r))),
    (PSFMoffat(1, 1.5),
     lambda r: (1 + r * r) ** (1 - 1.5)),
    (LSFGGauss(1, 0.5),
     lambda w: scipy.special.gammaincc(2, np.sqrt(w))),
    (LSFLorentz(1),
     lambda w: 1 - 2 / np.pi * np.arctan(w)),
    (LSFMoffat(1, 0.75),
     lambda w: moffat_1d_flux(w, np.inf) / moffat_1d_flux(0, np.inf))])
def test_wide_wings_are_drawn_until_they_hold_the_wing_flux(
        profile, wing_flux):
    # Profiles with wide wings extend beyond the minimum extent, until
    # their wings hold WING_FLUX of their flux (each wing_flux above is
    # the fraction of the flux beyond a radius, for scale lengths of 1)
    from gbkfit.instrument._detail import MIN_EXTENT, WING_FLUX
    extent = profile._extent()
    assert extent > MIN_EXTENT
    np.testing.assert_allclose(wing_flux(extent), WING_FLUX, rtol=1e-6)


@pytest.mark.parametrize('make, message', [
    (lambda: PSFGauss(0), "sigma must be greater than 0"),
    (lambda: PSFGauss(1, ratio=0), "ratio must be greater than 0"),
    (lambda: PSFGGauss(1, 1, ratio=1.5), "ratio must be greater than 0"),
    (lambda: PSFGGauss(1, -1), "beta must be greater than 0"),
    (lambda: PSFMoffat(1, 1), "finite flux only for beta > 1"),
    (lambda: LSFLorentz(-1), "gamma must be greater than 0"),
    (lambda: LSFMoffat(1, 0.5), "finite flux only for beta > 0.5")])
def test_invalid_profiles_are_rejected(make, message):
    with pytest.raises(RuntimeError, match=message):
        make()


def test_arrays_of_the_default_size_take_no_odd_offset():
    # The default size is odd, so the centre needs offset 0: an odd
    # offset (for even sizes) is rejected, like for any other size
    with pytest.raises(RuntimeError, match="must be odd"):
        PSFGauss(2).asarray((1, 1), None, (-1, 0))
    with pytest.raises(RuntimeError, match="must be odd"):
        LSFGauss(2).asarray(1, None, -1)


def test_psf_sum():
    from gbkfit.instrument import psf_parser
    from gbkfit.instrument import PSFGauss, PSFMoffat, PSFSum
    step = (0.5, 0.5)
    # A sum of one PSF is that PSF
    gauss = PSFGauss(1.0)
    single = PSFSum([gauss], [3])
    np.testing.assert_allclose(
        single.asarray(step), gauss.asarray(step), rtol=1e-12)
    # The terms have their fractions of the light, on the grid of the
    # widest term
    moffat = PSFMoffat(2.0, 2.5)
    double = PSFSum([gauss, moffat], [0.7, 0.3])
    assert double.size(step) == moffat.size(step)
    size = double.size(step)
    expected = 0.7 * gauss.asarray(step, size) + 0.3 * moffat.asarray(
        step, size)
    data = double.asarray(step)
    np.testing.assert_allclose(data, expected, rtol=1e-12)
    np.testing.assert_allclose(data.sum(), 1, rtol=1e-12)
    # Round trip through the configuration
    info = psf_parser.dump(double)
    assert info['weights'] == [0.7, 0.3]
    np.testing.assert_allclose(
        psf_parser.load(info).asarray(step), data, rtol=1e-12)


def test_lsf_sum():
    from gbkfit.instrument import lsf_parser
    from gbkfit.instrument import LSFGauss, LSFSum
    narrow, wide = LSFGauss(10.0), LSFGauss(30.0)
    double = LSFSum([narrow, wide], [2, 2])
    size = double.size(5.0)
    assert size == wide.size(5.0)
    np.testing.assert_allclose(
        double.asarray(5.0),
        0.5 * narrow.asarray(5.0, size) + 0.5 * wide.asarray(5.0, size),
        rtol=1e-12)
    info = lsf_parser.dump(double)
    np.testing.assert_allclose(
        lsf_parser.load(info).asarray(5.0), double.asarray(5.0), rtol=1e-12)


@pytest.mark.parametrize('weights', [[1], [1, 0], [1, -1]])
def test_sum_weights_are_checked(weights):
    from gbkfit.instrument import PSFGauss, PSFSum
    with pytest.raises(RuntimeError, match="positive weight"):
        PSFSum([PSFGauss(1.0), PSFGauss(2.0)], weights)


def test_lsf_hanning():
    from gbkfit.instrument import LSFHanning
    # 1/4, 1/2 and 1/4 of the light, a channel width apart
    np.testing.assert_allclose(
        LSFHanning(10).asarray(5.0), [0.25, 0, 0.5, 0, 0.25])
    np.testing.assert_allclose(
        LSFHanning(10).asarray(10.0, 5), [0, 0.25, 0.5, 0.25, 0])
    with pytest.raises(RuntimeError, match="multiple of the step"):
        LSFHanning(10).asarray(3.0)


def test_lsf_and_psf_convolutions_of_gaussians():
    from gbkfit.instrument import LSFConvolution, LSFGauss
    from gbkfit.instrument import PSFConvolution, PSFGauss
    # The convolution of Gaussians is a Gaussian of the summed variances
    lsf = LSFConvolution([LSFGauss(6.0), LSFGauss(8.0)])
    lsf_size = lsf.size(1.0)
    np.testing.assert_allclose(
        lsf.asarray(1.0), LSFGauss(10.0).asarray(1.0, lsf_size), atol=1e-6)
    psf = PSFConvolution([PSFGauss(0.6), PSFGauss(0.8)])
    psf_size = psf.size((0.1, 0.1))
    np.testing.assert_allclose(
        psf.asarray((0.1, 0.1)), PSFGauss(1.0).asarray((0.1, 0.1), psf_size),
        atol=1e-6)
    # An even array has its centre offset like the analytic ones
    np.testing.assert_allclose(
        lsf.asarray(1.0, lsf_size + 1, -1)[:-1], lsf.asarray(1.0),
        atol=1e-12)


def test_psf_beam_from_a_header(tmp_path):
    # BMAJ and BMIN are full widths at half maximum in degrees, BPA the
    # position angle of the major axis: a Gaussian of sigma bmaj / 2.355,
    # ratio bmin / bmaj, at that position angle
    from astropy.io import fits
    from gbkfit.instrument import psf_parser
    from gbkfit.instrument import PSFGaussBeam, PSFGauss
    header = fits.Header(dict(BMAJ=3 / 3600, BMIN=1.5 / 3600, BPA=30.0))
    fits.writeto(tmp_path / 'cube.fits', np.zeros((2, 2)), header)
    beam = psf_parser.load(dict(type='gauss_beam', file=str(tmp_path / 'cube.fits')))
    assert isinstance(beam, PSFGaussBeam)
    gauss = PSFGauss(3 / np.sqrt(8 * np.log(2)), 0.5, 30)
    for rota in (0, 20):
        np.testing.assert_allclose(
            beam.asarray((0.2, 0.2), rota=rota),
            gauss.asarray((0.2, 0.2), rota=rota), rtol=1e-12)
    dumped = psf_parser.dump(beam)
    assert dumped['type'] == 'gauss_beam' and dumped['bpa'] == 30
    np.testing.assert_allclose([dumped['bmaj'], dumped['bmin']], [3, 1.5])
    assert psf_parser.dump(psf_parser.load(dict(dumped))) == dumped


@pytest.mark.parametrize('info, message', [
    (dict(bmaj=1, bmin=2), "at most bmaj"),
    (dict(bmaj=1, bmin=1, file='cube.fits'), "either a file")])
def test_psf_beam_options_are_checked(info, message):
    from gbkfit.instrument import psf_parser
    with pytest.raises(Exception, match=message):
        psf_parser.load(dict(type='gauss_beam') | info)


@pytest.mark.parametrize('parser', [psf_parser, lsf_parser])
def test_unknown_options_of_points_are_reported(parser):
    # (they used to be ignored silently)
    from gbkfit.utils import parseutils
    with parseutils.strict_mode():
        with pytest.raises(parseutils.ConfigError, match="'sigma'"):
            parser.load(dict(type='point', sigma=2))


def test_psfs_from_files_in_python(tmp_path):
    # Reading a file from Python gives what its configuration gives
    from astropy.io import fits
    from gbkfit.instrument import (
        PSFGauss, PSFGaussBeam, PSFImage, psf_parser)
    from gbkfit.utils import fitsutils, gridutils
    header = fits.Header(dict(BMAJ=3 / 3600, BMIN=1.5 / 3600, BPA=30.0))
    fits.writeto(tmp_path / 'cube.fits', np.zeros((2, 2)), header)
    image = PSFGauss(1.5).asarray((0.5, 0.5))
    fitsutils.write_data(
        str(tmp_path / 'psf.fits'), image,
        gridutils.make_grid(image.shape[::-1], 0.5).coords)
    for psf, info in (
            (PSFGaussBeam.from_file(str(tmp_path / 'cube.fits')),
             dict(type='gauss_beam', file=str(tmp_path / 'cube.fits'))),
            (PSFImage.from_file(str(tmp_path / 'psf.fits')),
             dict(type='image', file=str(tmp_path / 'psf.fits')))):
        loaded = psf_parser.load(info)
        np.testing.assert_allclose(
            psf.asarray((0.2, 0.2)), loaded.asarray((0.2, 0.2)))
    np.testing.assert_allclose(
        PSFImage.from_file(str(tmp_path / 'psf.fits'))._step, (0.5, 0.5))
