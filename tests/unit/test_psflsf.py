
import math

import numpy as np
import pytest

import gbkfit.math
from gbkfit.psflsf import *
from gbkfit.psflsf.lsfs import *
from gbkfit.psflsf.psfs import *


@pytest.mark.parametrize(
    "psf_type, psf_class, psf_params", [
        ('point', PSFPoint, dict()),
        ('gauss', PSFGauss, dict(
            sigma=5.0, ratio=1.0, posa=0.0)),
        ('ggauss', PSFGGauss, dict(
            alpha=5.0, beta=1.0, ratio=1.0, posa=0.0)),
        ('lorentz', PSFLorentz, dict(
            gamma=5.0, ratio=1.0, posa=0.0)),
        ('moffat', PSFMoffat, dict(
            alpha=5.0, beta=1.0, ratio=1.0, posa=0.0))
    ]
)
def test_psf_analytic(psf_type, psf_class, psf_params):
    step = (0.5, 0.2)
    offset = (0, 0)
    # Creation tests
    psf = psf_class(**psf_params)
    psf_arr = psf.asarray(step)
    psf_size = psf.size(step, offset)
    arr_max_index = np.unravel_index(np.argmax(psf_arr), psf_arr.shape)
    arr_max_index = tuple(i.item() for i in arr_max_index)
    arr_max_index = arr_max_index[::-1]
    assert psf_arr.shape[::-1] == psf_size
    assert gbkfit.math.is_odd(np.all(psf_size))
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
    offset = 0
    # Creation tests
    lsf = lsf_class(**lsf_params)
    lsf_arr = lsf.asarray(step)
    lsf_size = lsf.size(step, offset)
    arr_max_index = np.unravel_index(np.argmax(lsf_arr), lsf_arr.shape)
    arr_max_index = tuple(i.item() for i in arr_max_index)
    arr_max_index = arr_max_index[::-1][0]
    assert lsf_arr.shape[0] == lsf_size
    assert gbkfit.math.is_odd(np.all(lsf_size))
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


if __name__ == '__main__':
    test_psf_analytic()
    test_lsf_analytic()


def test_psf_image_round_trip():
    image = np.exp(-np.add.outer(np.arange(-4, 5) ** 2, np.arange(-3, 4) ** 2))
    psf = PSFImage(image, step=(0.2, 0.3))
    loaded = psf_parser.load(psf_parser.dump(psf))
    np.testing.assert_allclose(loaded._step, (0.2, 0.3), rtol=1e-12)
    np.testing.assert_array_equal(loaded._data, psf._data)


def test_psf_image_pixel_scale_is_in_arcsec():
    from astropy.io import fits
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CUNIT1='deg', CDELT1=-0.1 / 3600, CRVAL1=150.0,
        CTYPE2='DEC--TAN', CUNIT2='deg', CDELT2=0.1 / 3600, CRVAL2=2.0))
    fits.writeto('psf.fits', np.ones((9, 9)), header)
    psf = psf_parser.load(dict(type='image', data='psf.fits'))
    np.testing.assert_allclose(psf._step, (0.1, 0.1), rtol=1e-12)


def test_lsf_image_round_trip_and_channel_width_in_km_s():
    from astropy.io import fits
    lsf = LSFImage(np.exp(-np.arange(-5, 6) ** 2 / 4), step=2.5)
    loaded = lsf_parser.load(lsf_parser.dump(lsf))
    np.testing.assert_allclose(loaded._step, 2.5, rtol=1e-12)
    np.testing.assert_array_equal(loaded._data, lsf._data)
    header = fits.Header(dict(CTYPE1='VRAD', CUNIT1='m/s', CDELT1=2500.0))
    fits.writeto('lsf_m_s.fits', np.ones(11), header)
    loaded = lsf_parser.load(dict(type='image', data='lsf_m_s.fits'))
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
