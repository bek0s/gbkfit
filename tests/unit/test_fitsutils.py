"""
Tests for reading and writing data with world coordinates in the units
of the model (gbkfit.utils.fitsutils).
"""

import astropy.coordinates
import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs
import numpy as np
import pytest

from gbkfit.utils import fitsutils
from gbkfit.utils.parseutils import ConfigError


def position_angle(header, pixel_from, pixel_to):
    """The position angle (degrees) on the sky from one pixel to another."""
    wcs = astropy.wcs.WCS(header).celestial
    sky = wcs.pixel_to_world([pixel_from[0], pixel_to[0]],
                             [pixel_from[1], pixel_to[1]])
    return sky[0].position_angle(sky[1]).to_value(u.deg)


def same_angle(a, b, tolerance=1e-6):
    return np.isclose((a - b + 180) % 360 - 180, 0, atol=tolerance)


def write(header, shape=(20, 24)):
    fits.writeto('data.fits', np.zeros(shape, np.float32), fits.Header(header))
    return fitsutils.read_data('data.fits')[1]


CELESTIAL = dict(
    CTYPE1='RA---TAN', CRVAL1=150.0, CRPIX1=12.5, CUNIT1='deg',
    CTYPE2='DEC--TAN', CRVAL2=2.0, CRPIX2=10.5, CUNIT2='deg')


@pytest.mark.parametrize('rota', [0, 30, -60, 135])
@pytest.mark.parametrize('shape', [(20, 24), (12, 20, 24)])
def test_write_and_read_back(rota, shape):
    step = (2.0, 3.0, 10.0)[:len(shape)]
    rpix = (11.5, 9.5, 5.0)[:len(shape)]
    rval = (150.0, 2.0, 1500.0)[:len(shape)]
    coords = fitsutils.Coords(step, rpix, rval, rota)
    spectral_axis = 2 if len(shape) == 3 else None
    fitsutils.write_data(
        'data.fits', np.ones(shape, np.float32), coords, spectral_axis)
    data, coords_read = fitsutils.read_data('data.fits')
    assert data.shape == shape
    np.testing.assert_allclose(coords_read.step, step, rtol=1e-12)
    np.testing.assert_allclose(coords_read.rpix, rpix, rtol=1e-12)
    np.testing.assert_allclose(coords_read.rval, rval, rtol=1e-12)
    assert same_angle(coords_read.rota, rota)


@pytest.mark.parametrize('rota', [0, 30, -60, 135])
def test_rota_is_the_position_angle_of_the_y_axis(rota):
    # Independently of fitsutils: astropy's position angle from a pixel
    # to the next one along y is rota, and along x it is rota - 90 (east
    # is to the left of north)
    coords = fitsutils.Coords((2.0, 2.0), (10, 10), (150.0, 2.0), rota)
    fitsutils.write_data('data.fits', np.ones((20, 20), np.float32), coords)
    header = fits.getheader('data.fits')
    assert same_angle(position_angle(header, (10, 10), (10, 11)), rota)
    assert same_angle(position_angle(header, (10, 10), (11, 10)), rota - 90)


def test_write_and_read_back_long_slit():
    # The position along the slit is an offset from its reference pixel
    coords = fitsutils.Coords((2.0, 10.0), (15.5, 5.0), (0.0, 1500.0), 0.0)
    fitsutils.write_data(
        'data.fits', np.ones((11, 32), np.float32), coords, spectral_axis=1)
    _, coords_read = fitsutils.read_data('data.fits')
    assert coords_read == coords


def test_north_up_east_left():
    coords = write(CELESTIAL | dict(CDELT1=-2 / 3600, CDELT2=2 / 3600))
    np.testing.assert_allclose(coords.step, [2, 2])
    assert coords.rpix == (11.5, 9.5)
    assert coords.rval == (150.0, 2.0)
    assert same_angle(coords.rota, 0)


def test_cd_matrix():
    # A CD matrix (B26: this gave a step of 1), rotated by 40 degrees
    angle = np.radians(40)
    header = CELESTIAL | dict(
        CD1_1=-np.cos(angle) / 3600, CD1_2=np.sin(angle) / 3600,
        CD2_1=np.sin(angle) / 3600, CD2_2=np.cos(angle) / 3600)
    coords = write(header)
    np.testing.assert_allclose(coords.step, [1, 1])
    assert same_angle(
        coords.rota, position_angle(header, (11.5, 9.5), (11.5, 10.5)),
        tolerance=1e-4)


def test_crota():
    # CROTA2 is deprecated, but old files have it
    header = CELESTIAL | dict(
        CDELT1=-1 / 3600, CDELT2=1 / 3600, CROTA2=25.0)
    coords = write(header)
    np.testing.assert_allclose(coords.step, [1, 1])
    assert same_angle(
        coords.rota, position_angle(header, (11.5, 9.5), (11.5, 10.5)),
        tolerance=1e-4)


@pytest.mark.parametrize('cunit, cdelt, step', [
    ('km/s', 10.0, 10.0),
    ('m/s', 10000.0, 10.0),
    (None, 10000.0, 10.0)])  # m/s is the default unit of velocities
def test_velocity_axes_are_in_km_s(cunit, cdelt, step):
    header = CELESTIAL | dict(
        CDELT1=-1 / 3600, CDELT2=1 / 3600,
        CTYPE3='VRAD', CRVAL3=150 * cdelt, CRPIX3=1.0, CDELT3=cdelt)
    if cunit:
        header['CUNIT3'] = cunit
    coords = write(header, shape=(6, 20, 24))
    np.testing.assert_allclose(coords.step[2], step)
    np.testing.assert_allclose(coords.rval[2], 150 * step)


def test_axes_without_a_type_are_as_given():
    coords = write(dict(CDELT1=2.0, CDELT2=3.0, CRPIX1=4.0, CRVAL2=7.0))
    assert coords == fitsutils.Coords((2.0, 3.0), (3.0, 9.5), (0.0, 7.0), 0.0)


def test_rpix_or_rval_can_be_given():
    header = CELESTIAL | dict(CDELT1=-1 / 3600, CDELT2=1 / 3600)
    write(header)
    # The world position at a given pixel, and the pixel at that position
    _, coords = fitsutils.read_data('data.fits', rpix=(5.0, 6.0))
    expected = astropy.wcs.WCS(fits.Header(header)).pixel_to_world_values(
        5.0, 6.0)
    np.testing.assert_allclose(coords.rval, expected, rtol=1e-12)
    _, coords = fitsutils.read_data('data.fits', rval=coords.rval)
    np.testing.assert_allclose(coords.rpix, (5.0, 6.0), atol=1e-9)


def test_hdu_can_be_chosen():
    # e.g. the SCI extension of JWST data
    header = fits.Header(CELESTIAL | dict(CDELT1=-1 / 3600, CDELT2=1 / 3600))
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((20, 24), np.float32), header, name='SCI')
    ]).writeto('data.fits')
    data, coords = fitsutils.read_data('data.fits', hdu='SCI')
    assert data.shape == (20, 24)
    assert coords.rval == (150.0, 2.0)


@pytest.mark.parametrize('header, shape, message', [
    (CELESTIAL | dict(CDELT1=1 / 3600, CDELT2=1 / 3600), (20, 24),
     "mirrored"),
    (CELESTIAL | dict(
        CD1_1=-1 / 3600, CD1_2=0.5 / 3600, CD2_1=0.0, CD2_2=1 / 3600),
     (20, 24), "skewed"),
    (CELESTIAL | dict(
        CDELT1=-1 / 3600, CDELT2=1 / 3600, CTYPE3='WAVE', CUNIT3='um',
        CRVAL3=1.5, CDELT3=0.001), (6, 20, 24), "type 'WAVE'"),
    (CELESTIAL | dict(
        CDELT1=-1 / 3600, CDELT2=1 / 3600, CTYPE3='FREQ', CUNIT3='Hz',
        CRVAL3=1.4e9, CDELT3=1e4), (6, 20, 24), "type 'FREQ'"),
    (CELESTIAL | dict(
        CDELT1=-1 / 3600, CDELT2=1 / 3600, CTYPE3='VRAD', CUNIT3='KM/S',
        CRVAL3=1.0, CDELT3=1.0), (6, 20, 24), "invalid world coordinates"),
    (dict(CDELT1=1.0, CDELT2=1.0, CDELT3=1.0, PC3_1=0.5), (6, 20, 24),
     "couple axes")])
def test_coordinates_the_model_cannot_represent(header, shape, message):
    with pytest.raises(ConfigError, match=message):
        write(header, shape)
