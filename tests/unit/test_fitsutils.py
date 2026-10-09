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


def test_write_and_read_back_line_of_sight_axis():
    # A cube without a spectral axis has a third spatial axis, along the
    # line of sight: an offset in arcsec from its reference pixel
    coords = fitsutils.Coords((2.0, 3.0, 0.5), (11.5, 9.5, 7.5),
                              (150.0, 2.0, 0.0), 30.0)
    fitsutils.write_data('data.fits', np.ones((16, 20, 24), np.float32),
                         coords, spectral_axis=None)
    header = fits.getheader('data.fits')
    assert header['CTYPE3'] == 'OFFSET' and header['CUNIT3'] == 'arcsec'
    _, coords_read = fitsutils.read_data('data.fits')
    np.testing.assert_allclose(coords_read.step, coords.step, rtol=1e-12)
    np.testing.assert_allclose(coords_read.rpix, coords.rpix, rtol=1e-12)
    np.testing.assert_allclose(coords_read.rval, coords.rval, rtol=1e-12)


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


@pytest.mark.parametrize('header', [
    # Dec before RA
    dict(CELESTIAL, CTYPE1='DEC--TAN', CRVAL1=2.0,
         CTYPE2='RA---TAN', CRVAL2=150.0),
    # The velocity before RA and Dec
    dict(CTYPE1='VRAD', CUNIT1='km/s', CDELT1=10.0,
         CTYPE2='RA---TAN', CDELT2=-1 / 3600, CUNIT2='deg',
         CTYPE3='DEC--TAN', CDELT3=1 / 3600, CUNIT3='deg')])
def test_celestial_axes_must_be_first(header):
    # The model takes the first two axes as x and y on the sky
    with pytest.raises(ConfigError, match="celestial axes must be the first"):
        write(header, (6, 20, 24) if 'CTYPE3' in header else (20, 24))


def test_spectral_axis_must_be_where_the_dataset_has_it():
    # e.g. a long slit (position, velocity) given as (velocity, position)
    header = dict(
        CTYPE1='VRAD', CUNIT1='km/s', CDELT1=10.0, CTYPE2='LINEAR',
        CDELT2=1.0)
    fits.writeto('data.fits', np.zeros((8, 40), np.float32),
                 fits.Header(header))
    assert fitsutils.read_data('data.fits')[1].step == (10.0, 1.0)
    with pytest.raises(ConfigError, match="spectral axis must be the axis 2"):
        fitsutils.read_data('data.fits', spectral_axis=1)

def test_hdu_without_data():
    # e.g. the empty primary HDU of JWST data: a clear error, not a crash
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((20, 24), np.float32), name='SCI')
    ]).writeto('data.fits')
    with pytest.raises(ConfigError, match="HDU 0 has no data"):
        fitsutils.read_data('data.fits')


def test_decreasing_velocity_is_rejected():
    # Common in radio cubes; prep reverses the axis
    header = CELESTIAL | dict(
        CDELT1=-1 / 3600, CDELT2=1 / 3600,
        CTYPE3='VRAD', CUNIT3='km/s', CRVAL3=1500.0, CDELT3=-10.0)
    with pytest.raises(ConfigError, match="velocity decreases"):
        write(header, shape=(6, 20, 24))


def test_grid_zero_and_spatial_axes():
    # The spatial axes are measured from the reference pixel, and the
    # spectral axis from its world value there
    coords = fitsutils.Coords((2.0, 3.0, 10.0), (4.0, 5.0, 6.0),
                              (150.0, 2.0, 1500.0), 30.0)
    grid = fitsutils.Grid((8, 10, 12), coords, 2)
    assert grid.zero() == (-8.0, -15.0, 1440.0)
    spatial = grid.spatial()
    assert spatial.size == (8, 10) and spatial.spectral_axis is None
    assert spatial.coords == fitsutils.Coords(
        (2.0, 3.0), (4.0, 5.0), (150.0, 2.0), 30.0)


def _spectral_header(ctype, **rest):
    return CELESTIAL | dict(
        CDELT1=-1 / 3600, CDELT2=1 / 3600,
        CTYPE3=ctype, CUNIT3='km/s', CRVAL3=0.0, CRPIX3=1.0, CDELT3=10.0,
        **rest)


@pytest.mark.parametrize('ctype, keywords, rest', [
    # The velocities of a rest wavelength are optical, those of a rest
    # frequency radio; the rest follows the convention of the axis
    ('VOPT', dict(RESTWAV=6.5628e-7), 6.5628e-7 * u.m),
    ('VRAD', dict(RESTFRQ=1.420405752e9), 1.420405752e9 * u.Hz),
    ('VOPT', dict(RESTFRQ=1.420405752e9), (1.420405752e9 * u.Hz).to(
        u.m, u.spectral())),
    ('VRAD', dict(RESTWAV=0.21), (0.21 * u.m).to(u.Hz, u.spectral())),
    ('VRAD', {}, None),
    ('VELO', dict(RESTFRQ=1.420405752e9), None)])
def test_rest_of_the_spectral_axis(ctype, keywords, rest):
    coords = write(_spectral_header(ctype, **keywords), shape=(6, 20, 24))
    if rest is None:
        assert coords.rest is None
    else:
        assert coords.rest.unit == rest.unit
        np.testing.assert_allclose(coords.rest.value, rest.value, rtol=1e-12)


def test_rest_can_be_given():
    # A given rest replaces that of the header, in the convention of the
    # axis; without a spectral axis type it keeps its kind
    fits.writeto('data.fits', np.zeros((6, 20, 24), np.float32),
                 fits.Header(_spectral_header('VOPT', RESTWAV=1e-7)))
    coords = fitsutils.read_data('data.fits', rest='1420.405752 MHz')[1]
    np.testing.assert_allclose(
        coords.rest.to_value(u.m), (1420.405752 * u.MHz).to_value(
            u.m, u.spectral()), rtol=1e-12)
    fits.writeto('plain.fits', np.zeros((6, 20, 24), np.float32))
    coords = fitsutils.read_data('plain.fits', rest='6562.8 Angstrom')[1]
    assert coords.rest == 6562.8 * u.Angstrom
    with pytest.raises(ConfigError, match="relativistic velocities"):
        fits.writeto('velo.fits', np.zeros((6, 20, 24), np.float32),
                     fits.Header(_spectral_header('VELO')))
        fitsutils.read_data('velo.fits', rest='6562.8 Angstrom')


@pytest.mark.parametrize('rest, ctype, keyword', [
    ('6562.8 Angstrom', 'VOPT', 'RESTWAV'),
    ('1420.405752 MHz', 'VRAD', 'RESTFRQ'),
    (None, 'VRAD', None)])
def test_rest_round_trip(rest, ctype, keyword):
    grid = fitsutils.make_grid(
        (24, 20, 6), (1, 1, 10), spectral_axis=2, rest=rest)
    fitsutils.write_data(
        'data.fits', np.zeros((6, 20, 24)), grid.coords, 2)
    header = fits.getheader('data.fits')
    assert header['CTYPE3'] == ctype
    assert keyword is None or keyword in header
    coords = fitsutils.read_data('data.fits')[1]
    if rest is None:
        assert coords.rest is None
    else:
        # FITS keeps about 16 significant digits
        assert coords.rest.unit == grid.coords.rest.unit
        np.testing.assert_allclose(
            coords.rest.value, grid.coords.rest.value, rtol=1e-14)


@pytest.mark.parametrize('value', [
    '6562.8', '3 km/s', '-1 Angstrom', 'Halpha', [1, 2] * u.m])
def test_invalid_rest(value):
    with pytest.raises(ConfigError, match="positive wavelength or frequency"):
        fitsutils.make_rest(value)


def test_nonlinear_velocity_axes_are_rejected():
    # e.g. velocities derived from a linear frequency axis (F2W) are not
    # linear in the pixels
    header = _spectral_header('VOPT-F2W', RESTWAV=6.5628e-7)
    with pytest.raises(ConfigError, match="not linear in velocity"):
        write(header, shape=(6, 20, 24))
