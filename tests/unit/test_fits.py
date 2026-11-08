"""
Tests for reading and writing data with world coordinates in the units
of the model.
"""

import astropy.coordinates
import astropy.io.fits as fits
import astropy.units as u
import astropy.wcs
import numpy as np
import pytest

from gbkfit.grid import (
    Coords, Grid, air_refractive_index, make_convention, make_grid, make_rest)
from gbkfit.io import (
    Beam, casa_beam_table, model_header, read_arrays, read_data, read_image,
    read_oriented, read_sky_map, spectra_header, write_data, write_image,
    write_spectra)
from gbkfit.io.headers import _centre_missing_crpix


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
    return read_data('data.fits')[1]


CELESTIAL = dict(
    CTYPE1='RA---TAN', CRVAL1=150.0, CRPIX1=12.5, CUNIT1='deg',
    CTYPE2='DEC--TAN', CRVAL2=2.0, CRPIX2=10.5, CUNIT2='deg')


@pytest.mark.parametrize('rota', [0, 30, -60, 135])
@pytest.mark.parametrize('shape', [(20, 24), (12, 20, 24)])
def test_write_and_read_back(rota, shape):
    step = (2.0, 3.0, 10.0)[:len(shape)]
    rpix = (11.5, 9.5, 5.0)[:len(shape)]
    rval = (150.0, 2.0, 1500.0)[:len(shape)]
    coords = Coords(step, rpix, rval, rota)
    spectral_axis = 2 if len(shape) == 3 else None
    write_data(
        'data.fits', np.ones(shape, np.float32), coords, spectral_axis)
    data, coords_read = read_data('data.fits')
    assert data.shape == shape
    np.testing.assert_allclose(coords_read.step, step, rtol=1e-12)
    np.testing.assert_allclose(coords_read.rpix, rpix, rtol=1e-12)
    np.testing.assert_allclose(coords_read.rval, rval, rtol=1e-12)
    assert same_angle(coords_read.rota, rota)


@pytest.mark.parametrize('rota', [0, 30, -60, 135])
def test_rota_is_the_position_angle_of_the_y_axis(rota):
    # Independently of the readers of FITS files: astropy's position angle from a pixel
    # to the next one along y is rota, and along x it is rota - 90 (east
    # is to the left of north)
    coords = Coords((2.0, 2.0), (10, 10), (150.0, 2.0), rota)
    write_data('data.fits', np.ones((20, 20), np.float32), coords)
    header = fits.getheader('data.fits')
    assert same_angle(position_angle(header, (10, 10), (10, 11)), rota)
    assert same_angle(position_angle(header, (10, 10), (11, 10)), rota - 90)


def test_write_and_read_back_long_slit():
    # The position along the slit is an offset from its reference pixel
    coords = Coords(
        (2.0, 10.0), (15.5, 5.0), (0.0, 1500.0), 0.0,
        convention='relativistic')
    write_data(
        'data.fits', np.ones((11, 32), np.float32), coords, spectral_axis=1)
    _, coords_read = read_data('data.fits')
    assert coords_read == coords


def test_write_and_read_back_line_of_sight_axis():
    # A cube without a spectral axis has a third spatial axis, along the
    # line of sight: an offset in arcsec from its reference pixel
    coords = Coords((2.0, 3.0, 0.5), (11.5, 9.5, 7.5),
                              (150.0, 2.0, 0.0), 30.0)
    write_data('data.fits', np.ones((16, 20, 24), np.float32),
                         coords, spectral_axis=None)
    header = fits.getheader('data.fits')
    assert header['CTYPE3'] == 'OFFSET' and header['CUNIT3'] == 'arcsec'
    _, coords_read = read_data('data.fits')
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


@pytest.mark.parametrize('frame', [
    dict(CTYPE1='GLON-TAN', CTYPE2='GLAT-TAN'),
    dict(RADESYS='FK4', EQUINOX=1950.0)])
def test_other_celestial_frames_are_turned_into_icrs(frame):
    # The RA and Dec of the reference pixel, and the position angle of the
    # +y axis from north of ICRS there, as astropy turns the pixels
    header = CELESTIAL | dict(
        CDELT1=-2 / 3600, CDELT2=2 / 3600, CROTA2=10.0) | frame
    coords = write(header)
    sky = astropy.wcs.WCS(header).celestial.pixel_to_world(
        [11.5, 11.5], [9.5, 10.5]).icrs
    assert coords.rval == pytest.approx(
        (sky[0].ra.deg, sky[0].dec.deg), abs=1e-9)
    assert same_angle(
        coords.rota, sky[0].position_angle(sky[1]).deg, tolerance=1e-4)
    assert not same_angle(coords.rota, 10.0, tolerance=0.1)


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
    # (the world position of the second axis at the reference pixel of
    # FITS, 0, 1-based)
    coords = write(dict(CDELT1=2.0, CDELT2=3.0, CRPIX1=4.0, CRVAL2=7.0))
    assert coords == Coords((2.0, 3.0), (3.0, -1.0), (0.0, 7.0), 0.0)


def test_rpix_or_rval_can_be_given():
    header = CELESTIAL | dict(CDELT1=-1 / 3600, CDELT2=1 / 3600)
    write(header)
    # The world position at a given pixel, and the pixel at that position
    _, coords = read_data('data.fits', rpix=(5.0, 6.0))
    expected = astropy.wcs.WCS(fits.Header(header)).pixel_to_world_values(
        5.0, 6.0)
    np.testing.assert_allclose(coords.rval, expected, rtol=1e-12)
    _, coords = read_data('data.fits', rval=coords.rval)
    np.testing.assert_allclose(coords.rpix, (5.0, 6.0), atol=1e-9)


def test_rpix_or_rval_must_have_a_value_for_each_axis():
    write(CELESTIAL | dict(CDELT1=-1 / 3600, CDELT2=1 / 3600))
    with pytest.raises(ValueError, match="'rpix' must have one value"):
        read_data('data.fits', rpix=(5.0, 6.0, 0.0))


def test_hdu_can_be_chosen():
    # e.g. the SCI extension of JWST data
    header = fits.Header(CELESTIAL | dict(CDELT1=-1 / 3600, CDELT2=1 / 3600))
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((20, 24), np.float32), header, name='SCI')
    ]).writeto('data.fits')
    data, coords = read_data(('data.fits', 'SCI'))
    assert data.shape == (20, 24)
    assert coords.rval == (150.0, 2.0)


@pytest.mark.parametrize('header, shape, message', [
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
        CRVAL3=1.0, CDELT3=1.0), (6, 20, 24), "CUNIT3: 'KM/S' is not a valid"),
    (dict(CDELT1=1.0, CDELT2=1.0, CDELT3=1.0, PC3_1=0.5), (6, 20, 24),
     "couple axes")])
def test_coordinates_the_model_cannot_represent(header, shape, message):
    with pytest.raises(ValueError, match=message):
        write(header, shape)


def test_rounding_of_a_matrix_is_not_a_coupling():
    # e.g. what tools that compute the matrices of headers leave
    write(dict(CDELT1=1.0, CDELT2=1.0, CDELT3=1.0, PC3_1=1e-17), (6, 20, 24))
    _, coords = read_data('data.fits')
    assert coords.step == (1.0, 1.0, 1.0)


def test_the_first_image_is_read_past_tables():
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns([fits.Column('a', 'K', array=[1, 2])]),
        fits.ImageHDU(np.ones((20, 24), np.float32), name='SCI')
    ]).writeto('data.fits')
    data, _ = read_image('data.fits')
    assert data.shape == (20, 24)
    with pytest.raises(ValueError, match="HDU 1 is not an image"):
        read_image(('data.fits', 1))


# A cube in the orientation of the model: RA, Dec and increasing velocity,
# east to the left of north (reference pixels off the centre, so that a
# reversed axis moves them)
CUBE = CELESTIAL | dict(
    CRPIX1=10.0, CDELT1=-1 / 3600, CDELT2=1 / 3600,
    CTYPE3='VRAD', CUNIT3='km/s', CRVAL3=1500.0, CRPIX3=2.0, CDELT3=10.0)

CUBE_DATA = np.arange(6 * 20 * 24, dtype=np.float32).reshape(6, 20, 24)

DEC = dict(CTYPE='DEC--TAN', CRVAL=2.0, CRPIX=10.5, CUNIT='deg',
           CDELT=1 / 3600)
RA = dict(CTYPE='RA---TAN', CRVAL=150.0, CRPIX=10.0, CUNIT='deg',
          CDELT=-1 / 3600)
VEL = dict(CTYPE='VRAD', CUNIT='km/s', CRVAL=1500.0, CRPIX=2.0, CDELT=10.0)


def axes_header(*axes):
    """The header of the given axes, in FITS order."""
    return {f'{key}{n}': value
            for n, axis in enumerate(axes, start=1)
            for key, value in axis.items()}


@pytest.mark.parametrize('header, data', [
    # Dec before RA
    (axes_header(DEC, RA, VEL), CUBE_DATA.transpose(0, 2, 1)),
    # The velocity before RA and Dec
    (axes_header(VEL, RA, DEC), CUBE_DATA.transpose(1, 2, 0)),
    # Mirrored (east to the right of north)
    (CUBE | dict(CDELT1=1 / 3600, CRPIX1=15.0), CUBE_DATA[:, :, ::-1]),
    # Velocities decreasing along the spectral axis
    (CUBE | dict(CDELT3=-10.0, CRPIX3=5.0), CUBE_DATA[::-1])])
def test_data_are_read_in_the_orientation_of_the_model(header, data):
    # The same sky, stored in another legal way, reads the same
    fits.writeto('model.fits', CUBE_DATA, fits.Header(CUBE))
    fits.writeto('other.fits', data, fits.Header(header))
    expected_data, expected = read_data('model.fits')
    result_data, result = read_data('other.fits')
    np.testing.assert_array_equal(result_data, expected_data)
    for field in ('step', 'rpix', 'rval'):
        np.testing.assert_allclose(
            getattr(result, field), getattr(expected, field), atol=1e-9)
    assert result.rota == pytest.approx(expected.rota, abs=1e-9)


@pytest.mark.parametrize('header, data, sky', [
    # Dec before RA
    (axes_header(DEC, RA, VEL), CUBE_DATA.transpose(0, 2, 1), np.transpose),
    # The velocity before RA and Dec
    (axes_header(VEL, RA, DEC), CUBE_DATA.transpose(1, 2, 0), np.asarray),
    # Mirrored (east to the right of north)
    (CUBE | dict(CDELT1=1 / 3600, CRPIX1=15.0), CUBE_DATA[:, :, ::-1],
     lambda image: image[:, ::-1]),
    # Velocities decreasing along the spectral axis
    (CUBE | dict(CDELT3=-10.0, CRPIX3=5.0), CUBE_DATA[::-1], np.asarray)])
def test_an_image_of_the_sky_is_read_on_the_pixels_of_the_data(
        header, data, sky):
    # An image of the sky of a cube without world coordinates (e.g. the
    # bins of its pixels) is in the axes of the file of the cube, and so
    # is put on its pixels as the cube is
    image = np.arange(20 * 24, dtype=np.float32).reshape(20, 24)
    fits.writeto('cube.fits', data, fits.Header(header))
    fits.writeto('image.fits', sky(image))
    arrays = read_arrays('cube.fits')
    result, _ = read_sky_map('image.fits', arrays)
    np.testing.assert_array_equal(result, image)
    # With world coordinates, it must be on the pixels of the sky of the
    # cube; with those of the cube, it is
    fits.writeto('wcs.fits', image, fits.Header(CELESTIAL | dict(
        CRPIX1=10.0, CDELT1=-1 / 3600, CDELT2=1 / 3600)))
    result, _ = read_sky_map('wcs.fits', arrays)
    np.testing.assert_array_equal(result, image)
    fits.writeto('off.fits', image, fits.Header(CELESTIAL | dict(
        CRPIX1=12.0, CDELT1=-1 / 3600, CDELT2=1 / 3600)))
    with pytest.raises(ValueError, match="not put its pixels on"):
        read_sky_map('off.fits', arrays)


def test_spectral_axis_is_read_last():
    # e.g. a long slit (position, velocity) given as (velocity, position)
    header = dict(
        CTYPE1='VRAD', CUNIT1='km/s', CDELT1=10.0, CTYPE2='LINEAR',
        CDELT2=1.0)
    data = np.arange(8 * 40, dtype=np.float32).reshape(8, 40)
    fits.writeto('data.fits', data, fits.Header(header))
    result, coords = read_data('data.fits', spectral_axis=1)
    assert coords.step == (1.0, 10.0)
    np.testing.assert_array_equal(result, data.T)


@pytest.mark.parametrize('stokes_axis', [1, 3])
def test_axes_of_length_1_of_the_data_are_kept(stokes_axis):
    # Spectra of one region: the spectral axis is the axis 2, so the axis
    # of the region (of length 1) is the data's own, and the axis of
    # Stokes I (of length 1 too, wherever it is) is not
    axes = [dict(CTYPE='LINEAR', CDELT=1.0),
            dict(CTYPE='VRAD', CUNIT='km/s', CDELT=10.0, CRPIX=1.0)]
    axes.insert(stokes_axis - 1, dict(CTYPE='STOKES', CRVAL=1.0))
    # (8 channels along the axis of VRAD; numpy order)
    vrad = [axis['CTYPE'] for axis in axes].index('VRAD')
    shape = [1, 1, 1]
    shape[2 - vrad] = 8
    data = np.arange(8, dtype=np.float32).reshape(shape)
    fits.writeto('data.fits', data, fits.Header(axes_header(*axes)))
    result, coords = read_data('data.fits', spectral_axis=1)
    np.testing.assert_array_equal(result, np.arange(8).reshape(8, 1))
    assert coords.step == (1.0, 10.0)


def test_a_position_axis_in_degrees_is_in_arcsec():
    # e.g. the OFFSET of a PV slice in degrees (pvextractor): the model
    # takes positions in arcsec (an axis without a unit keeps its values)
    for unit, step in (('deg', 0.25 / 3600), ('arcmin', 0.25 / 60),
                       ('arcsec', 0.25)):
        header = dict(
            CTYPE1='OFFSET', CUNIT1=unit, CDELT1=step, CTYPE2='VRAD',
            CUNIT2='km/s', CDELT2=10.0)
        fits.writeto('data.fits', np.ones((8, 40)), fits.Header(header),
                     overwrite=True)
        _, coords = read_data('data.fits')
        assert coords.step[0] == pytest.approx(0.25)


def test_hdu_without_data():
    # e.g. the empty primary HDU of JWST data: asked for, a clear error, not
    # a crash; by default, the first HDU with data is read
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.ones((20, 24), np.float32), name='SCI')
    ]).writeto('data.fits')
    with pytest.raises(ValueError, match="HDU 0 has no data"):
        read_data(('data.fits', 0))
    assert read_data('data.fits')[0].shape == (20, 24)


def test_grid_zero_and_spatial_axes():
    # The spatial axes are measured from the reference pixel, and the
    # spectral axis from its world value there
    coords = Coords((2.0, 3.0, 10.0), (4.0, 5.0, 6.0),
                              (150.0, 2.0, 1500.0), 30.0)
    grid = Grid((8, 10, 12), coords, 2)
    assert grid.zero() == (-8.0, -15.0, 1440.0)
    spatial = grid.spatial()
    assert spatial.size == (8, 10) and spatial.spectral_axis is None
    assert spatial.coords == Coords(
        (2.0, 3.0), (4.0, 5.0), (150.0, 2.0), 30.0)


def test_a_missing_reference_pixel():
    # Of an axis without a world position (e.g. only the size of its
    # pixels), the centre (the convention of the model); of one with one,
    # 0 (that of FITS, as other tools read it)
    header = _centre_missing_crpix(
        fits.Header(dict(CTYPE1='RA---TAN', CRVAL1=150.0, CDELT2=1.0)),
        (20, 24))
    assert 'CRPIX1' not in header
    assert header['CRPIX2'] == 10.5


def test_the_frame_of_rest_of_the_velocities():
    # (SPECSYS, read and written back)
    coords = write(_spectral_header('VRAD', SPECSYS='LSRK'), shape=(6, 20, 24))
    assert coords.frame == 'LSRK'
    write_data('out.fits', np.zeros((6, 20, 24)), coords, 2)
    assert fits.getheader('out.fits')['SPECSYS'] == 'LSRK'


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
    # A given rest becomes that of the axis, in its convention (its
    # velocities, of another rest, are of the given one); without a
    # spectral axis type it keeps its kind
    fits.writeto('data.fits', np.zeros((6, 20, 24), np.float32),
                 fits.Header(_spectral_header('VOPT', RESTWAV=1e-7)))
    coords = read_data('data.fits', rest='1420.405752 MHz')[1]
    np.testing.assert_allclose(
        coords.rest.to_value(u.m), (1420.405752 * u.MHz).to_value(
            u.m, u.spectral()), rtol=1e-12)
    fits.writeto('plain.fits', np.zeros((6, 20, 24), np.float32))
    coords = read_data('plain.fits', rest='6562.8 Angstrom')[1]
    assert coords.rest == 6562.8 * u.Angstrom
    with pytest.raises(ValueError, match="relativistic velocities"):
        fits.writeto('velo.fits', np.zeros((6, 20, 24), np.float32),
                     fits.Header(_spectral_header('VELO')))
        read_data('velo.fits', rest='6562.8 Angstrom')


@pytest.mark.parametrize('rest, convention, ctype, keyword', [
    ('6562.8 Angstrom', None, 'VOPT', 'RESTWAV'),
    ('1420.405752 MHz', None, 'VRAD', 'RESTFRQ'),
    (None, None, 'VELO', None),
    # (conventions without a rest; log velocities as a linear axis)
    (None, 'radio', 'VRAD', None),
    (None, 'optical', 'VOPT', None),
    (None, 'log', 'VLOG', None)])
def test_rest_round_trip(rest, convention, ctype, keyword):
    grid = make_grid(
        (24, 20, 6), (1, 1, 10), spectral_axis=2, rest=rest,
        convention=convention)
    write_data(
        'data.fits', np.zeros((6, 20, 24)), grid.coords, 2)
    header = fits.getheader('data.fits')
    assert header['CTYPE3'] == ctype
    assert keyword is None or keyword in header
    coords = read_data('data.fits')[1]
    assert coords.convention == (
        make_convention(convention, coords.rest) or 'relativistic')
    if rest is None:
        assert coords.rest is None
    else:
        # FITS keeps about 16 significant digits
        assert coords.rest.unit == grid.coords.rest.unit
        np.testing.assert_allclose(
            coords.rest.value, grid.coords.rest.value, rtol=1e-14)


@pytest.mark.parametrize('ctype, keywords, medium, expected', [
    # The wavelengths of optical velocities are in vacuum (FITS), or in the
    # medium of AIRORVAC, or else in that given (one that AIRORVAC
    # contradicts is refused); a rest frequency has none
    ('VOPT', dict(RESTWAV=6.5628e-7), None, 'vacuum'),
    ('VOPT', dict(RESTWAV=6.5628e-7, AIRORVAC='air'), None, 'air'),
    ('VOPT', dict(RESTWAV=6.5628e-7, AIRORVAC='air'), 'air', 'air'),
    ('VOPT', dict(RESTWAV=6.5628e-7, AIRORVAC='air'), 'vacuum', ValueError),
    ('VOPT', dict(RESTWAV=6.5628e-7), 'air', 'air'),
    ('VRAD', dict(RESTFRQ=1.420405752e9, AIRORVAC='air'), None, None)])
def test_medium_of_the_wavelengths(ctype, keywords, medium, expected):
    fits.writeto('data.fits', np.zeros((6, 20, 24), np.float32),
                 fits.Header(_spectral_header(ctype, **keywords)))
    if expected is ValueError:
        with pytest.raises(ValueError, match="not the one the header"):
            read_data('data.fits', medium=medium)
        return
    coords = read_data('data.fits', medium=medium)[1]
    assert coords.medium == expected


def test_medium_round_trip():
    grid = make_grid(
        (24, 20, 6), (1, 1, 10), spectral_axis=2, rest='6562.8 Angstrom',
        medium='air')
    write_data('data.fits', np.zeros((6, 20, 24)), grid.coords, 2)
    assert fits.getheader('data.fits')['AIRORVAC'] == 'air'
    assert read_data('data.fits')[1].medium == 'air'


@pytest.mark.parametrize('rest, medium, message', [
    ('6562.8 Angstrom', 'water', "must be 'vacuum' or 'air'"),
    ('1420.405752 MHz', 'air', "only the wavelengths of a rest wavelength"),
    (None, 'vacuum', "only the wavelengths of a rest wavelength")])
def test_invalid_medium(rest, medium, message):
    with pytest.raises(ValueError, match=message):
        make_grid(
            (24, 20, 6), spectral_axis=2, rest=rest, medium=medium)


@pytest.mark.parametrize('value', [
    '6562.8', '3 km/s', '-1 Angstrom', 'Halpha', [1, 2] * u.m])
def test_invalid_rest(value):
    with pytest.raises(ValueError, match="positive wavelength or frequency"):
        make_rest(value)


def test_nonlinear_velocity_axes_are_rejected():
    # e.g. velocities derived from a linear frequency axis (F2W) are not
    # linear in the pixels
    header = _spectral_header('VOPT-F2W', RESTWAV=6.5628e-7)
    with pytest.raises(ValueError, match="not linear in velocity"):
        write(header, shape=(6, 20, 24))


def test_read_array_of_maps_cubes_and_stale_axes(tmp_path):
    # (a map whose header keeps the keywords of a third axis, as tools
    # write moment maps; audit 2026-10-31 #86, of item 18)
    header = fits.Header(dict(
        CTYPE1='RA---TAN', CTYPE2='DEC--TAN', CRVAL1=10.0, CRVAL2=-20.0,
        CRPIX1=1.0, CRPIX2=1.0, CDELT1=-1 / 3600, CDELT2=1 / 3600,
        CUNIT1='deg', CUNIT2='deg'))
    image = np.arange(12.0).reshape(3, 4)
    fits.writeto(tmp_path / 'map.fits', image, header)
    data, wcs = read_oriented(str(tmp_path / 'map.fits'))
    assert data.shape == (3, 4) and wcs.naxis == 2
    stale = header.copy()
    stale.update(CTYPE3='VRAD', CRVAL3=0.0, CRPIX3=1.0, CDELT3=1e3)
    fits.writeto(tmp_path / 'stale.fits', image, stale)
    data, _ = read_oriented(str(tmp_path / 'stale.fits'))
    assert data.shape == (3, 4)
    fits.writeto(tmp_path / 'cube.fits', np.ones((5, 3, 4)), stale)
    data, wcs = read_oriented(str(tmp_path / 'cube.fits'))
    assert data.shape == (5, 3, 4) and wcs.naxis == 3


# A field on the sky of 24 x 20 pixels of 10 arcsec, east to the left
SKY = dict(
    CTYPE1='RA---TAN', CRVAL1=150.0, CRPIX1=12.5, CDELT1=-10 / 3600,
    CUNIT1='deg', CTYPE2='DEC--TAN', CRVAL2=2.0, CRPIX2=10.5,
    CDELT2=10 / 3600, CUNIT2='deg')


def write_map(path, header, shape=(20, 24)):
    fits.writeto(path, np.zeros(shape, np.float32), fits.Header(header))
    return str(path)


@pytest.mark.parametrize('frame', [
    dict(RADESYS='FK4', EQUINOX=1950.0),
    dict(CTYPE1='GLON-TAN', CRVAL1=120.0, CTYPE2='GLAT-TAN', CRVAL2=30.0)])
def test_a_given_reference_value_is_of_icrs_whatever_the_frame(
        tmp_path, frame):
    # The position of ICRS of a pixel, given as the reference value, puts
    # the reference pixel at that pixel, and stays as given, whatever the
    # frame of the header
    header = SKY | frame
    path = write_map(tmp_path / 'map.fits', header)
    sky = astropy.wcs.WCS(fits.Header(header)).pixel_to_world(6, 4).icrs
    coords = read_data(path, rval=[sky.ra.deg, sky.dec.deg])[1]
    np.testing.assert_allclose(coords.rpix, [6, 4], atol=1e-6)
    np.testing.assert_allclose(
        coords.rval, [sky.ra.deg, sky.dec.deg], rtol=0, atol=1e-10)


def test_rota_is_that_of_the_reference_pixel(tmp_path):
    # At Dec 60, the +y axis of a grid whose north is up at its CRPIX turns
    # away from north 200 pixels east of it: the rotation is that at the
    # reference pixel given, as the position angle on the sky says
    header = SKY | dict(CRVAL2=60.0, CRPIX1=1, CRPIX2=1)
    path = write_map(tmp_path / 'map.fits', header, shape=(20, 240))
    expected = position_angle(fits.Header(header), (200, 0), (200, 1))
    assert not same_angle(expected, 0, tolerance=0.5)
    assert same_angle(
        read_data(path, rpix=[200, 0])[1].rota, expected, tolerance=1e-3)
    assert same_angle(read_data(path)[1].rota, 0)


def test_rota_follows_lonpole(tmp_path):
    # A zenithal projection of LONPOLE 170 turns the grid by 10 degrees
    header = SKY | dict(LONPOLE=170.0)
    path = write_map(tmp_path / 'map.fits', header)
    expected = position_angle(fits.Header(header), (11.5, 9.5), (11.5, 10.5))
    assert same_angle(expected, -10, tolerance=1e-3)
    assert same_angle(read_data(path)[1].rota, expected, tolerance=1e-4)


@pytest.mark.parametrize('option', [
    dict(rpix=[10, None]), dict(rval=[None, 2.0]), dict(rpix=[10, None],
                                                        rval=[None, 2.0])])
def test_the_reference_of_one_celestial_axis_alone_is_refused(
        tmp_path, option):
    path = write_map(tmp_path / 'map.fits', SKY)
    with pytest.raises(ValueError, match="both of them or for neither"):
        read_data(path, **option)


# A cube of 24 x 20 pixels and 30 channels of wavelengths in air
AIR_CUBE = SKY | dict(
    CTYPE3='AWAV', CUNIT3='Angstrom', CRVAL3=6600.0, CDELT3=2.0, CRPIX3=1)


def test_a_medium_given_that_the_header_contradicts_is_refused(tmp_path):
    path = write_map(tmp_path / 'cube.fits', AIR_CUBE, (30, 20, 24))
    rest = '6562.8 Angstrom'
    assert read_data(path, rest=rest)[1].medium == 'air'
    assert read_data(path, rest=rest, medium='air')[1].medium == 'air'
    with pytest.raises(ValueError, match="not the one the header states"):
        read_data(path, rest=rest, medium='vacuum')
    # (an axis of optical velocities does not state it)
    velocities = write_map(tmp_path / 'velocities.fits', SKY | dict(
        CTYPE3='VOPT', CUNIT3='km/s', CRVAL3=0.0, CDELT3=10.0, CRPIX3=1,
        RESTWAV=6.5628e-7), (30, 20, 24))
    assert read_data(velocities, medium='air')[1].medium == 'air'


def test_a_rest_in_the_medium_given_is_the_line_of_the_header(tmp_path):
    # Optical velocities of Halpha (RESTWAV, in vacuum) read with the rest
    # of Halpha in air and the medium air: the same line, so the same
    # velocities (the rest of the air, as given), for the data of a
    # dataset too
    from gbkfit.dataset import Data
    vacuum = 6564.614e-10
    air = vacuum / air_refractive_index(vacuum) * u.m
    path = write_map(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='VOPT', CUNIT3='km/s', CRVAL3=1000.0, CDELT3=50.0, CRPIX3=1,
        RESTWAV=vacuum), (30, 20, 24))
    plain = read_data(path)[1]
    for read in (read_data, Data.from_files):
        coords = read(path, rest=air, medium='air')[1]
        np.testing.assert_allclose(coords.rval, plain.rval, rtol=1e-12)
        np.testing.assert_allclose(coords.step, plain.step, rtol=1e-12)
        assert (coords.medium, coords.rest) == ('air', air)
    # (a medium that AIRORVAC contradicts is refused)
    stated = write_map(tmp_path / 'stated.fits', SKY | dict(
        CTYPE3='VOPT', CUNIT3='km/s', CRVAL3=1000.0, CDELT3=50.0, CRPIX3=1,
        RESTWAV=vacuum, AIRORVAC='vac'), (30, 20, 24))
    with pytest.raises(ValueError, match="not the one the header states"):
        read_data(stated, rest=air, medium='air')


def test_the_first_image_is_read_by_default(tmp_path):
    # A file of an empty primary HDU and an image after it
    path = str(tmp_path / 'data.fits')
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(
        np.zeros((20, 24), np.float32), fits.Header(SKY),
        name='SCI')]).writeto(path)
    assert read_data(path)[1] == read_data((path, 1))[1]
    assert read_oriented(path)[0].shape == (20, 24)


def test_another_orientation_leaves_no_stale_keywords_of_axes():
    # The axes (velocity, Dec, RA) reordered: the name and the error of
    # the first, and an alternate description of the axes, would be those
    # of other axes; they are left out
    header = fits.Header(dict(
        NAXIS=3, NAXIS1=30, NAXIS2=20, NAXIS3=24,
        CTYPE1='VRAD', CUNIT1='km/s', CRVAL1=0.0, CDELT1=10.0, CRPIX1=1,
        CNAME1='Velocity', CRDER1=0.5,
        CTYPE2='DEC--TAN', CUNIT2='deg', CRVAL2=2.0, CDELT2=10 / 3600,
        CRPIX2=10.5,
        CTYPE3='RA---TAN', CUNIT3='deg', CRVAL3=150.0, CDELT3=-10 / 3600,
        CRPIX3=12.5, CTYPE1F='FREQ', CRVAL1F=1.42e9))
    model, orientation = model_header('test', header, (24, 20, 30))
    assert orientation.order == (2, 1, 0)
    assert not {'CNAME1', 'CRDER1', 'CTYPE1F', 'CRVAL1F'} & set(model)
    assert model['CTYPE1'] == 'RA---TAN' and model['CTYPE3'] == 'VRAD'


def test_distortions_are_left_out_with_a_warning(tmp_path, caplog):
    path = write_map(tmp_path / 'map.fits', SKY | dict(
        CTYPE1='RA---TAN-SIP', CTYPE2='DEC--TAN-SIP', A_ORDER=2, B_ORDER=2,
        A_2_0=1e-5, B_0_2=1e-5))
    with caplog.at_level('WARNING'):
        read_data(path)
    assert "distortions" in caplog.text


def test_the_beam_of_aips_is_read_from_the_history(tmp_path):
    # (as AIPS writes it, e.g. in the maps and cubes of THINGS: in
    # degrees, the last of its cleanings)
    header = fits.Header(SKY)
    header.add_history('AIPS   CLEAN BMAJ=  1.0000E-03 BMIN=  1.0000E-03 '
                       'BPA=   0.00')
    header.add_history('AIPS   CLEAN BMAJ=  3.1753E-03 BMIN=  2.6007E-03 '
                       'BPA= -80.43')
    path = str(tmp_path / 'map.fits')
    fits.writeto(path, np.zeros((20, 24), np.float32), header)
    beam = read_arrays(path).beam
    assert beam.bmaj == pytest.approx(11.43108)
    assert beam.bmin == pytest.approx(9.36252)
    assert beam.bpa == pytest.approx(-80.43)


def test_a_cube_of_one_channel_has_the_beam_of_its_table(tmp_path):
    # A cube of CASA of one channel, which reading leaves out, with its
    # beam in its table BEAMS
    path = str(tmp_path / 'cube.fits')
    beams = fits.BinTableHDU.from_columns([
        fits.Column(name='BMAJ', format='E', unit='arcsec', array=[10.0]),
        fits.Column(name='BMIN', format='E', unit='arcsec', array=[5.0]),
        fits.Column(name='BPA', format='E', unit='deg', array=[30.0]),
        fits.Column(name='CHAN', format='J', array=[0]),
        fits.Column(name='POL', format='J', array=[0])], name='BEAMS')
    fits.HDUList([fits.PrimaryHDU(
        np.zeros((1, 20, 24), np.float32), fits.Header(SKY | dict(
            CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.42e9, CDELT3=1e5,
            CRPIX3=1))), beams]).writeto(path)
    arrays = read_arrays(path)
    assert arrays.data.shape == (20, 24)
    assert arrays.beam == Beam(10.0, 5.0, 30.0)


def test_a_cunit_that_is_not_text_is_no_unit(tmp_path):
    # e.g. CUNIT1 = 0, of the PSF images of KMOS3D: astropy leaves it out
    # (with a warning), and the celestial axes are in degrees
    path = write_map(tmp_path / 'psf.fits', SKY | dict(CUNIT1=0, CUNIT2=0))
    with pytest.warns(astropy.wcs.FITSFixedWarning):
        coords = read_data(path)[1]
    assert coords.step == pytest.approx((10.0, 10.0))


def test_the_header_of_a_frequency_axis_names_its_rest_if_not_given(
        tmp_path):
    path = write_map(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.42e9, CDELT3=1e5, CRPIX3=1,
        RESTFRQ=1420405752.0), (3, 20, 24))
    with pytest.raises(ValueError, match="states '1420405752 Hz', RESTFRQ"):
        read_data(path)


def test_an_image_is_written_in_the_unit_given_whatever_its_header(
        tmp_path):
    header = fits.Header(dict(BUNIT='Jy/beam'))
    write_image(str(tmp_path / 'mask.fits'), np.ones((2, 2)), header)
    assert 'BUNIT' not in fits.getheader(tmp_path / 'mask.fits')
    write_image(
        str(tmp_path / 'data.fits'), np.ones((2, 2)), header,
        unit=u.Jy / u.beam)
    assert fits.getheader(tmp_path / 'data.fits')['BUNIT'] == 'Jy beam-1'


def test_a_cube_without_a_table_of_beams_has_no_casa_beam_table(tmp_path):
    path = write_map(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.42e9, CDELT3=1e5, CRPIX3=1),
        (3, 20, 24))
    with pytest.raises(ValueError, match="no table of the beams"):
        casa_beam_table(path)


def test_spectra_of_regions_of_the_model_and_of_data_have_one_layout(
        tmp_path):
    # The regions along x by their index, from 0, as prep writes the
    # spectra of the bins of a cube
    path = write_map(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='VRAD', CUNIT3='km/s', CRVAL3=1000.0, CDELT3=10.0, CRPIX3=1),
        (3, 20, 24))
    of_data = spectra_header(path, fits.getheader(path), 5)
    coords = read_data(path)[1]
    write_spectra(
        str(tmp_path / 'spectra.fits'), np.zeros((3, 5)), coords._replace(
            step=coords.step[2:], rpix=coords.rpix[2:],
            rval=coords.rval[2:], rota=0.0))
    of_model = fits.getheader(tmp_path / 'spectra.fits')
    for key in ('CTYPE1', 'CRPIX1', 'CDELT1', 'CRVAL1'):
        assert of_model[key] == of_data[key]


def test_a_nonstandard_type_of_an_axis_beyond_the_array_is_left_out(
        tmp_path):
    # A map whose header keeps the spectral axis of its cube (as an axis
    # beyond those of the array), of a type FITS does not define
    path = write_map(tmp_path / 'map.fits', SKY | dict(
        WCSAXES=3, CTYPE3='WAVELENGTH', CUNIT3='Angstrom', CRVAL3=6600.0,
        CDELT3=2.0, CRPIX3=1))
    assert read_data(path)[0].shape == (20, 24)


def test_a_rest_for_a_map_of_one_velocity_of_no_rest(tmp_path):
    # A moment map that keeps an axis of one pixel of relativistic
    # velocities (VELO), which have no rest: given one, the map is read,
    # and the rest is its
    path = write_map(tmp_path / 'map.fits', SKY | dict(
        CTYPE3='VELO', CUNIT3='m/s', CRVAL3=0.0, CDELT3=1e4, CRPIX3=1),
        (1, 20, 24))
    data, coords = read_data(path, rest='1420.405752 MHz')
    assert data.shape == (20, 24)
    assert coords.rest.to_value(u.MHz) == pytest.approx(1420.405752)


def test_the_keywords_of_a_rest_wavelength_without_a_medium(tmp_path):
    # Coords of a rest wavelength and no medium (that of the default of
    # make_medium, vacuum) are written, and read back in vacuum
    coords = Coords(
        (10.0, 10.0, 10.0), (11.5, 9.5, 0.0), (150.0, 2.0, 0.0), 0.0,
        make_rest('6564.61 Angstrom'), None, 'optical')
    path = str(tmp_path / 'cube.fits')
    write_data(path, np.zeros((30, 20, 24), np.float32), coords, 2)
    assert read_data(path)[1].medium == 'vacuum'



def test_a_rest_in_air_is_in_vacuum_in_headers(tmp_path):
    # An axis of wavelengths in air, read in velocities of Halpha in air:
    # its rest is in air, as given, and its header has it in vacuum
    # (RESTWAV, as FITS defines it: about 6564.61 Angstrom); written and
    # read back, the same
    def in_air(vacuum):
        return vacuum / air_refractive_index(vacuum)
    path = write_map(tmp_path / 'cube.fits', AIR_CUBE, (30, 20, 24))
    rest = make_rest('6562.80 Angstrom')
    model, _ = model_header(path, fits.getheader(path), (30, 20, 24), rest)
    assert model['AIRORVAC'] == 'air'
    assert model['RESTWAV'] == pytest.approx(6564.61e-10, abs=1e-12)
    assert in_air(model['RESTWAV']) == pytest.approx(
        rest.to_value(u.m), rel=1e-12)
    data, coords = read_data(path, rest=rest)
    assert (coords.medium, coords.rest) == ('air', rest)
    copy = str(tmp_path / 'copy.fits')
    write_data(copy, data, coords, 2)
    assert fits.getheader(copy)['RESTWAV'] == pytest.approx(
        model['RESTWAV'], rel=1e-12)
    back = read_data(copy)[1]
    assert back.medium == 'air'
    assert back.rest.to_value(u.m) == pytest.approx(
        rest.to_value(u.m), rel=1e-12)
    np.testing.assert_allclose(back.rval, coords.rval, rtol=1e-12)


def test_logarithmic_wavelengths_in_air_of_a_rest_in_vacuum(tmp_path):
    # A cube of logarithmic wavelengths in air, of another program, whose
    # first channel is Halpha in air, and whose rest is Halpha (RESTWAV,
    # in vacuum): the first channel is at 0 km/s
    halpha = 6562.797e-10
    path = write_map(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='AWAV-LOG', CUNIT3='m', CRVAL3=halpha,
        CDELT3=halpha * 30 / 299792.458, CRPIX3=1, RESTWAV=6564.61e-10),
        (30, 20, 24))
    coords = read_data(path)[1]
    assert coords.medium == 'air'
    assert coords.rest.to_value(u.m) == pytest.approx(halpha, abs=1e-14)
    assert coords.rval[2] == pytest.approx(0, abs=1e-3)


def test_the_coordinates_of_a_map_have_the_rest_given(tmp_path):
    path = write_map(tmp_path / 'map.fits', SKY)
    rest = make_rest('6562.80 Angstrom')
    assert read_data(path, rest=rest)[1].rest == rest


def test_the_rest_of_the_kind_of_the_axis_comes_first(tmp_path):
    # Optical velocities whose header has a rest wavelength and a rest
    # frequency of another line: read in velocities of that wavelength,
    # they are as they are
    path = write_map(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='VOPT', CUNIT3='km/s', CRVAL3=100.0, CDELT3=10.0, CRPIX3=1,
        RESTWAV=6.5646e-7, RESTFRQ=1.420405752e9), (30, 20, 24))
    coords = read_data(path, rest='6.5646e-7 m')[1]
    assert coords.rval[2] == pytest.approx(100.0)
    assert coords.step[2] == pytest.approx(10.0)


@pytest.mark.parametrize('value, medium', [
    ('AIR', 'air'), (' air ', 'air'), ('Vac', 'vacuum')])
def test_airorvac_in_any_case(tmp_path, value, medium):
    path = write_map(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='VOPT', CUNIT3='km/s', CRVAL3=0.0, CDELT3=10.0, CRPIX3=1,
        RESTWAV=6.5628e-7, AIRORVAC=value), (30, 20, 24))
    assert read_data(path)[1].medium == medium


def test_the_type_of_the_axis_states_its_medium(tmp_path, caplog):
    # Wavelengths in vacuum (WAVE) whose AIRORVAC says air: read in
    # vacuum, as FITS defines WAVE, with a warning
    path = write_map(tmp_path / 'cube.fits', SKY | dict(
        CTYPE3='WAVE', CUNIT3='Angstrom', CRVAL3=6600.0, CDELT3=2.0,
        CRPIX3=1, AIRORVAC='air'), (30, 20, 24))
    with caplog.at_level('WARNING'):
        coords = read_data(path, rest='6564.61 Angstrom')[1]
    assert coords.medium == 'vacuum'
    assert "AIRORVAC says" in caplog.text


def test_an_axis_of_length_1_keeps_its_coupling(tmp_path):
    # A map whose third axis, of one pixel at the pixel 1 (FITS) of a
    # reference pixel 3, is coupled to RA: its sky is that of the full
    # world coordinates at that pixel
    header = SKY | dict(
        CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.42e9, CDELT3=1e5, CRPIX3=3,
        PC1_3=0.5)
    path = write_map(tmp_path / 'map.fits', header, (1, 20, 24))
    full = astropy.wcs.WCS(fits.Header(header))
    model, _ = model_header(path, fits.getheader(path), (1, 20, 24))
    sky = astropy.wcs.WCS(model).celestial.pixel_to_world(5, 4)
    ra, dec, _ = full.wcs_pix2world([[5, 4, 0]], 0)[0]
    assert abs(sky.ra.deg - ra) < 1e-10 and abs(sky.dec.deg - dec) < 1e-10
    # (at the reference pixel of the third axis, the sky is elsewhere)
    assert abs(full.wcs_pix2world([[5, 4, 2]], 0)[0][0] - ra) > 1e-4
