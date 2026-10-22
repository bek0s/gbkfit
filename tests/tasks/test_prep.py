"""
Tests for the data preparation tasks (gbkfit.tasks.prep): the prepared
files keep the world coordinates of the data they were cut from.
"""

import astropy.io.fits as fits
import astropy.wcs
import numpy as np
import pytest
import scipy.ndimage

from gbkfit.tasks import prep
from gbkfit.utils import fitsutils
from gbkfit.utils.parseutils import ConfigError


# A spectral cube with a degenerate Stokes axis, as radio cubes often have:
# RA/Dec rotated by 30 degrees, and a radio velocity axis
SHAPE = (1, 12, 20, 24)
HEADER = {
    'CTYPE1': 'RA---TAN', 'CRVAL1': 150.0, 'CRPIX1': 12.5,
    'CDELT1': -1 / 3600, 'CUNIT1': 'deg',
    'CTYPE2': 'DEC--TAN', 'CRVAL2': 2.0, 'CRPIX2': 10.5,
    'CDELT2': 1 / 3600, 'CUNIT2': 'deg',
    'PC1_1': np.cos(np.radians(30)), 'PC1_2': -np.sin(np.radians(30)),
    'PC2_1': np.sin(np.radians(30)), 'PC2_2': np.cos(np.radians(30)),
    'CTYPE3': 'VRAD', 'CRVAL3': 1500.0, 'CRPIX3': 6.0,
    'CDELT3': 10.0, 'CUNIT3': 'km/s',
    'CTYPE4': 'STOKES', 'CRVAL4': 1.0, 'CRPIX4': 1.0, 'CDELT4': 1.0,
    'BUNIT': 'Jy/beam'}


def write_cube(filename, header=HEADER):
    # A bright blob away from the centre, on a faint background
    z, y, x = np.indices(SHAPE[1:])
    blob = np.exp(-((x - 15) ** 2 + (y - 8) ** 2 + (z - 5) ** 2) / 8)
    data = (blob + 0.01)[None].astype(np.float32)
    fits.writeto(filename, data, fits.Header(header))


def prep_pixel_spectra(filename, **options):
    defaults = dict.fromkeys([
        'roi_spat', 'roi_spec', 'clip_min', 'clip_max', 'ccl_lcount',
        'ccl_pcount', 'ccl_lratio', 'sclip_sigma', 'sclip_iters', 'nanpad'])
    options = defaults | dict(minify=False, dtype='float32') | options
    prep.prep_pixel_spectra(filename, None, None, **options)
    return fits.getdata('prep_cube.fits'), fits.getheader('prep_cube.fits')


def assert_same_world(header_in, header_out, offset):
    """
    The pixel (x, y, z) of the output is the pixel (x, y, z) + offset of
    the input (whose Stokes axis was dropped), at the same world
    coordinates.
    """
    wcs_in = astropy.wcs.WCS(header_in).dropaxis(3)
    wcs_out = astropy.wcs.WCS(header_out)
    pixel = np.array([[0, 0, 0], [3, 2, 1], [7, 5, 4]], float).T
    world_out = wcs_out.pixel_to_world_values(*pixel)
    world_in = wcs_in.pixel_to_world_values(*(pixel + np.c_[offset]))
    np.testing.assert_allclose(world_out, world_in, rtol=0, atol=1e-9)


def test_crop_keeps_the_world_coordinates():
    write_cube('cube.fits')
    data, header = prep_pixel_spectra(
        'cube.fits', roi_spat=[4, 20, 3, 17], roi_spec=[2, 10])
    assert data.shape == (8, 14, 16)
    assert_same_world(HEADER, header, offset=[4, 3, 2])


def test_minify_keeps_the_world_coordinates():
    write_cube('cube.fits')
    data, header = prep_pixel_spectra('cube.fits', clip_min=0.5, minify=True)
    # The pixels above 0.5 are within 2 pixels of the blob centre
    assert data.shape == (5, 5, 5)
    assert_same_world(HEADER, header, offset=[13, 6, 3])


def test_nanpad_pads_and_keeps_the_world_coordinates():
    write_cube('cube.fits')
    data, header = prep_pixel_spectra('cube.fits', nanpad=2)
    assert data.shape == (16, 24, 28)
    assert np.isnan(data[:2]).all() and np.isnan(data[:, :, -2:]).all()
    np.testing.assert_allclose(data[2:-2, 2:-2, 2:-2].max(), 1.01, rtol=1e-6)
    assert_same_world(HEADER, header, offset=[-2, -2, -2])


def test_squeezed_axes_are_dropped_from_the_header():
    write_cube('cube.fits')
    data, header = prep_pixel_spectra('cube.fits')
    assert data.ndim == 3
    assert header['NAXIS'] == 3
    assert astropy.wcs.WCS(header).naxis == 3
    assert 'CTYPE4' not in header
    # The keywords that are not about coordinates are kept
    assert header['BUNIT'] == 'Jy/beam'
    assert_same_world(HEADER, header, offset=[0, 0, 0])


def test_headers_without_coordinates_get_none():
    write_cube('cube.fits', header={'BUNIT': 'Jy/beam'})
    _, header = prep_pixel_spectra('cube.fits', roi_spat=[4, 20, 3, 17])
    assert 'CRPIX1' not in header and 'CTYPE1' not in header
    assert header['BUNIT'] == 'Jy/beam'


@pytest.mark.parametrize('nanpad', [None, 1])
def test_image_crop_keeps_the_world_coordinates(nanpad):
    header_in = {
        k: v for k, v in HEADER.items()
        if k[-1] in '12' or k == 'BUNIT'}
    image = np.ones((20, 24), np.float32)
    fits.writeto('image.fits', image, fits.Header(header_in))
    prep.prep_pixel_brightness(
        'image.fits', None, None, [4, 20, 3, 17], None, None,
        None, None, None, None, None, False, nanpad, 'float32')
    header = fits.getheader('prep_image.fits')
    pad = nanpad or 0
    pixel = np.array([[0, 0], [5, 7]], float).T
    world_out = astropy.wcs.WCS(header).pixel_to_world_values(*pixel)
    world_in = astropy.wcs.WCS(fits.Header(header_in)).pixel_to_world_values(
        *(pixel + np.c_[[4 - pad, 3 - pad]]))
    np.testing.assert_allclose(world_out, world_in, rtol=0, atol=1e-9)


def test_pixel_moments_crop_and_nanpad_keep_the_world_coordinates():
    header_in = fits.Header({
        k: v for k, v in HEADER.items() if k[-1] in '12'})
    for name in ['moment0', 'moment1']:
        fits.writeto(f'{name}.fits', np.ones((20, 24), np.float32), header_in)
    prep.prep_pixel_moments(
        ['moment0.fits', 'moment1.fits'], None, None, [4, 20, 3, 17],
        None, None, None, None, None, None, None, False, 1, 'float32')
    pixel = np.array([[0, 0], [5, 7]], float).T
    world_in = astropy.wcs.WCS(header_in).pixel_to_world_values(
        *(pixel + np.c_[[3, 2]]))
    for name in ['moment0', 'moment1']:
        data = fits.getdata(f'prep_{name}.fits')
        header = fits.getheader(f'prep_{name}.fits')
        assert data.shape == (16, 18)
        assert np.isnan(data[0]).all()
        world_out = astropy.wcs.WCS(header).pixel_to_world_values(*pixel)
        np.testing.assert_allclose(world_out, world_in, rtol=0, atol=1e-9)


def prep_pixel_brightness(filename, **options):
    defaults = dict.fromkeys([
        'roi_spat', 'clip_min', 'clip_max', 'ccl_lcount', 'ccl_pcount',
        'ccl_lratio', 'sclip_sigma', 'sclip_iters', 'nanpad'])
    options = defaults | dict(minify=False, dtype='float32') | options
    file_m = options.pop('file_m', None)
    prep.prep_pixel_brightness(filename, None, file_m, **options)
    return fits.getdata('prep_image.fits')


def test_sigma_clipping_removes_the_outliers():
    data = np.random.default_rng(1).normal(size=(32, 32)).astype(np.float32)
    data[5, 5] = 50
    fits.writeto('image.fits', data)
    prepared = prep_pixel_brightness('image.fits', sclip_sigma=3, sclip_iters=5)
    assert np.isnan(prepared[5, 5])
    assert np.isfinite(prepared).sum() > 1000


def test_the_mask_file_is_applied():
    fits.writeto('image.fits', np.ones((8, 8), np.float32))
    mask = np.ones((8, 8), np.float32)
    mask[:, :4] = 0
    fits.writeto('mask.fits', mask)
    prepared = prep_pixel_brightness('image.fits', file_m='mask.fits')
    assert np.isnan(prepared[:, :4]).all()
    assert np.isfinite(prepared[:, 4:]).all()


def test_connected_components_see_the_clipping():
    # Two islands above a background that clip_min removes: the largest
    # island only is kept
    data = np.full((16, 16), -1, np.float32)
    data[2:8, 2:8] = 5
    data[10:13, 10:13] = 5
    fits.writeto('image.fits', data)
    prepared = prep_pixel_brightness('image.fits', clip_min=0, ccl_lcount=1)
    assert np.isfinite(prepared).sum() == 36
    # No island large enough: everything is masked, without an error
    prepared = prep_pixel_brightness('image.fits', clip_min=0, ccl_pcount=100,
                          ccl_lratio=0.5)
    assert np.isnan(prepared).all()


def test_sigma_clipping_sees_the_kept_pixels_only():
    # A source of values from 9.9 to 10.1 (no outliers) above a background
    # of -1 that clip_min removes: the sigma clipping of the source alone
    # keeps it all (with the background in its statistics, it clipped it)
    data = np.full((30, 30), -1, np.float32)
    data[10:20, 10:20] = np.linspace(9.9, 10.1, 100).reshape(10, 10)
    fits.writeto('image.fits', data)
    prepared = prep_pixel_brightness('image.fits', clip_min=0, sclip_sigma=3)
    assert np.isfinite(prepared).sum() == 100


def test_minify_without_valid_pixels_is_an_error():
    fits.writeto('image.fits', np.full((8, 8), -1, np.float32))
    with pytest.raises(Exception, match="no valid pixels"):
        prep_pixel_brightness('image.fits', clip_min=0, minify=True)

def test_pixel_moments_use_the_sigma_clipping_of_each_map():
    rng = np.random.default_rng(1)
    for name in ['moment0', 'moment1']:
        data = rng.normal(size=(32, 32)).astype(np.float32)
        data[5, 5] = 50
        fits.writeto(f'{name}.fits', data)
    prep.prep_pixel_moments(
        ['moment0.fits', 'moment1.fits'], None, None, None,
        None, None, None, None, None, [3, 3], [5, 5], False, None,
        'float32')
    for name in ['moment0', 'moment1']:
        prepared = fits.getdata(f'prep_{name}.fits')
        assert np.isnan(prepared[5, 5])
        assert np.isfinite(prepared).sum() > 1000


def with_cd(header):
    """The header with CDi_j (CDELTi times PCi_j) instead of PC and CDELT."""
    out = {
        k: v for k, v in header.items() if not k.startswith(('PC', 'CDELT'))}
    for i in range(1, 5):
        for j in range(1, 5):
            pc = header.get(f'PC{i}_{j}', float(i == j))
            if pc:
                out[f'CD{i}_{j}'] = header[f'CDELT{i}'] * pc
    return out


@pytest.mark.parametrize('header', [
    dict(HEADER, CDELT3=-10.0), with_cd(dict(HEADER, CDELT3=-10.0))],
    ids=['pc', 'cd'])
def test_decreasing_velocity_is_reversed(header):
    # The model needs increasing velocities; each channel keeps its
    # velocity (and each pixel its position on the sky)
    write_cube('cube.fits', header)
    data, header_out = prep_pixel_spectra('cube.fits')
    np.testing.assert_array_equal(data, fits.getdata('cube.fits')[0, ::-1])
    nz = SHAPE[1]
    pixel = np.array([[0, 0, 0], [3, 2, 4], [7, 5, nz - 1]], float).T
    flipped = pixel.copy()
    flipped[2] = nz - 1 - pixel[2]
    world_out = astropy.wcs.WCS(header_out).pixel_to_world_values(*pixel)
    world_in = astropy.wcs.WCS(fits.Header(header)).dropaxis(3) \
        .pixel_to_world_values(*flipped)
    np.testing.assert_allclose(world_out, world_in, rtol=0, atol=1e-9)
    # The model reads the prepared cube
    _, coords = fitsutils.read_data('prep_cube.fits')
    assert coords.step[2] == pytest.approx(10)


@pytest.mark.parametrize('roi_spat, offset', [
    (None, (0, 0)), ([4, 20, 3, 17], (4, 3))])
def test_missing_reference_pixel_stays_at_the_centre(roi_spat, offset):
    # The model puts a missing CRPIX at the centre of the axis; the
    # prepared file keeps that pixel at the same world coordinates
    header = {
        k: v for k, v in HEADER.items()
        if k[-1] in '12' and not k.startswith('CRPIX')}
    fits.writeto('image.fits', np.ones((20, 24), np.float32),
                 fits.Header(header))
    prep_pixel_brightness('image.fits', roi_spat=roi_spat)
    _, coords_in = fitsutils.read_data('image.fits')
    _, coords_out = fitsutils.read_data('prep_image.fits')
    np.testing.assert_allclose(
        coords_out.rpix, np.subtract(coords_in.rpix, offset))
    np.testing.assert_allclose(coords_out.rval, coords_in.rval)


C = 299792.458


def test_crops_keep_the_pixels_of_the_ranges():
    # (the world coordinates of the crops are tested above)
    data = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    fits.writeto('cube.fits', data, fits.Header(HEADER))
    prepared, _ = prep_pixel_spectra(
        'cube.fits', roi_spat=[4, 20, 3, 17], roi_spec=[2, 10])
    np.testing.assert_array_equal(prepared, data[0, 2:10, 3:17, 4:20])


@pytest.mark.parametrize('cd', [False, True], ids=['pc', 'cd'])
def test_wavelength_axis_to_optical_velocities(cd):
    # Each channel of a linear wavelength axis gets the optical velocity
    # c (w / rest - 1) of its wavelength, exactly
    header = dict(
        HEADER, CTYPE3='WAVE', CUNIT3='Angstrom', CRVAL3=6500.0,
        CRPIX3=1.0, CDELT3=1.25)
    write_cube('cube.fits', with_cd(header) if cd else header)
    _, header_out = prep_pixel_spectra('cube.fits', velocity_rest='6562.8 Angstrom')
    assert header_out['CTYPE3'] == 'VOPT'
    data, coords = fitsutils.read_data('prep_cube.fits')
    np.testing.assert_allclose(coords.rest.to_value('Angstrom'), 6562.8)
    wavelength = 6500 + 1.25 * np.arange(SHAPE[1])
    velocity = coords.rval[2] + (np.arange(SHAPE[1]) - coords.rpix[2]) \
        * coords.step[2]
    np.testing.assert_allclose(
        velocity, C * (wavelength / 6562.8 - 1), rtol=0, atol=1e-6)


@pytest.mark.parametrize('ctype, medium', [
    ('WAVE', 'vacuum'), ('AWAV', 'air')])
def test_wavelength_axis_keeps_its_medium(ctype, medium):
    # The velocities and the rest of air wavelengths are in air
    header = dict(
        HEADER, CTYPE3=ctype, CUNIT3='Angstrom', CRVAL3=6500.0,
        CRPIX3=1.0, CDELT3=1.25)
    write_cube('cube.fits', header)
    prep_pixel_spectra('cube.fits', velocity_rest='6562.8 Angstrom')
    _, coords = fitsutils.read_data('prep_cube.fits')
    assert coords.medium == medium


def test_frequency_axis_to_radio_velocities():
    # A frequency that increases along the axis gives decreasing radio
    # velocities c (1 - f / rest), so the axis is reversed too
    header = dict(
        HEADER, CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.419e9, CRPIX3=1.0,
        CDELT3=2e4)
    write_cube('cube.fits', header)
    data, _ = prep_pixel_spectra('cube.fits', velocity_rest='1420.405752 MHz')
    np.testing.assert_array_equal(data, fits.getdata('cube.fits')[0, ::-1])
    _, coords = fitsutils.read_data('prep_cube.fits')
    np.testing.assert_allclose(coords.rest.to_value('Hz'), 1.420405752e9)
    frequency = (1.419e9 + 2e4 * np.arange(SHAPE[1]))[::-1]
    velocity = coords.rval[2] + (np.arange(SHAPE[1]) - coords.rpix[2]) \
        * coords.step[2]
    np.testing.assert_allclose(
        velocity, C * (1 - frequency / 1.420405752e9), rtol=0, atol=1e-6)


def test_error_and_mask_are_reversed_with_the_data():
    # The error and mask files (here without world coordinates) are
    # reversed with the data: each channel keeps its error and mask
    header = dict(
        HEADER, CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.419e9, CRPIX3=1.0,
        CDELT3=2e4)
    write_cube('cube.fits', header)
    data = fits.getdata('cube.fits')
    fits.writeto('error.fits', data + 100)
    mask = np.ones_like(data)
    mask[0, 0] = 0
    fits.writeto('mask.fits', mask)
    defaults = dict.fromkeys([
        'roi_spat', 'roi_spec', 'clip_min', 'clip_max', 'ccl_lcount',
        'ccl_pcount', 'ccl_lratio', 'sclip_sigma', 'sclip_iters', 'nanpad'])
    prep.prep_pixel_spectra(
        'cube.fits', 'error.fits', 'mask.fits', **defaults, minify=False,
        dtype='float32', velocity_rest='1420.405752 MHz')
    data_out = fits.getdata('prep_cube.fits')
    error_out = fits.getdata('prep_error.fits')
    mask_out = fits.getdata('prep_mask.fits')
    np.testing.assert_allclose(error_out, data_out + 100, rtol=1e-6)
    np.testing.assert_array_equal(mask_out[-1], 0)
    np.testing.assert_array_equal(mask_out[:-1], 1)

def test_data_in_an_extension_keep_their_world_coordinates():
    # The data and the world coordinates come from the same HDU: here the
    # extension after an empty primary HDU (as in MUSE and JWST cubes)
    header = {k: v for k, v in HEADER.items() if not k.endswith('4')}
    data = np.ones(SHAPE[1:], np.float32)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data, fits.Header(header), name='DATA')]).writeto(
            'cube.fits')
    prep_pixel_spectra('cube.fits', roi_spat=[2, 22, 2, 18])
    _, coords = fitsutils.read_data('prep_cube.fits')
    np.testing.assert_allclose(coords.step, (1, 1, 10))

def test_velocity_axis_gets_the_rest():
    write_cube('cube.fits')
    _, header_out = prep_pixel_spectra('cube.fits', velocity_rest='21.106 cm')
    assert header_out['CTYPE3'] == 'VRAD'
    assert header_out['RESTFRQ'] == pytest.approx(
        C * 1e3 / 0.21106, rel=1e-9)
    _, coords = fitsutils.read_data('prep_cube.fits')
    assert coords.rval[2] == pytest.approx(HEADER['CRVAL3'])



def test_velocity_axis_keeps_its_rest():
    # A velocity axis with a rest refers to it: giving the same one is
    # fine, another one an error (relabelling would move every line)
    write_cube('cube.fits', dict(HEADER, RESTFRQ=1.420405752e9))
    _, header_out = prep_pixel_spectra('cube.fits', velocity_rest='1420.405752 MHz')
    assert header_out['RESTFRQ'] == pytest.approx(1.420405752e9, rel=1e-9)
    with pytest.raises(Exception, match="refer to the rest"):
        prep_pixel_spectra('cube.fits', velocity_rest='115.271 GHz')


def test_converted_axis_keeps_no_old_rest():
    # The rest of the converted axis replaces the old rest keywords (also
    # in their old forms), which would otherwise be read instead
    header = dict(
        HEADER, CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.15e11, CRPIX3=1.0,
        CDELT3=-1e6, RESTWAV=2.6e-3, RESTFREQ=1.42e9)
    write_cube('cube.fits', header)
    _, header_out = prep_pixel_spectra('cube.fits', velocity_rest='115.271 GHz')
    assert 'RESTWAV' not in header_out and 'RESTFREQ' not in header_out
    _, coords = fitsutils.read_data('prep_cube.fits')
    np.testing.assert_allclose(coords.rest.to_value('GHz'), 115.271)


def test_rest_of_the_convention_of_the_axis_first():
    # A header with both rests gives that of the convention of its axis
    write_cube('cube.fits', dict(
        HEADER, CTYPE3='VRAD', RESTWAV=2.6e-3, RESTFRQ=1.420405752e9))
    _, coords = fitsutils.read_data('cube.fits')
    np.testing.assert_allclose(coords.rest.to_value('Hz'), 1.420405752e9)

@pytest.mark.parametrize('velocity, ctype', [
    ('radio', 'VRAD'), ('optical', 'VOPT')])
def test_aips_velocity_axes_need_their_convention(velocity, ctype):
    # VELO-HEL without VELREF is optical to AIPS (and astropy), radio to
    # MIRIAD (e.g. LVHIS): reading it is an error, and prep takes it; a
    # header that states its convention takes none
    header = dict(
        HEADER, CTYPE3='VELO-HEL', CUNIT3='m/s', CRVAL3=1.5e6, CDELT3=1e4,
        RESTFREQ=1.420405752e9)
    write_cube('cube.fits', header)
    for read in (fitsutils.read_data, prep_pixel_spectra):
        with pytest.raises(Exception, match="may be radio or optical"):
            read('cube.fits')
    _, header_out = prep_pixel_spectra('cube.fits', velocity=velocity)
    assert header_out['CTYPE3'] == ctype
    _, coords = fitsutils.read_data('prep_cube.fits')
    assert coords.step[2] == pytest.approx(10)
    write_cube('vrad.fits')
    with pytest.raises(Exception, match="this header states its own"):
        prep_pixel_spectra('vrad.fits', velocity=velocity)


@pytest.mark.parametrize('header, message', [
    (dict(HEADER, CTYPE3='STOKES'), "no spectral axis"),
    (dict(HEADER, CTYPE3='VOPT-F2W', RESTWAV=6.5628e-7), "not linear")])
def test_spectral_axes_that_cannot_be_converted(header, message):
    write_cube('cube.fits', header)
    with pytest.raises(Exception, match=message):
        prep_pixel_spectra('cube.fits', velocity_rest='6562.8 Angstrom')


def test_integer_data_are_prepared():
    # Integer data (here with an invalid error of 0) become floats, which
    # hold the NaN of invalid pixels
    header = {k: v for k, v in HEADER.items() if k[-1] in '12'}
    fits.writeto('image.fits', np.arange(480, dtype=np.int32).reshape(
        20, 24), fits.Header(header))
    error = np.ones((20, 24), np.int16)
    error[0, 0] = 0
    fits.writeto('error.fits', error)
    defaults = dict.fromkeys([
        'roi_spat', 'clip_min', 'clip_max', 'ccl_lcount', 'ccl_pcount',
        'ccl_lratio', 'sclip_sigma', 'sclip_iters', 'nanpad'])
    prep.prep_pixel_brightness(
        'image.fits', 'error.fits', None, **defaults, minify=False,
        dtype='float32')
    data = fits.getdata('prep_image.fits')
    assert np.isnan(data[0, 0]) and data[0, 1] == 1

def test_slit_spectra_crops_positions_and_channels():
    # A slit of 40 positions (FITS x) and 8 channels (FITS y): roi_spat
    # crops the positions and roi_spec the channels, and the prepared
    # slit keeps the world coordinates of the pixels it was cut from
    header = fits.Header(dict(
        CTYPE1='LINEAR', CRVAL1=0.0, CRPIX1=20.5, CDELT1=0.5,
        CTYPE2='VRAD', CUNIT2='km/s', CRVAL2=1500.0, CRPIX2=4.5,
        CDELT2=10.0))
    fits.writeto('slit.fits', np.ones((8, 40), np.float32), header)
    defaults = dict.fromkeys([
        'clip_min', 'clip_max', 'ccl_lcount', 'ccl_pcount', 'ccl_lratio',
        'sclip_sigma', 'sclip_iters', 'nanpad'])
    prep.prep_slit_spectra(
        'slit.fits', None, None, roi_spat=[10, 30], roi_spec=[2, 6],
        **defaults, minify=False, dtype='float32')
    assert fits.getdata('prep_slit.fits').shape == (4, 20)
    _, coords_in = fitsutils.read_data('slit.fits')
    _, coords_out = fitsutils.read_data('prep_slit.fits')
    np.testing.assert_allclose(
        coords_out.rpix, np.subtract(coords_in.rpix, (10, 2)))

def test_binned_maps_to_one_value_per_bin():
    # Maps whose pixels hold the value of their bin (as DAP and GIST
    # write them) become a vector of one value per bin; the bins are
    # numbered from 0 in the order of their numbers
    bins = np.full((8, 10), -1)
    bins[1:4, 1:5] = 7
    bins[4:7, 2:9] = 3
    bins[0, 9] = 12
    velocity = np.where(bins == 7, 120.0, np.where(bins == 3, -40.0, 5.0))
    velocity[bins < 0] = np.nan
    error = np.where(bins == 7, 2.0, 3.0)
    mask = np.ones_like(velocity)
    mask[0, 9] = 0                          # bin 12 is masked
    header = {k: v for k, v in HEADER.items() if k[-1] in '12' or k == 'BUNIT'}
    fits.writeto('bins.fits', bins.astype(np.int32), fits.Header(header))
    fits.writeto('vel.fits', velocity.astype(np.float32))
    fits.writeto('evel.fits', error.astype(np.float32))
    fits.writeto('mvel.fits', mask.astype(np.float32))
    prep.prep_region_moments(
        'bins.fits', ['vel.fits'], ['evel.fits'], ['mvel.fits'], 'float32')
    np.testing.assert_array_equal(
        fits.getdata('prep_bins.fits'),
        np.searchsorted([3, 7, 12], bins) * (bins >= 0) - (bins < 0))
    np.testing.assert_array_equal(fits.getdata('prep_vel.fits'), [-40, 120, np.nan])
    from gbkfit.dataset import dataset_parser
    dataset = dataset_parser.load(dict(
        type='region_moments', regions=dict(type='bins', file='prep_bins.fits'),
        moment1=dict(data='prep_vel.fits', error='prep_evel.fits')))
    assert dataset.regions().nregions() == 3
    np.testing.assert_array_equal(dataset['moment1'].mask(), [1, 1, 0])
    np.testing.assert_array_equal(dataset['moment1'].error()[:2], [3, 2])
    # The world coordinates of the bins are kept
    _, coords = fitsutils.read_data('prep_bins.fits')
    assert coords.rpix == (11.5, 9.5)


def test_binned_maps_must_hold_one_value_per_bin():
    bins = np.zeros((4, 4))
    values = np.ones((4, 4))
    values[0, 0] = 2
    fits.writeto('bins.fits', bins)
    fits.writeto('vel.fits', values)
    with pytest.raises(Exception, match="bin 0 hold different values"):
        prep.prep_region_moments('bins.fits', ['vel.fits'], None, None, 'float32')


@pytest.mark.parametrize('keywords', [
    # FWHM = 0.8 - 2e-5 (w - 7000) arcsec, beta = 2.5 (of MPDAF: the
    # wavelength normalised to [-0.5, 0.5] from 5000 to 9000 Angstrom)
    dict(FSFMODE='MOFFAT1', FSF00BET=2.5, FSF00FWA=0.94, FSF00FWB=-2e-5),
    dict(FSFMODE='MOFFAT2', FSFLB1=5000, FSFLB2=9000, FSF00FNC=2,
         FSF00F00=-0.08, FSF00F01=0.8, FSF00BNC=1, FSF00B00=2.5)])
def test_psf_of_muse(tmp_path, keywords):
    import astropy.units as u
    from gbkfit.instrument import psf_parser
    path = tmp_path / 'cube.fits'
    fits.writeto(path, np.zeros((2, 2, 2)), fits.Header(keywords))
    prep.prep_psf_muse(str(path), str(tmp_path))
    table = str(tmp_path / 'prep_psf.ecsv')
    psf = psf_parser.load(dict(
        type='moffat', alpha=dict(table=table, column='alpha'),
        beta=dict(table=table, column='beta')))
    moffat = psf.at(7000 * u.AA)
    alpha, beta = moffat.dump()['alpha'], moffat.dump()['beta']
    assert beta == pytest.approx(2.5)
    assert 2 * alpha * np.sqrt(2 ** (1 / beta) - 1) == pytest.approx(0.8)


def test_lsf_of_muse(tmp_path):
    import astropy.units as u
    from gbkfit.instrument import lsf_parser
    prep.prep_lsf_muse(str(tmp_path))
    table = str(tmp_path / 'prep_lsf.ecsv')
    lsf = lsf_parser.load(dict(
        type='gauss', sigma=dict(table=table, column='sigma')))
    # (a width in wavelength becomes a velocity with the rest)
    gauss = lsf.at(7000 * u.AA, rest=7000 * u.AA)
    # A FWHM of 2.48 Angstrom at 7000 Angstrom, in km/s
    fwhm = 5.866e-8 * 7000 ** 2 - 9.187e-4 * 7000 + 6.040
    assert gauss.dump()['sigma'] == pytest.approx(
        299792.458 * fwhm / 7000 / np.sqrt(8 * np.log(2)))


def test_beams_of_casa(tmp_path):
    import astropy.units as u
    from gbkfit.instrument import psf_parser
    header = fits.Header(dict(
        CTYPE1='RA---SIN', CDELT1=-1 / 3600, CUNIT1='deg',
        CTYPE2='DEC--SIN', CDELT2=1 / 3600, CUNIT2='deg',
        CTYPE3='FREQ', CRVAL3=1.42e9, CDELT3=1e5, CRPIX3=1, CUNIT3='Hz',
        CASAMBM=True))
    beams = fits.BinTableHDU.from_columns([
        fits.Column(name='BMAJ', format='E', unit='arcsec',
                    array=[10.0, 12.0, 14.0]),
        fits.Column(name='BMIN', format='E', unit='arcsec',
                    array=[8.0, 9.0, 10.0]),
        fits.Column(name='BPA', format='E', unit='deg',
                    array=[30.0, 31.0, 32.0]),
        fits.Column(name='CHAN', format='J', array=[0, 1, 2]),
        fits.Column(name='POL', format='J', array=[0, 0, 0])], name='BEAMS')
    path = tmp_path / 'cube.fits'
    fits.HDUList(
        [fits.PrimaryHDU(np.zeros((3, 2, 2)), header), beams]).writeto(path)
    prep.prep_psf_casa_beams(str(path), str(tmp_path))
    table = str(tmp_path / 'prep_psf.ecsv')
    psf = psf_parser.load(dict(type='gauss_beam', **{
        name: dict(table=table, column=name)
        for name in ('bmaj', 'bmin', 'bpa')}))
    # (the frequency of the second channel)
    beam = psf.at(1.4201e9 * u.Hz)
    assert beam.dump() == pytest.approx(
        dict(type='gauss_beam', bmaj=12, bmin=9, bpa=31))


def test_region_spectra_are_the_sums_of_the_spectra_of_their_bins():
    # Bins 7 (left), 3 (right) and none (the first column): their spectra
    # are the sums of the fluxes of their valid pixels, with the errors
    # added in quadrature, and the velocities of the cube; in Jy/beam, the
    # flux of a pixel (of 1 arcsec^2) is its value over the beam (in
    # arcsec^2), whose sums are in Jy
    write_cube('cube.fits', HEADER | dict(BMAJ=4 / 3600, BMIN=2 / 3600))
    per_beam = 1 / (np.pi / (4 * np.log(2)) * 4 * 2)
    cube = fits.getdata('cube.fits')[0].astype(float) * per_beam
    error = np.full(SHAPE, 0.1)
    error[0, :, 2, 5] = np.nan  # (an invalid pixel, left out)
    fits.writeto('error.fits', error)
    bins = np.full(SHAPE[2:], 7.0)
    bins[:, 12:] = 3
    bins[:, 0] = -1
    header = {k: v for k, v in HEADER.items() if k[-1] in '12' or k == 'BUNIT'}
    fits.writeto('bins.fits', bins, fits.Header(header))
    prep.prep_region_spectra(
        'bins.fits', 'cube.fits', 'error.fits', None, None, 'float64')
    index = fits.getdata('prep_bins.fits')
    assert set(np.unique(index)) == {-1, 0, 1}
    spectra = fits.getdata('prep_cube.fits')
    errors = fits.getdata('prep_error.fits')
    assert spectra.shape == (SHAPE[1], 2)
    left = (bins == 7)
    left[2, 5] = False
    np.testing.assert_allclose(spectra[:, 1], cube[:, left].sum(axis=1))
    np.testing.assert_allclose(
        errors[:, 1], 0.1 * per_beam * np.sqrt(left.sum()))
    np.testing.assert_allclose(spectra[:, 0], cube[:, bins == 3].sum(axis=1))
    assert fits.getheader('prep_cube.fits')['BUNIT'] == 'Jy'

    _, coords = fitsutils.read_data('prep_cube.fits')
    np.testing.assert_allclose(
        coords.rval[1] + coords.step[1] * (np.arange(3) - coords.rpix[1]),
        1500 + 10 * (np.arange(3) - 5))


@pytest.mark.parametrize('unit, factor, flux', [
    # (pixels of 2 arcsec: 4 arcsec^2)
    ('1e-20 erg/s/cm2/Angstrom', 1, '1e-20 erg/s/cm2/Angstrom'),
    ('JY/PIXEL', 1, 'Jy'),
    ('1e-17 erg/s/cm2/Angstrom/arcsec2', 4, '1e-17 erg/s/cm2/Angstrom'),
    ('MJy/sr', 4 / 206264.806 ** 2, 'MJy')])
def test_the_flux_of_a_pixel(unit, factor, flux):
    # (in an extension after an empty primary HDU, as in MUSE and JWST)
    from gbkfit.prep import FitsArrays, pixel_fluxes
    from gbkfit.utils.unitutils import parse_unit
    header = HEADER | dict(CDELT1=-2 / 3600, CDELT2=2 / 3600, BUNIT=unit)
    write_cube('one.fits', header)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(
        fits.getdata('one.fits'), fits.getheader('one.fits'))]).writeto(
        'cube.fits')
    cube = FitsArrays.read('cube.fits')
    scale, flux_unit = pixel_fluxes(
        cube.header, 'cube.fits', cube.data.shape[0])
    assert scale == pytest.approx(factor, rel=1e-9)
    assert (1 * flux_unit).to_value(parse_unit(flux)) == pytest.approx(1)


def test_the_noise_of_a_cube_is_that_of_its_negative_values():
    # A blob of 1 on noise of 0.01 to 0.1 from channel to channel: the
    # error of each channel is its noise (the faint wings of the blob,
    # which sigma clipping keeps, raised the dispersion of all the values
    # by up to 17% on smaller channels), to its scatter (1.5% here)
    from gbkfit.prep import noise_errors
    rng = np.random.default_rng(1)
    z, y, x = np.indices((6, 100, 100))
    sigma = np.linspace(0.01, 0.1, 6)[:, None, None]
    blob = np.exp(-((x - 50) ** 2 + (y - 40) ** 2) / 50)
    data = blob + sigma * rng.standard_normal(z.shape)
    errors = noise_errors(data, np.ones(z.shape))
    np.testing.assert_allclose(
        errors, np.broadcast_to(sigma, errors.shape), rtol=0.08)


def test_the_errors_of_a_prepared_cube_can_be_its_noise():
    write_cube('cube.fits')
    data = fits.getdata('cube.fits')
    noisy = data + 0.05 * np.random.default_rng(2).standard_normal(data.shape)
    fits.writeto('cube.fits', noisy.astype(np.float32), overwrite=True)
    prep_pixel_spectra('cube.fits', noise_error=True)
    np.testing.assert_allclose(
        fits.getdata('prep_cube_noise.fits'), 0.05, rtol=0.3)
    with pytest.raises(ConfigError, match="not both"):
        prep.prep_pixel_spectra(
            'cube.fits', 'errors.fits', None, *[None] * 9, False, None,
            'float32', noise_error=True)


def test_grown_masks_keep_faint_values_next_to_bright_ones():
    # From the seeds (S/N >= 5), the values of S/N >= 2 that touch them;
    # not those apart
    import gbkfit.prep
    snr = np.zeros((7, 9))
    snr[3, 2] = 6
    snr[3, 3:6] = 3
    snr[0, 8] = 3
    mask = gbkfit.prep.grown_mask(snr, 5, 2)
    assert mask[3, 2:6].all() and mask.sum() == 4
    with pytest.raises(ValueError, match="at least the threshold"):
        gbkfit.prep.grown_mask(snr, 1, 2)


def test_the_continuum_of_spectra_is_subtracted():
    # A line on a linear continuum, different in each spectrum: fitted
    # outside the line, the continuum goes, the line stays
    import gbkfit.prep
    z = np.arange(40)[:, None, None]
    rng = np.random.default_rng(3)
    slope, offset = rng.uniform(-1, 1, (2, 1, 3, 4))
    line = 5 * np.exp(-0.5 * ((z - 20) / 2) ** 2)
    cube = line + slope * z + offset
    cube[0, 1, 1] = np.nan
    channels = np.abs(np.arange(40) - 20) <= 12
    less, continuum = gbkfit.prep.subtract_continuum(cube, channels, 1)
    good = np.isfinite(cube)
    np.testing.assert_allclose(less[good], np.broadcast_to(line, cube.shape)[good],
                               atol=1e-6)


@pytest.mark.parametrize('tessellation', ['wvt', 'cvt'])
def test_voronoi_bins_are_of_the_target_sn(tessellation):
    # An exponential disk on noise, in a circle: each pixel of the circle
    # in one bin, of connected pixels, the bins of about the target S/N
    # (to some 10%: Cappellari & Copin 2003)
    import gbkfit.prep
    rng = np.random.default_rng(4)
    y, x = np.indices((60, 60)) - 29.5
    r = np.hypot(x, 1.4 * y)
    signal = 100 * np.exp(-r / 6) + 2 * rng.standard_normal(r.shape)
    signal[r > 27] = np.nan
    noise = np.full(r.shape, 2.0)
    bins = gbkfit.prep.voronoi_bins(signal, noise, 20, tessellation)
    assert np.array_equal(bins >= 0, np.isfinite(signal))
    nbins = bins.max() + 1
    for i in range(nbins):
        assert scipy.ndimage.label(bins == i, np.ones((3, 3)))[1] == 1
    binned = bins >= 0
    sn = (np.bincount(bins[binned], signal[binned])
          / np.sqrt(np.bincount(bins[binned], noise[binned] ** 2)))
    assert np.median(sn) == pytest.approx(20, rel=0.1)
    assert sn.min() > 0.6 * 20
    with pytest.raises(ValueError, match="below the target"):
        gbkfit.prep.voronoi_bins(signal, noise, 1e6)


def test_voronoi_bins_are_the_regions_of_spectra():
    # Bins of the summed cube (of no unit: its spectra are sums), written
    # with its spatial coordinates: the bins of region spectra, of every
    # pixel
    import gbkfit.prep
    from gbkfit.region.regions import RegionsBins
    write_cube('cube.fits', {k: v for k, v in HEADER.items() if k != 'BUNIT'})
    cube = fits.getdata('cube.fits')[0].astype(float)
    signal = cube.sum(axis=0)
    bins = gbkfit.prep.voronoi_bins(signal, np.full(signal.shape, 0.05), 30)
    header = {k: v for k, v in HEADER.items() if k[-1] in '12'}
    fits.writeto('bins.fits', bins, fits.Header(header))
    prep.prep_region_spectra(
        'bins.fits', 'cube.fits', None, None, None, 'float64')
    nbins = bins.max() + 1
    spectra = fits.getdata('prep_cube.fits')
    assert nbins > 1 and spectra.shape == (SHAPE[1], nbins)
    np.testing.assert_allclose(spectra.sum(axis=1), cube.sum(axis=(1, 2)))
    regions = RegionsBins.load(dict(file='prep_bins.fits'))
    assert regions.nregions() == nbins
