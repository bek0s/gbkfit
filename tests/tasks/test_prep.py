"""
Tests for the data preparation tasks (gbkfit-cli prep): the prepared
files keep the world coordinates of the data they were cut from.
"""

import subprocess
import sys

import astropy.io.fits as fits
import astropy.wcs
import numpy as np
import pytest

from gbkfit.tasks import prep
from gbkfit.utils import fitsutils


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


def prep_scube(filename, **options):
    defaults = dict.fromkeys([
        'roi_spat', 'roi_spec', 'clip_min', 'clip_max', 'ccl_lcount',
        'ccl_pcount', 'ccl_lratio', 'sclip_sigma', 'sclip_iters', 'nanpad'])
    options = defaults | dict(minify=False, dtype='float32') | options
    prep.prep_scube(filename, None, None, **options)
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
    data, header = prep_scube(
        'cube.fits', roi_spat=[4, 20, 3, 17], roi_spec=[2, 10])
    assert data.shape == (8, 14, 16)
    assert_same_world(HEADER, header, offset=[4, 3, 2])


def test_minify_keeps_the_world_coordinates():
    write_cube('cube.fits')
    data, header = prep_scube('cube.fits', clip_min=0.5, minify=True)
    # The pixels above 0.5 are within 2 pixels of the blob centre
    assert data.shape == (5, 5, 5)
    assert_same_world(HEADER, header, offset=[13, 6, 3])


def test_nanpad_pads_and_keeps_the_world_coordinates():
    write_cube('cube.fits')
    data, header = prep_scube('cube.fits', nanpad=2)
    assert data.shape == (16, 24, 28)
    assert np.isnan(data[:2]).all() and np.isnan(data[:, :, -2:]).all()
    np.testing.assert_allclose(data[2:-2, 2:-2, 2:-2].max(), 1.01, rtol=1e-6)
    assert_same_world(HEADER, header, offset=[-2, -2, -2])


def test_squeezed_axes_are_dropped_from_the_header():
    write_cube('cube.fits')
    data, header = prep_scube('cube.fits')
    assert data.ndim == 3
    assert header['NAXIS'] == 3
    assert astropy.wcs.WCS(header).naxis == 3
    assert 'CTYPE4' not in header
    # The keywords that are not about coordinates are kept
    assert header['BUNIT'] == 'Jy/beam'
    assert_same_world(HEADER, header, offset=[0, 0, 0])


def test_headers_without_coordinates_get_none():
    write_cube('cube.fits', header={'BUNIT': 'Jy/beam'})
    _, header = prep_scube('cube.fits', roi_spat=[4, 20, 3, 17])
    assert 'CRPIX1' not in header and 'CTYPE1' not in header
    assert header['BUNIT'] == 'Jy/beam'


@pytest.mark.parametrize('nanpad', [None, 1])
def test_image_crop_keeps_the_world_coordinates(nanpad):
    header_in = {
        k: v for k, v in HEADER.items()
        if k[-1] in '12' or k == 'BUNIT'}
    image = np.ones((20, 24), np.float32)
    fits.writeto('image.fits', image, fits.Header(header_in))
    prep.prep_image(
        'image.fits', None, None, [4, 20, 3, 17], None, None,
        None, None, None, None, None, False, nanpad, 'float32')
    header = fits.getheader('prep_image.fits')
    pad = nanpad or 0
    pixel = np.array([[0, 0], [5, 7]], float).T
    world_out = astropy.wcs.WCS(header).pixel_to_world_values(*pixel)
    world_in = astropy.wcs.WCS(fits.Header(header_in)).pixel_to_world_values(
        *(pixel + np.c_[[4 - pad, 3 - pad]]))
    np.testing.assert_allclose(world_out, world_in, rtol=0, atol=1e-9)


def test_mmaps_crop_and_nanpad_keep_the_world_coordinates():
    header_in = fits.Header({
        k: v for k, v in HEADER.items() if k[-1] in '12'})
    for name in ['mmap0', 'mmap1']:
        fits.writeto(f'{name}.fits', np.ones((20, 24), np.float32), header_in)
    prep.prep_mmaps(
        ['mmap0.fits', 'mmap1.fits'], None, None, [4, 20, 3, 17],
        None, None, None, None, None, None, None, False, 1, 'float32')
    pixel = np.array([[0, 0], [5, 7]], float).T
    world_in = astropy.wcs.WCS(header_in).pixel_to_world_values(
        *(pixel + np.c_[[3, 2]]))
    for name in ['mmap0', 'mmap1']:
        data = fits.getdata(f'prep_{name}.fits')
        header = fits.getheader(f'prep_{name}.fits')
        assert data.shape == (16, 18)
        assert np.isnan(data[0]).all()
        world_out = astropy.wcs.WCS(header).pixel_to_world_values(*pixel)
        np.testing.assert_allclose(world_out, world_in, rtol=0, atol=1e-9)


def prep_image(filename, **options):
    defaults = dict.fromkeys([
        'roi_spat', 'clip_min', 'clip_max', 'ccl_lcount', 'ccl_pcount',
        'ccl_lratio', 'sclip_sigma', 'sclip_iters', 'nanpad'])
    options = defaults | dict(minify=False, dtype='float32') | options
    file_m = options.pop('file_m', None)
    prep.prep_image(filename, None, file_m, **options)
    return fits.getdata('prep_image.fits')


def test_sigma_clipping_removes_the_outliers():
    data = np.random.default_rng(1).normal(size=(32, 32)).astype(np.float32)
    data[5, 5] = 50
    fits.writeto('image.fits', data)
    prepared = prep_image('image.fits', sclip_sigma=3, sclip_iters=5)
    assert np.isnan(prepared[5, 5])
    assert np.isfinite(prepared).sum() > 1000


def test_the_mask_file_is_applied():
    fits.writeto('image.fits', np.ones((8, 8), np.float32))
    mask = np.ones((8, 8), np.float32)
    mask[:, :4] = 0
    fits.writeto('mask.fits', mask)
    prepared = prep_image('image.fits', file_m='mask.fits')
    assert np.isnan(prepared[:, :4]).all()
    assert np.isfinite(prepared[:, 4:]).all()


def test_connected_components_see_the_clipping():
    # Two islands above a background that clip_min removes: the largest
    # island only is kept
    data = np.full((16, 16), -1, np.float32)
    data[2:8, 2:8] = 5
    data[10:13, 10:13] = 5
    fits.writeto('image.fits', data)
    prepared = prep_image('image.fits', clip_min=0, ccl_lcount=1)
    assert np.isfinite(prepared).sum() == 36
    # No island large enough: everything is masked, without an error
    prepared = prep_image('image.fits', clip_min=0, ccl_pcount=100,
                          ccl_lratio=0.5)
    assert np.isnan(prepared).all()


def test_sigma_clipping_sees_the_kept_pixels_only():
    # A source of values from 9.9 to 10.1 (no outliers) above a background
    # of -1 that clip_min removes: the sigma clipping of the source alone
    # keeps it all (with the background in its statistics, it clipped it)
    data = np.full((30, 30), -1, np.float32)
    data[10:20, 10:20] = np.linspace(9.9, 10.1, 100).reshape(10, 10)
    fits.writeto('image.fits', data)
    prepared = prep_image('image.fits', clip_min=0, sclip_sigma=3)
    assert np.isfinite(prepared).sum() == 100


def test_minify_without_valid_pixels_is_an_error():
    fits.writeto('image.fits', np.full((8, 8), -1, np.float32))
    with pytest.raises(Exception, match="no valid pixels"):
        prep_image('image.fits', clip_min=0, minify=True)

def test_mmaps_use_the_sigma_clipping_of_each_map():
    rng = np.random.default_rng(1)
    for name in ['mmap0', 'mmap1']:
        data = rng.normal(size=(32, 32)).astype(np.float32)
        data[5, 5] = 50
        fits.writeto(f'{name}.fits', data)
    prep.prep_mmaps(
        ['mmap0.fits', 'mmap1.fits'], None, None, None,
        None, None, None, None, None, [3, 3], [5, 5], False, None,
        'float32')
    for name in ['mmap0', 'mmap1']:
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
    data, header_out = prep_scube('cube.fits')
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
    prep_image('image.fits', roi_spat=roi_spat)
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
    prepared, _ = prep_scube(
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
    _, header_out = prep_scube('cube.fits', velocity_rest='6562.8 Angstrom')
    assert header_out['CTYPE3'] == 'VOPT'
    data, coords = fitsutils.read_data('prep_cube.fits')
    np.testing.assert_allclose(coords.rest.to_value('Angstrom'), 6562.8)
    wavelength = 6500 + 1.25 * np.arange(SHAPE[1])
    velocity = coords.rval[2] + (np.arange(SHAPE[1]) - coords.rpix[2]) \
        * coords.step[2]
    np.testing.assert_allclose(
        velocity, C * (wavelength / 6562.8 - 1), rtol=0, atol=1e-6)


def test_frequency_axis_to_radio_velocities():
    # A frequency that increases along the axis gives decreasing radio
    # velocities c (1 - f / rest), so the axis is reversed too
    header = dict(
        HEADER, CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.419e9, CRPIX3=1.0,
        CDELT3=2e4)
    write_cube('cube.fits', header)
    data, _ = prep_scube('cube.fits', velocity_rest='1420.405752 MHz')
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
    prep.prep_scube(
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
    prep_scube('cube.fits', roi_spat=[2, 22, 2, 18])
    _, coords = fitsutils.read_data('prep_cube.fits')
    np.testing.assert_allclose(coords.step, (1, 1, 10))

def test_velocity_axis_gets_the_rest():
    write_cube('cube.fits')
    _, header_out = prep_scube('cube.fits', velocity_rest='21.106 cm')
    assert header_out['CTYPE3'] == 'VRAD'
    assert header_out['RESTFRQ'] == pytest.approx(
        C * 1e3 / 0.21106, rel=1e-9)
    _, coords = fitsutils.read_data('prep_cube.fits')
    assert coords.rval[2] == pytest.approx(HEADER['CRVAL3'])



def test_velocity_axis_keeps_its_rest():
    # A velocity axis with a rest refers to it: giving the same one is
    # fine, another one an error (relabelling would move every line)
    write_cube('cube.fits', dict(HEADER, RESTFRQ=1.420405752e9))
    _, header_out = prep_scube('cube.fits', velocity_rest='1420.405752 MHz')
    assert header_out['RESTFRQ'] == pytest.approx(1.420405752e9, rel=1e-9)
    with pytest.raises(Exception, match="refer to the rest"):
        prep_scube('cube.fits', velocity_rest='115.271 GHz')


def test_converted_axis_keeps_no_old_rest():
    # The rest of the converted axis replaces the old rest keywords (also
    # in their old forms), which would otherwise be read instead
    header = dict(
        HEADER, CTYPE3='FREQ', CUNIT3='Hz', CRVAL3=1.15e11, CRPIX3=1.0,
        CDELT3=-1e6, RESTWAV=2.6e-3, RESTFREQ=1.42e9)
    write_cube('cube.fits', header)
    _, header_out = prep_scube('cube.fits', velocity_rest='115.271 GHz')
    assert 'RESTWAV' not in header_out and 'RESTFREQ' not in header_out
    _, coords = fitsutils.read_data('prep_cube.fits')
    np.testing.assert_allclose(coords.rest.to_value('GHz'), 115.271)


def test_rest_of_the_convention_of_the_axis_first():
    # A header with both rests gives that of the convention of its axis
    write_cube('cube.fits', dict(
        HEADER, CTYPE3='VRAD', RESTWAV=2.6e-3, RESTFRQ=1.420405752e9))
    _, coords = fitsutils.read_data('cube.fits')
    np.testing.assert_allclose(coords.rest.to_value('Hz'), 1.420405752e9)

@pytest.mark.parametrize('header, message', [
    (dict(HEADER, CTYPE3='STOKES'), "no spectral axis"),
    (dict(HEADER, CTYPE3='VOPT-F2W', RESTWAV=6.5628e-7), "not linear")])
def test_spectral_axes_that_cannot_be_converted(header, message):
    write_cube('cube.fits', header)
    with pytest.raises(Exception, match=message):
        prep_scube('cube.fits', velocity_rest='6562.8 Angstrom')


def run_cli(*args):
    """Run gbkfit-cli with the given arguments."""
    return subprocess.run(
        [sys.executable, '-m', 'gbkfit.apps.cli', *args],
        capture_output=True, text=True)


def write_image(filename):
    header = {k: v for k, v in HEADER.items() if k[-1] in '12'}
    fits.writeto(filename, np.ones((20, 24), np.float32), fits.Header(header))


def test_cli_writes_to_the_output_directory():
    write_image('image.fits')
    result = run_cli(
        'prep', 'image', '--data-d', 'image.fits', '--output-dir', 'prepared')
    assert result.returncode == 0
    assert fits.getdata('prepared/prep_image.fits').shape == (20, 24)
    # (its log, also under python -m)
    assert "output will be stored" in result.stderr + result.stdout


@pytest.mark.parametrize('args, message', [
    # One value for each moment order, also when the orders come last
    (['mmaps', '--data-d', 'image.fits', '--nanpad', '1', '0', '1'],
     "invalid length"),
    (['image', '--data-d', 'image.fits', '--ccl-lratio', 'nan'],
     "must be a number"),
    (['bmaps', '--bins', 'image.fits', '--data-d', 'image.fits', '--minify',
      '0'], "unrecognized arguments: --minify")])
def test_cli_rejects_invalid_arguments(args, message):
    write_image('image.fits')
    result = run_cli('prep', *args)
    assert result.returncode != 0 and message in result.stderr

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
    prep.prep_image(
        'image.fits', 'error.fits', None, **defaults, minify=False,
        dtype='float32')
    data = fits.getdata('prep_image.fits')
    assert np.isnan(data[0, 0]) and data[0, 1] == 1

def test_lslit_crops_positions_and_channels():
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
    prep.prep_lslit(
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
    prep.prep_bmaps(
        'bins.fits', ['vel.fits'], ['evel.fits'], ['mvel.fits'], 'float32')
    np.testing.assert_array_equal(
        fits.getdata('prep_bins.fits'),
        np.searchsorted([3, 7, 12], bins) * (bins >= 0) - (bins < 0))
    np.testing.assert_array_equal(fits.getdata('prep_vel.fits'), [-40, 120, np.nan])
    from gbkfit.dataset import dataset_parser
    dataset = dataset_parser.load(dict(
        type='bmaps', regions=dict(type='bins', file='prep_bins.fits'),
        mmap1=dict(data='prep_vel.fits', error='prep_evel.fits')))
    assert dataset.regions().nregions() == 3
    np.testing.assert_array_equal(dataset['mmap1'].mask(), [1, 1, 0])
    np.testing.assert_array_equal(dataset['mmap1'].error()[:2], [3, 2])
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
        prep.prep_bmaps('bins.fits', ['vel.fits'], None, None, 'float32')
