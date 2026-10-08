"""
Tests for the data preparation tasks (gbkfit-cli prep): the prepared
files keep the world coordinates of the data they were cut from.
"""

import astropy.io.fits as fits
import astropy.wcs
import numpy as np
import pytest

from gbkfit.tasks import prep


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
        [0, 1], ['mmap0.fits', 'mmap1.fits'], None, None, [4, 20, 3, 17],
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
