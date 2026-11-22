"""
Tests for the checks of the files of a dataset from their headers alone
(see check_files), and for the paths of the same errors as the files are
read.
"""

import numpy as np
import pytest
from astropy.io import fits

from gbkfit.config import ConfigProblem
from gbkfit.dataset import check_files, dataset_parser


SKY = dict(
    CTYPE1='RA---TAN', CUNIT1='deg', CDELT1=-1 / 3600, CRPIX1=3,
    CRVAL1=10.0, CTYPE2='DEC--TAN', CUNIT2='deg', CDELT2=1 / 3600,
    CRPIX2=2, CRVAL2=20.0)


def write(tmp_path, name, shape, header=None):
    path = tmp_path / f'{name}.fits'
    fits.writeto(path, np.ones(shape, np.float32), fits.Header(header or {}))
    return str(path)


def test_the_files_of_a_flux_map_are_on_the_pixels_of_its_data(tmp_path):
    # (each file that is not: its own problem, at its option)
    config = dict(
        type='pixel_brightness', data=write(tmp_path, 'data', (4, 5)),
        error=write(tmp_path, 'error', (5, 6)),
        mask=write(tmp_path, 'mask', (4, 5)),
        footprint=write(tmp_path, 'footprint', (2, 2)))
    assert check_files(config) == (
        ConfigProblem(
            'error', ['error'], 'errors 6×5 · must match the data, 5×4'),
        ConfigProblem(
            'error', ['footprint'],
            'footprint 2×2 · must match the data, 5×4'))


def test_the_data_have_the_axes_of_their_type(tmp_path):
    config = dict(
        type='pixel_brightness', data=write(tmp_path, 'cube', (3, 4, 5)))
    assert check_files(config) == (
        ConfigProblem('error', ['data'], 'data of 3 axes · needs 2'),)


def test_maps_of_moments_are_on_the_pixels_of_the_first(tmp_path):
    # (by their shape, and by their world coordinates; and the files of
    # each on the pixels of its own data)
    config = dict(
        type='pixel_moments',
        moment0=dict(data=write(tmp_path, 'm0', (4, 5), SKY)),
        moment1=dict(data=write(tmp_path, 'm1', (5, 6), SKY)),
        moment2=dict(
            data=write(tmp_path, 'm2', (4, 5), SKY | dict(CRPIX1=3.5)),
            error=write(tmp_path, 'e2', (4, 4))))
    assert check_files(config) == (
        ConfigProblem(
            'error', ['moment1', 'data'],
            'map 6×5 · must match that of moment 0, 5×4'),
        ConfigProblem(
            'error', ['moment2', 'error'],
            'errors 4×4 · must match the data, 5×4'),
        ConfigProblem(
            'error', ['moment2', 'data'],
            'map off the pixels of that of moment 0 (by their world '
            'coordinates)'))


def test_errors_on_other_pixels_of_the_sky_are_found(tmp_path):
    # (by their own world coordinates, as they are read)
    config = dict(
        type='pixel_brightness', data=write(tmp_path, 'data', (4, 5), SKY),
        error=write(tmp_path, 'error', (4, 5), SKY | dict(CRVAL1=10.1)))
    assert check_files(config) == (ConfigProblem(
        'error', ['error'],
        'errors off the pixels of the data (by their world coordinates)'),)


def test_the_footprint_of_a_cube_can_be_of_its_sky(tmp_path):
    cube = SKY | dict(
        CTYPE3='VRAD', CUNIT3='km/s', CDELT3=10.0, CRPIX3=1, CRVAL3=0.0,
        SPECSYS='BARYCENT')
    config = dict(
        type='pixel_spectra', data=write(tmp_path, 'cube', (6, 4, 5), cube),
        footprint=write(tmp_path, 'sky', (4, 5), SKY))
    assert check_files(config) == ()


def test_files_that_cannot_be_read_are_not_problems_of_the_check(tmp_path):
    # (they are of the files themselves: see describe_fits)
    config = dict(
        type='pixel_brightness', data=write(tmp_path, 'data', (4, 5)),
        error=str(tmp_path / 'none.fits'))
    assert check_files(config) == ()
    assert check_files(dict(
        type='pixel_brightness', data=str(tmp_path / 'none.fits'))) == ()


def test_a_type_without_a_check_of_its_files_has_none():
    # (its files are checked as they are read; the configuration itself
    # is checked)
    assert check_files(dict(
        type='profile', file='none.ecsv',
        radius=dict(column='r', unit='kpc'))) == ()
    assert [p.message for p in check_files(dict(
        type='profile', file='none.ecsv'))] == [
        "missing required options: 'radius'"]


def test_the_errors_of_reading_the_files_are_at_their_options(tmp_path):
    # (as the check finds them)
    config = dict(
        type='pixel_moments',
        moment0=dict(data=write(tmp_path, 'm0', (4, 5))),
        moment1=dict(
            data=write(tmp_path, 'm1', (4, 5)),
            mask=write(tmp_path, 'mask', (5, 5))))
    with pytest.raises(ValueError, match="mask 5×5 · must match") as error:
        dataset_parser.load(config)
    assert error.value.config_path == ['moment1', 'mask']


def test_the_shapes_are_checked_before_the_velocities_are_stated(tmp_path):
    # (they do not depend on how the velocities of the data are read: of
    # a cube of MIRIAD without VELREF, the shapes of its errors and its
    # footprint are checked, from the types of the axes; their world
    # coordinates are once the convention of the data is stated; that it
    # is not stated is a problem of the data, not of the check)
    cube = SKY | dict(
        CTYPE3='VELO-HEL', CRVAL3=1e6, CDELT3=1e4, CRPIX3=1, CUNIT3='m/s')
    config = dict(
        type='pixel_spectra', data=write(tmp_path, 'cube', (6, 4, 5), cube),
        error=write(tmp_path, 'error', (6, 4, 4), cube),
        footprint=write(tmp_path, 'sky', (3, 5), SKY))
    assert check_files(config) == (
        ConfigProblem(
            'error', ['error'], 'errors 4×4×6 · must match the data, 5×4×6'),
        ConfigProblem(
            'error', ['footprint'], 'footprint 5×3 · must match the data, 5×4'))
    shifted = cube | dict(CRVAL1=10.1)
    fixed = config | dict(
        error=write(tmp_path, 'shifted', (6, 4, 5), shifted),
        footprint=write(tmp_path, 'sky-of-the-cube', (4, 5), SKY))
    assert check_files(fixed) == ()
    assert check_files(fixed | dict(spectral=dict(convention='radio'))) == (
        ConfigProblem(
            'error', ['error'],
            'errors off the pixels of the data (by their world '
            'coordinates)'),)


def test_the_shape_of_an_array_in_the_model_needs_nothing_stated():
    # (the axes of a cube of MIRIAD without VELREF, in the order of the
    # model: the sky first, whatever the order of the file)
    from gbkfit.io import model_shape
    header = fits.Header(SKY | dict(
        CTYPE3='VELO-HEL', CRVAL3=1e6, CDELT3=1e4, CRPIX3=1, CUNIT3='m/s'))
    assert model_shape('cube', header, (6, 4, 5)) == (6, 4, 5)
    swapped = fits.Header(dict(
        CTYPE1='VELO-HEL', CRVAL1=1e6, CDELT1=1e4, CRPIX1=1, CUNIT1='m/s',
        CTYPE2='RA---TAN', CUNIT2='deg', CDELT2=-1 / 3600, CRPIX2=3,
        CRVAL2=10.0, CTYPE3='DEC--TAN', CUNIT3='deg', CDELT3=1 / 3600,
        CRPIX3=2, CRVAL3=20.0))
    assert model_shape('cube', swapped, (4, 5, 6)) == (6, 4, 5)
    assert model_shape('map', fits.Header(), (1, 4, 5)) == (4, 5)
