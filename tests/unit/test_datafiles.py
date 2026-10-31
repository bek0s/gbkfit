"""
Tests for the reading of data with their errors and mask (datafiles).
"""

import numpy as np
import pytest
from astropy.io import fits

from gbkfit.dataset import describe_fits
from gbkfit.utils import datafiles, fitsutils
from gbkfit.utils.parseutils import ConfigError


# A cube of 5 channels of radio velocities that decrease along the file,
# from 1000 to 960 km/s
CUBE = {
    'CTYPE1': 'RA---TAN', 'CRVAL1': 150.0, 'CRPIX1': 3.0,
    'CDELT1': -1 / 3600, 'CUNIT1': 'deg',
    'CTYPE2': 'DEC--TAN', 'CRVAL2': 2.0, 'CRPIX2': 2.0,
    'CDELT2': 1 / 3600, 'CUNIT2': 'deg',
    'CTYPE3': 'VRAD', 'CRVAL3': 1000.0, 'CRPIX3': 1.0,
    'CDELT3': -10.0, 'CUNIT3': 'km/s', 'BUNIT': 'Jy/beam'}
SHAPE = (5, 3, 4)


def write(filename, data, header=None):
    fits.writeto(
        filename, np.asarray(data, np.float32), fits.Header(header or {}))


@pytest.mark.parametrize('kind, values, unit', [
    ('sigma', 2.0, None),
    ('variance', 4.0, None),
    ('inverse_variance', 0.25, None),
    ('sigma', 2000.0, 'mJy/beam'),
    ('variance', 4e6, 'mJy2 / beam2')])
def test_errors_of_each_kind_become_standard_deviations(
        tmp_path, kind, values, unit):
    # In the unit of the data, whatever theirs
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    write(tmp_path / 'error.fits', np.full(SHAPE, values),
          {} if unit is None else dict(BUNIT=unit))
    arrays = datafiles.read_arrays(
        str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits'),
        error_kind=kind)
    np.testing.assert_allclose(arrays.error, 2.0, rtol=1e-6)


def test_errors_in_the_unit_of_another_kind_are_an_error(tmp_path):
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    write(tmp_path / 'error.fits', np.full(SHAPE, 4.0),
          dict(BUNIT='Jy2 / beam2'))
    with pytest.raises(ConfigError, match="error_kind: variance"):
        datafiles.read_arrays(
            str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits'))


def test_inverse_variances_of_zero_are_values_without_data(tmp_path):
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    inverse = np.full(SHAPE, 4.0)
    inverse[0, 0, 0] = 0
    write(tmp_path / 'error.fits', inverse)
    arrays = datafiles.read_arrays(
        str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits'),
        error_kind='inverse_variance')
    assert arrays.mask.sum() == arrays.mask.size - 1
    assert np.isnan(arrays.data[-1, 0, 0])


def test_errors_follow_their_own_world_coordinates(tmp_path):
    # The data decrease in velocity along their file, the errors (each
    # the velocity of its channel) increase along theirs: both read in
    # increasing velocity, each channel keeps its error; without world
    # coordinates, the errors follow the axes of the file of the data
    velocities = 1000 - 10 * np.arange(SHAPE[0])
    write(tmp_path / 'data.fits',
          np.broadcast_to(velocities[:, None, None], SHAPE), CUBE)
    write(tmp_path / 'error.fits',
          np.broadcast_to(velocities[::-1, None, None], SHAPE),
          CUBE | dict(CRVAL3=960.0, CDELT3=10.0))
    write(tmp_path / 'plain.fits',
          np.broadcast_to(velocities[:, None, None], SHAPE))
    for error in ('error.fits', 'plain.fits'):
        arrays = datafiles.read_arrays(
            str(tmp_path / 'data.fits'), str(tmp_path / error))
        np.testing.assert_array_equal(arrays.data[:, 0, 0], velocities[::-1])
        np.testing.assert_array_equal(arrays.error, arrays.data)


def test_errors_on_other_pixels_are_an_error(tmp_path):
    write(tmp_path / 'data.fits', np.ones(SHAPE), CUBE)
    write(tmp_path / 'error.fits', np.ones(SHAPE), CUBE | dict(CRVAL3=990.0))
    with pytest.raises(ConfigError, match="pixels"):
        datafiles.read_arrays(
            str(tmp_path / 'data.fits'), str(tmp_path / 'error.fits'))


def test_world_axes_beyond_the_array_are_flat(tmp_path):
    # A moment map written with the header of its cube (astropy keeps its
    # spectral axis) reads as a map
    filename = str(tmp_path / 'moment.fits')
    write(filename, np.ones(SHAPE[1:]), CUBE | dict(BMAJ=1e-3, BMIN=1e-3))
    data, coords = fitsutils.read_data(filename)
    assert data.shape == SHAPE[1:] and len(coords.step) == 2
    assert fitsutils.read_array(filename)[0].shape == SHAPE[1:]
    description, = describe_fits(filename)
    assert description.problems == ()
