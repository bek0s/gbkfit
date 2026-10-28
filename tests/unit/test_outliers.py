"""
Tests of the screening of data for outliers (see Dataset.screen_outliers).
"""

import logging

import numpy as np
import pytest
from astropy.io import fits

from gbkfit.dataset import (
    Data, DatasetPixelBrightness, DatasetPixelMoments, DatasetPixelSpectra,
    DatasetRegionMoments, DatasetRegionSpectra)
from gbkfit.region import (
    ApertureCircle, AperturePolygon, ApertureRectangle, RegionsApertures,
    RegionsBins)
from gbkfit.utils.parseutils import ConfigError


def test_a_spike_in_a_velocity_map_is_masked_not_its_gradient():
    # A rotating velocity map with noise of 2 km/s, a failed fit of 160
    # km/s where its neighbours have about 100, and a masked pixel
    j, i = np.indices((20, 30))
    velocity = 100 + 8 * (i - 15) + np.random.default_rng(1).normal(
        0, 2, (20, 30))
    velocity[10, 15] = 160
    velocity[3, 4] = np.nan
    maps = DatasetPixelMoments({1: Data(velocity)})
    screened = maps.screen_outliers([3, 3], 5)['moment1'].data()
    masked = np.isnan(screened) & ~np.isnan(velocity)
    assert np.argwhere(masked).tolist() == [[10, 15]]
    assert np.isnan(screened[3, 4])


def test_a_hot_pixel_is_masked_not_a_clump_seen_through_the_psf():
    j, i = np.indices((25, 25))
    image = 10 * np.exp(-((i - 8) ** 2 + (j - 12) ** 2) / (2 * 1.5 ** 2)) \
        + np.random.default_rng(2).normal(0, 0.01, (25, 25))
    image[18, 18] = 5
    screened = DatasetPixelBrightness(Data(image)).screen_outliers(
        [3, 3], 5)['brightness'].data()
    assert np.argwhere(np.isnan(screened)).tolist() == [[18, 18]]


def test_a_spike_in_a_cube_is_masked():
    cube = np.random.default_rng(3).normal(0, 1, (12, 10, 10))
    cube[6, 5, 5] = 40
    screened = DatasetPixelSpectra(Data(cube)).screen_outliers(
        [3, 3, 3], 6)['spectra'].data()
    assert np.argwhere(np.isnan(screened)).tolist() == [[6, 5, 5]]


@pytest.mark.parametrize('size, nsigma, message', [
    ([2, 3], 5, 'an odd positive integer for each of x and y'),
    ([3, 3, 3], 5, 'for each of x and y'),
    ([3, 3], -1, 'nsigma of the outliers is a positive number'),
])
def test_screening_options_refused(size, nsigma, message):
    image = DatasetPixelBrightness(Data(np.ones((5, 5))))
    with pytest.raises(ConfigError, match=message):
        image.screen_outliers(size, nsigma)


def test_a_bad_bin_is_masked_by_its_nearest_bins():
    # Bins of single pixels in a row (centres 1" apart): a velocity that
    # rises along the row, with one bin far off it
    bins = RegionsBins(np.arange(12).reshape(1, 12))
    velocity = 10 * np.arange(12) + np.random.default_rng(4).normal(0, 1, 12)
    velocity[6] = 300
    maps = DatasetRegionMoments({1: Data(velocity)}, bins)
    screened = maps.screen_outliers([4], 5)['moment1'].data()
    assert np.flatnonzero(np.isnan(screened)).tolist() == [6]
    spectra = np.random.default_rng(5).normal(0, 1, (9, 12))
    spectra[4, 3] = 50
    screened = DatasetRegionSpectra(Data(spectra), bins).screen_outliers(
        [2, 3], 6)['spectra'].data()
    assert np.argwhere(np.isnan(screened)).tolist() == [[4, 3]]


def test_apertures_have_centres():
    regions = RegionsApertures([
        ApertureCircle(1, 2, 1), ApertureRectangle(-3, 4, 2, 1, 30),
        AperturePolygon([[0, 0], [4, 0], [4, 2], [0, 2]])])
    np.testing.assert_allclose(regions.centres(), [[1, 2], [-3, 4], [2, 1]])


def test_the_data_of_an_observation_are_screened_as_they_are_read(
        tmp_path, caplog):
    from gbkfit.observation import observation_parser
    image = np.random.default_rng(6).normal(0, 1, (12, 16))
    image[5, 7] = 30
    fits.writeto(tmp_path / 'image.fits', image)
    data = dict(
        type='pixel_brightness', data=str(tmp_path / 'image.fits'),
        outliers=dict(size=[3, 3], nsigma=6))
    with caplog.at_level(logging.INFO):
        observation = observation_parser.load(dict(
            type='sky',
            line_of_sight=dict(type='flat'), data=data,
            observable=dict(type='pixel_brightness')))
    screened = observation.data()['brightness'].data()
    assert np.argwhere(np.isnan(screened)).tolist() == [[5, 7]]
    assert "outliers masked in the data: {'brightness': 1}" in caplog.text
