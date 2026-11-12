"""
Tests for the moment maps of spectral cubes measured on the host
(gbkfit.moments.moment_maps), as the observables of moments measure
those of their models.
"""

import numpy as np
import pytest

from gbkfit.moments import moment_maps


def lines(velocities, flux, centre, sigma):
    """A cube of Gaussian lines of the given maps of moments 0 to 2."""
    return flux / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -0.5 * ((velocities[:, None, None] - centre) / sigma) ** 2)


@pytest.mark.parametrize('method', ['moments', 'gauss_fit'])
def test_the_moment_maps_of_a_cube_are_its_lines(driver, method):
    # Gaussian lines, on channels of decreasing velocity (as those of many
    # radio cubes): the maps are their flux, centre and dispersion; a
    # spectrum without flux is masked, and one blanked whole has no value
    rng = np.random.default_rng(5)
    flux = rng.uniform(1, 10, (3, 4))
    centre = rng.uniform(-100, 100, (3, 4))
    sigma = rng.uniform(10, 40, (3, 4))
    velocities = np.arange(300, -301, -5.0)
    cube = lines(velocities, flux, centre, sigma)
    cube[:, 0, 0] = 0
    cube[:, 0, 1] = np.nan
    maps = moment_maps(
        cube, velocities, method=method, flux_floor=1e-3, driver=driver)
    assert not maps.mask[0, :2].any() and maps.mask.sum() == 10
    assert np.isnan(maps.data['moment0'][0, 1])
    good = maps.mask
    np.testing.assert_allclose(
        maps.data['moment0'][good], flux[good], rtol=1e-3)
    np.testing.assert_allclose(
        maps.data['moment1'][good], centre[good], atol=0.05)
    np.testing.assert_allclose(
        maps.data['moment2'][good], sigma[good], rtol=1e-3)


@pytest.mark.parametrize('method', ['gauss_fit', 'gauss_hermite'])
def test_the_fits_leave_missing_channels_out(driver, method):
    # (audit 2026-10-31 #20: zeroed, two missing channels at the peak
    # moved the centre by 4.5 km/s and the dispersion by 34)
    velocities = np.arange(-300, 301, 30.0)
    cube = lines(velocities, np.full((1, 2), 5.0), 40.0, 90.0)
    cube[[11, 12], 0, 1] = np.nan
    maps = moment_maps(cube, velocities, (0, 1, 2), method, driver=driver)
    assert maps.mask.all()
    for order, value in enumerate((5, 40, 90)):
        np.testing.assert_allclose(
            maps.data[f'moment{order}'], value, rtol=1e-4)


def test_the_moments_of_a_spectrum_with_a_missing_channel_are_masked(
        driver):
    velocities = np.arange(-300, 301, 30.0)
    cube = lines(velocities, np.full((1, 2), 5.0), 40.0, 90.0)
    cube[0, 0, 1] = np.nan
    maps = moment_maps(cube, velocities, driver=driver)
    assert maps.mask.tolist() == [[True, False]]
    assert np.isnan([maps.data[f'moment{i}'][0, 1] for i in range(3)]).all()


@pytest.mark.parametrize('method', ['gauss_fit', 'gauss_hermite'])
def test_the_fits_of_faint_lines_in_float32_are_those_of_bright_ones(
        driver, method):
    # (audit 2026-10-31 #21: in float32, the Gaussian fit of a line of
    # 1e-20, as in flux-calibrated cubes, returned its start)
    velocities = np.arange(-600, 601, 30.0)
    cube = lines(velocities, np.ones((1, 1)), 40.0, 90.0)
    cube += 0.1 * np.exp(-0.5 * ((velocities[:, None, None] - 200) / 60) ** 2)
    maps = [moment_maps(
        (scale * cube).astype(np.float32), velocities, (0, 1, 2), method,
        driver=driver) for scale in (1, 1e-20)]
    np.testing.assert_allclose(
        maps[1].data['moment0'], 1e-20 * maps[0].data['moment0'], rtol=1e-4)
    for order in (1, 2):
        np.testing.assert_allclose(
            maps[1].data[f'moment{order}'], maps[0].data[f'moment{order}'],
            atol=0.01)


def test_the_skewness_and_kurtosis_of_lines():
    # (of a skew-normal line, and 0 of a Gaussian one; audit 2026-10-31
    # #86)
    import scipy.stats
    velocities = np.arange(-600, 601, 5.0)
    skewed = scipy.stats.skewnorm(4, scale=60).pdf(velocities)
    gaussian = scipy.stats.norm(scale=60).pdf(velocities)
    cube = np.stack([skewed, gaussian], axis=-1)[:, None, :]
    maps = moment_maps(cube, velocities, (0, 1, 2, 3, 4))
    skewness, kurtosis = scipy.stats.skewnorm(4).stats(moments='sk')
    np.testing.assert_allclose(
        maps.skewness()[0], [skewness, 0], atol=1e-3)
    np.testing.assert_allclose(
        maps.kurtosis()[0], [kurtosis, 0], atol=1e-3)
