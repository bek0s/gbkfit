"""
Tests for point (unresolved) components: their flux at a point, shared
by the pixels around it, before the PSF, as surface brightness (the flux
of a pixel is its value times its area).
"""

import copy

import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group


FLAT = dict(type='flat')
GRID = dict(type='grid')

# A galaxy of a point that emits in a band (seen by images), and one of a
# point that emits a line (seen by spectral cubes)
BAND_POINT = dict(components=[dict(type='point', emission=dict(type='band'))])
LINE_POINT = dict(components=[dict(type='point', emission=dict(type='lines'))])


def evaluate(driver, galaxy, observable, properties, line_of_sight=FLAT):
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=copy.deepcopy(galaxy),
        line_of_sight=line_of_sight, observable=copy.deepcopy(observable))])
    params = gbkfit.params.ParamSpace(group.pdescs(), properties)
    data = group.model_h(params.evaluate())[0]
    return next(iter(data.values()))['d'].copy()


# The area of a pixel of the test images and cubes
PIXEL_AREA = 0.25


@pytest.mark.parametrize('line_of_sight', [FLAT, GRID], ids=['flat', 'grid'])
def test_point_in_an_image(driver, line_of_sight):
    # At the centre of a pixel (pixel (5, 6) is at (0.25, 1.25)), all its
    # flux is in that pixel; between pixels, it is shared bilinearly
    image = dict(type='pixel_brightness', size=[10, 8], step=[0.5, 0.5])
    centred = evaluate(
        driver, BAND_POINT, image, dict(xpos=0.25, ypos=1.25, flux=3),
        line_of_sight)
    assert centred[6, 5] * PIXEL_AREA == pytest.approx(3)
    assert centred.sum() * PIXEL_AREA == pytest.approx(3)
    shared = evaluate(
        driver, BAND_POINT, image, dict(xpos=0.5, ypos=1.25, flux=4),
        line_of_sight)
    np.testing.assert_allclose(shared[6, 5:7] * PIXEL_AREA, [2, 2], rtol=1e-6)
    # With a psf, the image of a point is the psf
    psf = dict(type='gauss', sigma=1.0)
    smeared = evaluate(
        driver, BAND_POINT, image | dict(psf=psf),
        dict(xpos=0.25, ypos=1.25, flux=1), line_of_sight)
    from gbkfit.instrument import psf_parser
    # (the psf drawn at its own size, around pixel (5, 6))
    kernel = psf_parser.load(dict(psf)).asarray((0.5, 0.5))
    cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
    np.testing.assert_allclose(
        smeared * PIXEL_AREA, kernel[cy - 6:cy + 2, cx - 5:cx + 5],
        rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('scale', [1, 2, 3])
def test_point_flux_does_not_depend_on_oversampling(driver, scale):
    # The model is drawn on a grid of pixels scale times smaller, and
    # averaged back to the pixels of the image: as for the disks, which
    # are surface brightness, the flux of the point does not change
    image = dict(type='pixel_brightness', size=[10, 8], step=[0.5, 0.5],
                 scale=[scale, scale])
    result = evaluate(
        driver, BAND_POINT, image, dict(xpos=0.25, ypos=1.25, flux=3))
    assert result.sum() * PIXEL_AREA == pytest.approx(3, rel=1e-6)


def mean_and_variance(values, positions):
    """The mean and variance of the positions, weighted by the values."""
    mean = np.sum(values * positions) / np.sum(values)
    return mean, np.sum(values * (positions - mean) ** 2) / np.sum(values)


@pytest.mark.parametrize('scale', [1, 2, 3])
def test_point_with_a_psf_does_not_depend_on_oversampling(driver, scale):
    # The psf is drawn on the oversampled grid at its pixel size: the
    # image of a point is the psf (variance sigma^2), at any oversampling
    image = dict(type='pixel_brightness', size=[32, 32], step=[0.5, 0.5],
                 scale=[scale, scale], psf=dict(type='gauss', sigma=2.0))
    result = evaluate(
        driver, BAND_POINT, image, dict(xpos=0.25, ypos=0.25, flux=1))
    position = (np.arange(32) - 15.5) * 0.5
    for profile in (result.sum(axis=0), result.sum(axis=1)):
        mean, variance = mean_and_variance(profile, position)
        assert mean == pytest.approx(0.25, abs=1e-3)
        assert variance == pytest.approx(2.0 ** 2, rel=0.01)


@pytest.mark.parametrize('scale', [1, 2, 3])
def test_point_with_an_lsf_does_not_depend_on_oversampling(driver, scale):
    # The lsf is drawn on the oversampled channels at their width: the
    # spectrum of a point has the variance of the line, the lsf and a
    # channel (10^2 / 12), at any oversampling
    scube = dict(type='pixel_spectra', size=[4, 4, 101], step=[0.5, 0.5, 10],
                 scale=[1, 1, scale], lsf=dict(type='gauss', sigma=30))
    cube = evaluate(driver, LINE_POINT, scube, dict(
        xpos=0.25, ypos=0.25, flux=1, vsys=0, disp=10))
    velocity = (np.arange(101) - 50) * 10
    mean, variance = mean_and_variance(cube.sum(axis=(1, 2)), velocity)
    assert mean == pytest.approx(0, abs=1e-3)
    assert variance == pytest.approx(10 ** 2 + 30 ** 2 + 10 ** 2 / 12, rel=1e-4)


@pytest.mark.parametrize('line_of_sight', [FLAT, GRID], ids=['flat', 'grid'])
def test_point_in_a_cube(driver, line_of_sight):
    # The spectrum of the point is a Gaussian of its flux, velocity and
    # dispersion (the mean of the line over each channel)
    scube = dict(type='pixel_spectra', size=[10, 8, 41], step=[0.5, 0.5, 10])
    cube = evaluate(driver, LINE_POINT, scube, dict(
        xpos=0.25, ypos=1.25, flux=2, vsys=33, disp=25), line_of_sight)
    spectrum = cube[:, 6, 5]
    np.testing.assert_allclose(spectrum.sum() * 10 * PIXEL_AREA, 2, rtol=1e-5)
    velocity = (np.arange(41) - 20) * 10
    mean = np.sum(spectrum * velocity) / spectrum.sum()
    assert mean == pytest.approx(33, abs=0.01)
    # The channels hold the mean of the line over them, which adds the
    # variance of a channel (10^2 / 12) to that of the line
    variance = np.sum(spectrum * (velocity - mean) ** 2) / spectrum.sum()
    assert variance == pytest.approx(25 ** 2 + 10 ** 2 / 12, rel=1e-3)
    assert cube.sum() == pytest.approx(spectrum.sum(), rel=1e-6)


def test_point_with_lines(driver):
    lines = [dict(name='ha', rest='6562.8 Angstrom'),
             dict(name='nii6583', rest='6583.45 Angstrom')]
    galaxy = dict(components=[dict(
        type='point', emission=dict(type='lines', lines=lines))])
    scube = dict(type='pixel_spectra', size=[10, 8, 301], step=[0.5, 0.5, 10],
                 rval=[0, 0, 500], rest='6562.8 Angstrom')
    cube = evaluate(driver, galaxy, scube, dict(
        xpos=0, ypos=0, flux=1, vsys=0, disp=30, nii6583_ratio=0.5))
    # Both lines, the second with half the flux of the first
    np.testing.assert_allclose(cube.sum() * 10 * PIXEL_AREA, 1.5, rtol=1e-4)
    # The second at c (k - 1) + k v, with k the ratio of the rest
    # wavelengths (v = 0): 943 km/s on the axis of the first
    spectrum = cube.sum(axis=(1, 2))
    velocity = 500 + (np.arange(301) - 150) * 10
    second = velocity > 600
    mean = np.sum(spectrum[second] * velocity[second]) / spectrum[second].sum()
    assert mean == pytest.approx(299792.458 * (6583.45 / 6562.8 - 1), abs=0.5)


def test_point_round_trip():
    from gbkfit.model import galaxy_parser
    info = dict(components=[dict(
        type='point', name='agn', emission=dict(
            type='lines', lines=[dict(name='ha', rest='6562.8 Angstrom')]))])
    galaxy = galaxy_parser.load(copy.deepcopy(info))
    # A named component prefixes its parameters with its name
    assert set(galaxy.pdescs()) == {
        'agn_xpos', 'agn_ypos', 'agn_flux', 'agn_vsys', 'agn_disp'}
    dumped = galaxy_parser.dump(galaxy)
    assert galaxy_parser.dump(galaxy_parser.load(dumped)) == dumped


def test_point_plan_takes_a_dtype_or_its_type(driver):
    # The plans of the components take the dtype as np.dtype or its type
    from gbkfit.model import galaxy_parser
    from gbkfit.model.components._component_set import IMAGE_SPECTRAL_AXIS
    from gbkfit.utils import gridutils
    galaxy = galaxy_parser.load(copy.deepcopy(BAND_POINT))
    grid = gridutils.make_grid((6, 4))
    image = driver.mem_alloc_d((4, 6), np.float32)
    driver.mem_fill(image, 0)
    plan = galaxy.plan(
        driver, grid, IMAGE_SPECTRAL_AXIS, None, np.float32, (0,), None,
        None)
    plan.evaluate(dict(xpos=0.5, ypos=0.5, flux=2), dict(image=image), None)
    assert driver.mem_copy_d2h(image).sum() == 2
