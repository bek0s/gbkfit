"""
Tests of the observable of visibilities: the transform of the cube of
the model (that of pixel_spectra on its grid), and analytic sources.
"""

import copy

import gbkfit.params
import numpy as np
import pytest
import scipy.special
from modelutils import observation_group

from gbkfit.dataset import UVCoverage
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError


C = 299792458.0
HI = 1420.405752e6

DISK = dict(components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))])

DISK_PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    pt_a=1, pt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=20)

# The channels of 20 km/s from -160 km/s, of increasing frequency (so
# against the cube, of increasing velocity); baselines of up to 15 km
# (7e4 wavelengths) at a phase centre 3.6 arcsec from the reference pixel
VELOCITIES = 140 - 20 * np.arange(16)
UVW = np.random.default_rng(1).uniform(-1, 1, (40, 3)) * [15e3, 15e3, 5e3]
COVERAGE = UVCoverage(
    UVW, HI * (1 - VELOCITIES * 1e3 / C), (150.001, -30.0005), 'LSRK',
    f'{HI} Hz')
GRID = dict(size=[32, 28], step=1, rval=[150, -30])


def evaluate(driver, galaxy, observable, properties):
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=copy.deepcopy(galaxy),
        line_of_sight=dict(type='flat'),
        observable=copy.deepcopy(observable))])
    params = gbkfit.params.ParamSpace(group.pdescs(), properties)
    data = group.model_h(params.evaluate())[0]
    return {key: value['d'].copy() for key, value in data.items()}


def visibilities(driver, galaxy, properties):
    COVERAGE.dump('coverage.npz', overwrite=True)
    data = evaluate(driver, galaxy, dict(
        type='visibilities', coverage='coverage.npz') | GRID, properties)
    return data['re'] + 1j * data['im']


def unit_vectors(ra, dec):
    return np.stack((
        np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)), -1)


def phases(grid):
    """
    Return exp(-2 pi i b . (s - p)) of the pixels of a grid (TAN) at the
    baselines of each channel, (nchannels, nrows, ny * nx).
    """
    ra, dec = gridutils._to_sky(
        *gridutils.sky_positions(grid), grid.coords.rval)
    directions = unit_vectors(ra, dec).reshape(-1, 3)
    centre = unit_vectors(*np.radians(COVERAGE.phase_centre()))
    east = np.cross([0, 0, 1], centre)
    east /= np.linalg.norm(east)
    axes = np.array([east, np.cross(centre, east), centre])
    return np.array([
        np.exp(-2j * np.pi * (UVW @ axes * (f / C)) @ (directions - centre).T)
        for f in COVERAGE.frequencies()])


def test_visibilities_are_the_transform_of_the_flux_of_the_cube(driver):
    # The cube of pixel_spectra on the grid of the visibilities, from the
    # highest frequency: V = sum of its surface brightness times the area
    # of the pixels, at their directions
    from gbkfit.observation import Visibilities
    grid = Visibilities(COVERAGE, **GRID).grid()
    coords = grid.coords
    cube = evaluate(driver, DISK, dict(
        type='pixel_spectra', size=list(grid.size), step=list(coords.step),
        rpix=list(coords.rpix), rval=list(coords.rval),
        rest=str(coords.rest)), DISK_PROPERTIES)['spectra']
    images = cube[::-1].reshape(len(cube), -1).astype(float)
    expected = np.einsum('ckj,cj->ck', phases(grid.spatial()), images)
    result = visibilities(driver, DISK, DISK_PROPERTIES)
    assert np.linalg.norm(result - expected) / np.linalg.norm(expected) < 1e-5


def test_a_point_source_is_its_flux_at_its_phase(driver):
    # A point of flux 2 at the centre of the pixel (20, 10), x = 4.5 and y
    # = -3.5 arcsec from the reference pixel: V = 2 exp(-2 pi i b . (s -
    # p)), per km/s of the channel of 0 km/s (without a unit of the data),
    # of the part of its line (of dispersion 5 km/s) in it
    point = dict(components=[dict(type='point', emission=dict(
        type='lines', dispersion=dict(type='uniform')))])
    result = visibilities(driver, point, dict(
        xpos=4.5, ypos=-3.5, flux=2, vsys=0, disp=5))
    channel = list(VELOCITIES).index(0)
    grid = gridutils.make_grid((32, 28), 1, rval=(150, -30))
    part = scipy.special.erf(10 / (5 * np.sqrt(2)))
    expected = 2 * part / 20 * phases(grid)[channel][:, 10 * 32 + 20]
    np.testing.assert_allclose(result[channel], expected, rtol=1e-5)


def test_visibilities_have_no_psf():
    from gbkfit.instrument import Instrument, PSFGauss
    from gbkfit.observation import Visibilities
    with pytest.raises(ConfigError, match="no PSF"):
        Visibilities(COVERAGE, (8, 8)).check_instrument(
            Instrument(psf=PSFGauss(sigma=1)))


def test_the_pixels_of_visibilities_are_not_oversampled(driver):
    COVERAGE.dump('coverage.npz', overwrite=True)
    with pytest.raises(ConfigError, match="not oversampled"):
        evaluate(driver, DISK, dict(
            type='visibilities', coverage='coverage.npz', scale=[2, 2, 1])
            | GRID, DISK_PROPERTIES)


def observation_of_data(driver, offset):
    """
    An observation group of visibilities whose data are those of the
    model plus offset in each part, of error 1, and its parameters.
    """
    from gbkfit.dataset import Data, DatasetVisibilities
    from gbkfit.model import galaxy_parser
    from gbkfit.observation import (
        Flat, Observation, ObservationGroup, Visibilities)
    from gbkfit.params import ParamSpace
    model = visibilities(driver, DISK, DISK_PROPERTIES)
    error = np.ones(model.shape)
    dataset = DatasetVisibilities(
        Data(model.real + offset, error=error),
        Data(model.imag + offset, error=error), COVERAGE)
    observation = Observation(
        Visibilities.from_data(dataset, **GRID), Flat(), driver, data=dataset)
    group = ObservationGroup(galaxy_parser.load(DISK), observation)
    return group, ParamSpace(group.pdescs(), DISK_PROPERTIES).evaluate()


def test_the_residuals_are_those_of_both_parts(driver):
    # Data 0.5 above the model in each part, of error 1: chi2 = 0.25 for
    # each part of each visibility
    from gbkfit.objective import Objective
    group, params = observation_of_data(driver, 0.5)
    nvalues = COVERAGE.nchannels() * COVERAGE.nrows()
    assert Objective(group).chi2(params) == pytest.approx(
        0.25 * 2 * nvalues, rel=1e-4)


def test_the_outputs_have_the_dirty_images(driver):
    # Of the data, the model and the data minus the model (on the grid,
    # from the highest frequency), and the dirty beam
    from gbkfit.objective import Objective
    from gbkfit.tasks import _detail
    group, params = observation_of_data(driver, 0.5)
    outputs = _detail.data_outputs(group) | _detail.evaluation_outputs(
        group, Objective(group), params, True, True)
    images = {name: outputs[name].data for name in (
        'data_0_dirty', 'data_0_dirty_beam', 'model_0_dirty',
        'residual_dirty')}
    assert all(x.shape == (16, 28, 32) for x in images.values())
    np.testing.assert_allclose(
        images['residual_dirty'],
        images['data_0_dirty'] - images['model_0_dirty'], atol=1e-6)


def test_mock_visibilities_are_the_model_plus_the_noise(driver):
    # Of rms 0.1 in each part, independent: their errors are 0.1, and
    # (mock - model) / 0.1 has a mean of 0 and a dispersion of 1
    from gbkfit.observation import NoiseGauss
    group, params = observation_of_data(driver, 0)
    [[dataset]] = group.mock(params, NoiseGauss(rms=0.1), seed=1)
    model = visibilities(driver, DISK, DISK_PROPERTIES)
    z = {}
    for key, part in (('re', model.real), ('im', model.imag)):
        np.testing.assert_allclose(dataset[key].error(), 0.1)
        z[key] = ((dataset[key].data() - part) / 0.1).ravel()
        assert abs(z[key].mean()) < 0.1 and abs(z[key].std() - 1) < 0.1
    assert abs(np.corrcoef(z['re'], z['im'])[0, 1]) < 0.1


def test_the_noise_of_visibilities_is_not_correlated_on_the_sky(driver):
    from gbkfit.instrument import PSFGauss
    from gbkfit.observation import NoiseGauss
    group, params = observation_of_data(driver, 0)
    noise = NoiseGauss(rms=0.1, spatial_correlation=PSFGauss(sigma=1))
    with pytest.raises(ConfigError, match="not on the sky"):
        group.mock(params, noise, seed=1)
