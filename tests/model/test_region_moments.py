"""
Tests for the moments of spectra in regions of the sky (bmaps): the
moments of the sums of a spectral cube in regions (bins or apertures).
"""

import copy

import astropy.units as u
import gbkfit.dataset
import gbkfit.params
import numpy as np
import pytest
from astropy.table import Table
from modelutils import group_plan

from gbkfit.dataset import Data
from gbkfit.dataset import DatasetRegionMoments
from gbkfit.region import RegionsBins
from gbkfit.utils import fitsutils, gridutils
from gbkfit.utils.gridutils import SpectralAxis
from gbkfit.utils.parseutils import ConfigError


GALAXY = dict(components=[dict(
    type='disk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        type='traits', vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))])

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    pt_a=1, pt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=20)

FLAT = dict(type='flat')

INSTRUMENT = dict(
    psf=dict(type='gauss', sigma=1.5), lsf=dict(type='gauss', sigma=15))

SPECTRAL = dict(size=101, step=5, convention='relativistic')


def evaluate(driver, observable):
    """The model data of an observable of the tests, by data item."""
    plan = group_plan([
        dict(driver=dict(type=driver.type()), galaxy=GALAXY,
             line_of_sight=FLAT, observable=copy.deepcopy(observable))])
    params = gbkfit.params.ParamSpace(plan.group().pdescs(), PROPERTIES)
    data = plan.model_h(params.evaluate())[0]
    return {key: value['d'].copy() for key, value in data.items()}


def write_bins(index):
    """A file of bins on the grid of the tests' cubes (32 x 41 pixels)."""
    fitsutils.write_data('bins.fits', index, gridutils.Coords(
        (1, 1), (15.5, 20), (0, 0), 0))
    return dict(type='bins', file='bins.fits')


def test_region_moments_of_single_pixel_bins_are_pixel_moments(driver):
    # With a bin for each pixel, the moments of the bins are the moment
    # maps
    regions = write_bins(np.arange(32 * 41).reshape(41, 32))
    mmaps = evaluate(driver, dict(
        type='pixel_moments', size=[32, 41], spectral=SPECTRAL, **INSTRUMENT))
    bmaps = evaluate(driver, dict(
        type='region_moments', regions=regions, spectral=SPECTRAL,
        **INSTRUMENT))
    for key in ('moment0', 'moment1', 'moment2'):
        assert bmaps[key].shape == (32 * 41,)
        np.testing.assert_allclose(
            bmaps[key], mmaps[key].ravel(), rtol=1e-5, atol=1e-4)


def test_region_moments_are_the_moments_of_the_summed_spectra(driver):
    # The moments of the sum of the spectra of a bin (not the mean of the
    # moments of its pixels)
    index = np.full((41, 32), -1)
    index[10:20, 5:15] = 0
    index[20:30, 10:25] = 1
    regions = write_bins(index)
    scube = evaluate(driver, dict(
        type='pixel_spectra', size=[32, 41], step=[1, 1],
        spectral=SPECTRAL, **INSTRUMENT))['spectra']
    bmaps = evaluate(driver, dict(
        type='region_moments', regions=regions, spectral=SPECTRAL,
        **INSTRUMENT))
    velocity = (np.arange(101) - 50) * 5
    for i in range(2):
        spectrum = scube[:, index == i].sum(axis=1)
        flux = spectrum.sum()
        mean = np.sum(spectrum * velocity) / flux
        dispersion = np.sqrt(np.sum(spectrum * (velocity - mean) ** 2) / flux)
        np.testing.assert_allclose(bmaps['moment0'][i], flux * 5, rtol=1e-4)
        np.testing.assert_allclose(bmaps['moment1'][i], mean, atol=1e-2)
        np.testing.assert_allclose(bmaps['moment2'][i], dispersion, rtol=1e-4)


def test_region_moments_of_apertures(driver):
    # The moments of the integrated spectrum of the field
    scube = evaluate(driver, dict(
        type='pixel_spectra', size=[32, 41], step=[1, 1],
        spectral=SPECTRAL, **INSTRUMENT))['spectra']
    bmaps = evaluate(driver, dict(
        type='region_moments', size=[32, 41], spectral=SPECTRAL,
        **INSTRUMENT,
        regions=dict(type='apertures', apertures=[dict(type='field')])))
    np.testing.assert_allclose(
        bmaps['moment0'], [scube.sum() * 5], rtol=1e-4)


def test_region_moments_output_is_a_map_of_the_bins():
    from gbkfit.observation import RegionMoments
    index = np.array([[0, 0, -1], [1, 2, 2]])
    bmaps = RegionMoments(
        RegionsBins(index, step=2, rota=10),
        spectral=SpectralAxis(convention='relativistic'))
    output = bmaps.output(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(output.data, [[1, 1, np.nan], [2, 3, 3]])
    assert output.coords == bmaps.regions().grid().coords


def test_region_moments_observation_from_data(tmp_path):
    # The regions come from the data, and the spectral axis covers the
    # velocities of moment1
    from gbkfit.observation import observation_parser
    index = np.array([[0, 0, -1], [1, 2, 2]])
    dataset = DatasetRegionMoments(
        {0: Data(np.ones(3)), 1: Data(np.array([1400.0, 1500, 1600]))},
        RegionsBins(index))
    data_info = dataset.dump(prefix=str(tmp_path / ''))
    data_info.pop('type')
    loaded = gbkfit.dataset.dataset_parser.load(
        dict(type='region_moments') | copy.deepcopy(data_info))
    assert loaded.regions() == dataset.regions()
    np.testing.assert_array_equal(loaded['moment1'].data(), [1400, 1500, 1600])
    info = dict(
        type='sky',
        data=dict(type='region_moments') | data_info, line_of_sight=FLAT,
        observable=dict(
            type='region_moments',
            spectral=dict(step=2, convention='relativistic')),
        likelihood=dict(type='gaussian', missing_residual=3))
    observation = observation_parser.load(copy.deepcopy(info))
    observable = observation.observable()
    assert observable.size() == (3, 2)
    assert observable.spectral().rval() == 1500
    # A dump of the observation does not repeat what the data give
    dumped = observation_parser.dump(
        observation, prefix=str(tmp_path / 'dump_'))
    assert 'regions' not in dumped['observable']
    assert 'size' not in dumped['observable']
    with pytest.raises(Exception, match="give its options \\['size'\\]"):
        observation_parser.load(info | dict(observable=dict(
            type='region_moments', size=[3, 2],
            spectral=dict(convention='relativistic'))))



def test_region_moments_observation_dumps_its_bins_with_the_prefix(tmp_path):
    # Without data, the bins are options of the observable: their file is
    # named with the prefix of the dump, which can be dumped again
    from gbkfit.observation import (
        Flat, ObservationSky, RegionMoments, observation_parser)
    observation = ObservationSky(RegionMoments(
        RegionsBins(np.arange(16).reshape(4, 4) % 3),
        spectral=SpectralAxis(size=11, convention='relativistic')), Flat())
    prefix = str(tmp_path / 'out_')
    for _ in range(2):
        dumped = observation_parser.dump(
            observation, prefix=prefix, overwrite=True)
    assert dumped['observable']['regions']['file'] == f'{prefix}bins.fits'

def test_region_moments_objective_residual(driver):
    # Data equal to the model plus 1, with errors of 2: every residual
    # (model - data) / error is -0.5
    from gbkfit.model import galaxy_parser
    from gbkfit.objective import Objective
    from gbkfit.instrument import instrument_parser
    from gbkfit.observation import (
        Flat, LikelihoodGaussian, RegionMoments, ObservationSky,
        ObservationGroup)
    index = np.full((41, 32), -1)
    index[10:20, 5:15] = 0
    index[20:30, 10:25] = 1
    regions = RegionsBins(index)
    observable = RegionMoments(
        regions, SpectralAxis(**SPECTRAL), orders=[1, 2])
    instrument = instrument_parser.load(copy.deepcopy(INSTRUMENT))
    group = ObservationGroup(
        [galaxy_parser.load(copy.deepcopy(GALAXY))],
        [ObservationSky(observable, Flat(), instrument=instrument)])
    params = gbkfit.params.ParamSpace(group.pdescs(), PROPERTIES)
    model = {
        key: value['d'].copy() for key, value in
        group.plan(driver).model_h(params.evaluate())[0].items()}
    dataset = DatasetRegionMoments({
        int(key.removeprefix('moment')):
            Data(value + 1, error=np.full_like(value, 2))
        for key, value in model.items()}, regions)
    group = ObservationGroup(
        [galaxy_parser.load(copy.deepcopy(GALAXY))],
        [ObservationSky(
            observable, Flat(), instrument=instrument,
            data=dataset, likelihood=LikelihoodGaussian(missing_residual=3))])
    residual = Objective(group.plan(driver)).residual_nddata_h(
        params.evaluate(), False)
    for key in ('moment1', 'moment2'):
        np.testing.assert_allclose(residual[0][key], -0.5, rtol=1e-4)


def test_region_moments_of_the_columns_of_a_table(tmp_path):
    # The V, sigma, h3 and h4 of the bins of pPXF, a row for each bin: the
    # velocities in km/s (converted from m/s), h3 and h4 without a unit
    Table(dict(
        V=[1.4e6, 1.5e6, 1.6e6] * u.m / u.s,
        V_ERR=[5e3, 6e3, 7e3] * u.m / u.s,
        SIGMA=[100.0, 120, 140] * u.km / u.s,
        H3=[0.01, -0.02, 0.03], H4=[0.0, 0.05, -0.01],
    )).write(tmp_path / 'kinematics.fits')
    fitsutils.write_data(
        str(tmp_path / 'bins.fits'), np.array([[0, 0, -1], [1, 2, 2]]),
        gridutils.Coords((1, 1), (1, 0.5), (0, 0), 0))
    dataset = gbkfit.dataset.dataset_parser.load(dict(
        type='region_moments',
        regions=dict(type='bins', file=str(tmp_path / 'bins.fits')),
        file=str(tmp_path / 'kinematics.fits'),
        moment1=dict(column='V', error='V_ERR'), moment2=dict(column='SIGMA'),
        moment3=dict(column='H3', error=0.01),
        moment4=dict(column='H4', error=0.01)))
    assert dataset.regions().nregions() == 3
    np.testing.assert_allclose(dataset['moment1'].data(), [1400, 1500, 1600])
    np.testing.assert_allclose(dataset['moment1'].error(), [5, 6, 7])
    np.testing.assert_allclose(dataset['moment2'].data(), [100, 120, 140])
    np.testing.assert_allclose(dataset['moment4'].data(), [0, 0.05, -0.01])
    assert dataset['moment3'].unit() is None


@pytest.mark.parametrize('method, unit, message', [
    ('gauss_hermite', 'km3/s3', "is h3, dimensionless"),
    ('moments', '%', "is a central moment"),
])
def test_the_moments_above_2_are_those_of_the_method(method, unit, message):
    # h3 of a Gauss-Hermite series has no unit, and the central moment of
    # order 3 is in (km/s)^3
    from gbkfit.observation import RegionMoments
    regions = RegionsBins(np.array([[0, 1]]))
    observable = RegionMoments(
        regions, orders=[3], method=method,
        spectral=SpectralAxis(convention='relativistic'))
    observable.require_matching(
        DatasetRegionMoments({3: Data(np.zeros(2))}, regions))
    with pytest.raises(ConfigError, match=message):
        observable.require_matching(
            DatasetRegionMoments({3: Data(np.zeros(2), unit=unit)}, regions))
