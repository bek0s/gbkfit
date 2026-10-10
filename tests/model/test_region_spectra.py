"""
Tests for spectra in regions of the sky (aspec): the sums of a spectral
cube in regions (apertures or bins), which must match those of an scube.
"""

import copy

import gbkfit.dataset
import gbkfit.region
import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group

from gbkfit.dataset import Data
from gbkfit.dataset import DatasetRegionSpectra
from gbkfit.region import RegionsApertures, RegionsBins
from gbkfit.utils import fitsutils, gridutils


GMODEL = dict(type='kinematics_2d', components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    bptraits=dict(type='exponential'),
    vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))])

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=20)

INSTRUMENT = dict(
    psf=dict(type='gauss', sigma=1.5), lsf=dict(type='gauss', sigma=15))

SCUBE = dict(type='pixel_spectra', size=[32, 41, 51], step=[1, 1, 10], **INSTRUMENT)

APERTURES = [
    dict(type='circle', x=2, y=-3, radius=2.5),
    dict(type='rectangle', x=-1, y=1, length=12, width=1, posa=50),
    dict(type='field')]


def evaluate(driver, dmodel):
    model_group = observation_group([
        dict(driver=dict(type=driver.type()), dmodel=copy.deepcopy(dmodel),
             gmodel=GMODEL)])
    params = gbkfit.params.EvaluationParams(model_group.pdescs(), PROPERTIES)
    data = model_group.model_h(params.evaluate())[0]
    return next(iter(data.values()))['d'].copy()


def aspec_of_apertures(apertures):
    return dict(
        type='region_spectra', regions=dict(type='apertures', apertures=apertures),
        size=[32, 41], step=[1, 1], spec_size=51, spec_step=10,
        **INSTRUMENT)


def observable_of(dmodel):
    """The observable of a dmodel of the tests, without the instrument."""
    from gbkfit.observation import observable_parser
    return observable_parser.load(
        {k: v for k, v in copy.deepcopy(dmodel).items()
         if k not in INSTRUMENT})


def test_region_spectra_of_apertures_are_the_sums_of_pixel_spectra(driver):
    # The cube of region_spectra is that of pixel_spectra, so its spectra
    # are the sums of its pixels weighted by the overlaps of the apertures
    # (and the area of a pixel, 1 here)
    scube = evaluate(driver, SCUBE)
    aspec = evaluate(driver, aspec_of_apertures(APERTURES))
    assert aspec.shape == (51, 3)
    grid = gridutils.make_grid((32, 41))
    weights = gbkfit.region.regions_parser.load(
        dict(type='apertures', apertures=APERTURES)).weights(grid)
    expected = (weights @ scube.reshape(51, -1).T).T
    np.testing.assert_allclose(aspec, expected, rtol=1e-5, atol=1e-6)
    # The field is the whole cube: the integrated spectrum
    np.testing.assert_allclose(
        aspec[:, 2], scube.sum(axis=(1, 2)), rtol=1e-5)


def test_region_spectra_of_bins_are_the_sums_of_pixel_spectra(driver):
    # Bins on the grid of pixel_spectra: the cube of region_spectra is on
    # their grid
    index = np.full((41, 32), -1)
    index[10:20, 5:15] = 0
    index[20:30, 10:25] = 1
    index[5, 5] = 2
    scube = evaluate(driver, SCUBE)
    dmodel = dict(
        type='region_spectra', spec_size=51, spec_step=10, **INSTRUMENT,
        regions=dict(type='bins', file='bins.fits'))
    fitsutils.write_data('bins.fits', index, gridutils.Coords(
        (1, 1), (15.5, 20), (0, 0), 0))
    aspec = evaluate(driver, dmodel)
    expected = np.stack(
        [scube[:, index == i].sum(axis=1) for i in range(3)], axis=1)
    np.testing.assert_allclose(aspec, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('step', [1.0, 0.5])
def test_region_spectra_of_a_point_is_its_flux(driver, step):
    # The spectra of regions are fluxes: around a point, its flux,
    # whatever the size of the pixels of the model
    gmodel = dict(type='kinematics_2d', components=[dict(type='point')])
    dmodel = dict(
        type='region_spectra', size=[int(16 / step)] * 2, step=[step, step],
        spec_size=51, spec_step=10, regions=dict(type='apertures', apertures=[
            dict(type='circle', x=0, y=0, radius=3)]))
    model_group = observation_group([dict(
        driver=dict(type=driver.type()), dmodel=dmodel, gmodel=gmodel)])
    params = gbkfit.params.EvaluationParams(
        model_group.pdescs(), dict(xpos=0, ypos=0, flux=2, vsys=0, disp=30))
    spectrum = model_group.model_h(params.evaluate())[0]['spectra']['d'][:, 0]
    assert spectrum.sum() * 10 == pytest.approx(2, rel=1e-4)


def test_region_spectra_objective_residual(driver):
    # Data equal to the model plus 1, with errors of 2: every residual
    # (model - data) / error is -0.5
    from gbkfit.model import gmodel_parser
    from gbkfit.objective import Objective
    from gbkfit.instrument import Instrument, instrument_parser
    from gbkfit.observation import Observation, ObservationGroup
    dmodel = aspec_of_apertures(APERTURES[:2])
    model = evaluate(driver, dmodel)
    observable = observable_of(dmodel)
    dataset = DatasetRegionSpectra(
        Data(model + 1, error=np.full_like(model, 2)),
        observable.regions(), step=10)
    observation = Observation(
        observable, driver, instrument=Instrument(), data=dataset)
    group = ObservationGroup(
        [gmodel_parser.load(copy.deepcopy(GMODEL))], [observation])
    params = gbkfit.params.EvaluationParams(group.pdescs(), PROPERTIES)
    residual = Objective(group).residual_nddata_h(params.evaluate(), False)
    # Without the instrument, the model differs from the data
    assert not np.allclose(residual[0]['spectra'], -0.5)
    observation = Observation(
        observable, driver,
        instrument=instrument_parser.load(copy.deepcopy(INSTRUMENT)),
        data=dataset)
    group = ObservationGroup(
        [gmodel_parser.load(copy.deepcopy(GMODEL))], [observation])
    residual = Objective(group).residual_nddata_h(params.evaluate(), False)
    np.testing.assert_allclose(residual[0]['spectra'], -0.5, rtol=1e-5)


def test_region_spectra_data_round_trip(tmp_path):
    # The spectra are written with the regions along x and the velocity
    # along y, and read back with the world coordinates of the velocity
    from gbkfit.dataset import dataset_parser
    regions = RegionsApertures([
        gbkfit.region.aperture_parser.load(info) for info in APERTURES])
    dataset = DatasetRegionSpectra(
        Data(np.arange(51 * 3.0).reshape(51, 3)), regions, step=10,
        rpix=4, rval=1500)
    info = dataset_parser.dump(dataset, prefix=str(tmp_path / ''))
    for key in ('step', 'rpix', 'rval'):
        info.pop(key)
    loaded = dataset_parser.load(info)
    assert loaded.regions() == regions
    assert loaded.spectral_grid().size == (51,)
    np.testing.assert_allclose(loaded.spectral_grid().coords.step, [10])
    np.testing.assert_allclose(loaded.spectral_grid().coords.rpix, [4])
    np.testing.assert_allclose(loaded.spectral_grid().coords.rval, [1500])
    np.testing.assert_array_equal(
        loaded['spectra'].data(), dataset['spectra'].data())


def test_region_spectra_options_replace_the_spectral_axis_of_files(tmp_path):
    from gbkfit.dataset import dataset_parser
    regions = RegionsApertures([
        gbkfit.region.aperture_parser.load(info) for info in APERTURES])
    dataset = DatasetRegionSpectra(
        Data(np.ones((51, 3))), regions, step=10, rpix=4, rval=1500)
    info = dataset_parser.dump(dataset, prefix=str(tmp_path / ''))
    for key in ('step', 'rpix', 'rval'):
        info.pop(key)
    coords = dataset_parser.load(
        info | dict(step=5, rpix=6)).spectral_grid().coords
    np.testing.assert_allclose(coords.step, [5])
    np.testing.assert_allclose(coords.rpix, [6])
    # (the velocity of the header at the channel rpix)
    np.testing.assert_allclose(coords.rval, [1520])


def test_region_spectra_observation_from_data(tmp_path):
    # The regions and the spectral axis come from the data; the spatial
    # grid of apertures is given
    from gbkfit.observation import observation_parser
    regions = RegionsApertures([
        gbkfit.region.aperture_parser.load(info) for info in APERTURES])
    dataset = DatasetRegionSpectra(
        Data(np.ones((51, 3))), regions, step=10, rval=1500)
    data_info = dataset.dump(prefix=str(tmp_path / ''))
    data_info.pop('type')
    info = dict(
        driver=dict(type='host'), data=data_info,
        observable=dict(type='region_spectra', size=[32, 41]))
    observation = observation_parser.load(copy.deepcopy(info))
    observable = observation.observable()
    assert observable.size() == (32, 41, 51)
    assert observable.spectral_grid() == dataset.spectral_grid()
    assert observable.regions() == regions
    # A dump of the observation does not repeat what the data give
    dumped = observation_parser.dump(
        observation, prefix=str(tmp_path / 'dump_'))
    assert set(dumped['observable']) == {
        'type', 'size', 'step', 'rpix', 'rval', 'rota'}
    with pytest.raises(Exception, match="give its options \\['spec_step'\\]"):
        observation_parser.load(info | dict(observable=dict(
            type='region_spectra', size=[32, 41], spec_step=10)))


def test_region_spectra_of_spectra_with_another_spectral_axis_is_an_error(driver):
    from gbkfit.observation import Observation
    observable = observable_of(aspec_of_apertures(APERTURES))
    dataset = DatasetRegionSpectra(
        Data(np.ones((51, 3))), observable.regions(), step=5)
    with pytest.raises(RuntimeError, match="spectral axis"):
        Observation(observable, driver, data=dataset)



@pytest.mark.parametrize('kind', ['apertures', 'bins'])
def test_region_spectra_observation_of_data_on_a_rotated_grid(driver, kind):
    # The spectral axis of a rotated spatial grid is not rotated: it
    # matches that of the spectra
    from gbkfit.region import aperture_parser
    from gbkfit.observation import RegionSpectra, Observation
    if kind == 'apertures':
        regions = RegionsApertures([aperture_parser.load(dict(type='field'))])
        observable = RegionSpectra(
            regions, spec_size=5, spec_step=10, size=[8, 8], rota=30)
    else:
        regions = RegionsBins(np.zeros((4, 4)), rota=20)
        observable = RegionSpectra(regions, spec_size=5, spec_step=10)
    dataset = DatasetRegionSpectra(Data(np.ones((5, 1))), regions, step=10)
    Observation(observable, driver, data=dataset)

def test_region_spectra_spatial_grid_options():
    from gbkfit.observation import RegionSpectra
    bins = RegionsBins(np.zeros((4, 4)))
    with pytest.raises(RuntimeError, match="remove the options \\['step'\\]"):
        RegionSpectra(bins, spec_size=5, step=[1, 1])
    apertures = RegionsApertures([
        gbkfit.region.aperture_parser.load(dict(type='field'))])
    with pytest.raises(RuntimeError, match="size of the grid"):
        RegionSpectra(apertures, spec_size=5)
    circle = RegionsApertures([gbkfit.region.aperture_parser.load(
        dict(type='circle', x=0, y=0, radius=5))])
    with pytest.raises(RuntimeError, match="not inside the grid"):
        RegionSpectra(circle, spec_size=5, size=[8, 8])
