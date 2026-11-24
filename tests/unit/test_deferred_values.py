"""
Datasets read the headers of their files when loaded, and their values
when first used (see gbkfit.dataset.Data): loading them, and checking an
observation of them, reads no values.
"""

import logging
import os
import pickle

import astropy.units as u
import numpy as np
import pytest

import gbkfit.io.formats.fits
from gbkfit.dataset import *
from gbkfit.grid import SpectralAxis
from gbkfit.observation import observation_parser
from gbkfit.config import FORMAT_VERSION
from gbkfit.setup import describe


@pytest.fixture
def reads(monkeypatch):
    """The files whose arrays are read (see read_image of FITS)."""
    read = []
    original = gbkfit.io.formats.fits.read_image

    def counted(filename, key):
        read.append(os.path.basename(filename))
        return original(filename, key)
    monkeypatch.setattr(gbkfit.io.formats.fits, 'read_image', counted)
    return read


def _cube(tmp_path, **options):
    """Return the configuration of a cube dumped to files, with a footprint."""
    rng = np.random.default_rng(3)
    values = rng.normal(1, 0.1, (6, 8, 20))
    footprint = np.ones((8, 20))
    footprint[:, :3] = 0
    dataset = DatasetPixelSpectra(
        Data(values, error=np.full(values.shape, 0.1)),
        spectral=SpectralAxis(step=10), footprint=footprint)
    return dataset_parser.dump(dataset, prefix=str(tmp_path / 'cube_')) \
        | options, values


def test_a_loaded_dataset_reads_its_values_when_first_used(tmp_path, reads):
    config, values = _cube(tmp_path)
    dataset = dataset_parser.load(config)
    # (its grid, shape and unit from the headers alone)
    assert reads == []
    assert dataset.grid().size == (20, 8, 6)
    assert dataset['spectra'].shape() == (6, 8, 20)
    assert dataset['spectra'].has_error()
    assert not dataset['spectra'].is_read()
    # (read once, with the footprint that masks them)
    spectra = dataset['spectra'].data()
    assert sorted(reads) == sorted(
        ['cube_spectra_d.fits', 'cube_spectra_e.fits',
         'cube_spectra_m.fits', 'cube_footprint.fits'])
    np.testing.assert_allclose(spectra[:, :, 3:], values[:, :, 3:])
    assert np.all(np.isnan(spectra[:, :, :3]))
    dataset['spectra'].mask()
    assert len(reads) == 4


def test_an_observation_of_data_is_built_from_their_headers(tmp_path, reads):
    config, _ = _cube(tmp_path)
    observation = observation_parser.load(
        dict(type='pixel_spectra',
             instrument=dict(psf=dict(type='gauss', sigma=1))),
        dataset=dataset_parser.load(config))
    assert reads == []
    assert observation.grid() == observation.dataset().grid()


def test_describing_a_setup_reads_no_values(tmp_path, monkeypatch):
    config, _ = _cube(tmp_path)

    def refused(filename, key):
        raise AssertionError(f"the values of {filename} were read")
    monkeypatch.setattr(gbkfit.io.formats.fits, 'read_image', refused)
    setup = dict(
        version=FORMAT_VERSION, galaxies=[dict(components=[dict(
            type='thin_disk', rnodes=[0, 1, 2],
            rptraits=dict(type='exponential'),
            vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform'), emission=dict(type='lines'))])],
        observations=[dict(type='pixel_spectra')],
        datasets=[config], params=dict(properties=dict(
            xpos=0, ypos=0, vsys=0, posa=0, incl=45, rpt_a=1, rpt_s=1,
            vpt_rt=1, vpt_vt=100, dpt_a=10)))
    described = describe(setup)
    assert not [p for p in described.problems if p.severity == 'error']
    assert described.group.observations()[0].dataset() is not None
    # (of data of another kind, the problem is of the dataset of the
    # observation)
    described = describe(setup | dict(
        observations=[dict(type='pixel_brightness')]))
    assert described.group is None
    [problem] = [p for p in described.problems if p.severity == 'error']
    assert list(problem.path) == ['datasets', 0]
    assert "data of pixel spectra (cube) · needs pixel brightness" \
        in problem.message


def test_the_problems_of_values_show_as_they_are_read(tmp_path):
    values = np.ones((8, 20))
    errors = np.full((8, 20), 0.1)
    errors[2, 3] = -1
    dataset = DatasetPixelBrightness(Data(values, error=np.abs(errors)))
    config = dataset_parser.dump(dataset, prefix=str(tmp_path / 'map_'))
    from astropy.io import fits
    fits.writeto(config['error'], errors, overwrite=True)
    loaded = dataset_parser.load(config)
    with pytest.raises(ValueError, match="negative"):
        loaded['brightness'].data()


def test_a_file_that_changes_after_it_is_opened_is_refused(tmp_path):
    config, values = _cube(tmp_path)
    dataset = dataset_parser.load(config)
    from astropy.io import fits
    fits.writeto(config['data'], values[:, :, :10], overwrite=True)
    with pytest.raises(ValueError, match="changed after it was opened"):
        dataset['spectra'].data()


def test_cleaning_is_logged_as_the_values_are_made(tmp_path, caplog):
    config, _ = _cube(tmp_path, mask_whole_spectra=True)
    with caplog.at_level(logging.INFO):
        dataset = dataset_parser.load(config)
        assert "whole spectra" not in caplog.text
        dataset['spectra'].data()
    assert "values masked with whole spectra of spectra" in caplog.text


def test_a_copy_for_another_process_has_the_values(tmp_path, reads):
    config, values = _cube(tmp_path)
    dataset = dataset_parser.load(config)
    copied = pickle.loads(pickle.dumps(dataset))
    assert copied['spectra'].is_read()
    os.remove(config['data'])
    np.testing.assert_allclose(
        copied['spectra'].data()[:, :, 3:], values[:, :, 3:])


def test_changes_of_items_not_read_are_made_as_they_are_read():
    data = Data.deferred(
        lambda: (np.arange(6.0).reshape(2, 3), None, None), (2, 3), False,
        'm/s')
    converted = data.to_unit(data.unit().to_string() and
                             __import__('astropy.units').units.km /
                             __import__('astropy.units').units.s)
    masked = converted.masked(np.array([[1, 0, 1], [1, 1, 1]]))
    assert not data.is_read() and not masked.is_read()
    assert str(masked.unit()) == 'km / s'
    np.testing.assert_allclose(
        masked.data(), [[0, np.nan, 0.002], [0.003, 0.004, 0.005]])
    # (what it is made of is not kept: it is read again if asked for)
    assert not data.is_read()


def test_a_deferred_array_is_read_once():
    calls = []

    def read():
        calls.append(1)
        return np.zeros((2, 3))
    deferred = Deferred((2, 3), read)
    assert deferred.shape == (2, 3) and calls == []
    deferred.values()
    deferred.values()
    assert calls == [1]
    with pytest.raises(ValueError, match="shape"):
        Deferred((4,), lambda: np.zeros(3)).values()


def test_the_valid_pixels_of_bins_are_read_when_used(tmp_path, reads):
    from astropy.io import fits
    index = np.zeros((4, 5))
    index[:, 3:] = 1
    fits.writeto(tmp_path / 'bins.fits', index)
    fits.writeto(tmp_path / 'valid.fits', np.ones((7, 4, 5)))
    bins = RegionsBins.from_file(
        str(tmp_path / 'bins.fits'), valid=str(tmp_path / 'valid.fits'))
    assert bins.valid_channels() == 7
    assert reads == ['bins.fits']
    assert bins.valid().shape == (7, 4, 5)
    assert reads == ['bins.fits', 'valid.fits']


def test_visibilities_read_their_coverage_when_loaded(tmp_path):
    nchannels, nrows = 3, 5
    rng = np.random.default_rng(1)
    arrays = dict(
        data=rng.normal(size=(nchannels, nrows))
        + 1j * rng.normal(size=(nchannels, nrows)),
        error=np.ones((nchannels, nrows)),
        uvw=rng.normal(size=(nrows, 3)) * 100,
        frequencies=np.linspace(1.42e9, 1.4201e9, nchannels),
        phase_centre=np.array([10.0, -20.0]), frame='LSRK',
        rest=1.420405752e9)
    np.savez(tmp_path / 'vis.npz', **arrays)
    dataset = DatasetVisibilities.from_file(str(tmp_path / 'vis.npz'))
    assert dataset['re'].shape() == (nchannels, nrows)
    assert not dataset['re'].is_read() and not dataset['im'].is_read()
    np.testing.assert_allclose(dataset['im'].data(), arrays['data'].imag)
    np.testing.assert_allclose(dataset['re'].data(), arrays['data'].real)
