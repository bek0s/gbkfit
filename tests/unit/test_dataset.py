
import numpy as np
import pytest
from astropy.io import fits

from gbkfit.dataset import *
from gbkfit.utils.gridutils import Coords


def test_data():
    shape = (2, 5)
    data_d = np.full(shape, 10.0)
    data_m = np.full(shape, 1.0)
    data_e = np.full(shape, 2.0)
    # Default value tests
    data01 = Data(data_d)
    assert data01.data() is not data_d
    assert data01.dtype() == np.float32
    assert np.array_equal(data01.data(), data_d)
    assert np.array_equal(data01.mask(), np.ones(shape))
    assert data01.error() is None
    assert data01.npix() == data_d.size
    assert data01.ndim() == 2
    assert data01.shape() == shape
    # Data tests
    data02 = Data(data_d, mask=data_m, error=data_e)
    assert data02.data() is not data_d
    assert data02.mask() is not data_m
    assert data02.error() is not data_e
    assert np.array_equal(data02.data(), data_d)
    assert np.array_equal(data02.mask(), data_m)
    assert np.array_equal(data02.error(), data_e)


def test_dataset_grid():
    data = Data(np.full((2, 5), 10.0))
    # Default value tests
    grid = DatasetPixelBrightness(data).grid()
    assert grid.size == (5, 2)
    assert grid.coords == Coords((1, 1), (2.0, 0.5), (0, 0), 0)
    assert grid.zero() == (-2.0, -0.5)
    # Scalar value tests
    grid = DatasetPixelBrightness(data, step=2, rpix=3, rval=4, rota=5).grid()
    assert grid.coords == Coords((2, 2), (3, 3), (4, 4), 5)
    # Vector value tests
    grid = DatasetPixelBrightness(
        data, step=(1, 2), rpix=(3, 4), rval=(5, 6), rota=7).grid()
    assert grid.coords == Coords((1, 2), (3, 4), (5, 6), 7)


def test_dataset_pixel_brightness(tmp_path):
    shape = (2, 5)
    data = Data(np.full(shape, 10.0), np.ones(shape), np.full(shape, 2.0))
    image01 = DatasetPixelBrightness(data)
    assert image01.shape() == data.shape()
    assert image01.dtype() == data.dtype()
    # Dump tests
    prefix = str(tmp_path / '')
    image01_info_dumped = dataset_parser.dump(
        image01, prefix=prefix, overwrite=True)
    # The options of its one data item are flat
    image01_info = dict(
        type='pixel_brightness',
        data=f'{prefix}brightness_d.fits',
        mask=f'{prefix}brightness_m.fits',
        error=f'{prefix}brightness_e.fits',
        step=(1, 1),
        rpix=(2.0, 0.5),
        rval=(0, 0),
        rota=0)
    assert image01_info_dumped == image01_info
    # Load tests
    image01_loaded = dataset_parser.load(image01_info)
    assert image01_loaded.grid().size == image01.grid().size
    np.testing.assert_allclose(
        image01_loaded.grid().coords.step, image01.grid().coords.step)
    assert image01_loaded.grid().coords.rpix == image01.grid().coords.rpix
    assert image01_loaded.grid().coords.rota == image01.grid().coords.rota
    np.testing.assert_array_equal(image01_loaded['brightness'].error(), 2)


def test_dataset_pixel_moments(tmp_path):
    # Every moment map, from moment0 to moment7, can be loaded from a file
    for name in ['moment0', 'moment7']:
        fits.writeto(tmp_path / f'{name}.fits', np.ones((8, 20)))
    dataset = dataset_parser.load(dict(
        type='pixel_moments',
        moment0=dict(data=str(tmp_path / 'moment0.fits')),
        moment7=dict(data=str(tmp_path / 'moment7.fits'))))
    assert set(dataset.keys()) == {'moment0', 'moment7'}
    assert dataset.grid().size == (20, 8)


def test_dataset_items_must_have_the_same_world_coordinates(tmp_path):
    fits.writeto(tmp_path / 'moment0.fits', np.ones((8, 20)))
    fits.writeto(
        tmp_path / 'moment1.fits', np.ones((8, 20)), fits.Header(dict(CRPIX1=3)))
    with pytest.raises(Exception, match="different world coordinates"):
        dataset_parser.load(dict(
            type='pixel_moments',
            moment0=dict(data=str(tmp_path / 'moment0.fits')),
            moment1=dict(data=str(tmp_path / 'moment1.fits'))))


def test_dataset_items_must_have_the_same_shape():
    with pytest.raises(RuntimeError, match="different shapes"):
        DatasetPixelMoments(Data(np.ones((8, 20))), Data(np.ones((8, 21))))


def test_dataset_reference_pixel_from_fits_header(tmp_path):
    # The FITS reference pixel (CRPIX) is 1-based; gbkfit's is 0-based.
    # Axes without a reference pixel use their centre.
    header = fits.Header(dict(CRPIX1=10, CDELT1=1, CDELT2=1))
    fits.writeto(tmp_path / 'data.fits', np.zeros((8, 20)), header)
    dataset = dataset_parser.load(dict(
        type='pixel_brightness', data=str(tmp_path / 'data.fits')))
    assert dataset.grid().coords.rpix == (9, 3.5)


def test_dataset_reference_pixel_without_fits_header(tmp_path):
    fits.writeto(tmp_path / 'data.fits', np.zeros((8, 20)))
    dataset = dataset_parser.load(dict(
        type='pixel_brightness', data=str(tmp_path / 'data.fits')))
    assert dataset.grid().coords.rpix == (9.5, 3.5)


def test_dataset_reference_pixel_survives_fits_round_trip(tmp_path):
    dataset = DatasetPixelBrightness(Data(np.zeros((8, 20))), rpix=(3, 4), step=(2, 3))
    info = dataset.dump(prefix=str(tmp_path / ''))
    for key in ('step', 'rpix', 'rval', 'rota'):
        info.pop(key)
    loaded = dataset_parser.load(info)
    assert loaded.grid().coords.rpix == (3, 4)
    np.testing.assert_allclose(loaded.grid().coords.step, (2, 3), rtol=1e-12)


def test_dataset_round_trip_keeps_the_rotation_and_the_velocities(tmp_path):
    dataset = DatasetPixelSpectra(
        Data(np.zeros((6, 8, 20))), step=(2, 2, 10), rpix=(3, 4, 2),
        rval=(150, 2, 1500), rota=30)
    info = dataset.dump(prefix=str(tmp_path / ''))
    for key in ('step', 'rpix', 'rval', 'rota'):
        info.pop(key)
    loaded = dataset_parser.load(info).grid()
    np.testing.assert_allclose(loaded.coords.rota, 30, atol=1e-9)
    np.testing.assert_allclose(
        loaded.coords.rval, (150, 2, 1500), rtol=1e-12)
    np.testing.assert_allclose(
        loaded.zero(), dataset.grid().zero(), atol=1e-9)
    # The spatial axes are measured from the reference pixel
    np.testing.assert_allclose(loaded.zero(), (-6, -8, 1480), atol=1e-9)


def test_dataset_options_replace_the_world_coordinates_of_files(tmp_path):
    # step and rota replace those of the headers of the data files; rpix
    # replaces the reference pixel, with the world position of the header
    # there as rval, and rval the reference value, at its pixel
    dataset = DatasetPixelSpectra(
        Data(np.zeros((6, 8, 20))), step=(2, 2, 10), rpix=(3, 4, 2),
        rval=(150, 2, 1500), rota=30)
    info = dataset.dump(prefix=str(tmp_path / ''))
    for key in ('step', 'rpix', 'rval', 'rota'):
        info.pop(key)
    coords = dataset_parser.load(
        info | dict(step=(1, 1, 5), rota=45)).grid().coords
    np.testing.assert_allclose(coords.step, (1, 1, 5))
    assert coords.rota == 45
    coords = dataset_parser.load(info | dict(rpix=(5, 6, 3))).grid().coords
    assert coords.rpix == (5, 6, 3)
    assert coords.rval[2] == pytest.approx(1510)
    coords = dataset_parser.load(
        info | dict(rval=(150, 2, 1520))).grid().coords
    np.testing.assert_allclose(coords.rpix, (3, 4, 4), atol=1e-6)
    np.testing.assert_allclose(coords.rval, (150, 2, 1520))


def test_data_can_be_read_from_extensions(tmp_path):
    # e.g. JWST data, with the data in SCI and the error in ERR
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.full((8, 20), 3.0), name='SCI'),
        fits.ImageHDU(np.full((8, 20), 0.5), name='ERR')
    ]).writeto(tmp_path / 'data.fits')
    filename = str(tmp_path / 'data.fits')
    data, _ = load_data(dict(
        data=dict(file=filename, hdu='SCI'),
        error=dict(file=filename, hdu='ERR')))
    assert data.data().mean() == 3.0
    assert data.error().mean() == 0.5


def test_data_does_not_change_the_callers_arrays():
    data = np.ones((4, 4))
    data[0, 0] = np.nan
    error = np.ones((4, 4))
    Data(data, error=error)
    assert error[0, 0] == 1


def test_data_masks_errors_that_are_not_positive():
    error = np.ones((2, 3))
    error[0, 0] = 0
    error[0, 1] = -1
    data = Data(np.ones((2, 3)), error=error)
    assert data.mask().sum() == 4
    assert np.isnan(data.data()[0, :2]).all()


def test_data_is_float32(tmp_path):
    # The drivers support float32 only, so float64 files (BITPIX = -64)
    # and integer files become float32. A scalar error is not rounded
    # to the type of the data.
    fits.writeto(tmp_path / 'data.fits', np.ones((8, 20), np.int16))
    data, _ = load_data(dict(data=str(tmp_path / 'data.fits'), error=0.5))
    assert data.dtype() == np.float32
    assert data.error()[0, 0] == 0.5
    assert Data(np.ones((8, 20))).dtype() == np.float32


def test_observation_of_a_float64_dataset_is_float32(tmp_path):
    from gbkfit.observation import observation_parser
    fits.writeto(tmp_path / 'image.fits', np.ones((8, 20), np.float64))
    observation = observation_parser.load(dict(
        driver=dict(type='host'), observable=dict(type='pixel_brightness'),
        data=dict(data=str(tmp_path / 'image.fits'))))
    assert observation.dtype() == np.float32
    assert observation.observable().size() == (20, 8)


def test_dataset_steps_must_be_positive():
    with pytest.raises(RuntimeError, match="step must be positive"):
        DatasetPixelBrightness(Data(np.ones((8, 20))), step=(1, -1))


@pytest.mark.parametrize('dataset_type, shape', [
    ('pixel_brightness', (3, 8, 20)), ('pixel_moments', (3, 8, 20)),
    ('pixel_spectra', (8, 20)), ('pixel_spectra', (1, 3, 8, 20)),
    ('slit_spectra', (3, 8, 20))])
def test_datasets_check_the_number_of_axes(tmp_path, dataset_type, shape):
    # e.g. a cube loaded as an image, or a radio cube with a Stokes axis
    fits.writeto(tmp_path / 'data.fits', np.ones(shape, np.float32))
    # Datasets of one data item take its options flat
    data = dict(data=str(tmp_path / 'data.fits'))
    if dataset_type == 'pixel_moments':
        data = dict(moment0=data)
    with pytest.raises(Exception, match="axes"):
        dataset_parser.load(dict(type=dataset_type) | data)


def test_observable_grid_comes_from_the_data(tmp_path):
    # The grid comes from the data, so the observable must not give it
    # too, and a dump of the observation does not repeat it
    from gbkfit.observation import observation_parser
    fits.writeto(tmp_path / 'image.fits', np.ones((8, 20)))
    info = dict(
        driver=dict(type='host'),
        data=dict(data=str(tmp_path / 'image.fits')))
    observation = observation_parser.load(
        info | dict(observable=dict(type='pixel_brightness')))
    assert observation.observable().size() == (20, 8)
    observable_info = observation_parser.dump(
        observation, prefix=str(tmp_path / 'dump_'))['observable']
    assert observable_info == dict(
        type='pixel_brightness', mask_cutoff=None, mask_apply=False)
    with pytest.raises(Exception, match="give its options \\['step'\\]"):
        observation_parser.load(
            info | dict(observable=dict(type='pixel_brightness', step=1)))


def test_observation_data_must_be_on_the_grid_of_the_observable():
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.observation import Observation, PixelBrightness
    data = DatasetPixelBrightness(Data(np.ones((8, 20))), rota=30)
    moments = DatasetPixelMoments(Data(np.ones((8, 20))))
    with pytest.raises(RuntimeError, match="the data are on the grid"):
        Observation(DriverHost(), PixelBrightness(size=(20, 8)), data=data)
    with pytest.raises(RuntimeError, match="cannot be compared"):
        Observation(
            DriverHost(), PixelBrightness(size=(20, 8)), data=moments)
    Observation(
        DriverHost(), PixelBrightness(size=(20, 8), rota=30), data=data)


def test_pixel_spectra_rest_comes_from_the_data(tmp_path):
    # The rest of the spectral axis of the data is that of the observable,
    # which must not be given it too
    from gbkfit.observation import observation_parser
    dataset = DatasetPixelSpectra(
        Data(np.ones((6, 8, 20))), step=(1, 1, 10), rest='6562.8 Angstrom')
    data_info = dataset.dump(prefix=str(tmp_path / ''))
    data_info.pop('type')
    loaded = dataset_parser.load(dict(type='pixel_spectra') | dict(data_info))
    assert loaded.grid().coords.rest == dataset.grid().coords.rest
    info = dict(driver=dict(type='host'), data=data_info)
    observation = observation_parser.load(
        info | dict(observable=dict(type='pixel_spectra')))
    assert observation.observable().rest() == dataset.grid().coords.rest
    with pytest.raises(Exception, match="give its options \\['rest'\\]"):
        observation_parser.load(info | dict(observable=dict(
            type='pixel_spectra', rest='6562.8 Angstrom')))


def test_pixel_brightness_has_no_rest(tmp_path, caplog):
    # An image has no spectral axis: rest is an unknown option
    fits.writeto(tmp_path / 'image.fits', np.ones((8, 20)))
    dataset_parser.load(dict(
        type='pixel_brightness', data=str(tmp_path / 'image.fits'),
        rest='6562.8 Angstrom'))
    assert "unknown options" in caplog.text and "'rest'" in caplog.text


def test_data_error_is_a_number_or_a_file():
    # (a bool is an int in Python, but not an error)
    from gbkfit.dataset.data import load_data
    fits.writeto('data.fits', np.ones((4, 4), np.float32))
    assert load_data(dict(data='data.fits', error=2))[0].error()[0, 0] == 2
    with pytest.raises(Exception, match="a number or a file"):
        load_data(dict(data='data.fits', error=True))
    for error in (0, -1, float('nan')):
        with pytest.raises(Exception, match="must be positive"):
            load_data(dict(data='data.fits', error=error))


def test_data_items_are_mappings_or_null():
    # A null item is absent; an item that is not a mapping (e.g. a file
    # name) is an error that says what an item has
    fits.writeto('m1.fits', np.ones((4, 4), np.float32))
    dataset = DatasetPixelMoments.load(dict(moment1=dict(data='m1.fits'), moment2=None))
    assert set(dataset.keys()) == {'moment1'}
    with pytest.raises(Exception, match="has the file of its data"):
        DatasetPixelMoments.load(dict(moment1='m1.fits'))


def test_data_mask_values_that_are_not_finite():
    # (their residual would be NaN, or infinite)
    data = Data(np.array([[np.nan, 1.0], [np.inf, 2.0]]))
    np.testing.assert_array_equal(data.mask(), [[0, 1], [0, 1]])
