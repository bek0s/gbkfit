
import numpy as np
import pytest

from gbkfit.dataset import *
from gbkfit.dataset.datasets import *


def test_data():
    size = (5, 2)
    shape = size[::-1]
    data_d = np.full(shape, 10.0)
    data_m = np.full(shape, 1.0)
    data_e = np.full(shape, 2.0)
    data_ones = np.ones(shape)
    # Default value tests
    data01 = Data(data_d)
    assert data01.data() is not data_d
    assert data01.dtype() == np.float32
    assert np.array_equal(data01.data(), data_d)
    assert np.array_equal(data01.mask(), data_ones)
    assert np.array_equal(data01.error(), None)
    assert data01.data().size == data01.npix()
    assert data01.ndim() == len(size)
    assert data01.size() == size
    assert data01.step() == (1, 1)
    assert data01.zero() == (-2.0, -0.5)
    assert data01.rpix() == (2.0, 0.5)
    assert data01.rval() == (0, 0)
    assert data01.rota() == 0
    # Data tests
    data02 = Data(data_d, mask=data_m, error=data_e)
    assert data02.data() is not data_d
    assert data02.mask() is not data_m
    assert data02.error() is not data_e
    assert np.array_equal(data02.data(), data_d)
    assert np.array_equal(data02.mask(), data_m)
    assert np.array_equal(data02.error(), data_e)
    # Scalar value wcs tests
    data03 = Data(data_d, step=2, rpix=3, rval=4, rota=5)
    assert data03.step() == (2, 2)
    assert data03.rpix() == (3, 3)
    assert data03.rval() == (4, 4)
    assert data03.rota() == 5
    # Vector value wcs tests
    data04 = Data(
        data_d, error=data_e, step=(1, 2), rpix=(3, 4), rval=(5, 6), rota=7)
    assert data04.step() == (1, 2)
    assert data04.rpix() == (3, 4)
    assert data04.rval() == (5, 6)
    assert data04.rota() == 7
    # Dump tests
    filename_data_d = 'data_d.fits'
    filename_data_m = 'data_m.fits'
    filename_data_e = 'data_e.fits'
    data_info_dumped = data_parser.dump(
        data04, filename_data_d, filename_data_m, filename_data_e,
        overwrite=True)
    data_info = dict(
        data=filename_data_d,
        mask=filename_data_m,
        error=filename_data_e,
        step=(1, 2),
        rpix=(3, 4),
        rval=(5, 6),
        rota=7)
    assert data_info_dumped == data_info
    # Load tests
    data04_loaded = data_parser.load(data_info)
    assert data04_loaded.size() == data04.size()
    assert data04_loaded.step() == data04.step()
    assert data04_loaded.rpix() == data04.rpix()
    assert data04_loaded.rval() == data04.rval()
    assert data04_loaded.rota() == data04.rota()


def test_dataset_image():
    size = (5, 2)
    shape = size[::-1]
    data_d = np.full(shape, 10.0)
    data_m = np.full(shape, 1.0)
    data_e = np.full(shape, 2.0)
    data01 = Data(data_d, data_m, data_e)
    # Default value tests
    image01 = DatasetImage(data01)
    assert image01.npix() == data01.npix()
    assert image01.size() == data01.size()
    assert image01.step() == data01.step()
    assert image01.zero() == data01.zero()
    assert image01.dtype() == data01.dtype()
    # Dump tests
    image01_info_dumped = dataset_parser.dump(image01, overwrite=True)
    image01_info = dict(
        type='image',
        image=dict(
            data='image_d.fits',
            mask='image_m.fits',
            error='image_e.fits'),
        step=(1.0, 1.0),
        rpix=(2.0, 0.5),
        rval=(0.0, 0.0),
        rota=0)
    assert image01_info_dumped == image01_info
    # Load tests
    image01_info_loaded = dataset_parser.load(image01_info)
    assert image01_info_loaded.size() == image01.size()
    assert image01_info_loaded.step() == image01.step()
    assert image01_info_loaded.rpix() == image01.rpix()
    assert image01_info_loaded.rval() == image01.rval()
    assert image01_info_loaded.rota() == image01.rota()


def test_dataset_lslit():
    pass


def test_dataset_mmaps(tmp_path):
    # Every moment map, from mmap0 to mmap7, can be loaded from a file
    from astropy.io import fits
    for name in ['mmap0', 'mmap7']:
        fits.writeto(tmp_path / f'{name}.fits', np.ones((8, 20)))
    dataset = dataset_parser.load(dict(
        type='mmaps',
        mmap0=dict(data=str(tmp_path / 'mmap0.fits')),
        mmap7=dict(data=str(tmp_path / 'mmap7.fits'))))
    assert set(dataset.keys()) == {'mmap0', 'mmap7'}
    assert dataset['mmap7'].size() == (20, 8)


def test_dataset_scube():
    pass


def test_data_reference_pixel_from_fits_header(tmp_path):
    # The FITS reference pixel (CRPIX) is 1-based; gbkfit's is 0-based.
    # Axes without a reference pixel use their centre.
    from astropy.io import fits
    header = fits.Header(dict(CRPIX1=10, CDELT1=1, CDELT2=1))
    fits.writeto(tmp_path / 'data.fits', np.zeros((8, 20)), header)
    data = data_parser.load(dict(data=str(tmp_path / 'data.fits')))
    assert data.rpix() == (9, 3.5)


def test_data_reference_pixel_without_fits_header(tmp_path):
    from astropy.io import fits
    fits.writeto(tmp_path / 'data.fits', np.zeros((8, 20)))
    data = data_parser.load(dict(data=str(tmp_path / 'data.fits')))
    assert data.rpix() == (9.5, 3.5)


def test_data_reference_pixel_survives_fits_round_trip(tmp_path):
    data = Data(np.zeros((8, 20)), rpix=(3, 4), step=(2, 3))
    info = data.dump(str(tmp_path / 'data.fits'), dump_wcs=False)
    loaded = data_parser.load(info)
    assert loaded.rpix() == (3, 4)
    np.testing.assert_allclose(loaded.step(), (2, 3), rtol=1e-12)



def test_data_round_trip_keeps_the_rotation_and_the_velocities(tmp_path):
    # rota is in degrees (Data.dump used to write it as radians)
    data = Data(
        np.zeros((6, 8, 20)), step=(2, 2, 10), rpix=(3, 4, 2),
        rval=(150, 2, 1500), rota=30, spectral_axis=2)
    info = data.dump(str(tmp_path / 'cube.fits'), dump_wcs=False)
    loaded = data_parser.load(info, spectral_axis=2)
    np.testing.assert_allclose(loaded.rota(), 30, atol=1e-9)
    np.testing.assert_allclose(loaded.rval(), (150, 2, 1500), rtol=1e-12)
    np.testing.assert_allclose(loaded.zero(), data.zero(), atol=1e-9)
    # The spatial axes are measured from the reference pixel
    np.testing.assert_allclose(loaded.zero(), (-6, -8, 1480), atol=1e-9)


def test_data_can_be_read_from_extensions(tmp_path):
    # e.g. JWST data, with the data in SCI and the error in ERR
    from astropy.io import fits
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(np.full((8, 20), 3.0), name='SCI'),
        fits.ImageHDU(np.full((8, 20), 0.5), name='ERR')
    ]).writeto(tmp_path / 'data.fits')
    filename = str(tmp_path / 'data.fits')
    data = data_parser.load(dict(
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
    from astropy.io import fits
    fits.writeto(tmp_path / 'data.fits', np.ones((8, 20), np.int16))
    data = data_parser.load(dict(data=str(tmp_path / 'data.fits'), error=0.5))
    assert data.dtype() == np.float32
    assert data.error()[0, 0] == 0.5
    assert Data(np.ones((8, 20))).dtype() == np.float32


def test_observation_of_a_float64_dataset_is_float32():
    from gbkfit.dataset.datasets import DatasetImage
    from gbkfit.observation import observation_parser
    dataset = DatasetImage(Data(np.ones((8, 20), np.float64)))
    observation = observation_parser.load(dict(
        driver=dict(type='host'), observable=dict(type='image')),
        dataset=dataset)
    assert observation.dtype() == np.float32


def test_data_steps_must_be_positive():
    with pytest.raises(RuntimeError, match="step must be positive"):
        Data(np.ones((8, 20)), step=(1, -1))


@pytest.mark.parametrize('dataset_type, shape', [
    ('image', (3, 8, 20)), ('mmaps', (3, 8, 20)), ('scube', (8, 20)),
    ('scube', (1, 3, 8, 20)), ('lslit', (3, 8, 20))])
def test_datasets_check_the_number_of_axes(tmp_path, dataset_type, shape):
    # e.g. a cube loaded as an image, or a radio cube with a Stokes axis
    from astropy.io import fits
    fits.writeto(tmp_path / 'data.fits', np.ones(shape, np.float32))
    name = 'mmap0' if dataset_type == 'mmaps' else dataset_type
    with pytest.raises(Exception, match="axes"):
        dataset_parser.load({
            'type': dataset_type,
            name: dict(data=str(tmp_path / 'data.fits'))})
