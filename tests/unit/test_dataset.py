
import numpy as np

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
    assert data01.dtype() == data_d.dtype
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


def test_dataset_mmaps():
    pass


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
    assert loaded.step() == (2, 3)

