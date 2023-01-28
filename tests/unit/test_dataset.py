
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
    assert np.array_equal(data01.error(), data_ones)
    assert data01.ndim() == 2
    assert data01.data().size == data01.npix()
    assert data01.size() == (5, 2)
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
    data04 = Data(data_d, step=(1, 2), rpix=(3, 4), rval=(5, 6), rota=7)
    assert data04.step() == (1, 2)
    assert data04.rpix() == (3, 4)
    assert data04.rval() == (5, 6)
    assert data04.rota() == 7
    # Dump tests
    filename_data_d = '/tmp/data_d.fits'
    filename_data_m = '/tmp/data_m.fits'
    filename_data_e = '/tmp/data_e.fits'
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
    data01 = Data(data_d)
    DatasetImage(data01)


def test_dataset_lslit():
    pass


def test_dataset_mmaps():
    pass


def test_dataset_scube():
    pass
