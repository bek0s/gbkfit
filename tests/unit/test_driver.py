
import numpy as np
import pytest


def _test_driver(driver):
    shape = (1000,)
    dtype = np.float32
    value = 1.0
    arr_h_1 = driver.mem_alloc_h(shape, dtype)
    arr_h_2 = driver.mem_alloc_h(shape, dtype)
    arr_d = driver.mem_alloc_d(shape, dtype)
    assert not (arr_h_1 is arr_d)
    assert not (arr_h_2 is arr_d)
    assert not (arr_h_1 is arr_h_2)
    arr_h_1[:] = value
    driver.mem_copy_h2d(arr_h_1, arr_d)
    driver.mem_copy_d2h(arr_d, arr_h_2)
    assert np.all(arr_h_2 == value)
    driver.mem_fill(arr_d, 2 * value)
    driver.mem_copy_d2h(arr_d, arr_h_1)
    assert np.all(arr_h_1 == 2 * value)


def test_driver_host():
    try:
        import gbkfit.driver.drivers.host
    except Exception as e:
        pytest.skip(f"host driver not available; reason: {e}")
    from gbkfit.driver.drivers.host import DriverHost
    _test_driver(DriverHost())


def test_driver_cuda():
    try:
        import gbkfit.driver.drivers.cuda
        import cupy
        assert cupy.cuda.is_available() == True
    except Exception as e:
        pytest.skip(f"cuda driver not available; reason: {e}")
    from gbkfit.driver.drivers.cuda import DriverCuda
    _test_driver(DriverCuda())


def test_driver_sycl():
    try:
        import gbkfit.driver.drivers.sycl
        import dpctl
        assert len(dpctl.device.get_devices()) > 0
    except Exception as e:
        pytest.skip(f"sycl driver not available; reason: {e}")
        from gbkfit.driver.drivers.sycl import DriverSycl
        _test_driver(DriverSycl())


if __name__ == '__main__':
    test_driver_host()
    test_driver_cuda()
    test_driver_sycl()
