
import numpy as np
import pytest


def test_driver_memory(driver):
    shape = (1000,)
    dtype = np.float32
    value = 1.0
    arr_h_1 = driver.mem_alloc_h(shape, dtype)
    arr_h_2 = driver.mem_alloc_h(shape, dtype)
    arr_d = driver.mem_alloc_d(shape, dtype)
    assert arr_h_1 is not arr_d
    assert arr_h_2 is not arr_d
    assert arr_h_1 is not arr_h_2
    arr_h_1[:] = value
    driver.mem_copy_h2d(arr_h_1, arr_d)
    driver.mem_copy_d2h(arr_d, arr_h_2)
    assert np.all(arr_h_2 == value)
    driver.mem_fill(arr_d, 2 * value)
    driver.mem_copy_d2h(arr_d, arr_h_1)
    assert np.all(arr_h_1 == 2 * value)


def test_device_arrays_are_copied_in_the_type_of_the_target(driver):
    values = np.array([1.5 + 2j, -3 + 4j])
    real = driver.mem_alloc_d(2, np.float32)
    driver.mem_copy_d2d(driver.mem_copy_h2d(values).real, real)
    np.testing.assert_array_equal(driver.mem_copy_d2h(real), [1.5, -3])


def test_unknown_driver_options_are_reported():
    # (they were dropped: e.g. a device, which the host driver does not
    # take)
    from gbkfit.driver import driver_parser
    from gbkfit.utils import parseutils
    with parseutils.strict_mode(), pytest.raises(
            parseutils.ConfigError, match="device"):
        driver_parser.load(dict(type='host', device=1))


def test_cuda_drivers_are_of_the_gpus_there_are(driver):
    if driver.type() != 'cuda':
        pytest.skip("a test of the cuda driver")
    import cupy
    from gbkfit.driver import driver_parser
    from gbkfit.utils import parseutils
    assert driver_parser.load(dict(type='cuda', device=0)).dump() \
        == dict(type='cuda', device=0)
    ndevices = cupy.cuda.runtime.getDeviceCount()
    with pytest.raises(parseutils.ConfigError, match="device"):
        driver_parser.load(dict(type='cuda', device=ndevices))


def test_arrays_too_large_for_the_kernels_are_refused(driver):
    # The kernels index arrays by 32-bit integers: an array of more values
    # is refused before it is allocated
    from gbkfit.utils.parseutils import ConfigError
    with pytest.raises(ConfigError, match="32-bit"):
        driver.mem_alloc_d((2 ** 16, 2 ** 16), np.float32)
