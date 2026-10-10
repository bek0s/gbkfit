
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


def test_unknown_driver_options_are_reported():
    # (they were dropped: e.g. a device, which the drivers do not take)
    from gbkfit.driver import driver_parser
    from gbkfit.utils import parseutils
    with parseutils.strict_mode(), pytest.raises(
            parseutils.ConfigError, match="device"):
        driver_parser.load(dict(type='host', device=1))
