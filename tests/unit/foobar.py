

# _storage = dict()
#
#
# def convolve(driver, image_h, kernel_h):
#
#     key = (driver, image_h, kernel_h)
#
#     image_d, kernel_d = \
#         driver.convolve_fft_prepare(image_h, kernel_h)
#
#     result = \
#         driver.convolve_fft_execute(image_d, kernel_d)

from gbkfit.driver.drivers.host.driver import DriverHost

import numpy as np


def foo1():

    driver = DriverHost()
    fft = driver.backend().make_fft(np.float32)

    size_r = [3, 3, 3]
    size_c = fft.fft3_complex_shape(size_r)

    # allocate host and device memory
    data_r = driver.mem_alloc_s(size_r, np.float32)
    data_c = driver.mem_alloc_s(size_c, np.complex64)

    # write data to host
    data_r[0][:] = 0
    data_c[0][:] = 0
    data_r[0][size_r[0] // 2, size_r[1] // 2, size_r[2] // 2] = 100

    # copy host to device
    driver.mem_copy_h2d(data_r[0], data_r[1])
    driver.mem_copy_h2d(data_c[0], data_c[1])

    # perform fft
    fft.fft3_r2c(data_r[1], data_c[1])
    fft.fft3_c2r(data_c[1], data_r[1])

    # copy device to host
    driver.mem_copy_d2h(data_r[1], data_r[0])
    driver.mem_copy_d2h(data_c[1], data_c[0])

    print(data_r[1] / (size_r[0]*size_r[1]*size_r[2]))


    # result = gbkfit.math.convolve(image, kernel, driver, out=result)

    pass

foo1()
