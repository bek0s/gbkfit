from abc import ABC
from typing import Any

import numpy as np
import pytest

from gbkfit.driver import Driver
from gbkfit.model.dmodels import DModelImage

from gbkfit.model import GModelImage
from gbkfit.psflsf.lsfs import LSFGauss


def set_middle_pixel(image_h):
    """Set all elements of an array to 0 except the central pixel(s).

    Supports 1D, 2D, and 3D arrays.
    """
    image_h[:] = 0  # Set everything to zero
    size = image_h.shape  # Get dimensions

    # Compute center indices for each dimension
    indices = [
        [s // 2] if s % 2 else [s // 2 - 1, s // 2]
        for s in size
    ]

    # Use np.ix_ to apply multi-dimensional indexing
    image_h[np.ix_(*indices)] = 1


def create_mock_image(driver: Driver, size: tuple[int, int], dtype):
    image = driver.mem_alloc_h(size[::-1], dtype)
    image[:] = 0
    set_middle_pixel(image)
    return image


class GModelImageMock(GModelImage, ABC):

    @staticmethod
    def type():
        return "mock"

    @classmethod
    def load(cls, info, *args, **kwargs):
        pass

    def dump(self):
        return dict()

    def pdescs(self):
        return dict()

    def has_weights(self):
        return False

    def evaluate_image(
            self, driver, params, image, weights, size, step, zero, rota,
            dtype, out_extra):
        driver.mem_copy_h2d(create_mock_image(driver, size, dtype), image)
        print(image)


def _test_dmodel_image():

    from gbkfit.driver.drivers.host import DriverHost

    from gbkfit.psflsf.psfs import PSFGauss

    driver = DriverHost()
    gmodel = GModelImageMock()
    psf = PSFGauss(5)

    size = (3, 3)
    step = (0.5, 0.5)

    dmodel = DModelImage(size, step, scale=(2, 2))

    output = dmodel.evaluate(driver, gmodel, dict(), None)

    foo = create_mock_image(driver, size, np.float32)

    # assert np.allclose(output['image']['d'], foo)


    assert 'image' in output
    assert 'd' in output['image']


def set_pixel(
        coord: tuple[int, int, int],
        value: float,
        cube: np.ndarray,
        size: tuple[int, int, int],
        edge: tuple[int, int, int]
) -> None:
    pass



def _test_dcube(driver: Driver, dtype):

    from gbkfit.model.dmodels._dcube import DCube
    from gbkfit.psflsf.psfs import PSFGauss
    from gbkfit.psflsf.lsfs import LSFGauss

    size = (21, 30, 41)
    step = (0.5, 1.5, 2.0)
    rpix = (5.0, 6.5, 7.5)
    rval = (1.0, 1.5, 2.0)
    scale = (1, 2, 3)
    psf = PSFGauss(2)
    lsf = LSFGauss(5)
    has_weights = True
    smooth_weights = True
    mask_cutoff = 1.0
    mask_apply = True

    args = dict(
        size=size,
        step=step,
        rpix=rpix,
        rval=rval,
        rota=0.0,
        scale=scale,
        psf=None,
        lsf=None,
        smooth_weights=False,
        mask_cutoff=mask_cutoff,
        mask_apply=mask_apply,
        dtype=dtype
    )

    dcube = DCube(**args)
    dcube.prepare(driver, True)

    zero = (
        rval[0] - rpix[0] * step[0],
        rval[1] - rpix[1] * step[1],
        rval[2] - rpix[2] * step[2])
    scaled_size = scratch_size = (
        size[0] * scale[0],
        size[1] * scale[1],
        size[2] * scale[2])
    scaled_step = scratch_step = (
        step[0] / scale[0],
        step[1] / scale[1],
        step[2] / scale[2])
    scratch_edge = (0, 0, 0)
    scratch_zero = (
        zero[0] - step[0] / 2 - (scratch_edge[0] - 0.5) * scratch_step[0],
        zero[1] - step[1] / 2 - (scratch_edge[1] - 0.5) * scratch_step[1],
        zero[2] - step[2] / 2 - (scratch_edge[2] - 0.5) * scratch_step[2])

    assert dcube.size() == size
    assert dcube.step() == step
    assert dcube.zero() == zero
    assert dcube.scale() == scale
    assert dcube.dcube() is not None
    assert dcube.mcube() is not None
    assert dcube.wcube() is not None
    assert dcube.dcube().shape == size[::-1]
    assert dcube.mcube().shape == size[::-1]
    assert dcube.wcube().shape == size[::-1]
    assert dcube.scratch_size() == scratch_size
    assert dcube.scratch_step() == scratch_step
    assert dcube.scratch_zero() == scratch_zero
    assert dcube.scratch_dcube() is not dcube.dcube()
    assert dcube.scratch_wcube() is not dcube.wcube()

    h_scratch_dcube = driver.mem_alloc_h(scratch_size[::-1], dtype)
    h_scratch_dcube[:] = 1
    x_min, y_min, z_min = scale
    x_max = scaled_size[0] - x_min
    y_max = scaled_size[1] - y_min
    z_max = scaled_size[2] - z_min
    h_scratch_dcube[z_min:z_max, y_min:y_max, x_min:x_max] = 2
    driver.mem_copy_h2d(h_scratch_dcube, dcube.scratch_dcube())

    out_extra = {}
    dcube.evaluate(out_extra)

    driver.mem_copy_d2h(dcube.scratch_dcube(), h_scratch_dcube)

    import astropy.io.fits as fits

    fits.writeto("test.fits", h_scratch_dcube, overwrite=True)

    # Dr Elke Wiseman
    # Georgios Bekiaris
    # 10:30 6 jun
    #
    return

    #
    #
    #

    size = (21, 30, 41)
    step = (0.5, 1.5, 2.0)
    rpix = (5.0, 6.5, 7.5)
    rval = (1.0, 1.5, 2.0)
    scale = (1, 2, 3)
    psf = PSFGauss(2)
    lsf = LSFGauss(5)
    has_weights = True
    smooth_weights = True
    mask_cutoff = 1.0
    mask_apply = True
    dtype = np.float32

    args = dict(
        size=size,
        step=step,
        rpix=rpix,
        rval=rval,
        rota=0.0,
        scale=scale,
        psf=psf,
        lsf=lsf,
        smooth_weights=smooth_weights,
        mask_cutoff=mask_cutoff,
        mask_apply=mask_apply,
        dtype=dtype
    )

    dcube = DCube(**args)
    dcube.prepare(driver, False)
    zero = (
        rval[0] - rpix[0] * step[0],
        rval[1] - rpix[1] * step[1],
        rval[2] - rpix[2] * step[2])

    assert dcube.size() == size
    assert dcube.step() == step
    assert dcube.zero() == zero
    assert dcube.scale() == scale
    assert dcube.dcube() is not None
    assert dcube.mcube() is not None
    assert dcube.wcube() is None
    assert dcube.dcube().shape == size[::-1]
    assert dcube.mcube().shape == size[::-1]

    scaled_size = (
        size[0] * scale[0],
        size[1] * scale[1],
        size[2] * scale[2])
    scaled_step = scratch_step = (
        step[0] / scale[0],
        step[1] / scale[1],
        step[2] / scale[2])
    scaled_step_spat = (scaled_step[0], scaled_step[1])
    scaled_step_spec = scaled_step[2]
    psf_size_hi = psf.size(scaled_step_spat)
    lsf_size_hi = lsf.size(scaled_step_spec)
    scratch_size, scratch_edge = (
        driver.backends().fft(dtype).fft_convolution_shape(
            scaled_size, psf_size_hi + (lsf_size_hi,)))
    scratch_zero = (
        zero[0] - step[0] / 2 - (scratch_edge[0] - 0.5) * scaled_step[0],
        zero[1] - step[1] / 2 - (scratch_edge[1] - 0.5) * scaled_step[1],
        zero[2] - step[2] / 2 - (scratch_edge[2] - 0.5) * scaled_step[2])

    assert dcube.scratch_size() == scratch_size
    assert dcube.scratch_step() == scratch_step
    assert dcube.scratch_zero() == scratch_zero
    assert dcube.scratch_dcube() is not dcube.dcube()
    assert dcube.scratch_wcube() is None
    assert dcube.scratch_dcube().shape == scratch_size[::-1]

    h_scratch_dcube = driver.mem_alloc_h(scratch_size[::-1], dtype)
    h_scratch_dcube[:] = 0
    x_min, y_min, z_min = scale
    x_max = scaled_size[0] - x_min
    y_max = scaled_size[1] - y_min
    z_max = scaled_size[2] - z_min
    h_scratch_dcube[z_min:z_max, y_min:y_max, x_min:x_max] = 1
    driver.mem_copy_h2d(h_scratch_dcube, dcube.scratch_dcube())

    out_extra = {}
    dcube.evaluate(out_extra)

    import astropy.io.fits as fits

    fits.writeto("psf_hi.fits", out_extra['psf_hi'], overwrite=True)


    # dcube_lo_h_1 = driver.mem_alloc_h(expected_shape_lo, dtype)
    # dcube_lo_h_2 = driver.mem_alloc_h(expected_shape_lo, dtype)
    # dcube_hi_h_1 = driver.mem_alloc_h(expected_shape_hi, dtype)
    # dcube_hi_h_2 = driver.mem_alloc_h(expected_shape_hi, dtype)
    # dcube_lo_h_1.fill(42)
    # dcube_hi_h_1.fill(42)
    # driver.mem_copy_h2d(dcube_hi_h_1, dcube.scratch_dcube())
    # out_extra = {}
    # dcube.evaluate(out_extra)
    # driver.mem_copy_d2h(dcube.scratch_dcube(), dcube_hi_h_2)
    # driver.mem_copy_d2h(dcube.dcube(), dcube_lo_h_2)
    #
    # assert np.array_equal(dcube_hi_h_1, dcube_hi_h_2)
    # assert np.array_equal(dcube_lo_h_1, dcube_lo_h_2)



    return

    from gbkfit.model.dmodels._dcube import DCube
    from gbkfit.psflsf.psfs import PSFGauss

    def set_range(data_d, value, range_x, range_y, range_z):
        data_h = driver.mem_alloc_h(data_d.shape, data_d.dtype)
        data_h[:, :, :] = 0
        x_min, x_max = range_x
        y_min, y_max = range_y
        z_min, z_max = range_z
        data_h[z_min:z_max, y_min:y_max, x_min:x_max] = value
        driver.mem_copy_h2d(data_h, data_d)

    size = (21, 30, 41)
    step = (0.5, 1.5, 2.0)
    rpix = (5.0, 6.5, 7.5)
    rval = (1.0, 1.5, 2.0)
    scale = (2, 3, 4)
    psf = PSFGauss(2)
    lsf = LSFGauss(5)

    args = dict(
        size=size,
        step=step,
        rpix=rpix,
        rval=rval,
        rota=0.0,
        scale=(1, 1, 1),
        psf=None,
        lsf=None,
        smooth_weights=False,
        mask_cutoff=None,
        mask_apply=False,
        dtype=np.float32
    )

    #
    # Test size, step, rpix, rval.
    #

    dcube = DCube(**args)
    dcube.prepare(driver, False)

    expected_zero = (-1.5, -8.25, -13.0)
    expected_shape = size[::-1]
    dtype = np.float32

    assert dcube.size() == size
    assert dcube.step() == step
    assert dcube.zero() == expected_zero
    assert dcube.dcube() is not None
    assert dcube.dcube().shape == expected_shape
    assert dcube.wcube() is None
    assert dcube.mcube() is None

    assert dcube.scratch_size() == size
    assert dcube.scratch_step() == step
    assert dcube.scratch_zero() == expected_zero
    assert dcube.scratch_dcube() is dcube.dcube()
    assert dcube.scratch_dcube().shape == expected_shape
    assert dcube.scratch_wcube() is None

    dcube_h_1 = driver.mem_alloc_h(expected_shape, dtype)
    dcube_h_2 = driver.mem_alloc_h(expected_shape, dtype)
    dcube_h_1.fill(42)
    driver.mem_copy_h2d(dcube_h_1, dcube.scratch_dcube())
    out_extra = {}
    dcube.evaluate(out_extra)
    driver.mem_copy_d2h(dcube.scratch_dcube(), dcube_h_2)
    assert np.array_equal(dcube_h_1, dcube_h_2)

    #
    # Test scale
    #

    args |= dict(scale=scale)

    dcube = DCube(**args)
    dcube.prepare(driver, False)

    expected_size_hi = tuple((np.array(size) * np.array(scale)).tolist())
    expected_step_hi = tuple((np.array(step) / np.array(scale)).tolist())
    expected_zero_hi = tuple((np.array(expected_zero)
                              - 0.5 * np.array(step)
                              + 0.5 * np.array(expected_step_hi)
                              ).tolist())
    expected_shape_lo = size[::-1]
    expected_shape_hi = expected_size_hi[::-1]

    assert dcube.size() == size
    assert dcube.step() == step
    assert dcube.zero() == expected_zero
    assert dcube.scale() == scale
    assert dcube.dcube() is not None
    assert dcube.dcube().shape == expected_shape_lo
    assert dcube.wcube() is None
    assert dcube.mcube() is None

    assert dcube.scratch_size() == expected_size_hi
    assert dcube.scratch_step() == expected_step_hi
    assert dcube.scratch_zero() == expected_zero_hi
    assert dcube.scratch_dcube() is not dcube.dcube()
    assert dcube.scratch_dcube().shape == expected_shape_hi
    assert dcube.scratch_wcube() is None

    def _fill_and_read_back(driver_: Driver, dcube: DCube, value: float):



        pass

    dcube_lo_h_1 = driver.mem_alloc_h(expected_shape_lo, dtype)
    dcube_lo_h_2 = driver.mem_alloc_h(expected_shape_lo, dtype)
    dcube_hi_h_1 = driver.mem_alloc_h(expected_shape_hi, dtype)
    dcube_hi_h_2 = driver.mem_alloc_h(expected_shape_hi, dtype)
    dcube_lo_h_1.fill(42)
    dcube_hi_h_1.fill(42)
    driver.mem_copy_h2d(dcube_hi_h_1, dcube.scratch_dcube())
    out_extra = {}
    dcube.evaluate(out_extra)
    driver.mem_copy_d2h(dcube.scratch_dcube(), dcube_hi_h_2)
    driver.mem_copy_d2h(dcube.dcube(), dcube_lo_h_2)

    assert np.array_equal(dcube_hi_h_1, dcube_hi_h_2)
    assert np.array_equal(dcube_lo_h_1, dcube_lo_h_2)

    #
    # Test psf
    #

    # psf_min_size =

    args |= dict(psf=psf, lsf=None)

    dcube = DCube(**args)
    dcube.prepare(driver, False)

    assert dcube.size() == size
    assert dcube.step() == step
    assert dcube.zero() == expected_zero
    assert dcube.scale() == scale
    assert dcube.dcube() is not None
    assert dcube.dcube().shape == expected_shape_lo
    assert dcube.wcube() is None
    assert dcube.mcube() is None

    expected_step_hi = tuple((np.array(step) / np.array(scale)).tolist())
    minimum_dcube_size = np.array(size) * np.array(scale)
    minimum_pcube_size = np.array(psf.size(expected_step_hi[:2]) + (1,))

    expected_dcube_size_hi, expected_edge_size_hi = (
        driver.backends().fft(dtype).fft_convolution_shape(
            minimum_dcube_size, minimum_pcube_size))

    assert dcube.scratch_size() == expected_dcube_size_hi

    dcube_lo_h_1 = driver.mem_alloc_h(expected_shape_lo, dtype)
    dcube_lo_h_2 = driver.mem_alloc_h(expected_shape_lo, dtype)
    dcube_hi_h_1 = driver.mem_alloc_h(expected_shape_hi, dtype)
    dcube_hi_h_2 = driver.mem_alloc_h(expected_shape_hi, dtype)
    dcube_lo_h_1.fill(0)
    dcube_hi_h_1.fill(0)

    # dcube_hi_h_1[expected_dcube_size_hi[1] * 15 + 10] = 1

    out_extra = {}
    dcube.evaluate(out_extra)

    #
    # Test lsf
    #

    args |= dict(psf=None, lsf=lsf)
    dcube = DCube(**args)
    dcube.prepare(driver, False)

    #
    # Test psf + lsf
    #

    args |= dict(psf=psf, lsf=lsf)
    dcube = DCube(**args)
    dcube.prepare(driver, True)

    scaled_size = (
        size[0] * scale[0],
        size[1] * scale[1],
        size[2] * scale[2])
    scaled_step = (
        step[0] / scale[0],
        step[1] / scale[1],
        step[2] / scale[2])
    scaled_step_spat = (scaled_step[0], scaled_step[1])
    scaled_step_spec = scaled_step[2]
    psf_size_hi = psf.size(scaled_step_spat)
    lsf_size_hi = lsf.size(scaled_step_spec)
    scratch_size, scratch_edge = (
        driver.backends().fft(dtype).fft_convolution_shape(
            scaled_size, psf_size_hi + (lsf_size_hi,)))

    assert dcube.size() == size
    assert dcube.step() == step
    assert dcube.zero() == expected_zero
    assert dcube.scale() == scale
    assert dcube.dcube() is not None
    assert dcube.dcube().shape == expected_shape_lo
    assert dcube.wcube() is not None
    assert dcube.wcube().shape == expected_shape_lo
    assert dcube.mcube() is None
    assert dcube.scratch_size() is not None
    assert dcube.scratch_step() is not None
    assert dcube.scratch_zero() is not None
    assert dcube.scratch_edge() is not None
    assert dcube.scratch_dcube() is not None
    assert dcube.scratch_wcube() is not None
    assert dcube.psf() is psf
    assert dcube.lsf() is lsf
    assert dcube.dtype() == np.float32


    # set_range(dcube.scratch_dcube(), 1, )


    # Hi res dcube and weights:
    # total sum must be equal to the value
    # max value at the center
    # check if result and psf are almost the same.
    # Low res dcube and weights:
    # max value at the center
    # check if result subset of low-res psf
    # total sum less or equal to the given value

    out_extra = {}
    dcube.evaluate(out_extra)




def test_dcube_host():
    try:
        import gbkfit.driver.drivers.host
    except Exception as e:
        pytest.skip(f"host driver not available; reason: {e}")
    from gbkfit.driver.drivers.host import DriverHost
    _test_dcube(DriverHost(), np.float32)


# def test_driver_cuda():
#     try:
#         import gbkfit.driver.drivers.cuda
#         import cupy
#         assert cupy.cuda.is_available()
#     except Exception as e:
#         pytest.skip(f"cuda driver not available; reason: {e}")
#     from gbkfit.driver.drivers.cuda import DriverCuda
#     _test_dcube(DriverCuda())


if __name__ == '__main__':
    # test_dmodel_image()
    test_dcube_host()
    # test_driver_cuda()
