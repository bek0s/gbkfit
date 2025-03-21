from abc import ABC

import numpy as np

from gbkfit.driver import Driver
from gbkfit.model.dmodels import DModelImage

from gbkfit.model import GModelImage


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

    def is_weighted(self):
        return False

    def evaluate_image(
            self, driver, params, image, weights, size, step, zero, rota,
            dtype, out_extra):
        driver.mem_copy_h2d(create_mock_image(driver, size, dtype), image)
        print(image)


def test_dmodel_image():

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


def test_dcube():

    from gbkfit.model.dmodels._dcube import DCube
    from gbkfit.psflsf.psfs import PSFGauss

    size = (51, 51, 1)
    step = (1.0, 1.0, 1.0)
    rpix = (0.0, 0.0, 0.0)
    rval = (0.0, 0.0, 0.0)
    rota = 0.0
    scale = (1, 1, 1)
    psf = PSFGauss(5)
    lsf = None
    weight = None
    mask_cutoff = None
    mask_create = None
    mask_apply = None
    dtype = np.float32



    dcube = DCube(size, step, rpix, rval, rota, scale, psf, lsf, weight, mask_cutoff, mask_create, mask_apply, dtype)

    pass


if __name__ == '__main__':
    # test_dmodel_image()
    test_dcube()
