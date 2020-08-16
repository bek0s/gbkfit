
import numpy as np

import gbkfit.math
from gbkfit.dataset.datasets import DatasetMMaps
from gbkfit.model import DModel, GModelSCubeSupport
from . import _dcube, _detail


__all__ = ['DModelMMaps']


class DModelMMaps(DModel):

    @staticmethod
    def desc():
        return 'moment maps'

    @staticmethod
    def type():
        return 'mmaps'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCubeSupport)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 3, dataset, DatasetMMaps)
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1), cval=(0, 0), rota=0,
            scale=(1, 1), psf=None, lsf=None, orders=(0, 1, 2),
            dtype=np.float32):
        super().__init__()
        size = tuple(size)
        step = tuple(step)
        cval = tuple(cval)
        scale = tuple(scale)
        orders = tuple(sorted(set(orders)))
        if any(order < 0 or order > 2 for order in orders):
            raise RuntimeError("moment orders must be between 0 and 2")
        if len(step) == 2:
            step = step + (1,)
        if len(size) == 2:
            size = size + (int(gbkfit.math.roundu_odd(1000/step[2])),)
        if len(cval) == 2:
            cval = cval + (0,)
        if len(scale) == 2:
            scale = scale + (1,)
        self._orders = orders
        self._dcube = _dcube.DCube(
            size, step, cval, rota, scale, psf, lsf, dtype)
        self._mmaps = None
        self._s_mmap_data = None
        self._d_mmap_order = None

    def size(self):
        return self._dcube.size()[:2]

    def step(self):
        return self._dcube.step()[:2]

    def cpix(self):
        return self._dcube.cpix()[:2]

    def cval(self):
        return self._dcube.cval()[:2]

    def zero(self):
        return self._dcube.zero()[:2]

    def scale(self):
        return self._dcube.scale()[:2]

    def orders(self):
        return self._orders

    def psf(self):
        return self._dcube.psf()

    def lsf(self):
        return self._dcube.lsf()

    def dtype(self):
        return self._dcube.dtype()

    def onames(self):
        return tuple([f'mmap{i}' for i in self._orders])

    def _submodel_impl(self, size, cval):
        size = size + (self._dcube.size()[2],)
        cval = cval + (self._dcube.cval()[2],)
        return DModelMMaps(
            size, self._dcube.step(), cval, self._dcube.scale(),
            self.orders(), self.psf(), self.lsf(), self.dtype())

    def _prepare_impl(self):
        # Allocate buffers for moment map data
        shape = (self.size() + (len(self.orders()),))[::-1]
        self._s_mmap_data = self._driver.mem_alloc_s(shape, self.dtype())
        self._d_mmap_order = self._driver.mem_copy_h2d(
            np.array(self.orders(), np.int32))
        # Prepare dcube
        self._dcube.prepare(self._driver)
        # Create and prepare moment map backend
        # Preparation must be done after dcube preparation
        self._mmaps = self._driver.make_dmodel_mmaps(self.dtype())
        self._mmaps.prepare(
            self._dcube.size()[:2],
            self._dcube.size()[2],
            self._dcube.step()[2],
            self._dcube.zero()[2],
            self._dcube.data(),
            self._s_mmap_data[1],
            self._d_mmap_order)

    def _evaluate_impl(self, params, out_dextra, out_gextra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        mmaps = self._mmaps
        driver.mem_fill(dcube.scratch_data(), 0)
        gmodel.evaluate_scube(
            driver, params,
            dcube.scratch_data(),
            dcube.scratch_size(),
            dcube.scratch_step(),
            dcube.scratch_zero(),
            dcube.rota(),
            dcube.dtype(),
            out_gextra)
        dcube.evaluate(out_dextra)
        mmaps.moments()
        out = dict()
        for i, oname in enumerate(self.onames()):
            out[oname] = driver.mem_copy_d2h(
                self._s_mmap_data[0][i, :, :],
                self._s_mmap_data[1][i, :, :])
        return out
