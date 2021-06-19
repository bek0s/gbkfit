
import numpy as np

import gbkfit.math
from gbkfit.dataset.datasets import DatasetMMaps
from gbkfit.model.core import DModel, GModelSCube
from . import _dcube, _detail


__all__ = ['DModelMMaps']


class DModelMMaps(DModel):

    @staticmethod
    def type():
        return 'mmaps'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 3, dataset, DatasetMMaps)
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1), rpix=None, rval=(0, 0), rota=0,
            scale=(1, 1), psf=None, lsf=None,
            weights=False, weights_conv=False,
            orders=(0, 1, 2),
            dtype=np.float32):
        super().__init__()
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        size = tuple(size)
        step = tuple(step)
        rpix = tuple(rpix)
        rval = tuple(rval)
        scale = tuple(scale)
        orders = tuple(sorted(set(orders)))
        if any(order < 0 or order > 7 for order in orders):
            raise RuntimeError("moment orders must be between 0 and 7")
        if len(step) == 2:
            step = step + (1,)
        if len(size) == 2:
            size = size + (int(gbkfit.math.roundu_odd(1000/step[2])),)
        if len(rpix) == 2:
            rpix = rpix + (size[2] / 2 - 0.5,)
        if len(rval) == 2:
            rval = rval + (0,)
        if len(scale) == 2:
            scale = scale + (1,)
        self._orders = orders
        self._dcube = _dcube.DCube(
            size, step, rpix, rval, rota, scale, psf, lsf,
            weights, weights_conv, dtype)
        self._mmaps = None
        self._s_mmap_data = None
        self._s_mmap_mask = None
        self._d_mmap_order = None

    def size(self):
        return self._dcube.size()[:2]

    def step(self):
        return self._dcube.step()[:2]

    def zero(self):
        return self._dcube.zero()[:2]

    def rota(self):
        return self._dcube.rota()

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

    def _prepare_impl(self):
        # Allocate buffers for moment map data
        shape = (self.size() + (len(self.orders()),))[::-1]
        self._s_mmap_data = self._driver.mem_alloc_d(shape, self.dtype())
        self._s_mmap_mask = self._driver.mem_alloc_d(self.size()[::-1], self.dtype())
        self._driver.mem_fill(self._s_mmap_data, np.nan)
        self._d_mmap_order = self._driver.mem_copy_h2d(
            np.array(self.orders(), np.int32))
        # Prepare dcube
        self._dcube.prepare(self._driver)
        # Create moment maps backend
        self._mmaps = self._driver.make_backend_dmodel_mmaps(self.dtype())

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
        mmaps.moments(
            self._dcube.size(),
            self._dcube.step(),
            self._dcube.zero(),
            self._dcube.data(),
            self._s_mmap_data,
            self._s_mmap_mask,
            self._d_mmap_order)
        out = dict()
        for i, oname in enumerate(self.onames()):
            out[oname] = dict(
                data=self._s_mmap_data[i, :, :],
                mask=self._s_mmap_mask)
        return out
