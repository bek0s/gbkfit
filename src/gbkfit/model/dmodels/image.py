
import numpy as np

from gbkfit.dataset.datasets import DatasetImage
from gbkfit.model import DModel, GModelImageSupport
from . import _dcube, _detail


__all__ = ['DModelImage']


class DModelImage(DModel):

    @staticmethod
    def type():
        return 'image'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelImageSupport)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 2, dataset, DatasetImage)
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1), cval=(0, 0), rota=0,
            scale=(1, 1), psf=None,
            dtype=np.float32):
        super().__init__()
        size = tuple(size) + (1,)
        step = tuple(step) + (0,)
        cval = tuple(cval) + (0,)
        scale = tuple(scale) + (1,)
        self._dcube = _dcube.DCube(
            size, step, cval, rota, scale, psf, None, dtype)

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

    def rota(self):
        return self._dcube.rota()

    def scale(self):
        return self._dcube.scale()[:2]

    def psf(self):
        return self._dcube.psf()

    def dtype(self):
        return self._dcube.dtype()

    def onames(self):
        return ['image']

    def _submodel_impl(self, size, cval):
        return DModelImage(
            size, self.step(), cval, self.scale(),
            self.psf(), self.dtype())

    def _prepare_impl(self):
        self._dcube.prepare(self._driver)

    def _evaluate_impl(self, params, out_dextra, out_gextra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        driver.mem_fill(dcube.scratch_data(), 0)
        gmodel.evaluate_image(
            driver, params,
            dcube.scratch_data()[0, :, :],
            dcube.scratch_size()[:2],
            dcube.scratch_step()[:2],
            dcube.scratch_zero()[:2],
            dcube.rota(),
            dcube.dtype(),
            out_gextra)
        dcube.evaluate(out_dextra)
        return dict(image=dcube.data()[0, :, :])
