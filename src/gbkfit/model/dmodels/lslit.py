
import numpy as np

from gbkfit.dataset.datasets import DatasetLSlit
from gbkfit.model import DModel, GModelSCubeSupport
from . import _dcube, _detail


__all__ = ['DModelLSlit']


class DModelLSlit(DModel):

    @staticmethod
    def type():
        return 'lslit'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCubeSupport)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 3, dataset, DatasetLSlit)
        opts.update(size=opts['size'][:2])
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1, 1), cval=(0, 0, 0), rota=0,
            scale=(1, 1, 1), psf=None, lsf=None,
            dtype=np.float32):
        super().__init__()
        size = tuple([size[0], 1, size[1]])
        step = tuple(step)
        cval = tuple(cval)
        scale = tuple(scale)
        self._dcube = _dcube.DCube(
            size, step, cval, rota, scale, psf, lsf, dtype)

    def size(self):
        return self._dcube.size()

    def step(self):
        return self._dcube.step()

    def cpix(self):
        return self._dcube.cpix()

    def cval(self):
        return self._dcube.cval()

    def zero(self):
        return self._dcube.zero()

    def scale(self):
        return self._dcube.scale()

    def psf(self):
        return self._dcube.psf()

    def lsf(self):
        return self._dcube.lsf()

    def dtype(self):
        return self._dcube.dtype()

    def onames(self):
        return ['lslit']

    def _submodel_impl(self, size, cval):
        return DModelLSlit(
            size, self.step(), cval, self.scale(),
            self.psf(), self.lsf(), self.dtype())

    def _prepare_impl(self):
        self._dcube.prepare(self._driver)

    def _evaluate_impl(self, params, out_dextra, out_gextra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
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
        return dict(lslit=dcube.data()[0][:, 0, :])
