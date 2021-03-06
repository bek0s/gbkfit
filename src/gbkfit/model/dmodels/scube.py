
import numpy as np

from gbkfit.dataset.datasets import DatasetSCube
from gbkfit.model.core import DModel, GModelSCube
from . import _dcube, _detail


__all__ = ['DModelSCube']


class DModelSCube(DModel):

    @staticmethod
    def type():
        return 'scube'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 3, dataset, DatasetSCube)
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1, 1), rpix=None, rval=(0, 0, 0), rota=0,
            scale=(1, 1, 1), psf=None, lsf=None,
            dtype=np.float32):
        super().__init__()
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        size = tuple(size)
        step = tuple(step)
        rpix = tuple(rpix)
        rval = tuple(rval)
        scale = tuple(scale)
        self._dcube = _dcube.DCube(
            size, step, rpix, rval, rota, scale, psf, lsf, dtype)

    def size(self):
        return self._dcube.size()

    def step(self):
        return self._dcube.step()

    def zero(self):
        return self._dcube.zero()

    def rota(self):
        return self._dcube.rota()

    def scale(self):
        return self._dcube.scale()

    def psf(self):
        return self._dcube.psf()

    def lsf(self):
        return self._dcube.lsf()

    def dtype(self):
        return self._dcube.dtype()

    def onames(self):
        return ['scube']

    def _prepare_impl(self):
        self._dcube.prepare(self._driver)
        self._mask = self._driver.mem_alloc_d(self.size(), self.dtype())

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

        return dict(scube=dict(data=dcube.data(), mask=dcube.mask()))
