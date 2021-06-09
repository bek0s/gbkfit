
import numpy as np

from gbkfit.dataset.datasets import DatasetLSlit
from gbkfit.model.core import DModel
from . import _dcube, _detail


__all__ = ['DModelLSlit']


class DModelLSlit(DModel):

    @staticmethod
    def type():
        return 'lslit'

    @staticmethod
    def is_compatible(gmodel):
        return hasattr(gmodel, 'evaluate_scube')

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 3, dataset, DatasetLSlit)
        opts.update(size=opts['size'][:2])
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1, 1), rpix=None, rval=(0, 0, 0), rota=0,
            scale=(1, 1, 1), psf=None, lsf=None,
            weights=False, weights_conv=False,
            dtype=np.float32):
        super().__init__()
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        size = tuple([size[0], 1, size[1]])
        step = tuple(step)
        rpix = tuple(rpix)
        rval = tuple(rval)
        scale = tuple(scale)
        self._dcube = _dcube.DCube(
            size, step, rpix, rval, rota, scale, psf, lsf,
            weights, weights_conv, dtype)

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
        return ['lslit']

    def _prepare_impl(self):
        self._dcube.prepare(self._driver)

    def _evaluate_impl(self, params, out_dextra, out_gextra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        driver.mem_fill(dcube.scratch_dcube(), 0)
        gmodel.evaluate_scube(
            driver, params,
            dcube.scratch_dcube(),
            dcube.scratch_wcube() if dcube.weights() else None,
            dcube.scratch_size(),
            dcube.scratch_step(),
            dcube.scratch_zero(),
            dcube.rota(),
            dcube.dtype(),
            out_gextra)
        dcube.evaluate(out_dextra)
        return dict(lslit=dict(
            d=dcube.dcube()[0][:, 0, :],
            m=dcube.mcube()[0][:, 0, :],
            w=dcube.mcube()[0][:, 0, :] if dcube.weights() else None))
