
import numpy as np

from gbkfit.dataset.datasets import DatasetImage
from gbkfit.model.core import DModel
from . import _dcube, _detail


__all__ = ['DModelImage']


class DModelImage(DModel):

    @staticmethod
    def type():
        return 'image'

    @staticmethod
    def is_compatible(gmodel):
        return hasattr(gmodel, 'evaluate_image')

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 2, dataset, DatasetImage)
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1), rpix=None, rval=(0, 0), rota=0,
            scale=(1, 1), psf=None,
            weights=False, weights_conv=False,
            dtype=np.float32):
        super().__init__()
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        size = tuple(size) + (1,)
        step = tuple(step) + (0,)
        rpix = tuple(rpix) + (0,)
        rval = tuple(rval) + (0,)
        scale = tuple(scale) + (1,)
        self._dcube = _dcube.DCube(
            size, step, rpix, rval, rota, scale, psf, None,
            weights, weights_conv, dtype)

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

    def psf(self):
        return self._dcube.psf()

    def dtype(self):
        return self._dcube.dtype()

    def onames(self):
        return ['image']

    def _prepare_impl(self):
        self._dcube.prepare(self._driver)

    def _evaluate_impl(self, params, out_dextra, out_gextra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        driver.mem_fill(dcube.scratch_dcube(), 0)
        gmodel.evaluate_image(
            driver, params,
            dcube.scratch_dcube()[0, :, :],
            dcube.scratch_wcube()[0, :, :] if dcube.weights() else None,
            dcube.scratch_size()[:2],
            dcube.scratch_step()[:2],
            dcube.scratch_zero()[:2],
            dcube.rota(),
            dcube.dtype(),
            out_gextra)
        dcube.evaluate(out_dextra)
        return dict(image=dict(
            d=dcube.dcube()[0, :, :],
            m=dcube.mcube()[0, :, :],
            w=dcube.wcube()[0, :, :] if dcube.weights() else None))
