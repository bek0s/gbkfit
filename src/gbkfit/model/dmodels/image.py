
import numpy as np

from gbkfit.dataset.datasets import DatasetImage
from gbkfit.model.core import DModel, GModelImage
from . import _dcube, _detail


__all__ = ['DModelImage']


class DModelImage(DModel):

    @staticmethod
    def type():
        return 'image'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelImage)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 2, dataset, DatasetImage)
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1), rpix=None, rval=(0, 0), rota=0,
            scale=(1, 1), psf=None, weights=False,
            mask_cutoff=None, mask_create=False, mask_apply=False,
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
            weights, mask_cutoff, mask_create, mask_apply, dtype)

    def keys(self):
        return ['image']

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

    def _prepare_impl(self):
        self._dcube.prepare(self._driver, self._gmodel.is_weighted())

    def _evaluate_impl(self, params, out_dmodel_extra, out_gmodel_extra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        has_mcube = dcube.mcube() is not None
        has_wcube = dcube.wcube() is not None
        driver.mem_fill(dcube.scratch_dcube(), 0)
        gmodel.evaluate_image(
            driver, params,
            dcube.scratch_dcube(),
            dcube.scratch_wcube(),
            dcube.scratch_size()[:2],
            dcube.scratch_step()[:2],
            dcube.scratch_zero()[:2],
            dcube.rota(),
            dcube.dtype(),
            out_gmodel_extra)
        dcube.evaluate(out_dmodel_extra)
        return dict(image=dict(
            d=dcube.dcube()[0, :, :],
            m=dcube.mcube()[0, :, :] if has_mcube else None,
            w=dcube.wcube()[0, :, :] if has_wcube else None))
