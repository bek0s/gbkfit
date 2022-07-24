
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
            scale=(1, 1, 1), psf=None, lsf=None, weights=False,
            mask_cutoff=None, mask_create=False, mask_apply=False,
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
            size, step, rpix, rval, rota, scale, psf, lsf,
            weights, mask_cutoff, mask_create, mask_apply, dtype)

    def keys(self):
        return ['scube']

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

    def _prepare_impl(self):
        self._dcube.prepare(self._driver, self._gmodel.is_weighted())

    def _evaluate_impl(self, params, out_dmodel_extra, out_gmodel_extra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        has_mcube = dcube.mcube() is not None
        has_wcube = dcube.wcube() is not None
        driver.mem_fill(dcube.scratch_dcube(), 0)
        gmodel.evaluate_scube(
            driver, params,
            dcube.scratch_dcube(),
            dcube.scratch_wcube(),
            dcube.scratch_size(),
            dcube.scratch_step(),
            dcube.scratch_zero(),
            dcube.rota(),
            dcube.dtype(),
            out_gmodel_extra)
        dcube.evaluate(out_dmodel_extra)
        return dict(scube=dict(
            d=dcube.dcube(),
            m=dcube.mcube() if has_mcube else None,
            w=dcube.wcube() if has_wcube else None))
