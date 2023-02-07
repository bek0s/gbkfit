
from collections.abc import Sequence

import numpy as np

from gbkfit.dataset.datasets import DatasetImage
from gbkfit.model.core import DModel, GModelImage
from gbkfit.psflsf import PSF
from . import _dcube, _detail


__all__ = [
    'DModelImage'
]


class DModelImage(DModel):

    @staticmethod
    def type():
        return 'image'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelImage)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(
            cls, info, 2, True, False, dataset, DatasetImage)
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self,
            size: Sequence[int],
            step: Sequence[int | float] = (1, 1),
            rpix: Sequence[int | float] | None = None,
            rval: Sequence[int | float] = (0, 0),
            rota: int | float = 0,
            scale: Sequence[int] = (1, 1),
            psf: PSF | None = None,
            weight: int | float = 1,
            mask_cutoff: int | float | None = None,
            mask_create: bool = False,
            mask_apply: bool = False,
            dtype=np.float32
    ):
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
            weight, mask_cutoff, mask_create, mask_apply, dtype)

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

    def _prepare_impl(self, gmodel):
        self._dcube.prepare(self._driver, gmodel.is_weighted())

    def _evaluate_impl(self, params, out_dmodel_extra, out_gmodel_extra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        has_mcube = dcube.mcube() is not None
        has_wcube = dcube.wcube() is not None
        # Clear DCube arrays
        # todo: investigate if this step can be skipped
        driver.mem_fill(dcube.scratch_dcube(), 0)
        # Evaluate gmodel on DCube's arrays
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
        # Evaluate DCube (perform convolution, supersampling, etc)
        dcube.evaluate(out_dmodel_extra)
        # Model evaluation complete.
        # Return data, mask, and weight arrays (if available)
        return dict(image=dict(
            d=dcube.dcube()[0, :, :],
            m=dcube.mcube()[0, :, :] if has_mcube else None,
            w=dcube.wcube()[0, :, :] if has_wcube else None))
