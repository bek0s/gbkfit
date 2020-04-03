
import numpy as np

import gbkfit.model.dmodel
import gbkfit.model.gmodel
import gbkfit.psflsf
from gbkfit.utils import parseutils
from . import _dcube


class DModelImage(gbkfit.model.dmodel.DModel):

    @staticmethod
    def type():
        return 'image'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, gbkfit.model.gmodel.GModelImageSupport)

    @classmethod
    def load(cls, info, *args, **kwargs):
        dataset = kwargs.get('dataset')
        if dataset is not None:
            info.update(dict(
                size=dataset.size(),
                step=info.get('step', dataset.step()),
                cval=info.get('cval', dataset.cval())))
        args = parseutils.parse_class_args(cls, info)
        args.update(
            psf=gbkfit.psflsf.psf_parser.load(info.get('psf')))
        return cls(**args)

    def dump(self):
        return dict(
            type=self.type(),
            size=self.size(),
            step=self.step(),
            cval=self.cval(),
            scale=self.scale(),
            dtype=self.dtype(),
            psf=gbkfit.psflsf.psf_parser.dump(self.psf()))

    def __init__(
            self, size, step=(1, 1), cval=(0, 0), scale=(1, 1),
            psf=None, dtype=np.float32):
        super().__init__()
        size = tuple(size) + (1,)
        step = tuple(step) + (0,)
        cval = tuple(cval) + (0,)
        scale = tuple(scale) + (1,)
        self._dcube = _dcube.DCube(size, step, cval, scale, psf, None, dtype)

    def size(self):
        return self._dcube.size()[:2]

    def step(self):
        return self._dcube.step()[:2]

    def cval(self):
        return self._dcube.cval()[:2]

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
        image = dcube.data()[0][0, :, :]
        driver.mem_fill_d(dcube.scratch_data(), 0)
        gmodel.evaluate_image(
            driver, params,
            dcube.scratch_data()[0, :, :],
            dcube.dtype(),
            dcube.scratch_size()[:2],
            dcube.scratch_step()[:2],
            dcube.scratch_zero()[:2],
            out_gextra)
        dcube.evaluate(out_dextra)
        return {'image': image}
