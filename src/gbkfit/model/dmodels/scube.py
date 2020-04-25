
import numpy as np

import gbkfit.model.dmodel
import gbkfit.model.gmodel
import gbkfit.psflsf
from gbkfit.utils import parseutils
from . import _dcube


class DModelSCube(gbkfit.model.dmodel.DModel):

    @staticmethod
    def type():
        return 'scube'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, gbkfit.model.gmodel.GModelSCubeSupport)

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
            psf=gbkfit.psflsf.psf_parser.load(info.get('psf')),
            lsf=gbkfit.psflsf.lsf_parser.load(info.get('lsf')))
        return cls(**args)

    def dump(self):
        return dict(
            type=self.type(),
            size=self.size(),
            step=self.step(),
            cval=self.cval(),
            scale=self.scale(),
            dtype=self.dtype(),
            psf=gbkfit.psflsf.psf_parser.dump(self.psf()),
            lsf=gbkfit.psflsf.lsf_parser.dump(self.lsf()))

    def __init__(
            self, size, step=(1, 1, 1), cval=(0, 0, 0), scale=(1, 1, 1),
            psf=None, lsf=None, dtype=np.float32):
        super().__init__()
        size = tuple(size)
        step = tuple(step)
        cval = tuple(cval)
        scale = tuple(scale)
        self._dcube = _dcube.DCube(size, step, cval, scale, psf, lsf, dtype)

    def size(self):
        return self._dcube.size()

    def step(self):
        return self._dcube.step()

    def cval(self):
        return self._dcube.cval()

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

    def _submodel_impl(self, size, cval):
        return DModelSCube(
            size, self.step(), cval, self.scale(),
            self.psf(), self.lsf(), self.dtype())

    def _prepare_impl(self):
        self._dcube.prepare(self._driver)

    def _evaluate_impl(self, params, out_dextra, out_gextra):
        driver = self._driver
        dcube = self._dcube
        scube = dcube.data()
        driver.mem_fill_d(dcube.scratch_data(), 0)
        self._gmodel.evaluate_scube(
            driver, params,
            dcube.scratch_data(),
            dcube.dtype(),
            dcube.scratch_size(),
            dcube.scratch_step(),
            dcube.scratch_zero(),
            out_gextra)
        dcube.evaluate(out_dextra)
        return {'scube': scube}
