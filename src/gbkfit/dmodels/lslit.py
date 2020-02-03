
import numpy as np

import gbkfit.dmodel
import gbkfit.gmodel
import gbkfit.psflsf
from . import _dcube


class DModelLSlit(gbkfit.dmodel.DModel):

    @staticmethod
    def type():
        return 'lslit'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, gbkfit.gmodel.GModelSCubeSupport)

    @classmethod
    def load(cls, info):
        size = info['size']
        step = info.get('step')
        cval = info.get('cval')
        scale = info.get('scale')
        dtype = info.get('dtype')
        psf = gbkfit.psflsf.psf_parser.load(info.get('psf'))
        lsf = gbkfit.psflsf.lsf_parser.load(info.get('lsf'))
        return cls(size, step, cval, scale, psf, lsf, dtype)

    def dump(self):
        return {
            'type': self.type(),
            'size': self.size(),
            'step': self.step(),
            'cval': self.cval(),
            'scale': self.scale(),
            'dtype': self.dtype(),
            'psf': gbkfit.psflsf.psf_parser.dump(self.psf()),
            'lsf': gbkfit.psflsf.lsf_parser.dump(self.lsf())
        }

    def __init__(
            self, size, step, cval, scale,
            psf=None, lsf=None, dtype=None):
        super().__init__()
        if dtype is None:
            dtype = np.float32
        self._dcube = _dcube.DCube(size, step, cval, scale, psf, lsf, dtype)

    def size(self):
        size = self._dcube.size()
        return size[0], size[2]

    def step(self):
        step = self._dcube.step()
        return step[0], step[2]

    def cval(self):
        cval = self._dcube.cval()
        return cval[0], cval[2]

    def scale(self):
        scale = self._dcube.scale()
        return scale[0], scale[2]

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
        dcube = self._dcube
        lslit = dcube.data()[0][:, 0, :]
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
        return {'lslit': lslit}
