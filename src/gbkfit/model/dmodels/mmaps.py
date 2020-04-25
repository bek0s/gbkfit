
import numpy as np

import gbkfit.model.dmodel
import gbkfit.model.gmodel
import gbkfit.math
import gbkfit.psflsf
from gbkfit.utils import parseutils
from . import _dcube


class DModelMMaps(gbkfit.model.dmodel.DModel):

    @staticmethod
    def type():
        return 'mmaps'

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
            orders=self.orders(),
            dtype=self.dtype(),
            psf=gbkfit.psflsf.psf_parser.dump(self.psf()),
            lsf=gbkfit.psflsf.lsf_parser.dump(self.lsf()))

    def __init__(
            self, size, step=(1, 1), cval=(0, 0), scale=(1, 1), orders=(0, 1, 2),
            psf=None, lsf=None, dtype=np.float32):
        super().__init__()
        size = tuple(size)
        step = tuple(step)
        cval = tuple(cval)
        scale = tuple(scale)
        orders = tuple(sorted(set(orders)))
        if any(order < 0 or order > 2 for order in orders):
            raise RuntimeError("moment orders must be between 0 and 2")
        if len(step) == 2:
            step = step + (1,)
        if len(size) == 2:
            size = size + (int(gbkfit.math.roundu_odd(1000/step[2])),)
        if len(cval) == 2:
            cval = cval + (0,)
        if len(scale) == 2:
            scale = scale + (1,)
        self._orders = orders
        self._dcube = _dcube.DCube(size, step, cval, scale, psf, lsf, dtype)
        self._mmaps = None
        self._m_mmaps = None
        self._d_orders = None

    def size(self):
        return self._dcube.size()[:2]

    def step(self):
        return self._dcube.step()[:2]

    def cval(self):
        return self._dcube.cval()[:2]

    def scale(self):
        return self._dcube.scale()[:2]

    def scube_size(self):
        return self._dcube.size()

    def scube_step(self):
        return self._dcube.step()

    def scube_cval(self):
        return self._dcube.cval()

    def scube_scale(self):
        return self._dcube.scale()

    def orders(self):
        return self._orders

    def psf(self):
        return self._dcube.psf()

    def lsf(self):
        return self._dcube.lsf()

    def dtype(self):
        return self._dcube.dtype()

    def onames(self):
        return [f'mmap{i}' for i in self._orders]

    def _submodel_impl(self, size, cval):
        size = size + (self.scube_size()[2],)
        cval = cval + (self.scube_cval()[2],)
        return DModelMMaps(
            size, self.scube_step(), cval, self.scube_scale(),
            self.orders(), self.psf(), self.lsf(), self.dtype())

    def _prepare_impl(self):
        driver = self._driver
        self._dcube.prepare(driver)
        self._mmaps = driver.make_dmodel_mmaps(self.dtype())

        shape = (self.size() + (len(self.orders()),))[::-1]
        self._m_mmaps = driver.mem_alloc_s(shape, self.dtype())
        self._d_orders = driver.mem_copy_h2d(np.array(self.orders(), np.int32))

        self._mmaps.prepare(
            self._dcube.size()[:2],
            self._dcube.size()[2], self._dcube.step()[2], self._dcube.zero()[2],
            np.nan,
            self._dcube.data()[1], self._m_mmaps[1], self._d_orders)

    def _evaluate_impl(self, params, out_dextra, out_gextra):
        driver = self._driver
        dcube = self._dcube
        mmaps = self._m_mmaps
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
        self._mmaps.moments()
        output = {}
        for i, oname in enumerate(self.onames()):
            h_mmap = mmaps[0][i, :, :]
            d_mmap = mmaps[1][i, :, :]
            output[oname] = driver.mem_copy_d2h(d_mmap, h_mmap)
        return output