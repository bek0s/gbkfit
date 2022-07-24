
import logging

import numpy as np

import gbkfit.math
from gbkfit.dataset.datasets import DatasetMMaps
from gbkfit.model.core import DModel, GModelSCube
from gbkfit.utils import parseutils
from . import _dcube, _detail


__all__ = ['DModelMMaps']


_log = logging.getLogger(__name__)


class DModelMMaps(DModel):

    @staticmethod
    def type():
        return 'mmaps'

    @staticmethod
    def is_compatible(gmodel):
        return isinstance(gmodel, GModelSCube)

    @classmethod
    def load(cls, info, dataset=None):
        opts = _detail.load_dmodel_common(cls, info, 3, dataset, DatasetMMaps)
        return cls(**opts)

    def dump(self):
        return _detail.dump_dmodel_common(self)

    def __init__(
            self, size, step=(1, 1), rpix=None, rval=(0, 0), rota=0,
            scale=(1, 1), psf=None, lsf=None,
            mask_cutoff=1e-6,
            orders=(0, 1, 2),
            dtype=np.float32):
        super().__init__()
        if rpix is None:
            rpix = tuple((np.array(size) / 2 - 0.5).tolist())
        size = tuple(size)
        step = tuple(step)
        rpix = tuple(rpix)
        rval = tuple(rval)
        scale = tuple(scale)
        orders = tuple(sorted(set(orders)))
        if any(order < 0 or order > 7 for order in orders):
            raise RuntimeError("moment orders must be between 0 and 7")
        if len(step) == 2:
            step = step + (1,)
        if len(size) == 2:
            size = size + (int(gbkfit.math.roundu_odd(1000/step[2])),)
        if len(rpix) == 2:
            rpix = rpix + (size[2] / 2 - 0.5,)
        if len(rval) == 2:
            rval = rval + (0,)
        if len(scale) == 2:
            scale = scale + (1,)
        if mask_cutoff is None:
            desc = parseutils.make_typed_desc(self.__class__, 'dmodel')
            raise RuntimeError(
                f"masking cannot be disabled for {desc}; "
                f"set the mask_cutoff to a value greater or equal to 0")
        if (psf or lsf) and mask_cutoff == 0:
            _log.warning(
                "mask_cutoff is set to 0 either by default or by choice; "
                "fft-based convolution will be performed on the model "
                "because psf and/or lsf were provided; "
                "fft-based convolution can result in noise in the model; "
                "extracting moments from noisy low-SNR spectra can be "
                "erroneous and generate artefacts on the resulting maps; "
                "because of this it is highly recommended to enable masking "
                "by providing a value for mask_cutoff greater than 0")
        self._orders = orders
        self._dcube = _dcube.DCube(
            size, step, rpix, rval, rota, scale, psf, lsf,
            False, None, False, False, dtype)
        self._mmaps = None
        self._mmaps_o = None
        self._mmaps_d = None
        self._mmaps_m = None
        self._mmaps_w = None
        self._mask_cutoff = mask_cutoff

    def keys(self):
        return tuple([f'mmap{i}' for i in self._orders])

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

    def orders(self):
        return self._orders

    def psf(self):
        return self._dcube.psf()

    def lsf(self):
        return self._dcube.lsf()

    def dtype(self):
        return self._dcube.dtype()

    def _prepare_impl(self):
        # Some shortcuts
        driver = self._driver
        dtype = self.dtype()
        orders = self.orders()
        # Calculate data sizes
        mmaps_size_all = self.size() + (len(orders),)
        mmaps_size_one = self.size()
        # Allocate memory
        self._mmaps_o = driver.mem_alloc_d(len(orders), np.int32)
        self._mmaps_d = driver.mem_alloc_d(mmaps_size_all[::-1], dtype)
        self._mmaps_m = driver.mem_alloc_d(mmaps_size_one[::-1], dtype)
        self._mmaps_w = driver.mem_alloc_d(mmaps_size_one[::-1], dtype)
        # Initialize memory
        driver.mem_copy_h2d(np.array(orders, dtype=np.int32), self._mmaps_o)
        driver.mem_fill(self._mmaps_d, np.nan)
        driver.mem_fill(self._mmaps_m, 0)
        driver.mem_fill(self._mmaps_w, np.nan)
        # Prepare dcube
        self._dcube.prepare(driver, self._gmodel.is_weighted())
        # Create backend
        self._backend = driver.backends().dmodel(dtype)

    def _evaluate_impl(self, params, out_dmodel_extra, out_gmodel_extra):
        driver = self._driver
        gmodel = self._gmodel
        dcube = self._dcube
        backend = self._backend
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
        backend.moments(
            dcube.size(),
            dcube.step(),
            dcube.zero(),
            dcube.dcube(),
            dcube.wcube(),
            self._mask_cutoff,
            self._mmaps_o,
            self._mmaps_d,
            self._mmaps_w,
            self._mmaps_m)
        out = dict()
        for i, key in enumerate(self.keys()):
            out[key] = dict(
                d=self._mmaps_d[i, :, :],
                m=self._mmaps_m,
                w=self._mmaps_w)
        return out
