
import cupy as cp
import numpy as np

import gbkfit.driver
from gbkfit.driver import _detail

import gbkfit.native.libgbkfit_cuda


def _ptr(a):
    return a.__cuda_array_interface__['data'][0] if a is not None else 0


def _shape(a):
    return a.__cuda_array_interface__['shape'] if a is not None else (0,)


class DriverCUDA(gbkfit.driver.Driver):

    @staticmethod
    def type():
        return 'cuda'

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return dict(type=self.type())

    def mem_alloc_s(self, shape, dtype):
        h_data = np.empty(shape, dtype)
        d_data = cp.empty(shape, dtype)
        return h_data, d_data

    def mem_alloc_h(self, shape, dtype):
        return np.empty(shape, dtype)

    def mem_alloc_d(self, shape, dtype):
        return cp.empty(shape, dtype)

    def mem_copy_h2d(self, h_src, d_dst=None):
        if d_dst is None:
            d_dst = self.mem_alloc_d(h_src.shape, h_src.dtype)
        if h_src is not d_dst:
            d_dst.set(h_src)
        return d_dst

    def mem_copy_d2h(self, d_src, h_dst=None):
        if h_dst is None:
            h_dst = self.mem_alloc_h(d_src.shape, d_src.dtype)
        if d_src is not h_dst:
            h_dst[:] = d_src.get()
        return h_dst

    def mem_fill(self, x, value):
        x.fill(value)

    def math_abs(self, x, out=None):
        return np.abs(x, out=out)

    def math_sum(self, x, out=None):
        return np.sum(x, out=out, keepdims=True)

    def math_add(self, x1, x2, out=None):
        return np.add(x1, x2, out=out)

    def math_sub(self, x1, x2, out=None):
        return np.sub(x1, x2, out=out)

    def math_mul(self, x1, x2, out=None):
        return np.mul(x1, x2, out=out)

    def math_div(self, x1, x2, out=None):
        return np.div(x1, x2, out=out)

    def make_dmodel_dcube(self, dtype):
        return DModelDCube(dtype)

    def make_dmodel_mmaps(self, dtype):
        return DModelMMaps(dtype)

    def make_gmodel_mcdisk(self, dtype):
        return GModelMCDisk(dtype)

    def make_gmodel_smdisk(self, dtype):
        return GModelSMDisk(dtype)


class DModelDCube(gbkfit.driver.driver.DModelDCube):

    _CLASSES = {
        np.float32: gbkfit.native.libgbkfit_cuda.DModelDCubef32}

    def __deepcopy__(self, memodict):
        return DModelDCube(self._dtype)

    def __init__(self, dtype):
        _detail.check_dtype(self._CLASSES, dtype)
        self._dcube = self._CLASSES[dtype]()
        self._dtype = dtype

    def convolve(
            self,
            size_hi,
            scube_hi, scube_hi_fft,
            psf3d_hi, psf3d_hi_fft):
        self._dcube.convolve(
            size_hi,
            _ptr(scube_hi), _ptr(scube_hi_fft),
            _ptr(psf3d_hi), _ptr(psf3d_hi_fft))

    def downscale(
            self,
            scale,
            edge_hi,
            size_hi, size_lo,
            scube_hi, scube_lo):
        self._dcube.downscale(
            scale,
            edge_hi,
            size_hi, size_lo,
            _ptr(scube_hi), _ptr(scube_lo))

    def make_mask(self, size, data, mask):
        self._dcube.make_mask(size, _ptr(data), _ptr(mask))

    def apply_mask(self, size, data, mask):
        self._dcube.apply_mask(size, _ptr(data), _ptr(mask))


class DModelMMaps(gbkfit.driver.driver.DModelMMaps):

    _CLASSES = {
        np.float32: gbkfit.native.libgbkfit_cuda.DModelMMapsf32}

    def __deepcopy__(self, memodict):
        return DModelMMaps(self._dtype)

    def __init__(self, dtype):
        _detail.check_dtype(self._CLASSES, dtype)
        self._mmaps = self._CLASSES[dtype]()
        self._dtype = dtype

    def prepare(
            self,
            spat_size,
            spec_size, spec_step, spec_zero,
            scube,
            mmaps,
            mmaps_orders):
        self._mmaps.prepare(
            spat_size[0], spat_size[1],
            spec_size, spec_step, spec_zero,
            np.nan,
            _ptr(scube),
            _ptr(mmaps),
            _ptr(mmaps_orders),
            _shape(mmaps_orders)[0])

    def moments(self):
        self._mmaps.moments()


class GModelMCDisk(gbkfit.driver.driver.GModelMCDisk):

    _CLASSES = {
        np.float32: gbkfit.native.libgbkfit_cuda.GModelMCDiskf32}

    def __deepcopy__(self, memodict):
        return GModelMCDisk(self._dtype)

    def __init__(self, dtype):
        _detail.check_dtype(self._CLASSES, dtype)
        self._disk = self._CLASSES[dtype]()
        self._dtype = dtype

    def evaluate(
            self,
            cflux, nclouds, ncloudspt, hasordint,
            loose, tilted, rnodes,
            vsys, xpos, ypos, posa, incl,
            rpt_uids, rpt_cvalues, rpt_ccounts, rpt_pvalues, rpt_pcounts,
            rht_uids, rht_cvalues, rht_ccounts, rht_pvalues, rht_pcounts,
            vpt_uids, vpt_cvalues, vpt_ccounts, vpt_pvalues, vpt_pcounts,
            vht_uids, vht_cvalues, vht_ccounts, vht_pvalues, vht_pcounts,
            dpt_uids, dpt_cvalues, dpt_ccounts, dpt_pvalues, dpt_pcounts,
            dht_uids, dht_cvalues, dht_ccounts, dht_pvalues, dht_pcounts,
            wpt_uids, wpt_cvalues, wpt_ccounts, wpt_pvalues, wpt_pcounts,
            spt_uids, spt_cvalues, spt_ccounts, spt_pvalues, spt_pcounts,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube, rcube,
            rdata, vdata, ddata):
        self._disk.evaluate(
            cflux, nclouds, _ptr(ncloudspt), _shape(ncloudspt)[0], _ptr(hasordint),
            loose, tilted,
            _shape(rnodes)[0],
            _ptr(rnodes),
            _ptr(vsys), _ptr(xpos), _ptr(ypos), _ptr(posa), _ptr(incl),
            _shape(rpt_uids)[0],
            _ptr(rpt_uids),
            _ptr(rpt_cvalues), _ptr(rpt_ccounts),
            _ptr(rpt_pvalues), _ptr(rpt_pcounts),
            _ptr(rht_uids),
            _ptr(rht_cvalues), _ptr(rht_ccounts),
            _ptr(rht_pvalues), _ptr(rht_pcounts),
            _shape(vpt_uids)[0],
            _ptr(vpt_uids),
            _ptr(vpt_cvalues), _ptr(vpt_ccounts),
            _ptr(vpt_pvalues), _ptr(vpt_pcounts),
            _ptr(vht_uids),
            _ptr(vht_cvalues), _ptr(vht_ccounts),
            _ptr(vht_pvalues), _ptr(vht_pcounts),
            _shape(dpt_uids)[0],
            _ptr(dpt_uids),
            _ptr(dpt_cvalues), _ptr(dpt_ccounts),
            _ptr(dpt_pvalues), _ptr(dpt_pcounts),
            _ptr(dht_uids),
            _ptr(dht_cvalues), _ptr(dht_ccounts),
            _ptr(dht_pvalues), _ptr(dht_pcounts),
            _shape(wpt_uids)[0],
            _ptr(wpt_uids),
            _ptr(wpt_cvalues), _ptr(wpt_ccounts),
            _ptr(wpt_pvalues), _ptr(wpt_pcounts),
            _shape(spt_uids)[0],
            _ptr(spt_uids),
            _ptr(spt_cvalues), _ptr(spt_ccounts),
            _ptr(spt_pvalues), _ptr(spt_pcounts),
            spat_size[0], spat_size[1], spat_size[2],
            spat_step[0], spat_step[1], spat_step[2],
            spat_zero[0], spat_zero[1], spat_zero[2],
            spec_size,
            spec_step,
            spec_zero,
            _ptr(image), _ptr(scube), _ptr(rcube),
            _ptr(rdata), _ptr(vdata), _ptr(ddata))


class GModelSMDisk(gbkfit.driver.driver.GModelSMDisk):

    _CLASSES = {
        np.float32: gbkfit.native.libgbkfit_cuda.GModelSMDiskf32}

    def __deepcopy__(self, memodict):
        return GModelSMDisk(self._dtype)

    def __init__(self, dtype):
        _detail.check_dtype(self._CLASSES, dtype)
        self._disk = self._CLASSES[dtype]()
        self._dtype = dtype

    def evaluate(
            self,
            loose, tilted, rnodes,
            vsys, xpos, ypos, posa, incl,
            rpt_uids, rpt_cvalues, rpt_ccounts, rpt_pvalues, rpt_pcounts,
            rht_uids, rht_cvalues, rht_ccounts, rht_pvalues, rht_pcounts,
            vpt_uids, vpt_cvalues, vpt_ccounts, vpt_pvalues, vpt_pcounts,
            vht_uids, vht_cvalues, vht_ccounts, vht_pvalues, vht_pcounts,
            dpt_uids, dpt_cvalues, dpt_ccounts, dpt_pvalues, dpt_pcounts,
            dht_uids, dht_cvalues, dht_ccounts, dht_pvalues, dht_pcounts,
            wpt_uids, wpt_cvalues, wpt_ccounts, wpt_pvalues, wpt_pcounts,
            spt_uids, spt_cvalues, spt_ccounts, spt_pvalues, spt_pcounts,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube, rcube,
            rdata, vdata, ddata):
        self._disk.evaluate(
            loose, tilted,
            _shape(rnodes)[0],
            _ptr(rnodes),
            _ptr(vsys), _ptr(xpos), _ptr(ypos), _ptr(posa), _ptr(incl),
            _shape(rpt_uids)[0],
            _ptr(rpt_uids),
            _ptr(rpt_cvalues), _ptr(rpt_ccounts),
            _ptr(rpt_pvalues), _ptr(rpt_pcounts),
            _ptr(rht_uids),
            _ptr(rht_cvalues), _ptr(rht_ccounts),
            _ptr(rht_pvalues), _ptr(rht_pcounts),
            _shape(vpt_uids)[0],
            _ptr(vpt_uids),
            _ptr(vpt_cvalues), _ptr(vpt_ccounts),
            _ptr(vpt_pvalues), _ptr(vpt_pcounts),
            _ptr(vht_uids),
            _ptr(vht_cvalues), _ptr(vht_ccounts),
            _ptr(vht_pvalues), _ptr(vht_pcounts),
            _shape(dpt_uids)[0],
            _ptr(dpt_uids),
            _ptr(dpt_cvalues), _ptr(dpt_ccounts),
            _ptr(dpt_pvalues), _ptr(dpt_pcounts),
            _ptr(dht_uids),
            _ptr(dht_cvalues), _ptr(dht_ccounts),
            _ptr(dht_pvalues), _ptr(dht_pcounts),
            _shape(wpt_uids)[0],
            _ptr(wpt_uids),
            _ptr(wpt_cvalues), _ptr(wpt_ccounts),
            _ptr(wpt_pvalues), _ptr(wpt_pcounts),
            _shape(spt_uids)[0],
            _ptr(spt_uids),
            _ptr(spt_cvalues), _ptr(spt_ccounts),
            _ptr(spt_pvalues), _ptr(spt_pcounts),
            spat_size[0], spat_size[1], spat_size[2],
            spat_step[0], spat_step[1], spat_step[2],
            spat_zero[0], spat_zero[1], spat_zero[2],
            spec_size,
            spec_step,
            spec_zero,
            _ptr(image), _ptr(scube), _ptr(rcube),
            _ptr(rdata), _ptr(vdata), _ptr(ddata))
