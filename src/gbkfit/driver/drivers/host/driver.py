
import numpy as np

import gbkfit.math
import gbkfit.native.libgbkfit_host as native_module

from gbkfit.driver.core import (
    Driver, DModelDCube, DModelMMaps, GModelMCDisk, GModelSMDisk, Objective)


__all__ = ['DriverHost']


def _ptr(x):
    return x.__array_interface__['data'][0] if x is not None else 0


def _shape(x):
    return x.__array_interface__['shape'] if x is not None else (0,)


def _size(x):
    return gbkfit.math.prod(_shape(x))


def _get_class(classes, dtype):
    if dtype not in classes:
        requested = np.dtype(dtype).name
        supported = ', '.join([np.dtype(dt).name for dt in classes])
        raise RuntimeError(
            f"the requested dtype is not supported "
            f"(requested: {requested}; supported: {supported})")
    return classes[dtype]


class DriverHost(Driver):

    @staticmethod
    def type():
        return 'host'

    def mem_alloc_s(self, shape, dtype):
        h_data = np.empty(shape, dtype)
        d_data = h_data
        return h_data, d_data

    def mem_alloc_h(self, shape, dtype):
        return np.empty(shape, dtype)

    def mem_alloc_d(self, shape, dtype):
        return np.empty(shape, dtype)

    def mem_copy_h2d(self, h_src, d_dst=None):
        if d_dst is None:
            d_dst = self.mem_alloc_d(h_src.shape, h_src.dtype)
        if h_src is not d_dst:
            d_dst[:] = h_src
        return d_dst

    def mem_copy_d2h(self, d_src, h_dst=None):
        if h_dst is None:
            h_dst = self.mem_alloc_h(d_src.shape, d_src.dtype)
        if d_src is not h_dst:
            h_dst[:] = d_src
        return h_dst

    def mem_fill(self, x, value):
        x.fill(value)

    def math_abs(self, x, out=None):
        return np.abs(x, out=out)

    def math_sum(self, x, out=None):
        return np.nansum(x, out=out, keepdims=True)

    def math_add(self, x1, x2, out=None):
        return np.add(x1, x2, out=out)

    def math_sub(self, x1, x2, out=None):
        return np.sub(x1, x2, out=out)

    def math_mul(self, x1, x2, out=None):
        return np.mul(x1, x2, out=out)

    def math_div(self, x1, x2, out=None):
        return np.div(x1, x2, out=out)

    def math_pow(self, x1, x2, out=None):
        return np.power(x1, x2, out=out)

    def make_dmodel_dcube(self, dtype):
        return DModelDCube(dtype)

    def make_dmodel_mmaps(self, dtype):
        return DModelMMaps(dtype)

    def make_gmodel_mcdisk(self, dtype):
        return GModelMCDisk(dtype)

    def make_gmodel_smdisk(self, dtype):
        return GModelSMDisk(dtype)

    def make_objective(self, dtype):
        return Objective(dtype)


class DModelDCube(DModelDCube):

    _CLASSES = {
        np.float32: native_module.DModelDCubef32}

    def __deepcopy__(self, memodict):
        return DModelDCube(self._dtype)

    def __init__(self, dtype):
        self._backend = _get_class(self._CLASSES, dtype)()
        self._dtype = dtype

    def convolve(
            self,
            size_hi,
            scube_hi, scube_hi_fft,
            wcube_hi, wcube_hi_fft,
            psf3d_hi, psf3d_hi_fft):
        self._backend.convolve(
            size_hi,
            _ptr(scube_hi), _ptr(scube_hi_fft),
            _ptr(wcube_hi), _ptr(wcube_hi_fft),
            _ptr(psf3d_hi), _ptr(psf3d_hi_fft))

    def downscale(
            self,
            scale,
            edge_hi,
            size_hi, size_lo,
            scube_hi, scube_lo):
        self._backend.downscale(
            scale,
            edge_hi,
            size_hi, size_lo,
            _ptr(scube_hi), _ptr(scube_lo))

    def make_mask(self, mask_spat, mask_spec, mask_coef, size, cube, mask):
        self._backend.make_mask(
            mask_spat, mask_spec, mask_coef, size, _ptr(cube), _ptr(mask))


class DModelMMaps(DModelMMaps):

    _CLASSES = {
        np.float32: native_module.DModelMMapsf32}

    def __deepcopy__(self, memodict):
        return DModelMMaps(self._dtype)

    def __init__(self, dtype):
        self._backend = _get_class(self._CLASSES, dtype)()
        self._dtype = dtype

    def moments(self, size, step, zero, scube, mmaps, masks, orders):
        self._backend.moments(
            size, step, zero,
            _ptr(scube),
            _ptr(mmaps),
            _ptr(masks),
            _ptr(orders),
            _size(orders))


class GModelMCDisk(GModelMCDisk):

    _CLASSES = {
        np.float32: native_module.GModelMCDiskf32}

    def __deepcopy__(self, memodict):
        return GModelMCDisk(self._dtype)

    def __init__(self, dtype):
        self._backend = _get_class(self._CLASSES, dtype)()
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
            zpt_uids, zpt_cvalues, zpt_ccounts, zpt_pvalues, zpt_pcounts,
            spt_uids, spt_cvalues, spt_ccounts, spt_pvalues, spt_pcounts,
            wpt_uids, wpt_cvalues, wpt_ccounts, wpt_pvalues, wpt_pcounts,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube, rcube, wcube,
            rdata, vdata, ddata):
        self._backend.evaluate(
            cflux, nclouds, _ptr(ncloudspt), _size(ncloudspt), _ptr(hasordint),
            loose, tilted,
            _size(rnodes),
            _ptr(rnodes),
            _ptr(vsys), _ptr(xpos), _ptr(ypos), _ptr(posa), _ptr(incl),
            _size(rpt_uids),
            _ptr(rpt_uids),
            _ptr(rpt_cvalues), _ptr(rpt_ccounts),
            _ptr(rpt_pvalues), _ptr(rpt_pcounts),
            _ptr(rht_uids),
            _ptr(rht_cvalues), _ptr(rht_ccounts),
            _ptr(rht_pvalues), _ptr(rht_pcounts),
            _size(vpt_uids),
            _ptr(vpt_uids),
            _ptr(vpt_cvalues), _ptr(vpt_ccounts),
            _ptr(vpt_pvalues), _ptr(vpt_pcounts),
            _ptr(vht_uids),
            _ptr(vht_cvalues), _ptr(vht_ccounts),
            _ptr(vht_pvalues), _ptr(vht_pcounts),
            _size(dpt_uids),
            _ptr(dpt_uids),
            _ptr(dpt_cvalues), _ptr(dpt_ccounts),
            _ptr(dpt_pvalues), _ptr(dpt_pcounts),
            _ptr(dht_uids),
            _ptr(dht_cvalues), _ptr(dht_ccounts),
            _ptr(dht_pvalues), _ptr(dht_pcounts),
            _size(zpt_uids),
            _ptr(zpt_uids),
            _ptr(zpt_cvalues), _ptr(zpt_ccounts),
            _ptr(zpt_pvalues), _ptr(zpt_pcounts),
            _size(spt_uids),
            _ptr(spt_uids),
            _ptr(spt_cvalues), _ptr(spt_ccounts),
            _ptr(spt_pvalues), _ptr(spt_pcounts),
            _size(wpt_uids),
            _ptr(wpt_uids),
            _ptr(wpt_cvalues), _ptr(wpt_ccounts),
            _ptr(wpt_pvalues), _ptr(wpt_pcounts),
            spat_size[0], spat_size[1], spat_size[2],
            spat_step[0], spat_step[1], spat_step[2],
            spat_zero[0], spat_zero[1], spat_zero[2],
            spec_size,
            spec_step,
            spec_zero,
            _ptr(image), _ptr(scube), _ptr(rcube), _ptr(wcube),
            _ptr(rdata), _ptr(vdata), _ptr(ddata))


class GModelSMDisk(GModelSMDisk):

    _CLASSES = {
        np.float32: native_module.GModelSMDiskf32}

    def __deepcopy__(self, memodict):
        return GModelSMDisk(self._dtype)

    def __init__(self, dtype):
        self._backend = _get_class(self._CLASSES, dtype)()
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
            zpt_uids, zpt_cvalues, zpt_ccounts, zpt_pvalues, zpt_pcounts,
            spt_uids, spt_cvalues, spt_ccounts, spt_pvalues, spt_pcounts,
            wpt_uids, wpt_cvalues, wpt_ccounts, wpt_pvalues, wpt_pcounts,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube, rcube, wcube,
            rdata, vdata, ddata):
        self._backend.evaluate(
            loose, tilted,
            _size(rnodes),
            _ptr(rnodes),
            _ptr(vsys), _ptr(xpos), _ptr(ypos), _ptr(posa), _ptr(incl),
            _size(rpt_uids),
            _ptr(rpt_uids),
            _ptr(rpt_cvalues), _ptr(rpt_ccounts),
            _ptr(rpt_pvalues), _ptr(rpt_pcounts),
            _ptr(rht_uids),
            _ptr(rht_cvalues), _ptr(rht_ccounts),
            _ptr(rht_pvalues), _ptr(rht_pcounts),
            _size(vpt_uids),
            _ptr(vpt_uids),
            _ptr(vpt_cvalues), _ptr(vpt_ccounts),
            _ptr(vpt_pvalues), _ptr(vpt_pcounts),
            _ptr(vht_uids),
            _ptr(vht_cvalues), _ptr(vht_ccounts),
            _ptr(vht_pvalues), _ptr(vht_pcounts),
            _size(dpt_uids),
            _ptr(dpt_uids),
            _ptr(dpt_cvalues), _ptr(dpt_ccounts),
            _ptr(dpt_pvalues), _ptr(dpt_pcounts),
            _ptr(dht_uids),
            _ptr(dht_cvalues), _ptr(dht_ccounts),
            _ptr(dht_pvalues), _ptr(dht_pcounts),
            _size(zpt_uids),
            _ptr(zpt_uids),
            _ptr(zpt_cvalues), _ptr(zpt_ccounts),
            _ptr(zpt_pvalues), _ptr(zpt_pcounts),
            _size(spt_uids),
            _ptr(spt_uids),
            _ptr(spt_cvalues), _ptr(spt_ccounts),
            _ptr(spt_pvalues), _ptr(spt_pcounts),
            _size(wpt_uids),
            _ptr(wpt_uids),
            _ptr(wpt_cvalues), _ptr(wpt_ccounts),
            _ptr(wpt_pvalues), _ptr(wpt_pcounts),
            spat_size[0], spat_size[1], spat_size[2],
            spat_step[0], spat_step[1], spat_step[2],
            spat_zero[0], spat_zero[1], spat_zero[2],
            spec_size,
            spec_step,
            spec_zero,
            _ptr(image), _ptr(scube), _ptr(rcube), _ptr(wcube),
            _ptr(rdata), _ptr(vdata), _ptr(ddata))


class Objective(Objective):

    _CLASSES = {
        np.float32: native_module.Objectivef32}

    def __deepcopy__(self, memodict):
        return Objective(self._dtype)

    def __init__(self, dtype):
        self._backend = _get_class(self._CLASSES, dtype)()
        self._dtype = dtype

    def count_pixels(self, data, model, size, count):
        self._backend.count_pixels(_ptr(data), _ptr(model), size, _ptr(count))
