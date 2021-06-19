
import numpy as np

import gbkfit.math
import gbkfit.driver.native.libgbkfit_host as native_module
from gbkfit.driver import backend


__all__ = ['Backend']


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


class Backend(backend.Backend):

    def make_dmodel_dcube(self, dtype):
        return DModelDCube(dtype)

    def make_dmodel_mmaps(self, dtype):
        return DModelMMaps(dtype)

    def make_gmodel(self, dtype):
        return GModel(dtype)

    def make_gmodel_mcdisk(self, dtype):
        return GModelMCDisk(dtype)

    def make_gmodel_smdisk(self, dtype):
        return GModelSMDisk(dtype)

    def make_objective(self, dtype):
        return Objective(dtype)


class DModelDCube(backend.DModelDCube):

    _CLASSES = {
        np.float32: native_module.DModelDCubef32}

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

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


class DModelMMaps(backend.DModelMMaps):

    _CLASSES = {
        np.float32: native_module.DModelMMapsf32}

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

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


class GModel(backend.GModel):

    _CLASSES = {
        np.float32: native_module.GModelMCDiskf32}

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

    def __init__(self, dtype):
        self._backend = _get_class(self._CLASSES, dtype)()
        self._dtype = dtype

    def make_wcube(self, spat_size, spec_size, spat_data, spec_data):
        pass


class GModelMCDisk(backend.GModelMCDisk):

    _CLASSES = {
        np.float32: native_module.GModelMCDiskf32}

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

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


class GModelSMDisk(backend.GModelSMDisk):

    _CLASSES = {
        np.float32: native_module.GModelSMDiskf32}

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

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


class Objective(backend.Objective):

    _CLASSES = {
        np.float32: native_module.Objectivef32}

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

    def __init__(self, dtype):
        self._backend = _get_class(self._CLASSES, dtype)()
        self._dtype = dtype

    def count_pixels(self, data, model, size, count):
        self._backend.count_pixels(_ptr(data), _ptr(model), size, _ptr(count))
