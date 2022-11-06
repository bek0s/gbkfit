
import numpy as np

import gbkfit.driver.native.libgbkfit_host as native_module
from gbkfit.driver.modules import (
    DriverBackends,
    DriverBackendDModel,
    DriverBackendFFT,
    DriverBackendGModel)


__all__ = [
    'DriverBackendsHost',
    'DriverBackendFFTHost',
    'DriverBackendDModelHost',
    'DriverBackendGModelHost',
]


def _ptr(x):
    return x.__array_interface__['data'][0] if x is not None else 0


def _dtype(x):
    return x.__array_interface__['typestr'] if x is not None else None


def _shape(x):
    return x.__array_interface__['shape'] if x is not None else None


def _size(x):
    return x.size if x is not None else 0


class DriverBackendsHost(DriverBackends):

    def fft(self, dtype):
        return DriverBackendFFTHost(dtype)

    def dmodel(self, dtype):
        return DriverBackendDModelHost(dtype)

    def gmodel(self, dtype):
        return DriverBackendGModelHost(dtype)

    def objective(self, dtype):
        raise NotImplementedError()


def _get_class(cls, dtype, classes):
    if dtype not in classes:
        requested = np.dtype(dtype).name
        supported = ', '.join([np.dtype(dt).name for dt in classes])
        raise RuntimeError(
            f"could not create native module wrapper of type '{cls}'; "
            f"the requested dtype is not supported by the native module "
            f"(requested: {requested}; supported: {supported})")
    return classes[dtype]


class DriverBackendFFTHost(DriverBackendFFT):

    _CLASSES = {
        np.float32: native_module.FFTf32}

    def __init__(self, dtype):
        super().__init__(32)
        self._dtype = dtype
        self._module = _get_class(
            self.__class__.__qualname__, dtype, self._CLASSES)()

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

    def fft_r2c(self, data_r, data_c):
        self._module.fft_r2c(_shape(data_r)[::-1], _ptr(data_r), _ptr(data_c))

    def fft_c2r(self, data_c, data_r):
        self._module.fft_c2r(_shape(data_r)[::-1], _ptr(data_c), _ptr(data_r))

    def fft_convolve(self, data1_r, data1_c, data2_c):
        self._module.fft_convolve(
            _shape(data1_r)[::-1], _ptr(data1_r), _ptr(data1_c), _ptr(data2_c))

    def fft_convolve_cached(self, data1_r, data2_r):
        self._module.fft_convolve_cached(
            _shape(data1_r)[::-1], _ptr(data1_r), _ptr(data2_r))


class DriverBackendDModelHost(DriverBackendDModel):

    _CLASSES = {
        np.float32: native_module.DModelf32}

    def __init__(self, dtype):
        self._dtype = dtype
        self._module = _get_class(
            self.__class__.__qualname__, dtype, self._CLASSES)()

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

    def downscale(self, scale, edge_hi, cube_hi, cube_lo):
        size_hi = _shape(cube_hi)[::-1]
        size_lo = _shape(cube_lo)[::-1]
        self._module.dcube_downscale(
            scale, edge_hi, size_hi, size_lo, _ptr(cube_hi), _ptr(cube_lo))

    def mask(self, cutoff, apply, dcube, mcube):
        self._module.mask(
            cutoff, apply, _shape(dcube), _ptr(dcube), _ptr(mcube))

    def moments(
            self,
            size, step, zero, dcube, wcube, cutoff, orders,
            mmaps_d, mmaps_w, mmaps_m):
        self._module.mmaps_moments(
            size, step, zero,
            _ptr(dcube), _ptr(wcube), cutoff, _size(orders), _ptr(orders),
            _ptr(mmaps_d), _ptr(mmaps_w), _ptr(mmaps_m))


class DriverBackendGModelHost(DriverBackendGModel):

    _CLASSES = {
        np.float32: native_module.GModelf32}

    def __init__(self, dtype):
        self._dtype = dtype
        self._module = _get_class(
            self.__class__.__qualname__, dtype, self._CLASSES)()

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype)

    def wcube_evaluate(self, spat_size, spec_size, spat_data, spec_data):
        self._module.wcube_evaluate(
            spat_size[0], spat_size[1], spat_size[2],
            spec_size,
            _ptr(spat_data),
            _ptr(spec_data))

    def evaluate_mcdisk(
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
        self._module.evaluate(
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

    def evaluate_smdisk(
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
        self._module.smdisk_evaluate(
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
