
import abc

import numpy as np

from gbkfit.driver.modules import (
    DriverBackendDModel,
    DriverBackendFFT,
    DriverBackendGModel)


__all__ = [
    'NativeMemory',
    'DriverBackendFFTNative',
    'DriverBackendDModelNative',
    'DriverBackendGModelNative',
]


def _get_class(cls, dtype, classes):
    if dtype not in classes:
        requested = np.dtype(dtype).name
        supported = ', '.join([np.dtype(dt).name for dt in classes])
        raise RuntimeError(
            f"could not create native module wrapper of type '{cls}'; "
            f"the requested dtype is not supported by the native module "
            f"(requested: {requested}; supported: {supported})")
    return classes[dtype]


class NativeMemory(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def ptr(x):
        pass

    @staticmethod
    @abc.abstractmethod
    def size(x):
        pass

    @staticmethod
    @abc.abstractmethod
    def shape(x):
        pass

    @staticmethod
    @abc.abstractmethod
    def dtype(x):
        pass


class DriverBackendFFTNative(DriverBackendFFT):

    def __init__(self, dtype, memory, classes):
        super().__init__(32)
        self._dtype = dtype
        self._memory = memory
        self._module = _get_class(self.__class__.__qualname__, dtype, classes)()

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype, self._memory, self._module)

    def fft_r2c(self, data_r, data_c):
        _ptr = self._memory.ptr
        _shape = self._memory.shape
        self._module.fft_r2c(
            _shape(data_r)[::-1], _ptr(data_r), _ptr(data_c))

    def fft_c2r(self, data_c, data_r):
        _ptr = self._memory.ptr
        _shape = self._memory.shape
        self._module.fft_c2r(
            _shape(data_r)[::-1], _ptr(data_c), _ptr(data_r))

    def fft_convolve(self, data1_r, data1_c, data2_c):
        _ptr = self._memory.ptr
        _shape = self._memory.shape
        self._module.fft_convolve(
            _shape(data1_r)[::-1], _ptr(data1_r), _ptr(data1_c), _ptr(data2_c))

    def fft_convolve_cached(self, data1_r, data2_r):
        _ptr = self._memory.ptr
        _shape = self._memory.shape
        self._module.fft_convolve_cached(
            _shape(data1_r)[::-1], _ptr(data1_r), _ptr(data2_r))


class DriverBackendDModelNative(DriverBackendDModel):

    def __init__(self, dtype, memory, classes):
        super().__init__()
        self._dtype = dtype
        self._memory = memory
        self._module = _get_class(self.__class__.__qualname__, dtype, classes)()

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype, self._memory, self._module)

    def dcube_downscale(self, scale, edge_hi, cube_hi, cube_lo):
        _ptr = self._memory.ptr
        _shape = self._memory.shape
        self._module.dcube_downscale(
            scale, edge_hi, _shape(cube_hi)[::-1], _shape(cube_lo)[::-1],
            _ptr(cube_hi), _ptr(cube_lo))

    def dcube_mask(self, cutoff, apply, dcube, mcube, wcube):
        _ptr = self._memory.ptr
        _shape = self._memory.shape
        self._module.dcube_mask(
            cutoff, apply, _shape(dcube), _ptr(dcube), _ptr(mcube), _ptr(wcube))

    def mmaps_moments(
            self,
            size, step, zero, dcube, wcube, cutoff, orders,
            mmaps_d, mmaps_w, mmaps_m):
        _ptr = self._memory.ptr
        _size = self._memory.size
        self._module.mmaps_moments(
            size, step, zero,
            _ptr(dcube), _ptr(wcube), cutoff, _size(orders), _ptr(orders),
            _ptr(mmaps_d), _ptr(mmaps_w), _ptr(mmaps_m))


class DriverBackendGModelNative(DriverBackendGModel):

    def __init__(self, dtype, memory, classes):
        super().__init__()
        self._dtype = dtype
        self._memory = memory
        self._module = _get_class(self.__class__.__qualname__, dtype, classes)()

    def __deepcopy__(self, memodict):
        return self.__class__(self._dtype, self._memory, self._module)

    def convolve(self, data1, data2, result):
        _ptr = self._memory.ptr
        _shape = self._memory.shape
        data1_size = _shape(data1)[::-1]
        data2_size = _shape(data2)[::-1]
        assert _shape(data1) == _shape(result)
        self._module.convolve(
            data1_size[0], data1_size[1], data1_size[2],
            data2_size[0], data2_size[1], data2_size[2],
            _ptr(data1),
            _ptr(data2),
            _ptr(result))

    def wcube_evaluate(self, spat_size, spec_size, spat_data, spec_data):
        _ptr = self._memory.ptr
        self._module.wcube_evaluate(
            spat_size[0], spat_size[1], spat_size[2],
            spec_size,
            _ptr(spat_data),
            _ptr(spec_data))

    def mcdisk_evaluate(
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
            opacity,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube,
            wdata, wdata_cmp,
            rdata, rdata_cmp,
            ordata,ordata_cmp,
            vdata_cmp, ddata_cmp):
        _ptr = self._memory.ptr
        _size = self._memory.size
        self._module.mcdisk_evaluate(
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
            _ptr(opacity),
            spat_size[0], spat_size[1], spat_size[2],
            spat_step[0], spat_step[1], spat_step[2],
            spat_zero[0], spat_zero[1], spat_zero[2],
            spec_size,
            spec_step,
            spec_zero,
            _ptr(image), _ptr(scube),
            _ptr(wdata), _ptr(wdata_cmp),
            _ptr(rdata), _ptr(rdata_cmp),
            _ptr(ordata), _ptr(ordata_cmp),
            _ptr(vdata_cmp), _ptr(ddata_cmp))

    def smdisk_evaluate(
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
            opacity,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube,
            wdata, wdata_cmp,
            rdata, rdata_cmp,
            ordata, ordata_cmp,
            vdata_cmp, ddata_cmp):
        _ptr = self._memory.ptr
        _size = self._memory.size
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
            _ptr(opacity),
            spat_size[0], spat_size[1], spat_size[2],
            spat_step[0], spat_step[1], spat_step[2],
            spat_zero[0], spat_zero[1], spat_zero[2],
            spec_size,
            spec_step,
            spec_zero,
            _ptr(image), _ptr(scube),
            _ptr(wdata), _ptr(wdata_cmp),
            _ptr(rdata), _ptr(rdata_cmp),
            _ptr(ordata), _ptr(ordata_cmp),
            _ptr(vdata_cmp), _ptr(ddata_cmp))
