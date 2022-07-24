
import abc

import numpy as np

import gbkfit.math


__all__ = [
    'DriverBackends',
    'DriverBackendFFT',
    'DriverBackendDModel',
    'DriverBackendGModel'
]


class DriverBackends(abc.ABC):

    @abc.abstractmethod
    def fft(self, dtype):
        pass

    @abc.abstractmethod
    def dmodel(self, dtype):
        pass

    @abc.abstractmethod
    def gmodel(self, dtype):
        pass

    @abc.abstractmethod
    def objective(self, dtype):
        pass


class DriverBackendFFT(abc.ABC):

    @staticmethod
    def fft_complex_shape(shape):
        return tuple(shape[:-1] + [shape[-1] // 2 + 1])

    @staticmethod
    def fft_convolution_shift(data):
        axis = tuple(range(data.ndim))
        return np.roll(data, np.array(data.shape) // 2 + 1, axis=axis)

    def __init__(self, fft_shape_threshold):
        self._fft_shape_threshold = fft_shape_threshold

    def fft_optimal_shape(self, shape):
        threshold = self._fft_shape_threshold
        new_shape = []
        for dim in shape:
            dim_po2 = gbkfit.math.roundu_po2(dim)
            dim_mul = gbkfit.math.roundu_multiple(dim, threshold)
            dim_new = dim_po2 if dim_po2 <= threshold else dim_mul
            new_shape.append(dim_new)
        return np.array(new_shape)

    def fft_convolution_shape(self, data1_shape, data2_shape):
        shape = np.array(data1_shape) + np.array(data2_shape) - 1
        shape = self.fft_optimal_shape(shape)
        margin = np.array(data2_shape) // 2
        return tuple(shape), tuple(margin)

    @abc.abstractmethod
    def fft_r2c(self, data_r, data_c):
        pass

    @abc.abstractmethod
    def fft_c2r(self, data_c, data_r):
        pass

    @abc.abstractmethod
    def fft_convolve(self, data1_r, data1_c, data2_c):
        pass

    @abc.abstractmethod
    def fft_convolve_cached(self, data1_r, data2_r):
        pass


class DriverBackendDModel(abc.ABC):

    @abc.abstractmethod
    def downscale(self, scale, edge_hi, cube_hi, cube_lo):
        pass

    @abc.abstractmethod
    def mask(self, cutoff, apply, dcube, mcube):
        pass

    @abc.abstractmethod
    def moments(
            self,
            size, step, zero, dcube, wcube, cutoff, orders,
            mmaps_d, mmaps_w, mmaps_m):
        pass


class DriverBackendGModel(abc.ABC):

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass
