
import numpy as np

import gbkfit.driver
from gbkfit.drivers import _detail

import libgbkfit_openmp


def _ptr(a):
    return a.__array_interface__['data'][0] if a is not None else 0


def _shape(a):
    return a.__array_interface__['shape'] if a is not None else (0,)


class BackendOpenMP(gbkfit.driver.Driver):

    @staticmethod
    def type():
        return 'openmp'

    @classmethod
    def load(cls, info):
        return cls()

    def dump(self):
        return {'type': self.type()}

    def mem_alloc(self, shape, dtype):
        h_data = np.empty(shape, dtype)
        d_data = h_data
        return h_data, d_data

    def mem_alloc_h(self, shape, dtype):
        return np.empty(shape, dtype)

    def mem_alloc_d(self, shape, dtype):
        return np.empty(shape, dtype)

    def mem_copy_h2d(self, h_src, d_dst=None):
        if d_dst is None:
            d_dst = h_src
        if h_src is not d_dst:
            d_dst[:] = h_src
        return d_dst

    def mem_copy_d2h(self, d_src, h_dst=None):
        if h_dst is None:
            h_dst = d_src
        if d_src is not h_dst:
            h_dst[:] = d_src
        return h_dst

    def mem_fill_d(self, ary, value):
        ary.fill(value)

    def array_add_d(self, ary, value):
        ary += value

    def array_mul_d(self, ary, value):
        ary *= value

    def make_dmodel_dcube(self, dtype):
        return DModelDCube(dtype)

    def make_dmodel_mmaps(self, dtype):
        return DModelMMaps(dtype)

    def make_gmodel_mcdisk(self, dtype):
        return GModelMCDisk(dtype)

    def make_gmodel_smdisk(self, dtype):
        return GModelSMDisk(dtype)


class DModelDCube(gbkfit.driver.DModelDCube):

    _CLASSES = {
        np.float32: libgbkfit_openmp.DModelDCubef32}

    def __init__(self, dtype):
        _detail.check_dtype(self._CLASSES, dtype)
        self._dcube = self._CLASSES[dtype]()

    def downscale(self, scale, offset, src_size, dst_size, src_cube, dst_cube):
        self._dcube.downscale(
            scale[0], scale[1], scale[2],
            offset[0], offset[1], offset[2],
            src_size[0], src_size[1], src_size[2],
            dst_size[0], dst_size[1], dst_size[2],
            _ptr(src_cube), _ptr(dst_cube))


class DModelMMaps(gbkfit.driver.DModelMMaps):

    _CLASSES = {
        np.float32: libgbkfit_openmp.DModelMMapsf32}

    def __init__(self, dtype):
        _detail.check_dtype(self._CLASSES, dtype)
        self._mmaps = self._CLASSES[dtype]()

    def moments(
            self,
            spat_size,
            spec_size, spec_step, spec_zero,
            scube,
            morders, mmaps):
        self._mmaps.moments(
            spat_size[0], spat_size[1],
            spec_size,
            spec_step,
            spec_zero,
            np.nan,
            _ptr(scube),
            _shape(morders)[0], _ptr(morders), _ptr(mmaps))


class GModelMCDisk(gbkfit.driver.GModelMCDisk):

    _CLASSES = {
        np.float32: libgbkfit_openmp.GModelMCDiskf32}

    def __init__(self, dtype):
        _detail.check_dtype(self._CLASSES, dtype)
        self._disk = self._CLASSES[dtype]()

    def evaluate(
            self,
            cflux, nclouds, ncloudspt,
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
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            image, scube, rcube,
            rdata, vdata, ddata):
        self._disk.evaluate(
            cflux, nclouds, _ptr(ncloudspt),
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


class GModelSMDisk(gbkfit.driver.GModelSMDisk):

    _CLASSES = {
        np.float32: libgbkfit_openmp.GModelSMDiskf32}

    def __init__(self, dtype):
        _detail.check_dtype(self._CLASSES, dtype)
        self._disk = self._CLASSES[dtype]()

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
            spat_size, spat_step, spat_zero,
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
