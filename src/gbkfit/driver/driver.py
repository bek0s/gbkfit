
import abc

from gbkfit.utils import parseutils


class Driver(parseutils.SimpleParserSupport, abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def type():
        pass

    @abc.abstractmethod
    def mem_alloc_s(self, shape, dtype):
        pass

    @abc.abstractmethod
    def mem_alloc_h(self, shape, dtype):
        pass

    @abc.abstractmethod
    def mem_alloc_d(self, shape, dtype):
        pass

    @abc.abstractmethod
    def mem_copy_h2d(self, h_src, d_dst=None):
        pass

    @abc.abstractmethod
    def mem_copy_d2h(self, d_src, h_dst=None):
        pass

    @abc.abstractmethod
    def mem_fill(self, x, value):
        pass

    @abc.abstractmethod
    def math_abs(self, x, out=None):
        pass

    @abc.abstractmethod
    def math_sum(self, x, out=None):
        pass

    @abc.abstractmethod
    def math_add(self, x1, x2, out=None):
        pass

    @abc.abstractmethod
    def math_sub(self, x1, x2, out=None):
        pass

    @abc.abstractmethod
    def math_mul(self, x1, x2, out=None):
        pass

    @abc.abstractmethod
    def math_div(self, x1, x2, out=None):
        pass

    def make_dmodel_dcube(self, dtype):
        raise NotImplementedError()

    def make_dmodel_mmaps(self, dtype):
        raise NotImplementedError()

    def make_gmodel_mcdisk(self, dtype):
        raise NotImplementedError()

    def make_gmodel_smdisk(self, dtype):
        raise NotImplementedError()


class DModelDCube(abc.ABC):

    @abc.abstractmethod
    def prepare(
            self,
            size_lo, size_hi, edge_hi, scale,
            scube_lo, dmask_hi,
            scube_hi, scube_hi_fft,
            psf3d_hi, psf3d_hi_fft):
        pass

    @abc.abstractmethod
    def convolve(self):
        pass

    @abc.abstractmethod
    def downscale(self):
        pass

    @abc.abstractmethod
    def make_mask(self):
        pass

    @abc.abstractmethod
    def apply_mask(self):
        pass


class DModelMMaps(abc.ABC):

    @abc.abstractmethod
    def prepare(
            self,
            spat_size,
            spec_size, spec_step, spec_zero,
            scube,
            mmaps,
            mmaps_orders):
        pass

    @abc.abstractmethod
    def moments(self):
        pass


class GModelMCDisk(abc.ABC):

    @abc.abstractmethod
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
        pass


class GModelSMDisk(abc.ABC):

    @abc.abstractmethod
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
        pass


parser = parseutils.TypedParser(Driver)
