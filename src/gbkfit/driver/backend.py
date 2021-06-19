
import abc


class Backend(abc.ABC):

    @abc.abstractmethod
    def make_dmodel_dcube(self, dtype):
        pass

    @abc.abstractmethod
    def make_dmodel_mmaps(self, dtype):
        pass

    @abc.abstractmethod
    def make_gmodel(self, dtype):
        pass

    @abc.abstractmethod
    def make_gmodel_mcdisk(self, dtype):
        pass

    @abc.abstractmethod
    def make_gmodel_smdisk(self, dtype):
        pass

    @abc.abstractmethod
    def make_objective(self, dtype):
        pass


class DModelDCube(abc.ABC):

    @abc.abstractmethod
    def convolve(
            self,
            size_hi,
            wcube_hi, wcube_hi_fft,
            scube_hi, scube_hi_fft,
            psf3d_hi, psf3d_hi_fft):
        pass

    @abc.abstractmethod
    def downscale(
            self,
            scale,
            edge_hi,
            size_hi, size_lo,
            scube_hi, scube_lo):
        pass

    @abc.abstractmethod
    def make_mask(self, mask_spat, mask_spec, mask_coef, size, cube, mask):
        pass


class DModelMMaps(abc.ABC):

    @abc.abstractmethod
    def moments(self, size, step, zero, scube, mmaps, masks, orders):
        pass


class GModel(abc.ABC):

    @abc.abstractmethod
    def make_wcube(self, spat_size, spec_size, spat_data, spec_data):
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
            zpt_uids, zpt_cvalues, zpt_ccounts, zpt_pvalues, zpt_pcounts,
            spt_uids, spt_cvalues, spt_ccounts, spt_pvalues, spt_pcounts,
            wpt_uids, wpt_cvalues, wpt_ccounts, wpt_pvalues, wpt_pcounts,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube, rcube, wcube,
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
            zpt_uids, zpt_cvalues, zpt_ccounts, zpt_pvalues, zpt_pcounts,
            spt_uids, spt_cvalues, spt_ccounts, spt_pvalues, spt_pcounts,
            wpt_uids, wpt_cvalues, wpt_ccounts, wpt_pvalues, wpt_pcounts,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube, rcube, wcube,
            rdata, vdata, ddata):
        pass


class Objective(abc.ABC):

    @abc.abstractmethod
    def count_pixels(self, data, model, size, count):
        pass
