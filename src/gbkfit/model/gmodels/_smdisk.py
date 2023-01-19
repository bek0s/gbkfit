
from . import _disk


__all__ = ['SMDisk']


class SMDisk(_disk.Disk):

    def __init__(
            self,
            loose, tilted, rnodes, rstep, interp,
            vsys_nwmode,
            xpos_nwmode, ypos_nwmode,
            posa_nwmode, incl_nwmode,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            zptraits,
            sptraits,
            wptraits):

        super().__init__(
            loose, tilted, rnodes, rstep, interp,
            vsys_nwmode,
            xpos_nwmode, ypos_nwmode,
            posa_nwmode, incl_nwmode,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            zptraits,
            sptraits,
            wptraits)

    def _impl_prepare(self, driver, dtype):
        pass

    def _impl_evaluate(
            self, driver, params,
            odata,
            image, scube,
            wdata, wdata_cmp,
            rdata, rdata_cmp,
            ordata, ordata_cmp,
            vdata_cmp, ddata_cmp,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):

        if out_extra is not None:
            pass

        self._backend.smdisk_evaluate(
            self._loose,
            self._tilted,
            self._s_subrnodes[1],
            self._s_vsys_pvalues[1],
            self._s_xpos_pvalues[1],
            self._s_ypos_pvalues[1],
            self._s_posa_pvalues[1],
            self._s_incl_pvalues[1],
            self._s_rpt_uids[1],
            self._s_rpt_cvalues[1], self._s_rpt_ccounts[1],
            self._s_rpt_pvalues[1], self._s_rpt_pcounts[1],
            self._s_rht_uids[1],
            self._s_rht_cvalues[1], self._s_rht_ccounts[1],
            self._s_rht_pvalues[1], self._s_rht_pcounts[1],
            self._s_vpt_uids[1],
            self._s_vpt_cvalues[1], self._s_vpt_ccounts[1],
            self._s_vpt_pvalues[1], self._s_vpt_pcounts[1],
            self._s_vht_uids[1],
            self._s_vht_cvalues[1], self._s_vht_ccounts[1],
            self._s_vht_pvalues[1], self._s_vht_pcounts[1],
            self._s_dpt_uids[1],
            self._s_dpt_cvalues[1], self._s_dpt_ccounts[1],
            self._s_dpt_pvalues[1], self._s_dpt_pcounts[1],
            self._s_dht_uids[1],
            self._s_dht_cvalues[1], self._s_dht_ccounts[1],
            self._s_dht_pvalues[1], self._s_dht_pcounts[1],
            self._s_zpt_uids[1],
            self._s_zpt_cvalues[1], self._s_zpt_ccounts[1],
            self._s_zpt_pvalues[1], self._s_zpt_pcounts[1],
            self._s_spt_uids[1],
            self._s_spt_cvalues[1], self._s_spt_ccounts[1],
            self._s_spt_pvalues[1], self._s_spt_pcounts[1],
            self._s_wpt_uids[1],
            self._s_wpt_cvalues[1], self._s_wpt_ccounts[1],
            self._s_wpt_pvalues[1], self._s_wpt_pcounts[1],
            odata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube,
            wdata, wdata_cmp,
            rdata, rdata_cmp,
            ordata, ordata_cmp,
            vdata_cmp, ddata_cmp)

        if out_extra is not None:
            pass
