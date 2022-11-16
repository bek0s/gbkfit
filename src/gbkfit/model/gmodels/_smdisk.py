
import logging

import numpy as np

from . import _disk


log = logging.getLogger(__name__)


class SMDisk(_disk.Disk):

    def __init__(
            self,
            loose, tilted, rnodes, rnstep, interp,
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
            loose, tilted, rnodes, rnstep, interp,
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
        self._disk = driver.backends().gmodel(dtype)

    def _impl_evaluate(
            self, driver, params, image, scube, rcube, wcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):

        rdata = None
        vdata = None
        ddata = None

        if out_extra is not None:
            shape = spat_size[::-1]
            if self._rptraits:
                rdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(rdata, 0)
            if self._vptraits:
                vdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(vdata, np.nan)
            if self._dptraits:
                ddata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(ddata, np.nan)

        self._disk.smdisk_evaluate(
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
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            image, scube, rcube, wcube,
            rdata, vdata, ddata)

        if out_extra is not None:
            if self._rptraits:
                out_extra['rdata'] = driver.mem_copy_d2h(rdata)
            if self._vptraits:
                out_extra['vdata'] = driver.mem_copy_d2h(vdata)
            if self._dptraits:
                out_extra['ddata'] = driver.mem_copy_d2h(ddata)
            if self._rptraits:
                sumabs = np.nansum(np.abs(out_extra['rdata']))
                log.debug(f"sum(abs(rdata)): {sumabs}")
            if self._vptraits:
                sumabs = np.nansum(np.abs(out_extra['vdata']))
                log.debug(f"sum(abs(vdata)): {sumabs}")
            if self._dptraits:
                sumabs = np.nansum(np.abs(out_extra['ddata']))
                log.debug(f"sum(abs(ddata)): {sumabs}")
