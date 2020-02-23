
import itertools
import logging

import numpy as np

from . import _disk


log = logging.getLogger(__name__)


class MCDisk(_disk.Disk):

    def __init__(
            self,
            cflux,
            loose, tilted, rnodes,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            wptraits,
            sptraits):

        super().__init__(
            loose, tilted, rnodes,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            wptraits,
            sptraits)

        self._cflux = cflux
        self._ncloudsptor = None
        self._hasordint = None

    def cflux(self):
        return self._cflux

    def _impl_prepare(self, driver, dtype):
        self._disk = driver.make_gmodel_mcdisk(dtype)
        hasordint = [t.has_ordinary_integral() for t in self._rptraits]
        size = sum([1 if h else self._subrnodes - 2 for h in hasordint])
        self._hasordint = driver.mem_alloc_s(len(self._rptraits), np.bool)
        self._ncloudsptor = driver.mem_alloc_s(size, np.int32)
        self._hasordint[0][:] = hasordint
        driver.mem_copy_h2d(self._hasordint[0], self._hasordint[1])

    def _impl_evaluate(
            self, driver, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):

        # Calculate the number of clouds per trait and subring.
        # The latter happens when the trait has no ordinary integral.
        ncloudspt = []
        for trait, pnames, in zip(self._rptraits, self._rpt_pnames):
            tparams = {oname: params[nname] for oname, nname in pnames.items()}

            #
            for pdesc in trait.params_nw(self._nrnodes):
                tparams[pdesc.name()] = tparams[pdesc.name()][1:-1]

            integral = trait.integrate(tparams, self._m_subrnodes[0][1:-1], self._subrsep)

            tnclouds = integral / self._cflux
            if trait.has_ordinary_integral():
                ncloudspt.append(tnclouds)
            else:
                ncloudspt.extend(tnclouds.astype(np.int32))

        ncloudspt = list(itertools.accumulate(ncloudspt))
        nclouds = int(ncloudspt[-1])

        self._ncloudsptor[0][:] = ncloudspt
        driver.mem_copy_h2d(self._ncloudsptor[0], self._ncloudsptor[1])

        rdata = None
        vdata = None
        ddata = None

        if out_extra is not None:
            shape = spat_size[::-1]
            if self._rptraits:
                rdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill_d(rdata, 0)
            if self._vptraits:
                vdata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill_d(vdata, np.nan)
            if self._dptraits:
                ddata = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill_d(ddata, np.nan)

        self._disk.evaluate(
            self._cflux, nclouds, self._ncloudsptor[1], self._hasordint[1],
            self._loose,
            self._tilted,
            self._m_subrnodes[1],
            self._m_vsys_pvalues[1],
            self._m_xpos_pvalues[1],
            self._m_ypos_pvalues[1],
            self._m_posa_pvalues[1],
            self._m_incl_pvalues[1],
            self._m_rpt_uids[1],
            self._m_rpt_cvalues[1], self._m_rpt_ccounts[1],
            self._m_rpt_pvalues[1], self._m_rpt_pcounts[1],
            self._m_rht_uids[1],
            self._m_rht_cvalues[1], self._m_rht_ccounts[1],
            self._m_rht_pvalues[1], self._m_rht_pcounts[1],
            self._m_vpt_uids[1],
            self._m_vpt_cvalues[1], self._m_vpt_ccounts[1],
            self._m_vpt_pvalues[1], self._m_vpt_pcounts[1],
            self._m_vht_uids[1],
            self._m_vht_cvalues[1], self._m_vht_ccounts[1],
            self._m_vht_pvalues[1], self._m_vht_pcounts[1],
            self._m_dpt_uids[1],
            self._m_dpt_cvalues[1], self._m_dpt_ccounts[1],
            self._m_dpt_pvalues[1], self._m_dpt_pcounts[1],
            self._m_dht_uids[1],
            self._m_dht_cvalues[1], self._m_dht_ccounts[1],
            self._m_dht_pvalues[1], self._m_dht_pcounts[1],
            self._m_wpt_uids[1],
            self._m_wpt_cvalues[1], self._m_wpt_ccounts[1],
            self._m_wpt_pvalues[1], self._m_wpt_pcounts[1],
            self._m_spt_uids[1],
            self._m_spt_cvalues[1], self._m_spt_ccounts[1],
            self._m_spt_pvalues[1], self._m_spt_pcounts[1],
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            image, scube, rcube,
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
