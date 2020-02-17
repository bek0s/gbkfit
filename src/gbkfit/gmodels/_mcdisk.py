
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

        self._hasordint = driver.mem_alloc(len(self._rptraits), np.bool)

        size = 0
        for i, trait in enumerate(self._rptraits):
            size += 1 if trait.has_ordinary_integral() else self._nsubrnodes
            self._hasordint[0][i] = trait.has_ordinary_integral()

        self._ncloudsptor = driver.mem_alloc(size, np.int32)

    def _impl_evaluate(
            self, driver, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):

        # Calculate the number of clouds
        ncloudspt = []
        start = 0
        for trait, names, in zip(self._rptraits, self._rpt_pnames):
            tparams = {oname: params[nname] for oname, nname in names.items()}
            integral = trait.integrate(tparams, self._m_subrnodes[0])

            if trait.has_ordinary_integral():
                ncloudspt.append(integral / self._cflux)
                size = 1
                self._ncloudsptor[0][start] = integral / self._cflux
            else:
                ncloudspt.extend((integral / self._cflux).astype(np.int32))
                size = self._nsubrnodes
                self._ncloudsptor[0][start:start + size] = integral / self._cflux

            start += size

        #self._ncloudsptor[0][:] = self._ncloudsptor[0][:] / self._cflux


        print("integral:", integral)
        print("nclouds:", self._ncloudsptor[0])
        print("nclouds:", ncloudspt)


        from time import time_ns
        t1 = time_ns()
        np.cumsum(self._ncloudsptor[0], out=self._ncloudsptor[0])
        t2 = time_ns()
        print("T:", t2 - t1)
        print("foo:", self._ncloudsptor[0])

        import itertools
        t1 = time_ns()
        ncloudspt = list(itertools.accumulate(ncloudspt))
        t2 = time_ns()
        print("T:", t2 - t1)
        print("bar:", ncloudspt)


        nclouds = self._ncloudsptor[0][-1]
        print(self._ncloudsptor[0].size)

        log.info(
            f"nclouds (total): {nclouds}\n"
            f"nclouds (per trait): {ncloudspt}")

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
                print("sum(abs(rdata)): ", np.nansum(np.abs(out_extra['rdata'])))
            if self._vptraits:
                print("sum(abs(vdata)): ", np.nansum(np.abs(out_extra['vdata'])))
            if self._dptraits:
                print("sum(abs(ddata)): ", np.nansum(np.abs(out_extra['ddata'])))
