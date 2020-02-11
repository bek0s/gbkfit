
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

        if cflux is None:
            cflux = 0.1

        self._cflux = cflux
        self._ncloudspt = None

        """
        xdata = np.array([0, 20, 40, 60, 80, 90, 100, 120, 140, 160, 180, 200])
        ydata = np.array([0, 40, 80, 50, 20, 10, 100, 110, 170, 190, 120, 200])
        self._rnodes = xdata
        rnodes = xdata
        use_subnodes = True
        radsep = 1.0
        subnodes_rmid = []
        subnodes_frac = []
        subnodes_dict = []
        rcur = self._rnodes[0]
        for i in range(len(rnodes) - 1):
            while rcur < self._rnodes[i + 1] - radsep:
                rcur += radsep if subnodes_rmid else radsep / 2
                subnodes_dict.append(i)
                subnodes_rmid.append(rcur)
        nsubnodes = len(subnodes_rmid)
        print(subnodes_rmid)
        import scipy.interpolate
        import time
        xdata_interp = subnodes_rmid
        t1 = time.time()
        for i in range(100):
            interp = scipy.interpolate.Akima1DInterpolator(xdata, ydata)
            ydata_interp = interp(xdata_interp)
        t2 = time.time()
        print(t2 - t1)
        import matplotlib.pyplot as plt
        plt.plot(xdata, ydata)
        plt.plot(xdata_interp, ydata_interp)
        plt.show()
        """

        #exit()



    def cflux(self):
        return self._cflux

    def _evaluate_impl(
            self, backend, params, image, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):

        # Perform preparations if needed
        if self._backend is not backend or self._dtype is dtype:
            self._disk = backend.make_gmodel_mcdisk(dtype)
            self._ncloudspt = backend.mem_alloc(len(self._rptraits), np.int32)

        # Calculate the number of clouds
        nclouds = 0
        ncloudspt = []
        for i, (trait, onames, nnames), in enumerate(zip(
                self._rptraits, self._rpt_ponames, self._rpt_pnnames)):
            tparams = {o: params[n] for o, n in zip(onames, nnames)}
            integral = trait.integrate(tparams, self._rnodes)
            ncloudspt.append(int(integral / self._cflux))
            nclouds += ncloudspt[-1]

        log.info(
            f"nclouds (total): {nclouds}\n"
            f"nclouds (per trait): {ncloudspt}")

        self._ncloudspt[0][:] = ncloudspt
        backend.mem_copy_h2d(self._ncloudspt[0], self._ncloudspt[1])

        rdata = None
        vdata = None
        ddata = None

        if out_extra is not None:
            shape = spat_size[::-1]
            if self._rptraits:
                rdata = backend.mem_alloc_d(shape, dtype)
                backend.mem_fill_d(rdata, 0)
            if self._vptraits:
                vdata = backend.mem_alloc_d(shape, dtype)
                backend.mem_fill_d(vdata, np.nan)
            if self._dptraits:
                ddata = backend.mem_alloc_d(shape, dtype)
                backend.mem_fill_d(ddata, np.nan)

        self._disk.evaluate(
            self._cflux, nclouds, self._ncloudspt[1],
            self._loose,
            self._tilted,
            self._m_rnodes[1],
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
                out_extra['rdata'] = backend.mem_copy_d2h(rdata)
            if self._vptraits:
                out_extra['vdata'] = backend.mem_copy_d2h(vdata)
            if self._dptraits:
                out_extra['ddata'] = backend.mem_copy_d2h(ddata)
            if self._rptraits:
                print("sum(abs(rdata)): ", np.nansum(np.abs(out_extra['rdata'])))
            if self._vptraits:
                print("sum(abs(vdata)): ", np.nansum(np.abs(out_extra['vdata'])))
            if self._dptraits:
                print("sum(abs(ddata)): ", np.nansum(np.abs(out_extra['ddata'])))
