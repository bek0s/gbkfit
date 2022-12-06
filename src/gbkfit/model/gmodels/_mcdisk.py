
import itertools
import logging

import numpy as np

from . import _disk


__all__ = ['MCDisk']


_log = logging.getLogger(__name__)


class MCDisk(_disk.Disk):

    def __init__(
            self,
            cflux,
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

        self._cflux = cflux
        # Array with number of clouds per trait.
        # For traits without an analytical integral,
        # we store the number of clouds in each of their rings
        # Actually, we store the cumulative sum of the array.
        # This reduces the number of calculations during evaluation.
        self._s_nclouds = [None, None]
        # Has-analytical-integral flag per trait
        self._s_hasaintegral = [None, None]

    def cflux(self):
        return self._cflux

    def _impl_prepare(self, driver, dtype):
        self._s_hasaintegral = driver.mem_alloc_s(len(self._rptraits), np.bool)
        hasaintegral = [t.has_analytical_integral() for t in self._rptraits]
        self._s_hasaintegral[0][:] = hasaintegral
        driver.mem_copy_h2d(self._s_hasaintegral[0], self._s_hasaintegral[1])
        nrings = self._nsubrnodes - 2
        size = sum([1 if h else nrings for h in hasaintegral])
        self._s_nclouds = driver.mem_alloc_s(size, np.int32)

    def _impl_evaluate(
            self, driver, params,
            image, scube, rcube, wcube, rdata, vdata, ddata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):

        # Calculate the number of clouds per trait or subring.
        # The latter happens when the trait has no analytical integral.
        nclouds = []
        for trait, pnames, in zip(self._rptraits, self._rpt_pnames):
            tparams = {oname: params[nname] for oname, nname in pnames.items()}

            #
            for pdesc in trait.params_rnw(self._nrnodes):
                tparams[pdesc.name()] = tparams[pdesc.name()][1:-1]

            integral = trait.integrate(tparams, self._s_subrnodes[0][1:-1])

            tnclouds = integral / self._cflux
            if trait.has_analytical_integral():
                nclouds.append(tnclouds)
            else:
                nclouds.extend(tnclouds.astype(np.int32))

        ncloudspt = list(itertools.accumulate(nclouds))
        #print("ncloudspt:", ncloudspt)

        nclouds = int(ncloudspt[-1])

        self._s_nclouds[0][:] = ncloudspt
        driver.mem_copy_h2d(self._s_nclouds[0], self._s_nclouds[1])

        if nclouds < 0:
            raise RuntimeError('negative flux not working yet')
        print(nclouds)

        if out_extra is not None:
            out_extra['nclouds'] = nclouds

        self._backend.mcdisk_evaluate(
            self._cflux, nclouds, self._s_nclouds[1], self._s_hasaintegral[1],
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
            pass
