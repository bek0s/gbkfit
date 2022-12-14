
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
        # Array containing the cumulative sum of the number of clouds
        # per trait. For traits without an analytical integral we
        # calculate the number of clouds per trait ring. The center of
        # each ring coincides with a subnode. The first and last
        # subnodes are excepted, as they are the inner and outer edges
        # of the first and last rings. We use the cumulative sum in
        # order to reduce the number of calculations during model
        # evaluation.
        self._s_ncloudsptor = [None, None]
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
        self._s_ncloudsptor = driver.mem_alloc_s(size, np.int32)

    def _impl_evaluate(
            self, driver, params,
            image, scube, rcube, wcube, rdata, vdata, ddata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):

        # Calculate the number of clouds per trait or ring.
        ncloudsptor = []
        for trait, pnames, in zip(self._rptraits, self._rpt_pnames):
            # Make a parameter dict for the current trait
            # Use the original names and not the new/prefixed ones
            trait_params = {}
            for old_name, new_name in pnames.items():
                if self._rpt_isnw[new_name]:
                    trait_params[old_name] = params[new_name][1:-1]
                else:
                    trait_params[old_name] = params[new_name]

            # Calculate the integral of this trait.
            # If the trait has an analytical integral, this will return
            # a single value. Otherwise, it will return an iterable
            # with the integral of each ring for that trait.
            ring_centers = self._s_subrnodes[0][1:-1]
            integral = trait.integrate(trait_params, ring_centers)
            # Calculate the number of clouds per trait or ring
            trait_nclouds = integral / self._cflux
            ncloudsptor.extend(np.atleast_1d(trait_nclouds).astype(np.int32))

        # Calculate cumsum
        ncloudsptor_cumsum = list(itertools.accumulate(ncloudsptor))

        # Transfer cumsum to host and then device memory
        self._s_ncloudsptor[0][:] = ncloudsptor_cumsum
        driver.mem_copy_h2d(self._s_ncloudsptor[0], self._s_ncloudsptor[1])

        # Store the total number of clouds across the entire disk
        nclouds = ncloudsptor_cumsum[-1]

        # TODO: investigate negative flux

        if out_extra is not None:
            # out_extra['nclouds'] = nclouds
            pass

        print("rstep:", self._rstep)
        print("subrnodes:", self._s_subrnodes[1])
        print("nclouds:", nclouds)
        print("ncloudsptor:", ncloudsptor)
        print("ncloudsptor(cumsum):", self._s_ncloudsptor[1])
        print("rpt_uids:", self._s_rpt_uids[1])
        print("rpt_ccounts:", self._s_rpt_ccounts[1])
        print("rpt_cvalues:", self._s_rpt_cvalues[1])
        print("rpt_pcounts:", self._s_rpt_pcounts[1])
        print("rpt_pvalues:", self._s_rpt_pvalues[1])
        print("hasaintegral:", self._s_hasaintegral[1])
        print("params:", params)
        # exit()

        self._backend.mcdisk_evaluate(
            self._cflux, nclouds,
            self._s_ncloudsptor[1],
            self._s_hasaintegral[1],
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
