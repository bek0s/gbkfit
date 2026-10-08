
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
        rptraits = self.rptraits()
        self._s_hasaintegral = driver.mem_alloc_s(len(rptraits), bool)
        hasaintegral = [t.has_analytical_integral() for t in rptraits]
        self._s_hasaintegral[0][:] = hasaintegral
        driver.mem_copy_h2d(self._s_hasaintegral[0], self._s_hasaintegral[1])
        nrings = self._nsubrnodes - 2
        size = sum([1 if h else nrings for h in hasaintegral])
        self._s_ncloudsptor = driver.mem_alloc_s(size, np.int32)

    def _impl_evaluate(self, driver, params, grid_and_outputs, out_extra):

        # Calculate the number of clouds per trait or ring.
        ncloudsptor = []
        rpt_params = self._trait_params['rpt']
        ring_centers = np.array(self._subrnodes[1:-1], self._dtype)
        for trait, pnames in zip(self.rptraits(), rpt_params.pnames):
            # Make a parameter dict for the current trait
            # Use the original names and not the new/prefixed ones
            trait_params = {}
            for old_name, new_name in pnames.items():
                if rpt_params.isnw[new_name]:
                    trait_params[old_name] = params[new_name][1:-1]
                else:
                    trait_params[old_name] = params[new_name]

            # Calculate the integral of this trait.
            # If the trait has an analytical integral, this will return
            # a single value. Otherwise, it will return an iterable
            # with the integral of each ring for that trait.
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

        self._backend.mcdisk_evaluate(
            self._native_disk,
            cflux=self._cflux,
            nclouds=nclouds,
            ncloudscsum=self._s_ncloudsptor[1],
            hasordint=self._s_hasaintegral[1],
            **grid_and_outputs)
