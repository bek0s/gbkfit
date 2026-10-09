
import logging

import numpy as np

from . import _disk, traits


__all__ = ['MCDisk', 'MCDiskPlan']


_log = logging.getLogger(__name__)


class MCDisk(_disk.Disk):

    # The traits that the Monte Carlo disk does not support yet
    unsupported_traits = (
        traits.BPTraitMixtureExponential,
        traits.BPTraitMixtureGauss,
        traits.BPTraitMixtureGGauss,
        traits.BPTraitMixtureMoffat,
        traits.BPTraitNWDistortion,
        traits.DPTraitMixtureExponential,
        traits.DPTraitMixtureGauss,
        traits.DPTraitMixtureGGauss,
        traits.DPTraitMixtureMoffat,
        traits.DPTraitNWDistortion,
        traits.OPTraitMixtureExponential,
        traits.OPTraitMixtureGauss,
        traits.OPTraitMixtureGGauss,
        traits.OPTraitMixtureMoffat,
        traits.OPTraitNWDistortion)

    def __init__(
            self, cflux, seed,
            loose, tilted, rnodes, rstep, interp, nwmodes, traits_,
            prefixes, rdata_key):
        super().__init__(
            loose, tilted, rnodes, rstep, interp, nwmodes, traits_,
            prefixes, rdata_key)
        if seed < 0:
            raise RuntimeError(f"seed must be >= 0; supplied value: {seed}")
        self._cflux = cflux
        # The seed of the random numbers of the clouds
        self._seed = seed

    def cflux(self):
        return self._cflux

    def seed(self):
        return self._seed

    def options(self):
        return dict(cflux=self._cflux, seed=self._seed)

    def plan(self, driver, nlines, dtype):
        return MCDiskPlan(self, driver, nlines, dtype)


class MCDiskPlan(_disk.DiskPlan):

    def __init__(self, disk, driver, nlines, dtype):
        super().__init__(disk, driver, nlines, dtype)
        # The clouds are made in pools: one for each density trait with
        # an analytical integral, and one for each ring of the others
        # (the rings are centred on the subnodes between the first and
        # the last, which are the edges of the disk). For each pool: the
        # cumulative number of clouds, and the (signed) flux of each of
        # its clouds. Also, for each density trait, whether it has an
        # analytical integral.
        rptraits = disk.traits('rpt')
        analytical = [t.has_analytical_integral() for t in rptraits]
        self._s_has_analytical_integral = driver.mem_alloc_s(
            len(rptraits), bool)
        host, device = self._s_has_analytical_integral
        host[:] = analytical
        driver.mem_copy_h2d(host, device)
        nrings = len(disk.subrnodes()) - 2
        npools = sum([1 if h else nrings for h in analytical])
        self._s_ncloudscsum = driver.mem_alloc_s(npools, np.int32)
        self._s_cloud_flux = driver.mem_alloc_s(npools, dtype)

    def _impl_evaluate(self, params, grid_and_outputs, out_extra):

        disk = self._disk
        driver = self._driver

        # The flux of each pool
        pool_flux = []
        rpt_params = disk.trait_params('rpt')
        ring_centers = np.array(disk.subrnodes()[1:-1], self._dtype)
        for trait, pnames in zip(disk.traits('rpt'), rpt_params.pnames):
            # Make a parameter dict for the current trait
            # Use the original names and not the new/prefixed ones
            trait_params = {}
            for old_name, new_name in pnames.items():
                if rpt_params.sampling[new_name] is not None:
                    trait_params[old_name] = params[new_name][1:-1]
                else:
                    trait_params[old_name] = params[new_name]
            # The flux of the trait (one value if it has an analytical
            # integral), or of each of its rings
            pool_flux.extend(np.atleast_1d(
                trait.cloud_flux(trait_params, ring_centers)))
        pool_flux = np.asarray(pool_flux, np.float64)

        # Each pool has as many clouds as its flux needs at cflux each
        # (and at least one), which share its flux exactly: a negative
        # flux gives negative clouds
        nclouds = np.where(
            pool_flux != 0,
            np.maximum(np.rint(np.abs(pool_flux) / disk.cflux()), 1),
            0).astype(np.int32)
        cloud_flux = np.divide(
            pool_flux, nclouds, out=np.zeros_like(pool_flux),
            where=nclouds > 0)

        self._s_ncloudscsum[0][:] = np.cumsum(nclouds)
        self._s_cloud_flux[0][:] = cloud_flux
        driver.mem_copy_h2d(self._s_ncloudscsum[0], self._s_ncloudscsum[1])
        driver.mem_copy_h2d(self._s_cloud_flux[0], self._s_cloud_flux[1])

        self._backend.mcdisk_evaluate(
            self._native_disk,
            cloud_flux=self._s_cloud_flux[1],
            seed=disk.seed(),
            nclouds=int(nclouds.sum()),
            ncloudscsum=self._s_ncloudscsum[1],
            has_analytical_integral=self._s_has_analytical_integral[1],
            **grid_and_outputs)
