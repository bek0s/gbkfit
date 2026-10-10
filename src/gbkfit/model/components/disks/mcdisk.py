import logging
from collections.abc import Sequence

import numpy as np

from gbkfit.utils import parseutils
from .._detail import dump_lines, dump_name
from ..base import (
    BrightnessComponent3D, OpacityComponent3D, SpectralComponent3D)
from ..lines import Line, line_parser
from . import _detail, _disk, traits
from ._detail import (
    BHT, BPT, DHT, DPT, OHT, OPT, SPT, VHT, VPT, WPT, ZPT)


__all__ = [
    'BrightnessMCDisk3D',
    'SpectralMCDisk3D',
    'OpacityMCDisk3D'
]


_log = logging.getLogger(__name__)

# The most clouds the kernels draw: they count them in int32
_MAX_CLOUDS = np.iinfo(np.int32).max


class MCDisk(_disk.Disk):

    def __init__(
            self, cflux, seed,
            loose, tilted, rnodes, rstep, interp, traits_,
            prefixes, rdata_key):
        super().__init__(
            loose, tilted, rnodes, rstep, interp, traits_,
            prefixes, rdata_key)
        if seed < 0:
            raise RuntimeError(f"seed must be >= 0; supplied value: {seed}")
        if not cflux > 0:
            raise RuntimeError(f"cflux must be greater than 0; it is {cflux}")
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
            0)
        if not np.sum(nclouds) <= _MAX_CLOUDS:
            raise RuntimeError(
                f"the Monte Carlo disk needs {np.sum(nclouds):.3g} clouds "
                f"of flux cflux = {disk.cflux()}, but the kernels draw at "
                f"most {_MAX_CLOUDS}; give it a larger cflux")
        nclouds = nclouds.astype(np.int32)
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


class BrightnessMCDisk3D(BrightnessComponent3D):

    _slots = (BPT, BHT, ZPT, SPT, WPT)

    @staticmethod
    def type():
        return 'mcdisk'

    @classmethod
    def load(cls, info):
        return cls(**_detail.load_options(cls, info, cls._slots))

    def dump(self):
        return (
            dict(type=self.type())
            | dump_name(self)
            | _detail.dump_disk(self._disk, self._slots))

    def __init__(
            self,
            cflux: float,
            loose: bool,
            tilted: bool,
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
            bhtraits: traits.BHTrait | Sequence[traits.BHTrait],
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            wptraits: traits.WPTrait | Sequence[traits.WPTrait] | None = None,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None,
            rstep: float | None = None,
            interp: str = 'linear',
            seed: int = 0,
            name: str | None = None
    ):
        super().__init__(name)
        self._disk = _detail.make_disk(
            type(self), MCDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits),
            cflux=cflux, seed=seed)

    def pdescs(self):
        return self._disk.pdescs()

    def has_weights(self):
        return bool(self._disk.traits('wpt'))

    def constants(self):
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def plan(self, driver, spectral, dtype, lines):
        return _detail.DiskComponentPlan(
            self, self._disk.plan(driver, 0, dtype))

    def disk_outputs(self, outputs):
        """The outputs of the disk, from those of the component."""
        # The density of the disk is its brightness
        return dict(
            image=outputs['image'],
            wdata=outputs['wdata'],
            rdata=outputs['bdata'],
            opacity=outputs['odata'],
            ordata=outputs['obdata'])


class SpectralMCDisk3D(SpectralComponent3D):

    _slots = (BPT, BHT, VPT, VHT, DPT, DHT, ZPT, SPT, WPT)

    @staticmethod
    def type():
        return 'mcdisk'

    @classmethod
    def load(cls, info):
        parseutils.load_option_and_update_info(
            line_parser, info, 'lines')
        return cls(**_detail.load_options(cls, info, cls._slots))

    def dump(self):
        return (
            dict(type=self.type())
            | dump_name(self)
            | _detail.dump_disk(self._disk, self._slots)
            | dump_lines(self._lines))

    def __init__(
            self,
            cflux: float,
            loose: bool,
            tilted: bool,
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
            vptraits: traits.VPTrait | Sequence[traits.VPTrait],
            dptraits: traits.DPTrait | Sequence[traits.DPTrait],
            bhtraits: traits.BHTrait | Sequence[traits.BHTrait],
            vhtraits: traits.VHTrait | Sequence[traits.VHTrait] | None = None,
            dhtraits: traits.DHTrait | Sequence[traits.DHTrait] | None = None,
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            wptraits: traits.WPTrait | Sequence[traits.WPTrait] | None = None,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None,
            rstep: float | None = None,
            interp: str = 'linear',
            seed: int = 0,
            lines: Sequence[Line] | None = None,
            name: str | None = None
    ):
        """
        lines are the emission lines of the component (see Lines; by
        default one line at the velocity of the spectral axis).
        """
        super().__init__(name)
        self._disk = _detail.make_disk(
            type(self), MCDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, vptraits=vptraits,
                vhtraits=vhtraits, dptraits=dptraits, dhtraits=dhtraits,
                zptraits=zptraits, sptraits=sptraits, wptraits=wptraits),
            cflux=cflux, seed=seed)
        self._lines = _detail.make_lines(self._disk, lines)

    def pdescs(self):
        return self._disk.pdescs() | self._lines.pdescs()

    def circular_velocity_params(self):
        return self._disk.circular_velocity_params()

    def has_weights(self):
        return bool(self._disk.traits('wpt'))

    def constants(self):
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def line_names(self):
        return self._lines.names()

    def plan(self, driver, spectral, dtype, lines):
        selected = self._lines.select(lines)
        return _detail.SpectralDiskComponentPlan(
            self, self._disk.plan(driver, len(selected), dtype),
            self._lines, selected, spectral)

    def disk_outputs(self, outputs):
        """The outputs of the disk, from those of the component."""
        # The density of the disk is its brightness
        return dict(
            scube=outputs['scube'],
            wdata=outputs['wdata'],
            rdata=outputs['bdata'],
            opacity=outputs['odata'],
            ordata=outputs['obdata'])


class OpacityMCDisk3D(OpacityComponent3D):

    _slots = (OPT, OHT, ZPT, SPT, WPT)

    @staticmethod
    def type():
        return 'mcdisk'

    @classmethod
    def load(cls, info):
        return cls(**_detail.load_options(cls, info, cls._slots))

    def dump(self):
        return (
            dict(type=self.type())
            | dump_name(self)
            | _detail.dump_disk(self._disk, self._slots))

    def __init__(
            self,
            cflux: float,
            loose: bool,
            tilted: bool,
            optraits: traits.OPTrait | Sequence[traits.OPTrait],
            ohtraits: traits.OHTrait | Sequence[traits.OHTrait],
            zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            wptraits: traits.WPTrait | Sequence[traits.WPTrait] | None = None,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None,
            rstep: float | None = None,
            interp: str = 'linear',
            seed: int = 0,
            name: str | None = None
    ):
        super().__init__(name)
        self._disk = _detail.make_disk(
            type(self), MCDisk, self._slots,
            rdata_key='odata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                optraits=optraits, ohtraits=ohtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits),
            cflux=cflux, seed=seed)

    def pdescs(self):
        return self._disk.pdescs()

    def has_weights(self):
        return bool(self._disk.traits('wpt'))

    def constants(self):
        return dict(
            rnodes=self._disk.rnodes(), subrnodes=self._disk.subrnodes())

    def plan(self, driver, spectral, dtype, lines):
        return _detail.DiskComponentPlan(
            self, self._disk.plan(driver, 0, dtype))

    def disk_outputs(self, outputs):
        """The outputs of the disk, from those of the component."""
        # The density of the disk is the opacity
        return dict(
            rdata=outputs['odata'])
