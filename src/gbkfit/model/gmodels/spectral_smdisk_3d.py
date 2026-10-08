
from collections.abc import Sequence

from . import _smdisk, common, traits
from ._component import (
    BPT, BHT, VPT, VHT, DPT, DHT, ZPT, SPT, WPT, DiskComponent)
from .core import SpectralComponent3D


__all__ = [
    'SpectralSMDisk3D'
]


class SpectralSMDisk3D(DiskComponent, SpectralComponent3D):

    _disk_class = _smdisk.SMDisk
    _slots = (BPT, BHT, VPT, VHT, DPT, DHT, ZPT, SPT, WPT)

    @staticmethod
    def type():
        return 'smdisk'

    def __init__(
            self,
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
            rnmin: int | float | None = None,
            rnmax: int | float | None = None,
            rnsep: int | float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[int | float] | None = None,
            rstep: int | float | None = None,
            interp: str = 'linear',
            vsys_nwmode: common.NWMode | None = None,
            xpos_nwmode: common.NWMode | None = None,
            ypos_nwmode: common.NWMode | None = None,
            posa_nwmode: common.NWMode | None = None,
            incl_nwmode: common.NWMode | None = None
    ):
        super().__init__(
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            nwmodes=dict(
                vsys=vsys_nwmode, xpos=xpos_nwmode, ypos=ypos_nwmode,
                posa=posa_nwmode, incl=incl_nwmode),
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, vptraits=vptraits,
                vhtraits=vhtraits, dptraits=dptraits, dhtraits=dhtraits,
                zptraits=zptraits, sptraits=sptraits, wptraits=wptraits))

    def evaluate(
            self,
            driver, params, odata, scube, wdata, bdata, obdata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        self._disk.evaluate(
            driver, params, odata, None, scube, wdata, bdata, obdata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)
