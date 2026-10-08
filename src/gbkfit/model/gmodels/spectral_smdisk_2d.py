
from collections.abc import Sequence

from . import _smdisk, common, traits
from ._component import (
    BPT, VPT, DPT, SPT, WPT, SPECTRAL_NWMODES, EmissionDiskComponent)
from .core import SpectralComponent2D


__all__ = [
    'SpectralSMDisk2D'
]


class SpectralSMDisk2D(EmissionDiskComponent, SpectralComponent2D):

    _disk_class = _smdisk.SMDisk
    _slots = (BPT, VPT, DPT, SPT, WPT)
    _nwmodes = SPECTRAL_NWMODES

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
                bptraits=bptraits, vptraits=vptraits, dptraits=dptraits,
                sptraits=sptraits, wptraits=wptraits))
