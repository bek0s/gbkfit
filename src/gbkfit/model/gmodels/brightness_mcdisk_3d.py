
from collections.abc import Sequence

from . import _component, _mcdisk, common, traits
from ._component import BPT, BHT, ZPT, SPT, WPT, SPATIAL_NWMODES
from .core import BrightnessComponent3D


__all__ = [
    'BrightnessMCDisk3D'
]


class BrightnessMCDisk3D(BrightnessComponent3D):

    _slots = (BPT, BHT, ZPT, SPT, WPT)
    _nwmodes = SPATIAL_NWMODES

    @staticmethod
    def type():
        return 'mcdisk'

    @classmethod
    def load(cls, info):
        return cls(**_component.load_options(
            cls, info, cls._slots, cls._nwmodes))

    def dump(self):
        return (
            dict(type=self.type())
            | _component.dump_name(self)
            | _component.dump_disk(self._disk, self._slots, self._nwmodes))

    def __init__(
            self,
            cflux: int | float,
            loose: bool,
            tilted: bool,
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
            bhtraits: traits.BHTrait | Sequence[traits.BHTrait],
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
            xpos_nwmode: common.NWMode | None = None,
            ypos_nwmode: common.NWMode | None = None,
            posa_nwmode: common.NWMode | None = None,
            incl_nwmode: common.NWMode | None = None,
            seed: int = 0,
            name: str | None = None
    ):
        super().__init__(name)
        self._disk = _component.make_disk(
            type(self), _mcdisk.MCDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            nwmodes=dict(
                xpos=xpos_nwmode, ypos=ypos_nwmode,
                posa=posa_nwmode, incl=incl_nwmode),
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

    def evaluate(self, driver, params, grid, outputs, dtype, out_extra):
        # The density of the disk is its brightness
        disk_outputs = dict(
            image=outputs['image'],
            wdata=outputs['wdata'],
            rdata=outputs['bdata'],
            opacity=outputs['odata'],
            ordata=outputs['obdata'])
        self._disk.evaluate(
            driver, params, grid, disk_outputs, dtype, out_extra)
