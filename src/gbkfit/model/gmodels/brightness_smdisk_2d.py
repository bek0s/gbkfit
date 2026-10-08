
from collections.abc import Sequence

from . import _component, _smdisk, common, traits
from ._component import BPT, SPT, WPT, SPATIAL_NWMODES
from .core import BrightnessComponent2D


__all__ = [
    'BrightnessSMDisk2D'
]


class BrightnessSMDisk2D(BrightnessComponent2D):

    _slots = (BPT, SPT, WPT)
    _nwmodes = SPATIAL_NWMODES

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):
        return cls(**_component.load_options(
            cls, info, cls._slots, cls._nwmodes))

    def dump(self):
        return dict(type=self.type()) | _component.dump_disk(
            self._disk, self._slots, self._nwmodes)

    def __init__(
            self,
            loose: bool,
            tilted: bool,
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
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
            incl_nwmode: common.NWMode | None = None
    ):
        self._disk = _component.make_disk(
            type(self), _smdisk.SMDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            nwmodes=dict(
                xpos=xpos_nwmode, ypos=ypos_nwmode,
                posa=posa_nwmode, incl=incl_nwmode),
            traits_=dict(
                bptraits=bptraits, sptraits=sptraits, wptraits=wptraits))

    def pdescs(self):
        return self._disk.pdescs()

    def has_weights(self):
        return bool(self._disk.traits('wpt'))

    def constants(self):
        return dict(rnodes=self._disk.rnodes())

    def evaluate(self, driver, params, grid, outputs, dtype, out_extra):
        # The density of the disk is its brightness
        disk_outputs = dict(
            image=outputs['image'],
            wdata=outputs['wdata'],
            rdata=outputs['bdata'])
        self._disk.evaluate(
            driver, params, grid, disk_outputs, dtype, out_extra)
