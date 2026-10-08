
from collections.abc import Sequence

from . import _component, _mcdisk, common, traits
from ._component import OPT, OHT, ZPT, SPT, WPT, SPATIAL_NWMODES
from .core import OpacityComponent3D


__all__ = [
    'OpacityMCDisk3D'
]


class OpacityMCDisk3D(OpacityComponent3D):

    _slots = (OPT, OHT, ZPT, SPT, WPT)
    _nwmodes = SPATIAL_NWMODES

    @staticmethod
    def type():
        return 'mcdisk'

    @classmethod
    def load(cls, info):
        return cls(**_component.load_options(
            cls, info, cls._slots, cls._nwmodes))

    def dump(self):
        return dict(type=self.type()) | _component.dump_disk(
            self._disk, self._slots, self._nwmodes)

    def __init__(
            self,
            cflux: int | float,
            loose: bool,
            tilted: bool,
            optraits: traits.OPTrait | Sequence[traits.OPTrait],
            ohtraits: traits.OHTrait | Sequence[traits.OHTrait],
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
            seed: int = 0
    ):
        self._disk = _component.make_disk(
            type(self), _mcdisk.MCDisk, self._slots,
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            nwmodes=dict(
                xpos=xpos_nwmode, ypos=ypos_nwmode,
                posa=posa_nwmode, incl=incl_nwmode),
            traits_=dict(
                optraits=optraits, ohtraits=ohtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits),
            cflux=cflux, seed=seed)

    def pdescs(self):
        return self._disk.pdescs()

    def has_weights(self):
        return bool(self._disk.traits('wpt'))

    def constants(self):
        return dict(rnodes=self._disk.rnodes())

    def evaluate(self, driver, params, grid, outputs, dtype, out_extra):
        # The density of the disk is the opacity
        disk_outputs = dict(
            rdata=outputs['odata'])
        self._disk.evaluate(
            driver, params, grid, disk_outputs, dtype, out_extra)
