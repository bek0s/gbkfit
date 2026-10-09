
from collections.abc import Sequence

from gbkfit.utils import parseutils
from . import _component, _smdisk, common, traits
from ._component import BPT, VPT, DPT, SPT, WPT, SPECTRAL_NWMODES
from .base import SpectralComponent2D
from .lines import Line, line_parser


__all__ = [
    'SpectralSMDisk2D'
]


class SpectralSMDisk2D(SpectralComponent2D):

    _slots = (BPT, VPT, DPT, SPT, WPT)
    _nwmodes = SPECTRAL_NWMODES

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):
        parseutils.load_option_and_update_info(
            line_parser, info, 'lines')
        return cls(**_component.load_options(
            cls, info, cls._slots, cls._nwmodes))

    def dump(self):
        return (
            dict(type=self.type())
            | _component.dump_name(self)
            | _component.dump_disk(self._disk, self._slots, self._nwmodes)
            | _component.dump_lines(self._lines))

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
            incl_nwmode: common.NWMode | None = None,
            lines: Sequence[Line] | None = None,
            name: str | None = None
    ):
        """
        lines are the emission lines of the component (see Lines; by
        default one line at the velocity of the spectral axis).
        """
        super().__init__(name)
        self._disk = _component.make_disk(
            type(self), _smdisk.SMDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            nwmodes=dict(
                vsys=vsys_nwmode, xpos=xpos_nwmode, ypos=ypos_nwmode,
                posa=posa_nwmode, incl=incl_nwmode),
            traits_=dict(
                bptraits=bptraits, vptraits=vptraits, dptraits=dptraits,
                sptraits=sptraits, wptraits=wptraits))
        self._lines = _component.make_lines(self._disk, lines)

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
        return _component.SpectralDiskComponentPlan(
            self, self._disk.plan(driver, len(selected), dtype),
            self._lines, selected, spectral)

    def disk_outputs(self, outputs):
        """The outputs of the disk, from those of the component."""
        # The density of the disk is its brightness
        return dict(
            scube=outputs['scube'],
            wdata=outputs['wdata'],
            rdata=outputs['bdata'])
