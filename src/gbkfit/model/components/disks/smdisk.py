from collections.abc import Sequence

from gbkfit.utils import parseutils
from .._detail import dump_lines, dump_name
from ..base import (
    BrightnessComponent2D, BrightnessComponent3D, OpacityComponent3D,
    SpectralComponent2D, SpectralComponent3D)
from ..lines import Line, line_parser
from . import _detail, _disk, traits
from ._detail import (
    BHT, BPT, DHT, DPT, OHT, OPT, SPT, VHT, VPT, WPT, ZPT)


__all__ = [
    'BrightnessSMDisk2D',
    'BrightnessSMDisk3D',
    'SpectralSMDisk2D',
    'SpectralSMDisk3D',
    'OpacitySMDisk3D'
]


class SMDisk(_disk.Disk):

    def plan(self, driver, nlines, dtype):
        return SMDiskPlan(self, driver, nlines, dtype)


class SMDiskPlan(_disk.DiskPlan):

    def _impl_evaluate(self, params, grid_and_outputs, out_extra):
        self._backend.smdisk_evaluate(self._native_disk, **grid_and_outputs)


class BrightnessSMDisk2D(BrightnessComponent2D):

    _slots = (BPT, SPT, WPT)

    @staticmethod
    def type():
        return 'smdisk'

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
            loose: bool,
            tilted: bool,
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            wptraits: traits.WPTrait | Sequence[traits.WPTrait] | None = None,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None,
            rstep: float | None = None,
            interp: str = 'linear',
            name: str | None = None
    ):
        super().__init__(name)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, sptraits=sptraits, wptraits=wptraits))

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
            rdata=outputs['bdata'])


class BrightnessSMDisk3D(BrightnessComponent3D):

    _slots = (BPT, BHT, ZPT, SPT, WPT)

    @staticmethod
    def type():
        return 'smdisk'

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
            name: str | None = None
    ):
        super().__init__(name)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits))

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


class SpectralSMDisk2D(SpectralComponent2D):

    _slots = (BPT, VPT, DPT, SPT, WPT)

    @staticmethod
    def type():
        return 'smdisk'

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
            loose: bool,
            tilted: bool,
            bptraits: traits.BPTrait | Sequence[traits.BPTrait],
            vptraits: traits.VPTrait | Sequence[traits.VPTrait],
            dptraits: traits.DPTrait | Sequence[traits.DPTrait],
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            wptraits: traits.WPTrait | Sequence[traits.WPTrait] | None = None,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None,
            rstep: float | None = None,
            interp: str = 'linear',
            lines: Sequence[Line] | None = None,
            name: str | None = None
    ):
        """
        lines are the emission lines of the component (see Lines; by
        default one line at the velocity of the spectral axis).
        """
        super().__init__(name)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, vptraits=vptraits, dptraits=dptraits,
                sptraits=sptraits, wptraits=wptraits))
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
            rdata=outputs['bdata'])


class SpectralSMDisk3D(SpectralComponent3D):

    _slots = (BPT, BHT, VPT, VHT, DPT, DHT, ZPT, SPT, WPT)

    @staticmethod
    def type():
        return 'smdisk'

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
            lines: Sequence[Line] | None = None,
            name: str | None = None
    ):
        """
        lines are the emission lines of the component (see Lines; by
        default one line at the velocity of the spectral axis).
        """
        super().__init__(name)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='bdata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                bptraits=bptraits, bhtraits=bhtraits, vptraits=vptraits,
                vhtraits=vhtraits, dptraits=dptraits, dhtraits=dhtraits,
                zptraits=zptraits, sptraits=sptraits, wptraits=wptraits))
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


class OpacitySMDisk3D(OpacityComponent3D):

    _slots = (OPT, OHT, ZPT, SPT, WPT)

    @staticmethod
    def type():
        return 'smdisk'

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
            name: str | None = None
    ):
        super().__init__(name)
        self._disk = _detail.make_disk(
            type(self), SMDisk, self._slots,
            rdata_key='odata',
            loose=loose, tilted=tilted,
            rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
            rnodes=rnodes, rstep=rstep, interp=interp,
            traits_=dict(
                optraits=optraits, ohtraits=ohtraits, zptraits=zptraits,
                sptraits=sptraits, wptraits=wptraits))

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
