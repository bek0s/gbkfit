
from collections.abc import Sequence

from . import _detail, _smdisk, common, traits
from .core import BrightnessComponent3D
from gbkfit.utils import parseutils


__all__ = [
    'BrightnessSMDisk3D'
]


class BrightnessSMDisk3D(BrightnessComponent3D):

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info, *args, **kwargs):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        parseutils.load_option_and_update_info(
            traits.bpt_parser, info, 'bptraits', required=True)
        parseutils.load_option_and_update_info(
            traits.bht_parser, info, 'bhtraits', required=True)
        parseutils.load_option_and_update_info(
            traits.zpt_parser, info, 'zptraits', required=False)
        parseutils.load_option_and_update_info(
            traits.spt_parser, info, 'sptraits', required=False)
        parseutils.load_option_and_update_info(
            traits.wpt_parser, info, 'wptraits', required=False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'xpos_nwmode', required=False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'ypos_nwmode', required=False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'posa_nwmode', required=False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'incl_nwmode', required=False)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            loose=self._disk.loose(),
            tilted=self._disk.tilted(),
            rnodes=self._disk.rnodes(),
            rstep=self._disk.rstep(),
            interp=self._disk.interp().type(),
            xpos_nwmode=common.nwmode_parser.dump(self._disk.xpos_nwmode()),
            ypos_nwmode=common.nwmode_parser.dump(self._disk.ypos_nwmode()),
            posa_nwmode=common.nwmode_parser.dump(self._disk.posa_nwmode()),
            incl_nwmode=common.nwmode_parser.dump(self._disk.incl_nwmode()),
            bptraits=traits.bpt_parser.dump(self._disk.rptraits()),
            bhtraits=traits.bht_parser.dump(self._disk.rhtraits()),
            zptraits=traits.zpt_parser.dump(self._disk.zptraits()),
            sptraits=traits.spt_parser.dump(self._disk.sptraits()),
            wptraits=traits.wpt_parser.dump(self._disk.wptraits()))

    def __init__(
            self,
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
            incl_nwmode: common.NWMode | None = None
    ):
        rnode_args = _detail.parse_component_rnode_args(
            rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp)
        nwmode_geometry_args = _detail.validate_component_nwmodes_for_geometry(
            loose, tilted, xpos_nwmode, ypos_nwmode, posa_nwmode, incl_nwmode)
        trait_args = _detail.parse_component_b3d_traits(
            bptraits, bhtraits,
            zptraits,
            sptraits,
            wptraits)
        _detail.rename_bx_to_rx_traits(trait_args)
        all_traits = sum(trait_args.values(), ())
        _detail.check_traits_common(all_traits)
        self._disk = _smdisk.SMDisk(
            loose=loose, tilted=tilted,
            **rnode_args,
            vsys_nwmode=None,
            **nwmode_geometry_args,
            **trait_args,
            vptraits=(), vhtraits=(),
            dptraits=(), dhtraits=())

    def pdescs(self):
        return self._disk.pdescs()

    def is_weighted(self):
        return bool(self._disk.wptraits())

    def evaluate(
            self,
            driver, params, odata, image, wdata, bdata, obdata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        spec_size = 1
        spec_step = 0
        spec_zero = 0
        self._disk.evaluate(
            driver, params, odata, image, None, wdata, bdata, obdata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)
