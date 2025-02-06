
from collections.abc import Sequence

from . import _detail, _smdisk, common, traits
from .core import SpectralComponent2D
from gbkfit.utils import parseutils


__all__ = [
    'SpectralSMDisk2D'
]


class SpectralSMDisk2D(SpectralComponent2D):

    @staticmethod
    def type():
        return 'smdisk'

    # @classmethod
    # def load(cls, info):
    #     desc = parseutils.make_typed_desc(cls, 'gmodel component')
    #     info.update(dict(
    #         vsys_nwmode=common.nwmode_parser.load(info.get('vsys_nwmode')),
    #         xpos_nwmode=common.nwmode_parser.load(info.get('xpos_nwmode')),
    #         ypos_nwmode=common.nwmode_parser.load(info.get('ypos_nwmode')),
    #         posa_nwmode=common.nwmode_parser.load(info.get('posa_nwmode')),
    #         incl_nwmode=common.nwmode_parser.load(info.get('incl_nwmode')),
    #         bptraits=traits.bpt_parser.load(info.get('bptraits')),
    #         vptraits=traits.vpt_parser.load(info.get('vptraits')),
    #         dptraits=traits.dpt_parser.load(info.get('dptraits')),
    #         sptraits=traits.spt_parser.load(info.get('sptraits')),
    #         wptraits=traits.wpt_parser.load(info.get('wptraits'))))
    #     opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
    #     return cls(**opts)

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        parseutils.load_option_and_update_info(
            traits.bpt_parser, info, 'bptraits', required=True, allow_none=False)
        parseutils.load_option_and_update_info(
            traits.vpt_parser, info, 'vptraits', required=True, allow_none=False)
        parseutils.load_option_and_update_info(
            traits.dpt_parser, info, 'dptraits', required=True, allow_none=False)
        parseutils.load_option_and_update_info(
            traits.spt_parser, info, 'sptraits', required=False, allow_none=False)
        parseutils.load_option_and_update_info(
            traits.wpt_parser, info, 'wptraits', required=False, allow_none=False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'vsys_nwmode', False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'xpos_nwmode', False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'ypos_nwmode', False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'posa_nwmode', False)
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, 'incl_nwmode', False)
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
            vsys_nwmode=common.nwmode_parser.dump(self._disk.vsys_nwmode()),
            xpos_nwmode=common.nwmode_parser.dump(self._disk.xpos_nwmode()),
            ypos_nwmode=common.nwmode_parser.dump(self._disk.ypos_nwmode()),
            posa_nwmode=common.nwmode_parser.dump(self._disk.posa_nwmode()),
            incl_nwmode=common.nwmode_parser.dump(self._disk.incl_nwmode()),
            bptraits=traits.bpt_parser.dump(self._disk.rptraits()),
            vptraits=traits.vpt_parser.dump(self._disk.vptraits()),
            dptraits=traits.dpt_parser.dump(self._disk.dptraits()),
            sptraits=traits.spt_parser.dump(self._disk.sptraits()),
            wptraits=traits.wpt_parser.dump(self._disk.wptraits()))

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
        rnode_args = _detail.parse_component_rnode_args(
            rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp)
        nwmode_velocity_args = _detail.validate_component_nwmodes_for_velocity(
            loose, vsys_nwmode)
        nwmode_geometry_args = _detail.validate_component_nwmodes_for_geometry(
            loose, tilted, xpos_nwmode, ypos_nwmode, posa_nwmode, incl_nwmode)
        trait_args = _detail.parse_component_s2d_traits(
            bptraits,
            vptraits,
            dptraits,
            sptraits,
            wptraits)
        _detail.rename_bx_to_rx_traits(trait_args)
        all_traits = sum(trait_args.values(), ())
        _detail.check_traits_common(all_traits)
        self._disk = _smdisk.SMDisk(
            loose=loose, tilted=tilted,
            **rnode_args,
            **nwmode_velocity_args,
            **nwmode_geometry_args,
            **trait_args,
            rhtraits=(),
            vhtraits=(), dhtraits=(),
            zptraits=())

    def params(self):
        return self._disk.params()

    def is_weighted(self):
        return bool(self._disk.wptraits())

    def evaluate(
            self,
            driver, params, scube, wdata, bdata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        spat_size = spat_size + (1,)
        spat_step = spat_step + (0,)
        spat_zero = spat_zero + (0,)
        self._disk.evaluate(
            driver, params, None, None, scube, wdata, bdata, None,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)
