
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

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(dict(
            vsys_nwmode=common.nwmode_parser.load(opts.get('vsys_nwmode')),
            xpos_nwmode=common.nwmode_parser.load(opts.get('xpos_nwmode')),
            ypos_nwmode=common.nwmode_parser.load(opts.get('ypos_nwmode')),
            posa_nwmode=common.nwmode_parser.load(opts.get('posa_nwmode')),
            incl_nwmode=common.nwmode_parser.load(opts.get('incl_nwmode')),
            bptraits=traits.bpt_parser.load(opts.get('bptraits')),
            vptraits=traits.vpt_parser.load(opts.get('vptraits')),
            dptraits=traits.dpt_parser.load(opts.get('dptraits')),
            sptraits=traits.spt_parser.load(opts.get('sptraits')),
            wptraits=traits.wpt_parser.load(opts.get('wptraits'))))
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
            loose,
            tilted,
            bptraits,
            vptraits,
            dptraits,
            sptraits=None,
            wptraits=None,
            rnmin=None,
            rnmax=None,
            rnsep=None,
            rnlen=None,
            rnodes=None,
            rstep=None,
            interp='linear',
            vsys_nwmode=None,
            xpos_nwmode=None, ypos_nwmode=None,
            posa_nwmode=None, incl_nwmode=None):
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
