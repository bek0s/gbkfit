
from . import _detail, _mcdisk, common, traits
from .core import SpectralComponent3D
from gbkfit.utils import parseutils


__all__ = [
    'SpectralMCDisk3D'
]


class SpectralMCDisk3D(SpectralComponent3D):

    @staticmethod
    def type():
        return 'mcdisk'

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
            bhtraits=traits.bht_parser.load(opts.get('bhtraits')),
            vptraits=traits.vpt_parser.load(opts.get('vptraits')),
            vhtraits=traits.vht_parser.load(opts.get('vhtraits')),
            dptraits=traits.dpt_parser.load(opts.get('dptraits')),
            dhtraits=traits.dht_parser.load(opts.get('dhtraits')),
            zptraits=traits.zpt_parser.load(opts.get('zptraits')),
            sptraits=traits.spt_parser.load(opts.get('sptraits')),
            wptraits=traits.wpt_parser.load(opts.get('wptraits'))))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            cflux=self._disk.cflux(),
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
            bhtraits=traits.bht_parser.dump(self._disk.rhtraits()),
            vptraits=traits.vpt_parser.dump(self._disk.vptraits()),
            vhtraits=traits.vht_parser.dump(self._disk.vhtraits()),
            dptraits=traits.dpt_parser.dump(self._disk.dptraits()),
            dhtraits=traits.dht_parser.dump(self._disk.dhtraits()),
            zptraits=traits.zpt_parser.dump(self._disk.zptraits()),
            sptraits=traits.spt_parser.dump(self._disk.sptraits()),
            wptraits=traits.wpt_parser.dump(self._disk.wptraits()))

    def __init__(
            self,
            cflux,
            loose,
            tilted,
            bptraits,
            vptraits,
            dptraits,
            bhtraits,
            vhtraits=None,
            dhtraits=None,
            zptraits=None,
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
        trait_args = _detail.parse_component_s3d_traits(
            bptraits, bhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            zptraits,
            sptraits,
            wptraits)
        _detail.rename_bx_to_rx_traits(trait_args)
        all_traits = sum(trait_args.values(), ())
        _detail.check_traits_common(all_traits)
        _detail.check_traits_mcdisk(self, all_traits)
        self._disk = _mcdisk.MCDisk(
            cflux=cflux,
            loose=loose, tilted=tilted,
            **rnode_args,
            **nwmode_velocity_args,
            **nwmode_geometry_args,
            **trait_args)

    def params(self):
        return self._disk.params()

    def is_weighted(self):
        return bool(self._disk.wptraits())

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
