
from . import _detail, _mcdisk, common, traits
from .core import OpacityComponent3D
from gbkfit.utils import parseutils


__all__ = [
    'OpacityMCDisk3D'
]


class OpacityMCDisk3D(OpacityComponent3D):

    @staticmethod
    def type():
        return 'mcdisk'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(dict(
            xpos_nwmode=common.nwmode_parser.load(opts.get('xpos_nwmode')),
            ypos_nwmode=common.nwmode_parser.load(opts.get('ypos_nwmode')),
            posa_nwmode=common.nwmode_parser.load(opts.get('posa_nwmode')),
            incl_nwmode=common.nwmode_parser.load(opts.get('incl_nwmode')),
            optraits=traits.opt_parser.load(opts.get('optraits')),
            ohtraits=traits.oht_parser.load(opts.get('ohtraits')),
            zptraits=traits.zpt_parser.load(opts.get('zptraits')),
            sptraits=traits.spt_parser.load(opts.get('sptraits'))))
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
            optraits=traits.opt_parser.dump(self._disk.rptraits()),
            ohtraits=traits.oht_parser.dump(self._disk.rhtraits()),
            zptraits=traits.zpt_parser.dump(self._disk.zptraits()),
            sptraits=traits.spt_parser.dump(self._disk.sptraits()),
            wptraits=traits.wpt_parser.dump(self._disk.wptraits()))

    def __init__(
            self,
            cflux,
            loose,
            tilted,
            optraits,
            ohtraits,
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
        nwmode_geometry_args = _detail.validate_component_nwmodes_for_geometry(
            loose, tilted, xpos_nwmode, ypos_nwmode, posa_nwmode, incl_nwmode)
        rnode_args = _detail.parse_component_rnode_args(
            rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp)
        trait_args = _detail.parse_component_o3d_traits(
            optraits, ohtraits,
            zptraits,
            sptraits,
            wptraits)
        _detail.rename_ox_to_rx_traits(trait_args)
        all_traits = sum(trait_args.values(), ())
        _detail.check_traits_common(all_traits)
        _detail.check_traits_mcdisk(self, all_traits)
        self._disk = _mcdisk.MCDisk(
            cflux=cflux,
            loose=loose, tilted=tilted,
            **rnode_args,
            vsys_nwmode=None,
            **nwmode_geometry_args,
            **trait_args,
            vptraits=(), vhtraits=(),
            dptraits=(), dhtraits=())

    def params(self):
        return self._disk.params()

    def evaluate(
            self,
            driver, params,
            odata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        spec_size = 1
        spec_step = 0
        spec_zero = 0
        self._disk.evaluate(
            driver, params, None, None, None, None, odata, None,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)
