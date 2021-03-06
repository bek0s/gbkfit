
from . import _detail, _smdisk, traits
from .core import SpectralComponent3D
from gbkfit.utils import parseutils


__all__ = ['SpectralSMDisk3D']


class SpectralSMDisk3D(SpectralComponent3D):

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(dict(
            rptraits=traits.rpt_parser.load(opts.get('rptraits')),
            rhtraits=traits.rht_parser.load(opts.get('rhtraits')),
            vptraits=traits.vpt_parser.load(opts.get('vptraits')),
            vhtraits=traits.vht_parser.load(opts.get('vhtraits')),
            dptraits=traits.dpt_parser.load(opts.get('dptraits')),
            dhtraits=traits.dht_parser.load(opts.get('dhtraits')),
            wptraits=traits.wpt_parser.load(opts.get('wptraits')),
            sptraits=traits.spt_parser.load(opts.get('sptraits'))))
        return cls(**opts)

    def dump(self):
        return dict(
            type=self.type(),
            loose=self._disk.loose(),
            tilted=self._disk.tilted(),
            rnodes=self._disk.rnodes(),
            rnstep=self._disk.rnstep(),
            interp=self._disk.interp().type(),
            rptraits=traits.rpt_parser.dump(self._disk.rptraits()),
            rhtraits=traits.rht_parser.dump(self._disk.rhtraits()),
            vptraits=traits.vpt_parser.dump(self._disk.vptraits()),
            vhtraits=traits.vht_parser.dump(self._disk.vhtraits()),
            dptraits=traits.dpt_parser.dump(self._disk.dptraits()),
            dhtraits=traits.dht_parser.dump(self._disk.dhtraits()),
            wptraits=traits.wpt_parser.dump(self._disk.wptraits()),
            sptraits=traits.spt_parser.dump(self._disk.sptraits()))

    def __init__(
            self,
            loose,
            tilted,
            rptraits,
            vptraits,
            dptraits,
            rhtraits,
            vhtraits=None,
            dhtraits=None,
            wptraits=None,
            sptraits=None,
            rnmin=None,
            rnmax=None,
            rnsep=None,
            rnlen=None,
            rnodes=None,
            rnstep=None,
            interp='linear'):
        rnode_args = _detail.parse_component_rnode_args(
            rnmin, rnmax, rnsep, rnlen, rnodes, rnstep, interp)
        trait_args = _detail.parse_component_s3d_trait_args(
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            wptraits,
            sptraits)
        self._disk = _smdisk.SMDisk(
            loose=loose, tilted=tilted,
            **rnode_args, **trait_args)

    def params(self):
        return self._disk.params()

    def evaluate(
            self,
            driver, params, scube, rcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        self._disk.evaluate(
            driver, params, None, scube, rcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)
