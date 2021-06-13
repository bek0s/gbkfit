
from . import _detail, _mcdisk, traits
from .core import DensityComponent3D
from gbkfit.utils import parseutils


__all__ = ['DensityMCDisk3D']


class DensityMCDisk3D(DensityComponent3D):

    @staticmethod
    def type():
        return 'mcdisk'

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(dict(
            rptraits=traits.rpt_parser.load(opts.get('rptraits')),
            rhtraits=traits.rht_parser.load(opts.get('rhtraits')),
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
            rnstep=self._disk.rnstep(),
            interp=self._disk.interp().type(),
            rptraits=traits.rpt_parser.dump(self._disk.rptraits()),
            rhtraits=traits.rht_parser.dump(self._disk.rhtraits()),
            zptraits=traits.zpt_parser.dump(self._disk.zptraits()),
            sptraits=traits.spt_parser.dump(self._disk.sptraits()),
            wptraits=traits.wpt_parser.dump(self._disk.wptraits()))

    def __init__(
            self,
            cflux,
            loose,
            tilted,
            rptraits,
            rhtraits,
            zptraits=None,
            sptraits=None,
            wptraits=None,
            rnmin=None,
            rnmax=None,
            rnsep=None,
            rnlen=None,
            rnodes=None,
            rnstep=None,
            interp='linear'):
        rnode_args = _detail.parse_component_rnode_args(
            rnmin, rnmax, rnsep, rnlen, rnodes, rnstep, interp)
        trait_args = _detail.parse_component_d3d_trait_args(
            rptraits, rhtraits,
            zptraits,
            sptraits,
            wptraits)
        self._disk = _mcdisk.MCDisk(
            cflux=cflux,
            loose=loose, tilted=tilted,
            **rnode_args, **trait_args,
            vptraits=(), vhtraits=(),
            dptraits=(), dhtraits=())

    def params(self):
        return self._disk.params()

    def evaluate(
            self,
            driver, params, image, rcube, wcube,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        spec_size = 1
        spec_step = 0
        spec_zero = 0
        self._disk.evaluate(
            driver, params, image, None, rcube, wcube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)
