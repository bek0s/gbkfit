
from . import SpectralComponent2D, _detail, _smdisk, make_component_desc, traits
from gbkfit.utils import parseutils


__all__ = ['SpectralSMDisk2D']


class SpectralSMDisk2D(SpectralComponent2D):

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):
        desc = make_component_desc(cls)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(dict(
            rptraits=traits.rpt_parser.load(opts.get('rptraits')),
            vptraits=traits.vpt_parser.load(opts.get('vptraits')),
            dptraits=traits.dpt_parser.load(opts.get('dptraits')),
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
            vptraits=traits.vpt_parser.dump(self._disk.vptraits()),
            dptraits=traits.dpt_parser.dump(self._disk.dptraits()),
            sptraits=traits.spt_parser.dump(self._disk.sptraits()))

    def __init__(
            self,
            loose,
            tilted,
            rptraits,
            vptraits,
            dptraits,
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
        trait_args = _detail.parse_component_s2d_trait_args(
            rptraits,
            vptraits,
            dptraits,
            sptraits)
        self._disk = _smdisk.SMDisk(
            loose=loose, tilted=tilted,
            **rnode_args, **trait_args,
            rhtraits=(),
            vhtraits=(), dhtraits=(),
            wptraits=())

    def params(self):
        return self._disk.params()

    def evaluate(
            self,
            driver, params, scube,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra):
        spat_size = spat_size + (1,)
        spat_step = spat_step + (0,)
        spat_zero = spat_zero + (0,)
        self._disk.evaluate(
            driver, params, None, scube, None,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)
