
from . import DensityComponent2D, _detail, _smdisk, make_component_desc, traits
from gbkfit.utils import parseutils


__all__ = ['DensitySMDisk2D']


class DensitySMDisk2D(DensityComponent2D):

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):
        desc = make_component_desc(cls)
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        opts.update(dict(
            rptraits=traits.rpt_parser.load(opts.get('rptraits')),
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
            sptraits=traits.spt_parser.dump(self._disk.sptraits()))

    def __init__(
            self,
            loose,
            tilted,
            rptraits,
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
        trait_args = _detail.parse_component_d2d_trait_args(
            rptraits,
            sptraits)
        self._disk = _smdisk.SMDisk(
            loose=loose, tilted=tilted,
            **rnode_args, **trait_args,
            rhtraits=(),
            vptraits=(), vhtraits=(),
            dptraits=(), dhtraits=(),
            wptraits=())

    def params(self):
        return self._disk.params()

    def evaluate(
            self, driver, params, image,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, out_extra):
        spat_size = spat_size + (1,)
        spat_step = spat_step + (0,)
        spat_zero = spat_zero + (0,)
        spec_size = 1
        spec_step = 0
        spec_zero = 0
        self._disk.evaluate(
            driver, params, image, None, None,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, out_extra)
