
from . import _common, _smdisk, traits
from gbkfit.utils import parseutils


class SpectralSMDisk2D(_common.SpectralComponent2D):

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):
        info = parseutils.parse_class_args(cls, info)
        info.update(dict(
            rptraits=traits.rpt_parser.load(info.get('rptraits')),
            vptraits=traits.vpt_parser.load(info.get('vptraits')),
            dptraits=traits.dpt_parser.load(info.get('dptraits')),
            sptraits=traits.spt_parser.load(info.get('sptraits'))))
        return cls(**info)

    def dump(self):
        return dict(
            type=self.type(),
            loose=self._disk.loose(),
            tilted=self._disk.tilted(),
            rnodes=self._disk.rnodes(),
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
            rnodes=None):

        args = _common.parse_spectral_disk_2d_common_args(
            loose, tilted,
            rnmin, rnmax, rnsep, rnlen, rnodes,
            rptraits,
            vptraits,
            dptraits,
            sptraits)

        self._disk = _smdisk.SMDisk(
            **args,
            rhtraits=(),
            vhtraits=(), dhtraits=(),
            wptraits=())

    def params(self):
        return self._disk.params()

    def evaluate(
            self,
            driver, params, scube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):
        spat_size = spat_size + (1,)
        spat_step = spat_step + (0,)
        spat_zero = spat_zero + (0,)
        self._disk.evaluate(
            driver, params, None, scube, None, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra)
