
from . import _common, _mcdisk, traits
from gbkfit.utils import parseutils


class DensityMCDisk3D(_common.DensityComponent3D):

    @staticmethod
    def type():
        return 'mcdisk'

    @classmethod
    def load(cls, info):
        info = parseutils.parse_class_args(cls, info)
        info.update(dict(
            rptraits=traits.rpt_parser.load(info.get('rptraits')),
            rhtraits=traits.rht_parser.load(info.get('rhtraits')),
            wptraits=traits.wpt_parser.load(info.get('wptraits')),
            sptraits=traits.spt_parser.load(info.get('sptraits'))))
        return cls(**info)

    def dump(self):
        return dict(
            type=self.type(),
            cflux=self._disk.cflux(),
            loose=self._disk.loose(),
            tilted=self._disk.tilted(),
            rnodes=self._disk.rnodes(),
            rptraits=traits.rpt_parser.dump(self._disk.rptraits()),
            rhtraits=traits.rht_parser.dump(self._disk.rhtraits()),
            wptraits=traits.wpt_parser.dump(self._disk.wptraits()),
            sptraits=traits.spt_parser.dump(self._disk.sptraits()))

    def __init__(
            self,
            cflux,
            loose,
            tilted,
            rptraits,
            rhtraits,
            wptraits=None,
            sptraits=None,
            rnmin=None,
            rnmax=None,
            rnsep=None,
            rnlen=None,
            rnodes=None):

        args = _common.parse_density_disk_3d_common_args(
            loose, tilted,
            rnmin, rnmax, rnsep, rnlen, rnodes,
            rptraits, rhtraits,
            wptraits,
            sptraits)

        self._disk = _mcdisk.MCDisk(
            cflux, **args,
            vptraits=(), vhtraits=(),
            dptraits=(), dhtraits=())

    def params(self):
        return self._disk.params()

    def evaluate(
            self,
            driver, params, image, rcube, dtype,
            spat_size, spat_step, spat_zero,
            out_extra):
        spec_size = 1
        spec_step = 0
        spec_zero = 0
        self._disk.evaluate(
            driver, params, image, None, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra)
