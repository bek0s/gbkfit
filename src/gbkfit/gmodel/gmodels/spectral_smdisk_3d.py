
from . import _common, _smdisk, traits
from gbkfit.utils import parseutils


class SpectralSMDisk3D(_common.SpectralComponent3D):

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):
        info = parseutils.parse_class_args(cls, info)
        info.update(dict(
            rptraits=traits.rpt_parser.load(info.get('rptraits')),
            rhtraits=traits.rht_parser.load(info.get('rhtraits')),
            vptraits=traits.vpt_parser.load(info.get('vptraits')),
            vhtraits=traits.vht_parser.load(info.get('vhtraits')),
            dptraits=traits.dpt_parser.load(info.get('dptraits')),
            dhtraits=traits.dht_parser.load(info.get('dhtraits')),
            wptraits=traits.wpt_parser.load(info.get('wptraits')),
            sptraits=traits.spt_parser.load(info.get('sptraits'))))
        return cls(**info)

    def dump(self):
        return dict(
            type=self.type(),
            loose=self._disk.loose(),
            tilted=self._disk.tilted(),
            rnodes=self._disk.rnodes(),
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
            rnodes=None):

        args = _common.parse_spectral_disk_3d_common_args(
            loose, tilted,
            rnmin, rnmax, rnsep, rnlen, rnodes,
            rptraits, rhtraits,
            vptraits, vhtraits,
            dptraits, dhtraits,
            wptraits,
            sptraits)

        self._disk = _smdisk.SMDisk(**args)

    def params(self):
        return self._disk.params()

    def evaluate(
            self,
            driver, params, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):
        self._disk.evaluate(
            driver, params, None, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra)
