
from . import _common, _smdisk, traits


class SpectralSMDisk2D(_common.SpectralComponent2D):

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):
        loose = info.get('loose')
        tilted = info.get('tilted')
        rnodes = info['rnodes']
        rptraits = traits.rpt_parser.load(info.get('rptraits'))
        vptraits = traits.vpt_parser.load(info.get('vptraits'))
        dptraits = traits.dpt_parser.load(info.get('dptraits'))
        sptraits = traits.spt_parser.load(info.get('sptraits'))
        return cls(
            rnodes,
            rptraits, vptraits, dptraits,
            sptraits,
            tilted, loose)

    def dump(self):
        return {
            'type': self.type(),
            'loose': self._disk.loose(),
            'tilted': self._disk.tilted(),
            'rnodes': self._disk.rnodes(),
            'rptraits': traits.rpt_parser.dump(self._disk.rptraits()),
            'vptraits': traits.vpt_parser.dump(self._disk.vptraits()),
            'dptraits': traits.dpt_parser.dump(self._disk.dptraits()),
            'sptraits': traits.spt_parser.dump(self._disk.sptraits())}

    def __init__(
            self,
            rnodes,
            rptraits,
            vptraits,
            dptraits,
            sptraits=None,
            tilted=None,
            loose=None):

        args = _common.parse_spectral_disk_2d_common_args(
            loose, tilted, rnodes,
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