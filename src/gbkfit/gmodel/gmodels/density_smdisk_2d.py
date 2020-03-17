
from . import _common, _smdisk, traits


class DensitySMDisk2D(_common.DensityComponent2D):

    @staticmethod
    def type():
        return 'smdisk'

    @classmethod
    def load(cls, info):

        parseutils.validate_options(cls.__init__, info)

        loose = info.get('loose')
        tilted = info.get('tilted')
        interp = info.get('interp')
        rnodes = info['rnodes']
        rptraits = traits.rpt_parser.load(info.get('rptraits'))
        sptraits = traits.spt_parser.load(info.get('sptraits'))
        return cls(
            rnodes,
            rptraits,
            sptraits,
            tilted, loose)

    def dump(self):
        return {
            'type': self.type(),
            'loose': self._disk.loose(),
            'tilted': self._disk.tilted(),
            'rnodes': self._disk.rnodes(),
            'rptraits': traits.rpt_parser.dump(self._disk.rptraits()),
            'sptraits': traits.spt_parser.dump(self._disk.sptraits())}

    def __init__(
            self,
            rnodes,
            rptraits,
            sptraits=None,
            tilted=None,
            loose=None):

        args = _common.parse_density_disk_2d_common_args(
            loose, tilted, rnodes,
            rptraits,
            sptraits)

        self._disk = _smdisk.SMDisk(
            **args,
            rhtraits=(),
            vptraits=(), vhtraits=(),
            dptraits=(), dhtraits=(),
            wptraits=())

    def params(self):
        return self._disk.params()

    def evaluate(
            self, driver, params, image, dtype,
            spat_size, spat_step, spat_zero,
            out_extra):
        spat_size = spat_size + (1,)
        spat_step = spat_step + (0,)
        spat_zero = spat_zero + (0,)
        spec_size = 1
        spec_step = 0
        spec_zero = 0
        self._disk.evaluate(
            driver, params, image, None, None, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra)
