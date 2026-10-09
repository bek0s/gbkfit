from gbkfit.psflsf import LSF, PSF, lsf_parser, psf_parser
from gbkfit.utils import parseutils


__all__ = [
    'Instrument',
    'instrument_parser'
]


class Instrument(parseutils.BasicSerializable):
    """
    What happens to the light of a gmodel in the telescope and the
    instrument: the point and line spread functions. Each has its own
    slot, in the order the light meets them.
    """

    @classmethod
    def load(cls, info):
        desc = parseutils.make_basic_desc(cls, 'instrument')
        parseutils.load_option_and_update_info(psf_parser, info, 'psf')
        parseutils.load_option_and_update_info(lsf_parser, info, 'lsf')
        return cls(**parseutils.parse_options_for_callable(
            info, desc, cls.__init__))

    def dump(self):
        return dict(
            psf=psf_parser.dump(self._psf),
            lsf=lsf_parser.dump(self._lsf))

    def __init__(self, psf: PSF | None = None, lsf: LSF | None = None):
        self._psf = psf
        self._lsf = lsf

    def psf(self) -> PSF | None:
        return self._psf

    def lsf(self) -> LSF | None:
        return self._lsf


instrument_parser = parseutils.BasicParser(Instrument)
