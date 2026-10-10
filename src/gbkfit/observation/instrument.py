from gbkfit.psflsf import LSF, PSF, lsf_parser, psf_parser
from gbkfit.utils import parseutils
from .primary_beam import PrimaryBeam, primary_beam_parser


__all__ = [
    'Instrument',
    'instrument_parser'
]


class Instrument(parseutils.Serializable):
    """
    What happens to the light of a gmodel in the telescope and the
    instrument: the primary beam, and the point and line spread functions.
    Each has its own slot, in the order the light meets them.
    """

    @classmethod
    def load(cls, info):
        parseutils.load_option_and_update_info(
            primary_beam_parser, info, 'primary_beam')
        parseutils.load_option_and_update_info(psf_parser, info, 'psf')
        parseutils.load_option_and_update_info(lsf_parser, info, 'lsf')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self, prefix='', dump_path=True, overwrite=False):
        """
        The options of the instrument; the data of its parts (images) go
        to files whose names start with prefix (see PSF.dump).
        """
        kwargs = dict(prefix=prefix, dump_path=dump_path, overwrite=overwrite)
        return dict(
            primary_beam=primary_beam_parser.dump(
                self._primary_beam, **kwargs),
            psf=psf_parser.dump(self._psf, **kwargs),
            lsf=lsf_parser.dump(self._lsf, **kwargs))

    def __init__(
            self,
            primary_beam: PrimaryBeam | None = None,
            psf: PSF | None = None,
            lsf: LSF | None = None
    ):
        self._primary_beam = primary_beam
        self._psf = psf
        self._lsf = lsf

    def primary_beam(self) -> PrimaryBeam | None:
        return self._primary_beam

    def psf(self) -> PSF | None:
        return self._psf

    def lsf(self) -> LSF | None:
        return self._lsf


instrument_parser = parseutils.BasicParser(Instrument)
