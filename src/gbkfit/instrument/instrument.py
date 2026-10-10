"""
The instrument: the primary beam, the PSF and the LSF together.
"""

from gbkfit.utils import parseutils
from .lsfs import LSF, lsf_parser
from .primary_beams import PrimaryBeam, primary_beam_parser
from .psfs import PSF, psf_parser


__all__ = [
    'Instrument',
    'instrument_parser'
]


class Instrument(parseutils.Serializable):
    """
    What happens to the light of a gmodel in the telescope and the
    instrument, in the order the light meets them: the primary beam, and
    the point and line spread functions.

    Parameters
    ----------
    primary_beam : PrimaryBeam, optional
        The primary beam, if any.
    psf : PSF, optional
        The point spread function, if any.
    lsf : LSF, optional
        The line spread function, if any.
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
        Dump the instrument to its configuration.

        Parameters
        ----------
        prefix : str, optional
            The start of the names of the files of the data of its parts
            (e.g. images; see PSF.dump).
        dump_path : bool, optional
            Whether the configuration has the paths of the files, or only
            their names.
        overwrite : bool, optional
            Whether to overwrite existing files.

        Returns
        -------
        dict
            The options.
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
