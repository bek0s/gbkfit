"""
The instrument: the primary beam, the PSF and the LSF together.
"""

from typing import Any

from gbkfit.config import BasicParser, Serializable, check_options
from .lsfs import LSF, lsf_parser
from .primary_beams import PrimaryBeam, primary_beam_parser
from .psfs import PSF, psf_parser


__all__ = [
    'Instrument',
    'instrument_parser'
]


class Instrument(Serializable):
    """
    What happens to the light of a model in the telescope and the
    instrument, in the order the light meets them: the primary beam, and
    the point and line spread functions.

    Their widths are fixed in a fit, not parameters: an error of the width
    of the PSF moves the dispersion and the rotation of a disk it smears
    (e.g. some 3 km/s of dispersion per 10% of the width at z ~ 1.5). To
    quote it, fit again with the widths off by their uncertainty (e.g.
    +-10%), and take the spread of the results as a systematic error (as
    KMOS3D does).

    Parameters
    ----------
    primary_beam : PrimaryBeam, optional
        The primary beam, if any.
    psf : PSF, optional
        The point spread function, if any.
    lsf : LSF, optional
        The line spread function, if any.
    """

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
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
        check_options(Instrument.__init__, locals())
        self._primary_beam = primary_beam
        self._psf = psf
        self._lsf = lsf

    def primary_beam(self) -> PrimaryBeam | None:
        """Return the primary beam, if any."""
        return self._primary_beam

    def psf(self) -> PSF | None:
        """Return the point spread function, if any."""
        return self._psf

    def lsf(self) -> LSF | None:
        """Return the line spread function, if any."""
        return self._lsf


instrument_parser = BasicParser(Instrument)
