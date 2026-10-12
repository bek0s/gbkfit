"""
The roles of the components of a galaxy: the light they emit (Emission),
and what they remove from the light behind them (Opacity).
"""

import abc
from collections.abc import Sequence
from typing import Any, Self

from gbkfit.utils import iterutils, parseutils
from .lines import Line, LineCatalogue, Lines, line_parser


__all__ = [
    'BandEmission',
    'DustOpacity',
    'Emission',
    'LineEmission',
    'Opacity',
    'emission_parser',
    'opacity_parser'
]


def _load_line(
        info: Any, path: tuple[str, int], catalogue: LineCatalogue
) -> Line:
    """
    Return the line of a configuration: the line of the catalogue of the
    given name, or a line (its name and rest). path is the place of the
    line in the configuration.
    """
    with parseutils.config_path(*path):
        if isinstance(info, str):
            return catalogue.line(info)
        return line_parser.load(info)


class Emission(parseutils.TypedSerializable, abc.ABC):
    """
    The light that a component emits: its density, with a spectral shape.
    Spectral cubes see the components that emit lines, and images those
    that emit in a band.

    In configurations, emission is loaded with the catalogue of the lines
    it can name (see load).
    """

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> Self:
        """
        Load it from its configuration, with the catalogue of the names of
        its lines (that of GBKFIT by default).
        """
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())


class LineEmission(Emission):
    """
    Emission lines: each a Gaussian at the velocity and of the dispersion
    of the kinematics of its component (see Lines). Seen by spectral
    cubes.

    Parameters
    ----------
    lines : Sequence of Line, optional
        The lines (the flux of each after the first, relative to the
        first, is a parameter '{name}_ratio'); by default one line at the
        velocity of the spectral axis, whatever the line is.

    Raises
    ------
    ConfigError
        If there are no lines, or their names repeat.
    """

    @staticmethod
    def type() -> str:
        return 'lines'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> Self:
        """
        Load it from its configuration, in which each line is the name of
        a line of the catalogue, or a line (its name and rest).

        Parameters
        ----------
        info : dict
            The configuration.
        catalogue : LineCatalogue, optional
            The catalogue of the names; that of GBKFIT by default.

        Returns
        -------
        LineEmission
            The emission.

        Raises
        ------
        ConfigError
            If the configuration has an error, or a line is not in the
            catalogue.
        """
        if catalogue is None:
            catalogue = LineCatalogue()
        if info.get('lines') is not None:
            info['lines'] = [
                _load_line(line, ('lines', i), catalogue)
                for i, line in enumerate(iterutils.listify(info['lines']))]
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        lines = self._lines.lines()
        return dict(type=self.type()) | (
            {} if lines is None else dict(lines=line_parser.dump(list(lines))))

    def __init__(self, lines: Sequence[Line] | None = None):
        self._lines = Lines(lines)

    def lines(self) -> Lines:
        """Return the lines."""
        return self._lines


class BandEmission(Emission):
    """
    Light in the band of an image: the density, flat across the band (e.g.
    starlight). Seen by images.
    """

    @staticmethod
    def type() -> str:
        return 'band'


class Opacity(parseutils.TypedSerializable, abc.ABC):
    """
    What a component removes from the light behind it: its density, with
    a spectral shape. It needs a line of sight with a depth axis.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())


class DustOpacity(Opacity):
    """
    Dust: the density is the optical depth of each voxel, the same at
    every wavelength, which dims the light of the voxels behind it and
    that of its own voxels (as a uniform mixture of light and dust).
    """

    @staticmethod
    def type() -> str:
        return 'dust'


emission_parser = parseutils.TypedParser(Emission, [
    LineEmission,
    BandEmission])

opacity_parser = parseutils.TypedParser(Opacity, [
    DustOpacity])
