"""
The roles of the components of a galaxy: the light they emit (Emission),
and what they remove from the light behind them (Opacity).
"""

import abc
from collections.abc import Sequence
from typing import Any, Self

import astropy.units

from gbkfit.utils import iterutils, parseutils
from .extinction import ExtinctionLaw, extinction_law_parser
from .lines import (
    Band, Bands, Line, LineCatalogue, Lines, band_parser, line_parser)


__all__ = [
    'BandEmission',
    'DustOpacity',
    'Emission',
    'LineEmission',
    'Opacity',
    'emission_parser',
    'opacity_parser'
]


def _load_line(info: Any, catalogue: LineCatalogue) -> Line:
    """
    Return the line of a configuration: the line of the catalogue of the
    given name, or a line (its name and rest).
    """
    if isinstance(info, str):
        return catalogue.line(info)
    return line_parser.load(info)


def load_band(info: Any, catalogue: LineCatalogue) -> Band:
    """
    Return the band of a configuration: the band of the catalogue of the
    given name, or a band (its name and wavelength).
    """
    if isinstance(info, str):
        return catalogue.band(info)
    return band_parser.load(info)


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

    def emits_in(self, band: Band | None) -> bool:
        """
        Check whether it emits in the band of an image (None if not
        known); here, in every band.
        """
        return True

    @abc.abstractmethod
    def wavelengths(
            self,
            lines: Sequence[str] | None,
            band: Band | None,
            redshift: float
    ) -> tuple[float | None, ...]:
        """
        Return the rest wavelength (um) of the light of the component that
        an observation sees: of each of its lines of the given names (all
        if None), in their order, or in the band of an image (if known)
        of a component at the given redshift; None where it is not known.
        """


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
            lines = iterutils.listify(info['lines'])
            info['lines'] = []
            for i, line in enumerate(lines):
                with parseutils.config_path('lines', i):
                    info['lines'].append(_load_line(line, catalogue))
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

    def wavelengths(
            self,
            lines: Sequence[str] | None,
            band: Band | None,
            redshift: float
    ) -> tuple[float | None, ...]:
        # (one line at the velocity of the spectral axis: whatever line)
        if self._lines.lines() is None:
            return (None,)
        return tuple(
            self._lines.lines()[i].rest().to_value(
                astropy.units.um, astropy.units.spectral())
            for i in self._lines.select(lines))


class BandEmission(Emission):
    """
    Light in the bands of images: the density, flat across a band (e.g.
    starlight). Seen by images (see Bands): in every band, or with bands,
    in those, of the flux ratios '{band}_ratio' to the first.

    Parameters
    ----------
    bands : Sequence of Band, optional
        The bands; every band if None.

    Raises
    ------
    ConfigError
        If there are no bands, or their names repeat.
    """

    @staticmethod
    def type() -> str:
        return 'band'

    @classmethod
    def load(
            cls,
            info: dict[str, Any],
            catalogue: LineCatalogue | None = None
    ) -> Self:
        """
        Load it from its configuration, in which each band is the name of
        a band of the catalogue, or a band (its name and wavelength).

        Parameters
        ----------
        info : dict
            The configuration.
        catalogue : LineCatalogue, optional
            The catalogue of the names; that of GBKFIT by default.

        Returns
        -------
        BandEmission
            The emission.

        Raises
        ------
        ConfigError
            If the configuration has an error, or a band is not in the
            catalogue.
        """
        if catalogue is None:
            catalogue = LineCatalogue()
        if info.get('bands') is not None:
            bands = iterutils.listify(info['bands'])
            info['bands'] = []
            for i, band in enumerate(bands):
                with parseutils.config_path('bands', i):
                    info['bands'].append(load_band(band, catalogue))
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        bands = self._bands.bands()
        return dict(type=self.type()) | (
            {} if bands is None else dict(bands=band_parser.dump(list(bands))))

    def __init__(self, bands: Sequence[Band] | None = None):
        self._bands = Bands(bands)

    def bands(self) -> Bands:
        """Return the bands."""
        return self._bands

    def emits_in(self, band: Band | None) -> bool:
        return self._bands.emits_in(band)

    def wavelengths(
            self,
            lines: Sequence[str] | None,
            band: Band | None,
            redshift: float
    ) -> tuple[float | None, ...]:
        if band is None:
            return (None,)
        return (band.wavelength().to_value(astropy.units.um) / (1 + redshift),)


class Opacity(parseutils.TypedSerializable, abc.ABC):
    """
    What a component removes from the light behind it: its density, with
    a spectral shape. It needs a line of sight with a depth axis.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())


class DustOpacity(Opacity):
    """
    Dust: the density is the optical depth of each voxel at V, which dims
    the light of the voxels behind it and that of its own voxels (as a
    uniform mixture of light and dust). At the wavelength of a line or a
    band, the optical depth is that at V times the ratio of the extinction
    law there (at the rest wavelength of the line).

    Parameters
    ----------
    law : ExtinctionLaw, optional
        The extinction law; by default, none: the optical depth is the
        same at every wavelength (grey dust).
    """

    @staticmethod
    def type() -> str:
        return 'dust'

    @classmethod
    def load(cls, info: dict[str, Any]) -> Self:
        parseutils.load_option_and_update_info(
            extinction_law_parser, info, 'law')
        return cls(**parseutils.parse_options_for_callable(
            info, cls.__init__))

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type()) | (
            {} if self._law is None
            else dict(law=extinction_law_parser.dump(self._law)))

    def __init__(self, law: ExtinctionLaw | None = None):
        self._law = law

    def law(self) -> ExtinctionLaw | None:
        """Return the extinction law, if any (none: grey dust)."""
        return self._law


emission_parser = parseutils.TypedParser(Emission, [
    LineEmission,
    BandEmission])

# (the disks register their line opacity, LineOpacity)
opacity_parser = parseutils.TypedParser(Opacity, [
    DustOpacity])
