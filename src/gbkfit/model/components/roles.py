"""
The roles of the components of a galaxy: the light they emit (Light),
and what they remove from the light behind them (Opacity). What an
observation sees of their light is its tracer (see tracers).
"""

import abc
from collections.abc import Callable, Sequence
from typing import Any

import astropy.units

import gbkfit.units
from gbkfit.config import (
    TypedParser, TypedSerializable, check_options, config_path)
from gbkfit.utils import iterutils
from . import _units
from .extinction import ExtinctionLaw, extinction_law_parser
from .lines import (
    Band, Bands, Line, Lines, Spectrum, band_parser, catalogue_in_use,
    line_parser)


__all__ = [
    'ContinuumLight',
    'DustOpacity',
    'Light',
    'LineLight',
    'Opacity',
    'light_parser',
    'opacity_parser'
]


def _named(
        items: Any,
        lookup: Callable[[str], Line | Band],
        option: str
) -> list[Line | Band]:
    """
    Return the items of an option (one or a sequence), each name replaced
    by its line or band of the catalogue (lookup); the error of an
    unknown name has the path of its item (e.g. lines[1]).
    """
    named = []
    for i, item in enumerate(iterutils.listify(items)):
        with config_path(option, i):
            named.append(lookup(item) if isinstance(item, str) else item)
    return named


class Light(TypedSerializable, abc.ABC):
    """
    The light that a component emits: its density, with a spectrum (see
    Spectrum): emission lines (LineLight) or a continuum (ContinuumLight).
    Each observation sees the light of the components that its tracer
    traces (see tracers.Tracer), which asks it only for these facts.
    """

    config_titles = dict(unit='Unit')

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def lines(self) -> Lines | None:
        """Return its emission lines, if it has lines; none here."""
        return None

    def bands(self) -> Bands | None:
        """Return its bands, if it is a continuum; none here."""
        return None

    def line_names(self) -> tuple[str, ...]:
        """Return the names of its emission lines; none here."""
        return ()

    @abc.abstractmethod
    def needs_velocities(self) -> bool:
        """
        Check whether its component must move (have kinematics) to emit
        it: lines, which are at the velocities of their component.
        """

    @abc.abstractmethod
    def spectrum(self) -> Spectrum:
        """Return its spectrum: its lines or its bands (see Spectrum)."""

    @abc.abstractmethod
    def unit(self) -> astropy.units.UnitBase | None:
        """
        Return the physical unit of its density, if it has one: without
        one, its density is in the units of the data of each observation
        of it.
        """


class LineLight(Light):
    """
    Emission lines: each a Gaussian at the velocity and of the dispersion
    of the kinematics of its component, which must move (see Lines). The
    tracer lines sees them (see tracers.LinesTracer): on a spectral axis,
    or in an image, the light of the lines it names.

    Parameters
    ----------
    lines : str or Line or Sequence of them, optional
        The lines, or names of lines of the catalogue in use (that of
        GBKFIT, or of a configuration, see lines_from; the flux of each
        after the first, relative to the first, is a parameter
        '{name}_ratio'); by default one line at the velocity of the
        spectral axis, whatever the line is.
    unit : str or astropy.units.UnitBase, optional
        The physical unit of the density (that of the first line; the
        ratios have none): the light of the line integrated over it, per
        solid angle (e.g. '1e-17 erg/s/cm2/arcsec2', 'Jy km/s/arcsec2',
        'K km/s'), or for HI (lines: [hi21]) its column density ('cm-2') or
        the surface density of its mass ('Msun/pc2'); that of the flux of
        a point is this unit times arcsec^2. It is the light as the galaxy
        emits it, in its rest frame: each observation sees it dimmed by the
        redshift (see _units.line_factor). Without a unit, the density is
        in the units of the data of each observation of it (see
        ObservationSky), as observed.

    Raises
    ------
    ValueError
        If there are no lines, a name is not in the catalogue, the names
        repeat, or the unit is not one of lines (see
        _units.check_line_unit).
    """

    config_titles = dict(lines='Lines')

    @staticmethod
    def type() -> str:
        return 'lines'

    @staticmethod
    def type_title() -> str:
        return 'Emission lines'

    def dump(self) -> dict[str, Any]:
        lines = self._lines.lines()
        return dict(type=self.type()) | (
            {} if lines is None else dict(lines=line_parser.dump(list(lines)))
        ) | _dump_unit(self._unit)

    def __init__(
            self,
            lines: str | Line | Sequence[str | Line] | None = None,
            unit: str | astropy.units.UnitBase | None = None
    ):
        check_options(LineLight.__init__, locals())
        self._lines = Lines(None if lines is None else _named(
            lines, catalogue_in_use().line, 'lines'))
        self._unit = _parse_unit(unit)
        if self._unit is not None:
            _units.check_line_unit(self._unit, self._lines.names())

    def unit(self) -> astropy.units.UnitBase | None:
        return self._unit

    def lines(self) -> Lines:
        return self._lines

    def spectrum(self) -> Lines:
        return self._lines

    def line_names(self) -> tuple[str, ...]:
        return self._lines.names()

    def needs_velocities(self) -> bool:
        return True


class ContinuumLight(Light):
    """
    A continuum: the density, flat across a band (e.g. starlight), in
    every band, or with bands, in those, of the flux ratios
    '{band}_ratio' to the first (see Bands). The tracer continuum sees it
    in its band (see tracers.ContinuumTracer), and that of the stars'
    velocities (losvd) sees its velocities, weighted by its light in its
    band, when its component moves (see tracers.LosvdTracer).

    Parameters
    ----------
    bands : str or Band or Sequence of them, optional
        The bands, or names of bands of the catalogue in use (those of a
        configuration, see lines_from); every band if None.
    unit : str or astropy.units.UnitBase, optional
        The physical unit of the density (that of the first band; the
        ratios have none): a flux density per solid angle (e.g.
        'Jy/arcsec2', 'MJy/sr', '1e-20 erg/s/cm2/Angstrom/arcsec2') or a
        brightness temperature ('K'). It is the light as the galaxy emits
        it, in its rest frame, at the frequency of each band times 1 + z:
        each observation sees it dimmed by the redshift (see
        _units.BandFactors), which its galaxy must give unless the
        component has a systemic velocity (see tracers). Without a unit,
        the density is in the units of the data of each observation of it,
        as observed.

    Raises
    ------
    ValueError
        If there are no bands, a name is not in the catalogue, the names
        repeat, or the unit is not one of bands (see
        _units.check_band_unit).
    """

    config_titles = dict(bands='Bands')

    @staticmethod
    def type() -> str:
        return 'continuum'

    @staticmethod
    def type_title() -> str:
        return 'Continuum'

    def dump(self) -> dict[str, Any]:
        bands = self._bands.bands()
        return dict(type=self.type()) | (
            {} if bands is None else dict(bands=band_parser.dump(list(bands)))
        ) | _dump_unit(self._unit)

    def __init__(
            self,
            bands: str | Band | Sequence[str | Band] | None = None,
            unit: str | astropy.units.UnitBase | None = None
    ):
        check_options(ContinuumLight.__init__, locals())
        self._bands = Bands(None if bands is None else _named(
            bands, catalogue_in_use().band, 'bands'))
        self._unit = _parse_unit(unit)
        if self._unit is not None:
            _units.check_band_unit(self._unit)

    def unit(self) -> astropy.units.UnitBase | None:
        return self._unit

    def bands(self) -> Bands:
        return self._bands

    def spectrum(self) -> Bands:
        return self._bands

    def needs_velocities(self) -> bool:
        # (an image sees it at rest; the tracer losvd sees its velocities
        # only if its component moves)
        return False


def _parse_unit(
        unit: str | astropy.units.UnitBase | None
) -> astropy.units.UnitBase | None:
    """Return a unit, read from its text if given as text."""
    return gbkfit.units.parse_unit(unit) if isinstance(unit, str) else unit


def _dump_unit(unit: astropy.units.UnitBase | None) -> dict[str, str]:
    """Return the option unit of a configuration, if there is a unit."""
    return {} if unit is None else dict(unit=gbkfit.units.format_unit(unit))


class Opacity(TypedSerializable, abc.ABC):
    """
    What a component removes from the light behind it: its density, with
    a spectral shape. It needs a line of sight with a depth axis.
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    @abc.abstractmethod
    def dims_others(self) -> bool:
        """
        Check whether it dims the light of the other components too (e.g.
        dust), which every observation of its galaxy then sees, or only
        that of its own component (e.g. line opacity), which only the
        observations that see the light of the component see.
        """


class DustOpacity(Opacity):
    """
    Dust, which dims the light behind it and within it (grey, or by an
    extinction law). The density is the optical depth of each voxel at V,
    which dims the light of the voxels behind it and that of its own
    voxels (as a uniform mixture of light and dust). At the wavelength of
    a line or a band, the optical depth is that at V times the ratio of
    the extinction law there (at the rest wavelength of the line).

    Parameters
    ----------
    law : ExtinctionLaw, optional
        The extinction law; by default, none: the optical depth is the
        same at every wavelength (grey dust).
    """

    config_titles = dict(law='Extinction law')

    @staticmethod
    def type() -> str:
        return 'dust'

    @staticmethod
    def type_title() -> str:
        return 'Dust'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type()) | (
            {} if self._law is None
            else dict(law=extinction_law_parser.dump(self._law)))

    def __init__(self, law: ExtinctionLaw | None = None):
        check_options(DustOpacity.__init__, locals())
        self._law = law

    def dims_others(self) -> bool:
        return True

    def law(self) -> ExtinctionLaw | None:
        """Return the extinction law, if any (none: grey dust)."""
        return self._law


light_parser = TypedParser(Light, [
    LineLight,
    ContinuumLight], family='light')

# (the disks register their line opacity, LineOpacity)
opacity_parser = TypedParser(Opacity, [
    DustOpacity], family='opacity')
