"""
What the data of an observation trace of the light of the components of
the galaxies it sees (Tracer): the light of emission lines (LinesTracer),
the continuum in a band (ContinuumTracer), or the distribution of the
velocities of the stars along the line of sight, weighted by their light
in a band (LosvdTracer, the data of stellar kinematics). An observation
sees the light of the components that its tracer traces, on its spectral
axis or in its image (View), and asks only facts of their light (see
Light).
"""

import abc
import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import astropy.units
import numpy as np

from gbkfit.config import (
    TypedParser, TypedSerializable, check_name, check_options, config_path)
from ._common import Fluxes
from ._units import DataUnit, OpacityScales, band_factors, line_factors
from .lines import Band, LinePlacement, band_parser, catalogue_in_use
from .roles import Light

if TYPE_CHECKING:
    from gbkfit.grid import Grid


__all__ = [
    'ContinuumTracer',
    'LightSetup',
    'LinesTracer',
    'LosvdTracer',
    'Tracer',
    'View',
    'tracer_parser'
]


@dataclasses.dataclass(frozen=True)
class LightSetup:
    """
    What the plan of a component evaluates of its light, seen in a view
    (see View.setup): the number of lines its plan adds to spectral axes
    (0 in an image, or without light), the row of the line of each of its
    traits of line opacity among them (-1 for a line it does not
    evaluate; None without lines), the fluxes of its rows of light (of
    its lines on a spectral axis; in an image, their sum is its light),
    the place of each line on the spectral axis, and the scales of its
    line opacity.
    """
    nlines: int = 0
    opacity_rows: list[int] | None = None
    fluxes: Fluxes | None = None
    lines: LinePlacement | None = None
    opacity_scales: OpacityScales | None = None


def _band(band: str | Band | None) -> Band | None:
    """Return a band, looked up in the catalogue in use if a name."""
    if isinstance(band, str):
        with config_path('band'):
            return catalogue_in_use().band(band)
    return band


class Tracer(TypedSerializable, abc.ABC):
    """
    What the data of an observation trace of the light of the components
    (see the module): which components it sees (sees), and how their
    light is drawn (setup) and dimmed by dust (wavelengths), on a
    spectral axis or in an image (spectral; see View). Each kind of
    observation takes some tracers, and has one by default (see
    ObservationSky).
    """

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type())

    def band(self) -> Band | None:
        """Return its band, if it has one."""
        return None

    def line_names(self) -> tuple[str, ...] | None:
        """
        Return the names of the lines it traces (None for all of them);
        none here, as it traces no lines.
        """
        return ()

    def dims_by_redshift(self) -> bool:
        """
        Check whether it sees the continuum with a unit dimmed by the
        redshift of its galaxy (see _units.BandFactors), which must then
        be known (see check_frame).
        """
        return False

    def check(self, lights: Sequence[Light]) -> None:
        """
        Check what it sees, of the light of all the components it sees
        (e.g. that each line it names is one of theirs); nothing here.

        Raises
        ------
        ValueError
            If it cannot see it.
        """

    @abc.abstractmethod
    def sees(self, light: Light, moves: bool) -> bool:
        """
        Check whether it sees the light of a component that moves or not
        (has kinematics, see Component.moves).

        Raises
        ------
        ValueError
            If it would see the light only in part (e.g. without the
            velocities that it traces): the component must be left out of
            what the observation sees.
        """

    @abc.abstractmethod
    def setup(
            self,
            light: Light,
            spectral: bool,
            axis: 'Grid',
            data_unit: DataUnit | None,
            redshift: float,
            opacity_lines: Sequence[int] = ()
    ) -> LightSetup:
        """
        Return what a plan of a component evaluates of its light, which it
        sees (see LightSetup), on a spectral axis (spectral, of the grid
        axis) or in an image, in the unit of the data (if any; see
        Component.plan), at the redshift of its galaxy. opacity_lines has
        the line of each trait of line opacity of the component: its index
        among its lines.
        """

    @abc.abstractmethod
    def wavelengths(
            self, light: Light, spectral: bool, redshift: float
    ) -> tuple[float | None, ...]:
        """
        Return the rest wavelength (um) of each row of the light of a
        component that it sees, on a spectral axis (spectral; a row for
        each line it draws) or in an image (one row), of a component at
        the given redshift; None where it is not known. Dust of an
        extinction law dims each row at its wavelength.
        """

    @abc.abstractmethod
    def rows(self, light: Light, spectral: bool) -> tuple[str | None, ...]:
        """
        Return what it draws on a spectral axis of the light of a component
        that it sees, one entry for each row: the name of each line (None
        for the one line without a name, and for a row without a rest:
        e.g. the velocities of stars); none in an image (spectral false).
        """


class LinesTracer(Tracer):
    """
    The light of emission lines (e.g. of the gas): of the lines of the
    given names, or all of them. On a spectral axis, each line is at its
    place and its velocities; in an image, the light of the lines,
    integrated over their velocities (e.g. a narrow-band image of Halpha,
    without its continuum). It sees the components of lines that have
    one of the lines (a component of one line without a name, only
    without names).

    An image of the lines sees their light dimmed by dust at one
    wavelength: that of its one line (dust of an extinction law cannot
    dim several lines of an image, each at its wavelength); it cannot see
    lines with line opacity (their light depends on their profile), nor
    yet the light of lines with a unit in data with a unit (see setup).

    Parameters
    ----------
    lines : Sequence of str, optional
        The names of the lines; all if None.

    Raises
    ------
    ValueError
        If the names are empty or not valid.
    """

    config_titles = dict(lines='Lines')

    @staticmethod
    def type() -> str:
        return 'lines'

    @staticmethod
    def type_title() -> str:
        return 'Emission lines'

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type()) | (
            {} if self._lines is None else dict(lines=list(self._lines)))

    def __init__(self, lines: Sequence[str] | None = None):
        check_options(LinesTracer.__init__, locals())
        if lines is not None:
            lines = tuple(lines)
            if not lines:
                raise ValueError(
                    "lines needs at least one name; leave it out to trace "
                    "all the lines")
            for i, line in enumerate(lines):
                with config_path('lines', i):
                    check_name(line)
        self._lines = lines

    def line_names(self) -> tuple[str, ...] | None:
        return self._lines

    def sees(self, light: Light, moves: bool) -> bool:
        lines = light.lines()
        if lines is None:
            return False
        if self._lines is None:
            return True
        return any(name in self._lines for name in lines.names())

    def check(self, lights: Sequence[Light]) -> None:
        if self._lines is None:
            return
        known = {name for light in lights for name in light.line_names()}
        if unknown := [name for name in self._lines if name not in known]:
            raise ValueError(
                f"no component has the lines {unknown}; the lines of the "
                f"components are {sorted(known)}")

    def setup(
            self,
            light: Light,
            spectral: bool,
            axis: 'Grid',
            data_unit: DataUnit | None,
            redshift: float,
            opacity_lines: Sequence[int] = ()
    ) -> LightSetup:
        lines, unit = light.lines(), light.unit()
        selected = lines.select(self._lines)
        ratios = [
            (row, lines.ratio_name(index))
            for row, index in enumerate(selected)
            if lines.ratio_name(index) is not None]
        if not spectral:
            if opacity_lines:
                raise ValueError(
                    "an image of lines cannot see a component with line "
                    "opacity: the light of its lines depends on their "
                    "profile; see them in a spectral cube (e.g. its "
                    "moment 0)")
            if unit is not None and data_unit is not None:
                raise ValueError(
                    "an image of lines whose light has a unit, in data "
                    "with a unit, is not supported yet: give the light no "
                    "unit (its density is then in the unit of the image), "
                    "or see the lines in a spectral cube")
            return LightSetup(fluxes=Fluxes(len(selected), ratios))
        # (and the row of the line of each trait of line opacity among the
        # lines it evaluates)
        opacity_rows = [
            selected.index(line) if line in selected else -1
            for line in opacity_lines]
        factors = lines.factors(axis, redshift)[list(selected)]
        # (with a unit, the frequency at which the galaxy emits each line)
        frequencies = None if unit is None \
            else lines.frequencies(axis, redshift)[list(selected)]
        # (with a unit, the coefficients of line opacity are 1/S in 1/K)
        opacity_scales = None
        if opacity_lines and unit is not None:
            opacity_scales = OpacityScales(
                unit, data_unit, factors, axis, frequencies)
        return LightSetup(
            len(selected), opacity_rows,
            Fluxes(len(factors), ratios, line_factors(
                unit, data_unit, factors, axis, frequencies)),
            LinePlacement(factors, axis), opacity_scales)

    def wavelengths(
            self, light: Light, spectral: bool, redshift: float
    ) -> tuple[float | None, ...]:
        lines = light.lines()
        # (one line at the velocity of the spectral axis: whatever line)
        if lines.lines() is None:
            return (None,)
        rests = tuple(
            lines.lines()[i].rest().to_value(
                astropy.units.um, astropy.units.spectral())
            for i in lines.select(self._lines))
        if spectral:
            return rests
        # (an image dims its light at one wavelength: that of one line;
        # several have none, which dust of an extinction law needs)
        return rests if len(rests) == 1 else (None,)

    def rows(self, light: Light, spectral: bool) -> tuple[str | None, ...]:
        if not spectral:
            return ()
        lines = light.lines()
        names = lines.names()
        if not names:
            return (None,)
        return tuple(names[i] for i in lines.select(self._lines))


class _BandTracer(Tracer, abc.ABC):
    """
    A tracer of the continuum of the components in a band (see
    ContinuumTracer and LosvdTracer): of those that emit in the band (a
    continuum without bands emits in every band). Its band, if known,
    gives the flux ratio of the light of a component of bands, and the
    wavelength at which dust of an extinction law dims it; it is needed
    to see a component of bands.
    """

    config_titles = dict(band='Band')

    def dump(self) -> dict[str, Any]:
        return dict(type=self.type()) | (
            {} if self._band is None
            else dict(band=band_parser.dump(self._band)))

    def __init__(self, band: str | Band | None):
        self._band = _band(band)

    def band(self) -> Band | None:
        return self._band

    def dims_by_redshift(self) -> bool:
        return True

    def _emits_in(self, light: Light) -> bool:
        """
        Check whether a light is a continuum that emits in its band.

        Raises
        ------
        ValueError
            If the continuum has bands, and it has no band.
        """
        bands = light.bands()
        if bands is None:
            return False
        if self._band is None and bands.bands() is not None:
            raise ValueError(
                f"it emits in the bands {list(bands.names())}, and the "
                f"tracer of the observation has no band: give it the band "
                f"of its data (e.g. tracer: {{type: {self.type()}, band: "
                f"{bands.names()[0]}}})")
        return bands.emits_in(self._band)

    def _fluxes(
            self,
            light: Light,
            data_unit: DataUnit | None,
            redshift: float
    ) -> Fluxes:
        """
        Return the flux of the one row of the light of a continuum that
        emits in its band: its flux ratio in the band, and the factor of
        its unit (see _units.BandFactors).
        """
        ratio = light.bands().ratio_name(self._band)
        return Fluxes(
            1, [(0, ratio)] if ratio is not None else [],
            band_factors(light.unit(), data_unit, self._band, redshift))

    def wavelengths(
            self, light: Light, spectral: bool, redshift: float
    ) -> tuple[float | None, ...]:
        if self._band is None:
            return (None,)
        return (self._band.wavelength().to_value(astropy.units.um)
                / (1 + redshift),)


class ContinuumTracer(_BandTracer):
    """
    The continuum in a band (e.g. the light of the stars in an image): of
    the components of a continuum that emit in the band (see _BandTracer),
    flat across it.

    Parameters
    ----------
    band : str or Band, optional
        The band, or the name of a band of the catalogue in use (see
        lines_from); not known if None, which can see only components of
        a continuum without bands.
    """

    @staticmethod
    def type() -> str:
        return 'continuum'

    @staticmethod
    def type_title() -> str:
        return 'Continuum'

    def __init__(self, band: str | Band | None = None):
        check_options(ContinuumTracer.__init__, locals())
        super().__init__(band)

    def sees(self, light: Light, moves: bool) -> bool:
        return self._emits_in(light)

    def setup(
            self,
            light: Light,
            spectral: bool,
            axis: 'Grid',
            data_unit: DataUnit | None,
            redshift: float,
            opacity_lines: Sequence[int] = ()
    ) -> LightSetup:
        return LightSetup(fluxes=self._fluxes(light, data_unit, redshift))

    def rows(self, light: Light, spectral: bool) -> tuple[str | None, ...]:
        return ()


class LosvdTracer(_BandTracer):
    """
    The distribution of the velocities of the stars along the line of
    sight (LOSVD; e.g. the kinematics of pPXF), weighted by their light in
    a band: on a spectral axis of velocities, one row of light without a
    rest for each component of a continuum that emits in the band (see
    _BandTracer), at its velocities, of its flux in the band. Every
    component that emits in the band must move: the velocities of the
    stars are those of all their light, so a component without
    kinematics must be left out of what the observation sees (the
    section seen of its setup).

    Parameters
    ----------
    band : str or Band, optional
        The band of the light that weights the velocities, or the name of
        a band of the catalogue in use (see lines_from); not known if None
        (as ContinuumTracer).
    """

    @staticmethod
    def type() -> str:
        return 'losvd'

    @staticmethod
    def type_title() -> str:
        return 'Stellar velocities (LOSVD)'

    def __init__(self, band: str | Band | None = None):
        check_options(LosvdTracer.__init__, locals())
        super().__init__(band)

    def sees(self, light: Light, moves: bool) -> bool:
        if not self._emits_in(light):
            return False
        if not moves:
            raise ValueError(
                "it has no kinematics, and the velocities of the stars are "
                "those of all their light in the band: give it kinematics, "
                "or leave it out of what the observation sees (the section "
                "seen of its setup)")
        return True

    def setup(
            self,
            light: Light,
            spectral: bool,
            axis: 'Grid',
            data_unit: DataUnit | None,
            redshift: float,
            opacity_lines: Sequence[int] = ()
    ) -> LightSetup:
        # (one row without a rest, at the velocities of the component)
        return LightSetup(
            nlines=1, fluxes=self._fluxes(light, data_unit, redshift),
            lines=LinePlacement(np.ones(1), axis))

    def rows(self, light: Light, spectral: bool) -> tuple[str | None, ...]:
        return (None,) if spectral else ()


@dataclasses.dataclass(frozen=True)
class View:
    """
    What an observation sees of the light of the components: its tracer
    (see Tracer), on a spectral axis (spectral) or in an image. Its
    methods are those of its tracer, of its axis.
    """
    spectral: bool
    tracer: Tracer

    def band(self) -> Band | None:
        """Return the band of its tracer, if any."""
        return self.tracer.band()

    def sees(self, light: Light, moves: bool) -> bool:
        """See Tracer.sees."""
        return self.tracer.sees(light, moves)

    def setup(
            self,
            light: Light,
            axis: 'Grid',
            data_unit: DataUnit | None,
            redshift: float,
            opacity_lines: Sequence[int] = ()
    ) -> LightSetup:
        """See Tracer.setup."""
        return self.tracer.setup(
            light, self.spectral, axis, data_unit, redshift, opacity_lines)

    def wavelengths(
            self, light: Light, redshift: float
    ) -> tuple[float | None, ...]:
        """See Tracer.wavelengths."""
        return self.tracer.wavelengths(light, self.spectral, redshift)

    def rows(self, light: Light) -> tuple[str | None, ...]:
        """See Tracer.rows."""
        return self.tracer.rows(light, self.spectral)


tracer_parser = TypedParser(Tracer, [
    LinesTracer,
    ContinuumTracer,
    LosvdTracer], family='tracer')
