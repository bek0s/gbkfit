"""
The physical units of the light of components (see LineEmission and
BandEmission): which units each kind of emission takes, and the factors
that turn them into the units of the data of an observation (see
DataUnit, LineFactors and BandFactors). The light of a component with a
unit is that of the galaxy as it emits it, in its rest frame; each
observation sees it dimmed by the redshift (see line_factor), so that
physical quantities follow from it at any redshift: the mass of its gas
or stars (see mass_per_unit), and the source function of its line
opacity (see OpacityScales).
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import astropy.constants as const
import astropy.units as u
import numpy as np

from gbkfit.utils import gridutils, unitutils
from gbkfit.utils.parseutils import ConfigError
from .lines import Band, LinePlacement

if TYPE_CHECKING:
    from .base import Component
    from .roles import Emission


__all__ = [
    'DataUnit',
    'LineFactors',
    'BandFactors',
    'OpacityScales',
    'band_factors',
    'check_band_unit',
    'check_line_unit',
    'check_line_units_agree',
    'is_hi_unit',
    'line_factor',
    'line_factors',
    'mass_per_unit'
]


# Note (2026-10-15; the user asked for it, to revise later): there is no
# option for the light of a component as we observe it (e.g. an emission
# option 'brightness: observed', its parameters the observed brightness).
# A component without a unit already fits the observed brightness, in
# the units of the data. Such an option would skip the dimming here (in
# line_factor and BandFactors), and undo it, times (1 + z)^n with n of
# each kind of unit, in mass_per_unit and OpacityScales, which take the
# light as the galaxy emits it.


# The units of the lines: their light integrated over the line, per
# solid angle (as energy, as flux density times velocity, or as
# brightness temperature times velocity), and those of HI (21 cm) only:
# its column density and the surface density of its mass
_LINE_UNITS = (
    u.W / u.m ** 2 / u.sr, u.Jy * u.km / u.s / u.sr, u.K * u.km / u.s)
_HI_UNITS = (u.cm ** -2, u.Msun / u.pc ** 2)

# The units of the bands: flux density per solid angle (per frequency or
# per wavelength), or brightness temperature
_BAND_UNITS = (u.Jy / u.sr, u.W / u.m ** 3 / u.sr, u.K)

# The name of the HI line in the catalogue (see LineCatalogue)
HI_LINE = 'hi21'

# The column density of HI per K km/s of its brightness temperature, for
# optically thin gas (Dickey & Lockman 1990, ARA&A 28, 215), and the mass
# of a hydrogen atom (for the surface density of the mass of HI, without
# helium)
_HI_COLUMN = 1.823e18 * u.cm ** -2 / (u.K * u.km / u.s)
_H_MASS = 1.00794 * u.u

# The units that the conversions go through: intensity (flux density per
# solid angle), and its integral over the velocity of a line
_INTENSITY = u.Jy / u.sr
_INTEGRATED = u.Jy * u.km / u.s / u.sr

# The speed of light (km/s)
_C = const.c.to_value(u.km / u.s)

# Two frequencies at which the factor of a band must agree when the band
# is not known (see BandFactors)
_NOMINAL_BANDS = (1e14 * u.Hz, 2e14 * u.Hz)


def is_hi_unit(unit: u.UnitBase) -> bool:
    """Check whether a unit is one of HI only (column or mass density)."""
    return any(unit.is_equivalent(hi) for hi in _HI_UNITS)


def check_line_unit(unit: u.UnitBase, lines: Sequence[str] | None) -> None:
    """
    Raise ConfigError unless a unit is one of the light of emission lines
    of the given names (None: one line, whatever it is): integrated over
    the line, per solid angle (e.g. 'erg/s/cm2/arcsec2', 'Jy km/s/arcsec2',
    'K km/s'), or of HI only, whose first line must then be hi21 (e.g.
    'Msun/pc2', 'cm-2').
    """
    if is_hi_unit(unit):
        if not lines or lines[0] != HI_LINE:
            raise ConfigError(
                f"the unit {unitutils.format_unit(unit)!r} is one of HI only: "
                f"its first line must be {HI_LINE!r} (lines: [{HI_LINE}]); "
                f"the lines are {lines}")
        return
    if not any(unit.is_equivalent(kind) for kind in _LINE_UNITS):
        raise ConfigError(
            f"the unit of emission lines is that of their light integrated "
            f"over the line, per solid angle (e.g. 'erg/s/cm2/arcsec2', "
            f"'Jy km/s/arcsec2', 'K km/s'), or of HI only ('Msun/pc2', "
            f"'cm-2'); {unitutils.format_unit(unit)!r} is not (a unit per "
            f"beam is one of data, not of a galaxy)")


def check_band_unit(unit: u.UnitBase) -> None:
    """
    Raise ConfigError unless a unit is one of light in bands: flux density
    per solid angle (e.g. 'Jy/arcsec2', 'MJy/sr', 'erg/s/cm2/Angstrom/
    arcsec2'), or brightness temperature ('K').
    """
    if not any(unit.is_equivalent(kind) for kind in _BAND_UNITS):
        raise ConfigError(
            f"the unit of light in bands is a flux density per solid angle "
            f"(e.g. 'Jy/arcsec2', 'MJy/sr', 'erg/s/cm2/Angstrom/arcsec2') or "
            f"a brightness temperature ('K'); "
            f"{unitutils.format_unit(unit)!r} is not (a unit per beam is one "
            f"of data, not of a galaxy)")


def check_line_units_agree(components: Sequence['Component']) -> None:
    """
    Raise ConfigError if components of a galaxy give one named line
    different units (e.g. a typo: hi21 beside halpha in a component of
    another unit). The lines of a component have the unit of its emission
    (its further lines are ratios); unnamed lines can be different lines,
    and are not compared.
    """
    # (imported here: roles imports this module)
    from .roles import LineEmission
    seen = {}
    for i, component in enumerate(components):
        emission = component.emission()
        if not isinstance(emission, LineEmission) or emission.unit() is None:
            continue
        unit = emission.unit()
        label = component.name() or f'component{i}'
        for line in emission.lines().names():
            if line in seen and seen[line][0] != unit:
                first, other = (
                    unitutils.format_unit(x) for x in (seen[line][0], unit))
                raise ConfigError(
                    f"the line {line!r} has the unit {first!r} in component "
                    f"{seen[line][1]!r} and {other!r} in component "
                    f"{label!r}: give it one unit, or emit it from a "
                    f"component of its own")
            seen.setdefault(line, (unit, label))


def mass_per_unit(emission: 'Emission | None') -> float | None:
    """
    Return the mass surface density (Msun/kpc^2) of one unit of the
    density of a component (see mass.SurfaceComponent), if its emission has
    a unit: for HI (its first line hi21), that of the HI of one unit, by
    N_HI = 1.823e18 cm^-2 per K km/s (at the rest frequency of the line,
    for units of flux density or energy), so that upsilon is the mass of
    gas per mass of HI (1.33 with helium); for other units 1 Msun/pc^2, so
    that upsilon is in Msun/pc^2 per unit (e.g. alpha_CO per K km/s, or a
    mass-to-light ratio at the wavelength at which the galaxy emits the
    light). The light is that of the galaxy as it emits it (see
    line_factor), so this holds at any redshift. None without a unit.
    """
    # (imported here: roles imports this module)
    from .roles import LineEmission
    unit = None if emission is None else emission.unit()
    if unit is None:
        return None
    per_kpc2 = u.Msun / u.kpc ** 2
    if not isinstance(emission, LineEmission) \
            or emission.lines().names()[:1] != (HI_LINE,):
        return (1 * u.Msun / u.pc ** 2).to_value(per_kpc2)
    frequency = emission.lines().lines()[0].rest().to(u.Hz, u.spectral())
    frame = _frame(gridutils.VelocityConvention(frequency), 0.0, frequency)
    intensity = _axis_integral(1 * unit, frame) / (u.km / u.s)
    temperature = intensity.to(u.K, u.brightness_temperature(frequency))
    column = temperature * (u.km / u.s) * _HI_COLUMN
    return (column * _H_MASS).to_value(per_kpc2)


class DataUnit(NamedTuple):
    """
    The unit of the light in the data of an observation, which its model
    takes (see Observable.data_unit): the unit (None if the data have
    none); the solid angle of a pixel of the data (arcsec^2), for units
    per pixel; that of the beam of the data (arcsec^2), for units per
    beam: one, or one for each of the spectral points beam_spectral
    (frequencies, wavelengths or velocities; e.g. CASA cubes); and whether
    the light is integrated over the lines (e.g. moment 0), rather than
    per km/s of the spectral axis (spectral cubes) or in a band (images).
    """
    unit: u.UnitBase | None
    pixel: float
    beam: float | np.ndarray | None = None
    beam_spectral: u.Quantity | None = None
    integrated: bool = False


class LineFrame(NamedTuple):
    """
    Where a line is on the spectral axis of an observation (see
    line_factor): its frequency there (in vacuum), how many Hz of it and
    how many metres of the wavelengths of the axis (in their medium; in
    vacuum for an axis of frequencies) one km/s of the axis spans
    (hz_per_kms, m_per_kms), how many km/s of the axis one km/s of the
    velocity of its component spans there (scale; see LinePlacement), and
    the frequency at which the galaxy emits it, in its rest frame
    (emitted; emitted / frequency is 1 + z, the redshift of the line).
    """
    frequency: u.Quantity
    hz_per_kms: u.Quantity
    m_per_kms: u.Quantity
    scale: float
    emitted: u.Quantity


def line_factor(
        source: u.UnitBase,
        target: u.UnitBase,
        frame: LineFrame,
        beam: float | None,
        pixel: float,
        integrated: bool
) -> float:
    """
    Return how many target units one source unit of the light of a line
    is: of the light of a component (source, see check_line_unit) in the
    units of the data of an observation (target), as the model of a
    spectral cube holds it, per km/s of its spectral axis, or integrated
    over the line (integrated: e.g. moment 0, the integral over the axis).
    frame tells where the line is on the axis.

    The data are per solid angle (e.g. MJy/sr, K), per beam (e.g.
    Jy/beam: beam is its solid angle, arcsec^2), or per pixel (e.g. 1e-20
    erg/s/cm2/Angstrom per spaxel: pixel is its solid angle, arcsec^2), per
    frequency or per wavelength (of the axis), or brightness temperature,
    or for integrated light also its integral (e.g. Jy/beam km/s, K km/s,
    erg/s/cm2/arcsec2). The light of a component in units of velocity
    (e.g. K km/s, Jy km/s/arcsec2) is integrated over the velocity of the
    component, of which each km/s spans scale km/s of the axis.

    The light of the component is that of the galaxy as it emits it, in
    its rest frame, at the frequency emitted of the frame (e.g. its
    brightness temperature there). The observation sees it dimmed by the
    redshift of the line, 1 + z = emitted / frequency: its intensity by
    exactly (1 + z)^-3 (I_nu / nu^3 does not change along the path of the
    light), that is (1 + z)^-4 for light integrated over frequency
    (energy), (1 + z)^-5 per wavelength and (1 + z)^-1 for brightness
    temperature; it is as wide in km/s of the galaxy (see LinePlacement).

    The conversions that depend on the frequency of the light within the
    line (brightness temperature ~ nu^2, the dimming ~ nu^3, the km/s of
    the axis that a km/s of the galaxy spans, and on axes of optical
    velocities the frequency that a km/s spans ~ nu^2) are made at the
    frequency of the frame: that of the line at the systemic velocity of
    its component (see LineFactors). Across a line of +-v they change by
    about +-p v / c, p the power of the frequency of the factor (4 for HI
    in Jy/beam, or Halpha in erg/s/cm2/Angstrom: +-0.27% at 200 km/s),
    in opposite ways on the two sides of a rotating disk; one factor per
    line leaves this out: an approximation that the user accepted
    (2026-10-14 for brightness temperature, 2026-10-15 for the dimming);
    exact factors for each channel would need weights for each channel
    in the kernels.

    Raises
    ------
    ConfigError
        If the target is not a unit of the light of lines, or it is per
        beam and there is no beam.
    """
    try:
        light = _axis_integral(1 * source, frame)
        if integrated and _is_energy(target):
            return _per_element(
                (light * frame.hz_per_kms).to(u.W / u.m ** 2 / u.sr), target,
                beam, pixel)
        channel = target / (u.km / u.s) if integrated else target
        return _channel(light / (u.km / u.s), channel, frame, beam, pixel)
    except u.UnitConversionError as e:
        raise ConfigError(
            f"the light of a line in {unitutils.format_unit(source)!r} "
            f"cannot be data in {unitutils.format_unit(target)!r}: {e}") from e


def line_factors(
        source: u.UnitBase | None,
        target: DataUnit | None,
        redshifts: np.ndarray,
        spectral: gridutils.Grid,
        frequencies: np.ndarray | None
) -> 'LineFactors | None':
    """
    Return the factors of the lines of a component (see LineFactors), or
    None if it needs none: without a unit (source), the light of the
    component is in the unit of the data; without data (target), the
    model is in the unit of its components.

    Raises
    ------
    ConfigError
        If the component has a unit and the data have none, or a factor
        cannot be computed (see LineFactors).
    """
    if source is None or target is None:
        return None
    if target.unit is None:
        raise ConfigError(
            f"the light of the component is in "
            f"{unitutils.format_unit(source)!r}, but the data have no unit: "
            f"give them one (BUNIT in the header, or the option unit of the "
            f"data)")
    return LineFactors(source, target, redshifts, spectral, frequencies)


class LineFactors:
    """
    The factors that turn the light of the lines of a component, in the
    unit of its emission (source), into the unit of the data of an
    observation (target), as the model holds them (see line_factor): one
    for each line, of the given redshifts (the 1 + z of each on the
    spectral axis, see Lines.factors), which the galaxy emits at the given
    frequencies (Hz, see Lines.frequencies), at the frequency of the line
    at the systemic velocity of the component (see __call__), which places
    it on the axis.

    Each factor is a power of that frequency (brightness temperature
    ~ nu^-2; the dimming by the redshift ~ nu^3; the frequency or
    wavelength that a km/s of the axis spans ~ nu^0 or nu^2; the km/s of
    the axis that a km/s of the component spans ~ nu or 1/nu, see
    LinePlacement), which two frequencies give exactly; with a beam for
    each channel, the factor is also that of the beam at the line,
    interpolated between the channels.

    Raises
    ------
    ConfigError
        If a factor cannot be computed (see line_factor), or the spectral
        axis has no rest: the data see the light of the component dimmed
        by the redshift of each line, which takes where the line is.
    """

    def __init__(
            self,
            source: u.UnitBase,
            target: DataUnit,
            redshifts: np.ndarray,
            spectral: gridutils.Grid,
            frequencies: np.ndarray
    ):
        coords = spectral.coords
        if coords.rest is None:
            raise ConfigError(
                f"the light of the component in "
                f"{unitutils.format_unit(source)!r} is that of the galaxy as "
                f"it emits it, which the data see dimmed by the redshift of "
                f"each line: the spectral axis needs its rest (RESTFRQ or "
                f"RESTWAV in the header of the data, or spectral: {{rest: "
                f"...}})")
        self._convention = gridutils.VelocityConvention.of(coords)
        self._placement = LinePlacement(redshifts, spectral)
        # (the velocity of each line on the axis at the systemic velocity 0)
        self._offsets = self._convention.velocity(redshifts)
        self._emitted = np.array(frequencies, dtype=float)
        self._beam = _BeamCurve(target, self._convention)
        # The factor of each line at its frequency at the systemic velocity
        # 0 (with a beam of 1 arcsec^2), and its power of the frequency
        factors = []
        powers = []
        for offset, emitted in zip(self._offsets, self._emitted):
            factor, power = self._line(source, target, offset, emitted)
            factors.append(factor)
            powers.append(power)
        self._factors = np.array(factors)
        self._powers = np.array(powers)
        self._frequencies = self._frequency(0.0)

    def _line(
            self,
            source: u.UnitBase,
            target: DataUnit,
            offset: float,
            emitted: float
    ) -> tuple[float, float]:
        """
        Return the factor of a line of the given offset, emitted at the
        given frequency (Hz), at its frequency at the velocity 0, and its
        power of the frequency.
        """
        def factor(velocity):
            return line_factor(
                source, target.unit,
                _frame(self._convention, velocity, emitted * u.Hz),
                None if target.beam is None else 1.0, target.pixel,
                target.integrated)
        frequency = self._convention.frequency(offset)
        first = factor(offset)
        second = factor(self._convention.velocity_at(2 * frequency))
        return first, float(np.log2(second / first))

    def _frequency(self, vsys: float) -> np.ndarray:
        """Return the frequency (Hz) of each line at a systemic velocity."""
        return self._convention.frequency(self._placement.centres(vsys))

    def redshifts(self, vsys: float) -> np.ndarray:
        """
        Return 1 + z of each line (the frequency at which the galaxy emits
        it over that on the axis), with the component at the given
        systemic velocity (km/s).
        """
        return self._emitted / self._frequency(vsys)

    def __call__(self, vsys: float) -> np.ndarray:
        """
        Return the factor of each line, with the component at the given
        systemic velocity (km/s).
        """
        factors = self._factors * self._beam(self._placement.centres(vsys))
        return factors * (self._frequency(vsys) / self._frequencies) \
            ** self._powers


class OpacityScales:
    """
    The factors that turn the light of the lines of a component with a
    unit (source), as the model holds it (in the unit of the data of an
    observation, target; or without data, in its own unit, see
    line_factors), into its brightness temperature as the galaxy emits it
    (K per km/s of the spectral axis; 1 + z times that observed, see
    line_factor): the coefficients of its line opacity are then 1/S in
    1/K, S the source function of the gas in its rest frame (e.g.
    J(T_ex), see LineOpacity), and the light of a voxel in K over S is its
    optical depth. Like the factors of the light, they are taken at the
    frequency of each line at the systemic velocity of the component (see
    line_factor).

    Raises
    ------
    ConfigError
        If a factor cannot be computed (see LineFactors).
    """

    def __init__(
            self,
            source: u.UnitBase,
            target: DataUnit | None,
            redshifts: np.ndarray,
            spectral: gridutils.Grid,
            frequencies: np.ndarray
    ):
        self._kelvin = LineFactors(
            source, DataUnit(u.K, 1.0), redshifts, spectral, frequencies)
        self._light = line_factors(
            source, target, redshifts, spectral, frequencies)

    def __call__(self, vsys: float) -> np.ndarray:
        """
        Return the factor of each line, with the component at the given
        systemic velocity (km/s).
        """
        kelvin = self._kelvin(vsys) * self._kelvin.redshifts(vsys)
        return kelvin if self._light is None else kelvin / self._light(vsys)


def band_factors(
        source: u.UnitBase | None,
        target: DataUnit | None,
        band: Band | None,
        redshift: float
) -> 'BandFactors | None':
    """
    Return the factor of the light of a component in a band (see
    BandFactors), of a galaxy at the given redshift, or None if it needs
    none: without a unit (source), the light of the component is in the
    unit of the data; without data (target), the model is in the unit of
    its components.

    Raises
    ------
    ConfigError
        If the component has a unit and the data have none, or the factor
        cannot be computed (see BandFactors).
    """
    if source is None or target is None:
        return None
    if target.unit is None:
        raise ConfigError(
            f"the light of the component is in "
            f"{unitutils.format_unit(source)!r}, but the data have no unit: "
            f"give them one (BUNIT in the header, or the option unit of the "
            f"data)")
    return BandFactors(source, target, band, redshift)


class BandFactors:
    """
    The factor that turns the light of a component in a band, in the unit
    of its emission (source, see check_band_unit), into the unit of the
    data of an image (target), at the band of the image (its effective
    wavelength; without a band, the factor must not depend on it), with
    the component at a systemic velocity (see __call__).

    The light of the component is that of the galaxy as it emits it, in
    its rest frame, at the frequency of the band times 1 + z (which band
    that is there, e.g. for a mass-to-light ratio, is up to the user): the
    image sees its intensity dimmed by (1 + z)^-3 (see line_factor). 1 + z
    is (1 + redshift) doppler(vsys): the redshift of the galaxy, and the
    systemic velocity of the component, if it has one (relativistic, as
    on every spectral axis; see LinePlacement). The factor is a power of
    1 + z, which two redshifts give exactly.

    Raises
    ------
    ConfigError
        If the target is not a unit of light in bands, it is per beam and
        there is no beam, or the factor depends on the frequency and the
        band is not known.
    """

    def __init__(
            self,
            source: u.UnitBase,
            target: DataUnit,
            band: Band | None,
            redshift: float
    ):
        one_plus_z = 1 + redshift
        if band is not None:
            frequency = band.wavelength().to(u.Hz, u.spectral())
        else:
            frequency, other = _NOMINAL_BANDS
            if not np.isclose(
                    _band_factor(source, target, frequency, one_plus_z),
                    _band_factor(source, target, other, one_plus_z),
                    rtol=1e-9, atol=0):
                raise ConfigError(
                    f"light in a band in {unitutils.format_unit(source)!r} "
                    f"becomes data in {unitutils.format_unit(target.unit)!r} "
                    f"at the wavelength of the band: the image needs its band")
        self._factor = _band_factor(source, target, frequency, one_plus_z)
        self._power = float(np.log2(
            _band_factor(source, target, frequency, 2 * one_plus_z)
            / self._factor))

    def __call__(self, vsys: float) -> np.ndarray:
        """
        Return the factor (an array of one), with the component at the
        given systemic velocity (km/s).
        """
        return np.array(
            [self._factor * gridutils.doppler(vsys) ** self._power])


def _band_factor(
        source: u.UnitBase,
        target: DataUnit,
        frequency: u.Quantity,
        one_plus_z: float
) -> float:
    """
    Return the factor of BandFactors at the given frequency of the band,
    with the galaxy at the given 1 + z.
    """
    try:
        # (the intensity of the galaxy where it emits the light, and that
        # which the image sees)
        emitted = frequency * one_plus_z
        light = 1 * source
        if source.is_equivalent(u.K):
            intensity = light.to(
                _INTENSITY, u.brightness_temperature(emitted))
        else:
            intensity = (light * u.sr).to(
                u.Jy, u.spectral_density(emitted)) / u.sr
        intensity = intensity / one_plus_z ** 3
        if target.unit.is_equivalent(u.K):
            return intensity.to_value(
                target.unit, u.brightness_temperature(frequency))
        spatial = unitutils.spatial_unit(target.unit)
        beam = None if target.beam is None else float(target.beam)
        return _element(intensity, spatial, beam, target.pixel).to_value(
            target.unit * spatial, u.spectral_density(frequency))
    except u.UnitConversionError as e:
        raise ConfigError(
            f"light in a band in {unitutils.format_unit(source)!r} cannot be "
            f"data in {unitutils.format_unit(target.unit)!r}: {e}") from e


def _frame(
        convention: gridutils.VelocityConvention,
        velocity: float,
        emitted: u.Quantity
) -> LineFrame:
    """
    Return where a line, which the galaxy emits at the given frequency, is
    on a spectral axis of the given convention at a velocity of the axis
    (see LineFrame; its scale as in LinePlacement).
    """
    frequency = convention.frequency(velocity)
    slope = convention.log_slope(velocity)
    kms = u.km / u.s
    return LineFrame(
        frequency * u.Hz, frequency * slope * u.Hz / kms,
        convention.wavelength(velocity) * slope * u.m / kms,
        convention.stretch(velocity), emitted)


class _BeamCurve:
    """
    The solid angle of the beam of data (arcsec^2) at velocities of the
    spectral axis, for data per beam: that of one beam, or that of the
    beam of each channel, interpolated linearly in the velocity of the
    axis (see DataUnit); 1 for data of other units or without a beam (the
    factors take none).
    """

    def __init__(
            self, target: DataUnit, convention: gridutils.VelocityConvention
    ):
        self._velocities = None
        self._areas = 1.0
        if target.beam is None or u.beam not in target.unit.bases:
            return
        self._areas = target.beam
        if target.beam_spectral is None:
            return
        spectral = target.beam_spectral
        if spectral.unit.is_equivalent(u.km / u.s):
            velocities = spectral.to_value(u.km / u.s)
        else:
            velocities = convention.velocity_at(
                spectral.to_value(u.Hz, u.spectral()))
        order = np.argsort(velocities)
        self._velocities = velocities[order]
        self._areas = np.asarray(target.beam, dtype=float)[order]

    def __call__(self, velocity: np.ndarray) -> float | np.ndarray:
        """Return the solid angle at the given velocities of the axis."""
        if self._velocities is None:
            return self._areas
        return np.interp(velocity, self._velocities, self._areas)


def _axis_integral(light: u.Quantity, frame: LineFrame) -> u.Quantity:
    """
    Return the light of a line (see check_line_unit), as the galaxy emits
    it, as the intensity that the observation sees integrated over the
    velocity of the spectral axis (Jy km/s/sr), at the given frame (see
    line_factor).
    """
    # (as the galaxy emits it: its intensity integrated over the velocity
    # of the component, at the frequency emitted)
    if light.unit.is_equivalent(u.Msun / u.pc ** 2):
        light = (light / _H_MASS).to(u.cm ** -2)
    if light.unit.is_equivalent(u.cm ** -2):
        light = light / _HI_COLUMN
    if light.unit.is_equivalent(u.W / u.m ** 2 / u.sr):
        light = (light / (frame.emitted / (_C * u.km / u.s))).to(_INTEGRATED)
    if light.unit.is_equivalent(u.K * u.km / u.s):
        temperature = light.to(u.K * u.km / u.s) / (u.km / u.s)
        light = temperature.to(
            _INTENSITY, u.brightness_temperature(frame.emitted)) \
            * (u.km / u.s)
    # (as seen: dimmed by the redshift, each km/s of the component spanning
    # scale km/s of the axis)
    dimming = float((frame.frequency / frame.emitted).decompose()) ** 3
    return light.to(_INTEGRATED) * dimming * frame.scale


def _channel(
        intensity: u.Quantity,
        target: u.UnitBase,
        frame: LineFrame,
        beam: float | None,
        pixel: float
) -> float:
    """
    Return the light of a channel per km/s of the axis, as intensity (flux
    density per frequency per solid angle), in the target unit:
    brightness temperature, or flux density per frequency or per
    wavelength of the axis (that a km/s of it spans, see LineFrame), per
    solid angle, beam or pixel.
    """
    if target.is_equivalent(u.K):
        return intensity.to_value(
            target, u.brightness_temperature(frame.frequency))
    spatial = unitutils.spatial_unit(target)
    if (target * spatial).is_equivalent(u.W / u.m ** 3):
        intensity = (intensity * frame.hz_per_kms / frame.m_per_kms).to(
            u.W / u.m ** 3 / u.sr)
    return _element(intensity, spatial, beam, pixel).to_value(
        target * spatial)


def _per_element(
        light: u.Quantity,
        target: u.UnitBase,
        beam: float | None,
        pixel: float
) -> float:
    """
    Return light per solid angle (with no other conversion) in the target
    unit, per solid angle, beam or pixel.
    """
    spatial = unitutils.spatial_unit(target)
    return _element(light, spatial, beam, pixel).to_value(target * spatial)


def _element(
        light: u.Quantity,
        spatial: u.UnitBase,
        beam: float | None,
        pixel: float
) -> u.Quantity:
    """
    Return light per solid angle as the light of one element (see
    unitutils.spatial_unit): of a beam or a pixel (of the given solid
    angles, arcsec^2; a pixel named so, pix, or not), or of a solid angle
    of 1 sr.
    """
    if spatial == u.beam:
        if beam is None:
            raise ConfigError(
                "the data are per beam, but have no beam (BMAJ and BMIN in "
                "the header, or the option beam of the data)")
        return light * beam * u.arcsec ** 2
    if spatial == u.sr:
        return light * u.sr
    return light * pixel * u.arcsec ** 2


def _is_energy(target: u.UnitBase) -> bool:
    """
    Check whether a unit is of energy flux (per solid angle, beam or
    pixel): of light integrated over frequency (e.g. erg/s/cm2/arcsec2).
    """
    return (target * unitutils.spatial_unit(target)).is_equivalent(u.W / u.m ** 2)
