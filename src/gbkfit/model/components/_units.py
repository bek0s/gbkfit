"""
The physical units of the light of components (see LineEmission and
BandEmission): which units each kind of emission takes.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import astropy.constants as const
import astropy.units as u

from gbkfit.utils import unitutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .base import Component


__all__ = [
    'band_factor',
    'check_band_unit',
    'check_line_unit',
    'check_line_units_agree',
    'is_hi_unit',
    'line_factor'
]


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


def line_factor(
        source: u.UnitBase,
        target: u.UnitBase,
        frequency: u.Quantity,
        beam_area: u.Quantity | None,
        pixel_area: u.Quantity,
        integrated: bool
) -> float:
    """
    Return how many target units one source unit of the light of a line
    is: of the light of a component (source, see check_line_unit) in the
    units of the data of an observation (target), per km/s (the model of
    a spectral cube: its light per unit of velocity), or integrated over
    the line (integrated: e.g. moment 0).

    The data are per solid angle (e.g. MJy/sr, K), per beam (e.g.
    Jy/beam: beam_area, the solid angle of the beam, is required), or per
    pixel (e.g. 1e-20 erg/s/cm2/Angstrom per spaxel: of pixel_area), per
    frequency or per wavelength (or brightness temperature), or for
    integrated light also its integral (e.g. Jy/beam km/s, K km/s,
    erg/s/cm2/arcsec2).

    The conversions that depend on the frequency (velocity to frequency or
    wavelength; flux density to brightness temperature) are made at the
    given frequency: that of the line at the systemic velocity of its
    component. Across the line they change by about 2 v / c (0.1% for
    lines of 150 km/s), which one factor per line leaves out: an
    approximation that the user accepted (2026-10-14); exact factors for
    each channel would need weights for each channel in the kernels.

    Raises
    ------
    ConfigError
        If the target is not a unit of the light of lines, or it is per
        beam and there is no beam.
    """
    try:
        light = _integrated_intensity(1 * source, frequency)
        if not integrated:
            return _from_intensity(
                light / (u.km / u.s), target, frequency, beam_area,
                pixel_area)
        if (target / (u.km / u.s)).is_equivalent(u.K) or _is_intensity(
                target / (u.km / u.s)):
            return _from_intensity(
                light / (u.km / u.s), target / (u.km / u.s), frequency,
                beam_area, pixel_area)
        # (integrated light as energy, e.g. erg/s/cm2/arcsec2, or per
        # pixel or beam)
        energy = (light * frequency / const.c).to(u.W / u.m ** 2 / u.sr)
        return _spatial(energy, target, beam_area, pixel_area)
    except u.UnitConversionError as e:
        raise ConfigError(
            f"the light of a line in {unitutils.format_unit(source)!r} "
            f"cannot be data in {unitutils.format_unit(target)!r}: {e}") from e


def band_factor(
        source: u.UnitBase,
        target: u.UnitBase,
        frequency: u.Quantity,
        beam_area: u.Quantity | None,
        pixel_area: u.Quantity
) -> float:
    """
    Return how many target units one source unit of the light in a band
    is (see check_band_unit and line_factor; at the given frequency, that
    of the band at the redshift of the component).

    Raises
    ------
    ConfigError
        If the target is not a unit of light in bands, or it is per beam
        and there is no beam.
    """
    try:
        light = 1 * source
        if source.is_equivalent(u.K):
            light = light.to(_INTENSITY, u.brightness_temperature(frequency))
        else:
            light = (light * u.sr).to(
                u.Jy, u.spectral_density(frequency)) / u.sr
        return _from_intensity(
            light, target, frequency, beam_area, pixel_area)
    except u.UnitConversionError as e:
        raise ConfigError(
            f"light in a band in {unitutils.format_unit(source)!r} cannot be "
            f"data in {unitutils.format_unit(target)!r}: {e}") from e


def _integrated_intensity(
        light: u.Quantity, frequency: u.Quantity
) -> u.Quantity:
    """
    Return the light of a line (see check_line_unit) as intensity
    integrated over the velocity of the line (Jy km/s/sr), at the given
    frequency.
    """
    if light.unit.is_equivalent(u.Msun / u.pc ** 2):
        light = (light / _H_MASS).to(u.cm ** -2)
    if light.unit.is_equivalent(u.cm ** -2):
        light = light / _HI_COLUMN
    if light.unit.is_equivalent(u.K * u.km / u.s):
        temperature = light.to(u.K * u.km / u.s) / (u.km / u.s)
        return temperature.to(
            _INTENSITY, u.brightness_temperature(frequency)) * u.km / u.s
    if light.unit.is_equivalent(u.W / u.m ** 2 / u.sr):
        return (light / (frequency / const.c)).to(_INTEGRATED)
    return light.to(_INTEGRATED)


def _from_intensity(
        intensity: u.Quantity,
        target: u.UnitBase,
        frequency: u.Quantity,
        beam_area: u.Quantity | None,
        pixel_area: u.Quantity
) -> float:
    """
    Return an intensity (flux density per solid angle) in the target
    unit: brightness temperature, or flux density (per frequency or per
    wavelength) per solid angle, beam or pixel.
    """
    if target.is_equivalent(u.K):
        return intensity.to_value(
            target, u.brightness_temperature(frequency))
    spatial = _spatial_unit(target)
    return _per_element(intensity, spatial, beam_area, pixel_area).to_value(
        target * spatial, u.spectral_density(frequency))


def _spatial(
        light: u.Quantity,
        target: u.UnitBase,
        beam_area: u.Quantity | None,
        pixel_area: u.Quantity
) -> float:
    """
    Return light per solid angle (with no other conversion) in the target
    unit, per solid angle, beam or pixel.
    """
    spatial = _spatial_unit(target)
    return _per_element(light, spatial, beam_area, pixel_area).to_value(
        target * spatial)


def _per_element(
        light: u.Quantity,
        spatial: u.UnitBase,
        beam_area: u.Quantity | None,
        pixel_area: u.Quantity
) -> u.Quantity:
    """
    Return light per solid angle as the light of one element (see
    _spatial_unit): of a beam (of the given solid angle), of a solid angle
    of 1 sr, or of a pixel (of the given solid angle).
    """
    if spatial == u.beam:
        if beam_area is None:
            raise ConfigError(
                "the data are per beam, but have no beam (give the option "
                "beam of the data)")
        return light * beam_area
    if spatial == u.sr:
        return light * u.sr
    return light * pixel_area


def _spatial_unit(target: u.UnitBase) -> u.UnitBase:
    """
    Return what a unit of data is per: a beam, a solid angle (sr), or a
    pixel (neither: dimensionless).
    """
    if u.beam in target.bases:
        return u.beam
    if any(base.physical_type in ('solid angle', 'angle')
           for base in target.bases):
        return u.sr
    return u.dimensionless_unscaled


def _is_intensity(unit: u.UnitBase) -> bool:
    """
    Check whether a unit is of flux density (per frequency or per
    wavelength) per solid angle, beam or pixel.
    """
    return (unit * _spatial_unit(unit)).is_equivalent(
        u.Jy, u.spectral_density(1 * u.Hz))
