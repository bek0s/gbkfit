"""
The physical units of the light of components (see LineEmission and
BandEmission): which units each kind of emission takes.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import astropy.units as u

from gbkfit.utils import unitutils
from gbkfit.utils.parseutils import ConfigError

if TYPE_CHECKING:
    from .base import Component


__all__ = [
    'check_band_unit',
    'check_line_unit',
    'check_line_units_agree',
    'is_hi_unit'
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
