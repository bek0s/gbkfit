"""
Physical units of data and models: parsing them from the text of
configurations and FITS headers (BUNIT).
"""

import re

import astropy.units

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'format_unit',
    'parse_unit',
    'spatial_unit'
]


# The terms of a unit that multiply each other: separated by spaces, '*'
# (not '**', a power) or '.' (FITS: 'JY/BEAM.KM/S'; not the point of a
# number)
_TERM_SEPARATOR = re.compile(r'\s+|(?<!\*)\*(?!\*)|(?<![0-9])\.(?![0-9])')

# The units of FITS headers that are written in capitals (e.g. 'JY/BEAM')
_CAPITALS = {
    'JY': 'Jy', 'MJY': 'MJy', 'BEAM': 'beam', 'KM': 'km', 'M': 'm',
    'S': 's', 'HZ': 'Hz', 'KHZ': 'kHz', 'MHZ': 'MHz', 'GHZ': 'GHz',
    'SR': 'sr', 'ERG': 'erg', 'CM': 'cm', 'ANGSTROM': 'Angstrom',
    'ARCSEC': 'arcsec', 'DEG': 'deg', 'PIXEL': 'pix', 'PIX': 'pix'}


def parse_unit(text: str) -> astropy.units.UnitBase:
    """
    Return the unit of the given text (e.g. 'Jy/beam', 'MJy/sr',
    '1e-17 erg/s/cm2/arcsec2', or BUNIT of a FITS header).

    Terms separated by spaces, '*' or '.' multiply each other, each read
    on its own: astropy reads 'Jy/beam km/s' as Jy / (beam km s), but the
    unit of a moment 0 map is Jy km/s per beam. Units written in capitals
    (e.g. 'JY/BEAM.KM/S') are read in the case of their names. Else the
    text is read by astropy as a whole (e.g. its own 'Jy / beam' or
    'Jy km / (beam s)', whose spaces are not between terms).

    Parameters
    ----------
    text : str
        The unit.

    Returns
    -------
    astropy.units.UnitBase
        The unit.

    Raises
    ------
    ConfigError
        If the text is not a unit, or is one of electricity, which no data
        are in (e.g. 'erg/s/cm2/A': A is the ampere, the angstrom is
        Angstrom).
    """
    # A unit of electricity may be the capitals of another (e.g. S, the
    # siemens, for s), so the other readings are tried before it is refused
    electric = False
    for parse in (_product, astropy.units.Unit):
        for candidate in (text, _uncapitalised(text)):
            try:
                unit = parse(candidate)
            except ValueError:
                continue
            if astropy.units.A in unit.decompose().bases:
                electric = True
                continue
            return unit
    if electric:
        raise ConfigError(
            f"{text!r} is a unit of electricity, which no data are in: if A "
            f"is meant as the angstrom, write Angstrom (e.g. "
            f"'erg/s/cm2/Angstrom')")
    raise ConfigError(
        f"{text!r} is not a unit (e.g. 'Jy/beam', 'K', 'MJy/sr', "
        f"'1e-17 erg/s/cm2/arcsec2')")


def format_unit(unit: astropy.units.UnitBase) -> str:
    """
    Return a unit as text that parse_unit reads as the same unit: its
    scale and its terms with their powers (e.g. 'Jy km beam-1 s-1').
    """
    # repr gives the shortest text of the scale that reads back exactly
    terms = [] if unit.scale == 1 else [repr(float(unit.scale))]
    for base, power in zip(unit.bases, unit.powers):
        name = base.to_string()
        terms.append(name if power == 1 else f'{name}{power:g}')
    return ' '.join(terms)


def spatial_unit(unit: astropy.units.UnitBase) -> astropy.units.UnitBase:
    """
    Return what light in a unit is per: a beam (e.g. Jy/beam), a solid
    angle (sr; e.g. MJy/sr, erg/s/cm2/arcsec2), a pixel named so (pix;
    e.g. Jy/pixel), or neither (dimensionless): a pixel, or a region (e.g.
    1e-20 erg/s/cm2/Angstrom of a spaxel).

    Raises
    ------
    ConfigError
        If the unit is per an angle that is not a solid angle (e.g.
        Jy/arcsec).
    """
    if astropy.units.beam in unit.bases:
        return astropy.units.beam
    if astropy.units.pix in unit.bases:
        return astropy.units.pix
    # The power of the angles of the unit: -2 per solid angle (an angle
    # squared, e.g. arcsec2, or a solid angle, e.g. sr)
    power = 0
    for base, base_power in zip(unit.bases, unit.powers):
        if base.physical_type == 'solid angle':
            power += 2 * base_power
        elif base.physical_type == 'angle':
            power += base_power
    if power == -2:
        return astropy.units.sr
    if power != 0:
        raise ConfigError(
            f"{format_unit(unit)!r} is per an angle that is not a solid "
            f"angle (e.g. per arcsec, not per arcsec2)")
    return astropy.units.dimensionless_unscaled


def _product(text: str) -> astropy.units.UnitBase:
    """
    Return the product of the terms of a unit (see parse_unit). Raise
    ValueError if a term is not a unit.
    """
    terms = [term for term in _TERM_SEPARATOR.split(text.strip()) if term]
    if not terms:
        raise ValueError("no unit")
    unit = astropy.units.dimensionless_unscaled
    for term in terms:
        unit = unit * _divided(term)
    return unit


def _divided(term: str) -> astropy.units.UnitBase:
    """
    Return the unit of a term whose parts are divided by '/', from the
    left (e.g. 'erg/s/cm2' is erg / (s cm2)). Raise ValueError if a part
    is empty or not a unit.

    The parts are read one by one because astropy warns of a text with
    several '/' (as the FITS standard discourages it), and silencing that
    warning would change the warnings of every thread.
    """
    first, *divisors = term.split('/')
    if not first and not divisors or '' in divisors:
        raise ValueError(f"{term!r} has an empty part")
    unit = astropy.units.Unit(first) if first else (
        astropy.units.dimensionless_unscaled)
    for divisor in divisors:
        unit = unit / astropy.units.Unit(divisor)
    return unit


def _uncapitalised(text: str) -> str:
    """Return a unit with the names of _CAPITALS in their case."""
    return re.sub(
        r'[A-Za-z]+',
        lambda match: _CAPITALS.get(match.group(0).upper(), match.group(0)),
        text)
