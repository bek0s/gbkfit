"""
Physical units of data and models: reading them from the text of
configurations and of FITS headers (BUNIT), and writing them back as text.
"""

import re

import astropy.units


__all__ = [
    'format_unit',
    'parse_unit',
    'spatial_unit'
]


# A regular expression for the separators between the terms of a unit.
# Terms multiply each other: 'Jy/beam km/s' is 'Jy/beam' times 'km/s'.
# A separator is one of:
# - a space, as in 'Jy/beam km/s';
# - a '*', as in 'Jy/beam*km/s' ('**' is a power, as in 'm**2');
# - a '.', as in 'Jy/beam.km/s' (a '.' between digits is a decimal
#   point, as in '1.5').
_TERM_SEPARATOR = re.compile(r'\s+|(?<!\*)\*(?!\*)|(?<![0-9])\.(?![0-9])')

# A map from invalid unit names to the valid names of the FITS standard.
# astropy reads only valid names, which are case-sensitive: to it, 'JY'
# is not a unit, 'S' is the siemens and 'B' is the byte.
# Invalid names come from software that ignores the standard:
# - radio software, in capitals: 'JY/BEAM';
# - AIPS, with names of its own: 'JY/B*M/S', 'METR/SEC';
# - SDSS (MaNGA): 'ang' for the angstrom, 'spaxel' for a pixel;
# - HST and CCD software: 'ELECTRONS/S', 'COUNTS/S', 'ADU'.
_STANDARD_NAMES = {
    'JY': 'Jy', 'MJY': 'MJy', 'BEAM': 'beam', 'B': 'beam', 'KM': 'km',
    'M': 'm', 'METR': 'm', 'S': 's', 'SEC': 's', 'HZ': 'Hz', 'KHZ': 'kHz',
    'MHZ': 'MHz', 'GHZ': 'GHz', 'SR': 'sr', 'ERG': 'erg', 'CM': 'cm',
    'ANGSTROM': 'Angstrom', 'ANG': 'Angstrom', 'ARCSEC': 'arcsec',
    'DEG': 'deg', 'PIXEL': 'pix', 'PIX': 'pix', 'SPAXEL': 'pix',
    'ELECTRON': 'electron', 'ELECTRONS': 'electron', 'COUNT': 'ct',
    'COUNTS': 'ct', 'ADU': 'adu'}

# Units that no data are in: the ampere and the bit. If a reading of a
# unit contains either, a name in it was misread, and the next reading is
# tried. For example:
# - 'A' is the ampere, but is meant as the angstrom;
# - 'S' is the siemens, which contains the ampere (A2 s3 / (kg m2)), but
#   is meant as the second;
# - 'B' is the byte, which is 8 bits, but is meant as the beam.
_MISREAD_UNITS = frozenset({astropy.units.A, astropy.units.bit})


def parse_unit(text: str) -> astropy.units.UnitBase:
    """
    Return the unit that a text states (e.g. 'Jy/beam', 'MJy/sr',
    '1e-17 erg/s/cm2/arcsec2', or the BUNIT of a FITS header).

    The text as it is is read, then the text with its invalid names made
    valid (e.g. 'JY/BEAM.KM/S' as 'Jy/beam.km/s'; see _STANDARD_NAMES),
    until a reading gives a unit that data can be in (see _MISREAD_UNITS).
    Each is read in two ways, the first that works:

    1. As terms that multiply each other, separated by spaces, '*' or '.',
       each read on its own. The FITS standard (4.0, 4.3.1) reads a unit
       by the normal precedence of operations, from the left:
       'Jy/beam km/s' is (Jy/beam) (km/s), as maps of moment 0 mean it.
       astropy reads everything after a '/' as dividing: Jy / (beam km s).
    2. As a whole, by astropy: e.g. astropy's own 'Jy / beam' and
       'Jy km / (beam s)', whose spaces do not separate terms.

    An empty text is dimensionless, as format_unit writes it.

    Parameters
    ----------
    text : str
        The text of the unit.

    Returns
    -------
    astropy.units.UnitBase
        The unit.

    Raises
    ------
    ValueError
        If the text is not a unit, or every reading of it is a unit that
        no data are in (see _MISREAD_UNITS). For example, in
        'erg/s/cm2/A', A is the ampere: the angstrom is written Angstrom.
    """
    refused = None
    for candidate in dict.fromkeys((text, _standardised(text))):
        unit = _reading(candidate)
        if unit is None:
            continue
        if _MISREAD_UNITS.isdisjoint(unit.decompose().bases):
            return unit
        refused = unit
    if refused is not None:
        raise ValueError(
            f"{text!r} reads as {refused.to_string()!r}, which no data are "
            f"in (a unit of the ampere or of the bit: if A is meant as the "
            f"angstrom, write Angstrom; if B as the beam, beam)")
    raise ValueError(
        f"{text!r} is not a unit (e.g. 'Jy/beam', 'K', 'MJy/sr', "
        f"'1e-17 erg/s/cm2/arcsec2')")


def format_unit(unit: astropy.units.UnitBase) -> str:
    """
    Return a unit as text that parse_unit reads back as the same unit: its
    scale, then each of its units with its power (e.g. 'Jy km beam-1
    s-1').
    """
    # (repr gives the shortest text of the scale that reads back as the
    # same number)
    terms = [] if unit.scale == 1 else [repr(float(unit.scale))]
    for base, power in zip(unit.bases, unit.powers):
        name = base.to_string()
        terms.append(name if power == 1 else f'{name}{power:g}')
    return ' '.join(terms)


def spatial_unit(unit: astropy.units.UnitBase) -> astropy.units.UnitBase:
    """
    Return what a unit of light is per, on the sky:

    - a beam (beam), e.g. Jy/beam;
    - a solid angle (sr), e.g. MJy/sr or erg/s/cm2/arcsec2;
    - a pixel, if the unit names it (pix), e.g. Jy/pixel;
    - nothing else (dimensionless): the light is that of a whole pixel or
      region, e.g. 1e-20 erg/s/cm2/Angstrom of a spaxel.

    Raises
    ------
    ValueError
        If the unit is per an angle that is not a solid angle (e.g.
        Jy/arcsec).
    """
    if astropy.units.beam in unit.bases:
        return astropy.units.beam
    if astropy.units.pix in unit.bases:
        return astropy.units.pix
    # The sum of the powers of the angles of the unit, a solid angle (e.g.
    # sr) counting as an angle squared: -2 for a unit per solid angle (per
    # sr, or per arcsec2)
    power = 0
    for base, base_power in zip(unit.bases, unit.powers):
        if base.physical_type == 'solid angle':
            power += 2 * base_power
        elif base.physical_type == 'angle':
            power += base_power
    if power == -2:
        return astropy.units.sr
    if power != 0:
        raise ValueError(
            f"{format_unit(unit)!r} is per an angle that is not a solid "
            f"angle (e.g. per arcsec, not per arcsec2)")
    return astropy.units.dimensionless_unscaled


def _reading(text: str) -> astropy.units.UnitBase | None:
    """
    Return the unit of a text read as terms that multiply each other (see
    _product), or else as a whole by astropy, or None if neither reads it.
    """
    for parse in (_product, astropy.units.Unit):
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def _product(text: str) -> astropy.units.UnitBase:
    """
    Return the unit of a text as terms that multiply each other (see
    parse_unit). Raise ValueError if a term is not a unit.
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
    Return the unit of a term of parts separated by '/', each dividing all
    before it (e.g. 'erg/s/cm2' is erg / (s cm2)). Raise ValueError if a
    part is empty or not a unit.

    The parts are read one at a time because astropy warns of a text with
    several '/' (the FITS standard discourages them), and silencing that
    warning here would silence it in every other thread too.
    """
    first, *divisors = term.split('/')
    if not first and not divisors or '' in divisors:
        raise ValueError(f"{term!r} has an empty part")
    unit = astropy.units.Unit(first) if first else (
        astropy.units.dimensionless_unscaled)
    for divisor in divisors:
        unit = unit / astropy.units.Unit(divisor)
    return unit


def _standardised(text: str) -> str:
    """
    Return the text of a unit with its invalid names, in any case, made
    valid (see _STANDARD_NAMES; e.g. 'JY/BEAM' as 'Jy/beam'); other names
    are left as they are.
    """
    return re.sub(
        r'[A-Za-z]+',
        lambda match: _STANDARD_NAMES.get(
            match.group(0).upper(), match.group(0)),
        text)
