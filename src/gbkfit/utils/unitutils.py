"""
Physical units of data and models: reading them from the text of
configurations and of FITS headers (BUNIT), and writing them back as text.
"""

import re
from fractions import Fraction

import astropy.units
import pyparsing as pp


__all__ = [
    'format_unit',
    'parse_unit',
    'spatial_unit'
]


# Regular expressions for the parts of the text of a unit (see _GRAMMAR):
# - a name, of any characters but digits, signs, spaces, operators and
#   brackets, as in 'km', 'M_sun' or '%';
# - a number, as in '2', '1.5' or '1e-17';
# - a power, after a name, a number or brackets: a number, alone or after
#   '**' or '^', in brackets or not, as in 'cm2', 's-1', 'm**2',
#   '10**(-20)' or 'Hz(1/2)'.
_NAME = r'[^\s*./()^+\-0-9]+'
_NUMBER = r'[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
_EXPONENT = (
    r'[+-]?[0-9]+(?:\.[0-9]+)?|\([+-]?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?\)')
_POWER = rf'\s*(?:\*\*|\^)\s*(?:{_EXPONENT})|(?:{_EXPONENT})'

# A map from invalid unit names to the valid names of the FITS standard.
# astropy reads only valid names, which are case-sensitive: to it, 'JY'
# is not a unit, 'S' is the siemens and 'B' is the byte.
# A name is looked up in any case ('Jy/Beam' has 'BEAM'), and replaced
# only if astropy does not read it as a unit that makes sense for data
# (see _NON_DATA_BASES): 'MJY' becomes MJy, but 'mJy' stays.
# Invalid names come from software that ignores the standard:
# - some radio software, in capitals: 'JY/BEAM';
# - AIPS, with names of its own: 'JY/B*M/S', 'METR/SEC';
# - SDSS (MaNGA): 'ang' for the angstrom, 'spaxel' for a pixel;
# - some optical spectra: 'A' for the angstrom, as in 'erg/s/cm2/A';
# - the pipelines of HST: 'ELECTRONS/S', 'COUNTS/S';
# - some CCD software: 'ADU'.
_NAME_CORRECTIONS = {
    'JY': 'Jy', 'MJY': 'MJy', 'BEAM': 'beam', 'B': 'beam', 'KM': 'km',
    'M': 'm', 'METR': 'm', 'S': 's', 'SEC': 's', 'HZ': 'Hz', 'KHZ': 'kHz',
    'MHZ': 'MHz', 'GHZ': 'GHz', 'SR': 'sr', 'ERG': 'erg', 'CM': 'cm',
    'ANGSTROM': 'Angstrom', 'ANG': 'Angstrom', 'A': 'Angstrom',
    'ARCSEC': 'arcsec', 'DEG': 'deg', 'PIXEL': 'pix', 'PIX': 'pix',
    'SPAXEL': 'pix',
    'ELECTRON': 'electron', 'ELECTRONS': 'electron', 'COUNT': 'ct',
    'COUNTS': 'ct', 'ADU': 'adu'}

# The base units that no astronomical data are made of: the ampere, the
# mole, the candela and the bit.
# A unit made of one of them makes no sense for data. So if astropy reads
# a name as such a unit, the name means another unit, which
# _NAME_CORRECTIONS gives:
# - 'A' is the ampere to astropy, but means the angstrom;
# - 'S' is the siemens (made of the ampere), but means the second;
# - 'B' is the byte (made of bits), but means the beam.
# A text that is still made of one of them after the corrections is
# refused, e.g. 'G' (the gauss, made of the ampere).
_NON_DATA_BASES = frozenset({
    astropy.units.A, astropy.units.mol, astropy.units.cd, astropy.units.bit})


def parse_unit(text: str) -> astropy.units.UnitBase:
    """
    Return the unit that a text states, e.g. 'Jy/beam', 'MJy/sr' or
    '1e-17 erg/s/cm2/arcsec2'.

    First, the invalid names of the text are made valid: those of
    _NAME_CORRECTIONS, in any case, that astropy does not read as a unit
    that makes sense for astronomical data (see _NON_DATA_BASES). For
    example, 'JY/BEAM' becomes 'Jy/beam' and 'erg/s/cm2/A' becomes
    'erg/s/cm2/Angstrom', but 'mJy' stays (the table would make it 'MJy').

    Then the text is read by the precedence of the FITS standard
    (4.0, 4.3.1): from the left, each '/' divides by what follows it, and
    each space, '*' or '.' multiplies by it; brackets are read first, at
    any depth (see _GRAMMAR). For example, 'Jy/beam km/s' is
    (Jy/beam) (km/s), as maps of moment 0 mean it; astropy alone reads it
    as Jy / (beam km s).

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
        If the text is not a unit, or is one that makes no sense for
        astronomical data (see _NON_DATA_BASES; e.g. 'G', the gauss).
    """
    unit = _unit_of(_corrected(text))
    if unit is None:
        raise ValueError(
            f"{text!r} is not a unit (e.g. 'Jy/beam', 'K', 'MJy/sr', "
            f"'1e-17 erg/s/cm2/arcsec2')")
    if not _makes_sense(unit):
        raise ValueError(
            f"{text!r} reads as {unit.to_string()!r}, which no data are in "
            f"(a unit of the ampere, the mole, the candela or the bit)")
    return unit


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


def _unit_of(text: str) -> astropy.units.UnitBase | None:
    """
    Return the unit of a text (see _GRAMMAR), or None if it is not a unit.
    An empty text is dimensionless.
    """
    if not text.strip():
        return astropy.units.dimensionless_unscaled
    try:
        return _GRAMMAR.parse_string(text, parse_all=True)[0]
    except (pp.ParseBaseException, ValueError):
        return None


def _power(text: str) -> Fraction:
    """
    Return the number of a power (see _POWER), e.g. 2 of '**2' and 1/2 of
    '^(1/2)'. Raise ValueError if it divides by 0.
    """
    number = text.strip().removeprefix('**').removeprefix('^').strip()
    try:
        return Fraction(number.removeprefix('(').removesuffix(')'))
    except ZeroDivisionError:
        raise ValueError(f"{text!r} divides by 0") from None


def _product(tokens: pp.ParseResults) -> astropy.units.UnitBase:
    """
    Return the unit of the operands of an expression, each after its
    operator: '/' divides by it, and anything else multiplies by it.
    """
    items = tokens.as_list()
    unit = astropy.units.dimensionless_unscaled
    for operator, operand in zip(items[::2], items[1::2]):
        unit = unit / operand if operator == '/' else unit * operand
    return unit


def _grammar() -> pp.ParserElement:
    """
    Return the grammar of the text of a unit, by the precedence of the FITS
    standard (4.0, 4.3.1): from the left, a '/' divides by the operand after
    it, and a space, '*' or '.' multiplies by it.
    For example, 'Jy/beam km/s' is Jy km / (beam s).

    An operand is one of:
    - a name or a number, with its power, if any, which astropy reads (e.g.
      'km', 'cm2', 's-1', '1e-17' or '10**(-20)');
    - an expression in brackets, at any depth, with its power, if any (e.g.
      '(km/s)' or '(km/s)2');
    - the square root of an expression in brackets (e.g. 'sqrt(Hz)').

    A text may start with a '/': '/s' is 1/s.

    astropy never reads a '/' between operands, so it never warns of a
    text with several '/' (the FITS standard discourages them); silencing
    that warning here would silence it in every other thread too.
    """
    expression = pp.Forward()
    # (a power after brackets, with no space before it unless it has '**' or
    # '^': '(km/s) 2' is 2 km/s)
    power = pp.Regex(_POWER).leave_whitespace()
    atom = pp.Regex(rf'(?:{_NAME}|{_NUMBER})(?:{_POWER})?')
    atom.set_parse_action(lambda tokens: astropy.units.Unit(tokens[0]))
    group = (
        pp.Suppress('(') + expression + pp.Suppress(')')
        + pp.Opt(power, default='1'))
    group.set_parse_action(lambda tokens: tokens[0] ** _power(tokens[1]))
    root = (
        pp.Suppress(pp.Keyword('sqrt') + '(') + expression
        + pp.Suppress(')'))
    root.set_parse_action(lambda tokens: tokens[0] ** Fraction(1, 2))
    operand = root | group | atom
    first = pp.Opt('/', default='') + operand
    rest = pp.Opt(pp.one_of('* . /'), default='') + operand
    expression <<= first + rest[...]
    expression.set_parse_action(_product)
    return expression


_GRAMMAR = _grammar()


def _corrected(text: str) -> str:
    """
    Return the text of a unit with its invalid names corrected. A name is
    a run of letters and underscores, as 'M_sun'; digits are not part of
    it, so 'cm2' is the name 'cm' squared.
    """
    def corrected_name(match: re.Match) -> str:
        name = match.group(0)
        correction = _NAME_CORRECTIONS.get(name.upper())
        if correction is None:
            return name
        unit = _unit_of(name)
        if unit is not None and _makes_sense(unit):
            return name
        return correction
    return re.sub(r'[A-Za-z_]+', corrected_name, text)


def _makes_sense(unit: astropy.units.UnitBase) -> bool:
    """
    Return whether a unit makes sense for astronomical data (see
    _NON_DATA_BASES).
    """
    return _NON_DATA_BASES.isdisjoint(unit.decompose().bases)
