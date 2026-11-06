"""
Physical units of data and models: reading them from the text of
configurations and of FITS headers (BUNIT), and writing them back as text.
"""

import re
from fractions import Fraction

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
_TERM_SEPARATOR = re.compile(r'\s+|(?<!\*)\*(?!\*)|(?<![0-9])\.|\.(?![0-9])')

# A regular expression for what splits the text of a unit into operands
# (see _split): a separator of terms, a '/' or a bracket.
_SPLITTER = re.compile(rf'{_TERM_SEPARATOR.pattern}|[/()]')

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
    any depth (see _expression). For example, 'Jy/beam km/s' is
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
    scale, then each of its units with its power.
    For example, 'Jy km beam-1 s-1', or 'Hz(0.5)' for a power that is not
    whole.
    """
    # (repr gives the shortest text of the scale that reads back as the
    # same number)
    terms = [] if unit.scale == 1 else [repr(float(unit.scale))]
    for base, power in zip(unit.bases, unit.powers):
        name = base.to_string()
        if power == 1:
            terms.append(name)
        elif power == int(power):
            terms.append(f'{name}{int(power)}')
        else:
            # The FITS standard allows a power that is not whole only in
            # brackets: 'Hz(0.5)', not 'Hz0.5'
            terms.append(f'{name}({power})')
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
    Return the unit of a text (see _expression), or None if it is not a
    unit.
    """
    try:
        return _expression(text)
    except ValueError:
        return None


def _expression(text: str) -> astropy.units.UnitBase:
    """
    Return the unit of a text of operands and operators, read from the left
    by the precedence of the FITS standard (4.0, 4.3.1): a '/' divides by
    the operand after it, and a separator of terms (see _TERM_SEPARATOR)
    multiplies by it.
    For example, 'Jy/beam km/s' is Jy km / (beam s).

    An operand is either an expression in brackets (see _bracketed), or
    else a unit that astropy reads, e.g. 'km', 'cm2' or '10**(-17)'.
    astropy never reads a '/' between operands, so it never warns of a text
    with several '/' (the FITS standard discourages them); silencing that
    warning here would silence it in every other thread too.

    An empty text is dimensionless, and so is the operand before a '/' that
    starts the text: '/s' is 1/s.

    Raise ValueError if an operand is not a unit or is missing (e.g. 'Jy/'
    or 'Jy..km'), or if the brackets do not match.
    """
    # Remove the spaces around each '/', '*' and '.', and at the ends of the
    # text, as they do not separate terms.
    # For example, 'Jy/beam km / s' becomes 'Jy/beam km/s'.
    for operator in '/*.':
        text = operator.join(part.strip() for part in text.split(operator))
    unit = astropy.units.dimensionless_unscaled
    for index, (operator, operand) in enumerate(_split(text)):
        if not operand:
            if text and not (index == 0 and text.startswith('/')):
                raise ValueError(
                    f"{text!r} has an operator without an operand")
            continue
        if operand.startswith('('):
            value = _bracketed(operand)
        else:
            value = astropy.units.Unit(operand)
        unit = unit / value if operator == '/' else unit * value
    return unit


def _split(text: str) -> list[tuple[str, str]]:
    """
    Return the operands of a text, split at the operators outside brackets,
    each with the operator before it: '/', or '*' for a separator of terms
    and for the first operand.
    For example, 'Jy/(beam pix) km' gives
    [('*', 'Jy'), ('/', '(beam pix)'), ('*', 'km')].

    Raise ValueError if the brackets do not match.
    """
    operands = []
    operator, start, depth = '*', 0, 0
    for match in _SPLITTER.finditer(text):
        token = match.group()
        if token == '(':
            depth += 1
        elif token == ')':
            depth -= 1
            if depth < 0:
                raise ValueError(f"{text!r} has a ')' without a '('")
        elif depth == 0:
            operands.append((operator, text[start:match.start()]))
            operator = '/' if token == '/' else '*'
            start = match.end()
    if depth > 0:
        raise ValueError(f"{text!r} has a '(' without a ')'")
    operands.append((operator, text[start:]))
    return operands


def _bracketed(operand: str) -> astropy.units.UnitBase:
    """
    Return the unit of an operand that starts with a bracket: the
    expression in the brackets (see _expression), to the power after them,
    if any.
    For example, '(km/s)', '(km/s)2', '(km/s)**2' or '(km/s)^(-1/2)'.

    Raise ValueError if the expression is empty or not a unit, or if the
    power is not a number.
    """
    # The index of the bracket that closes the first one (the brackets of
    # an operand match, see _split)
    depth = 0
    for close, char in enumerate(operand):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth == 0:
                break
    inner, power = operand[1:close], operand[close + 1:]
    if not inner.strip():
        raise ValueError(f"{operand!r} has nothing in its brackets")
    unit = _expression(inner)
    if not power:
        return unit
    # The power may follow '**' or '^', or nothing, as in '(km/s)2'
    if power.startswith('**'):
        power = power[2:]
    elif power.startswith('^'):
        power = power[1:]
    return unit ** _power(power)


def _power(text: str) -> Fraction:
    """
    Return the number of a power, in brackets or braces or not.
    For example, '2', '(1/2)' or '{-2}' (as MaNGA writes it).

    Raise ValueError if it is not a number.
    """
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    if text.startswith('{') and text.endswith('}'):
        text = text[1:-1]
    try:
        return Fraction(text)
    except ZeroDivisionError:
        raise ValueError(f"{text!r} divides by 0") from None


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
    Return whether a unit makes sense for astronomical data.
    """
    return _NON_DATA_BASES.isdisjoint(unit.decompose().bases)
