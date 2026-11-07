"""
Physical units: reading them from text and writing them as text, and
telling what a unit of light is per on the sky.

All text is read and written by the rules of the FITS standard (4.0,
section 4.3): that of the headers of FITS files, of configurations and
of outputs. The names of units are those that astropy reads, which
include those of the standard. Nothing is corrected.
"""

import re
from fractions import Fraction

import astropy.units


__all__ = [
    'format_quantity',
    'format_unit',
    'parse_quantity',
    'parse_unit',
    'spatial_unit'
]


# A regular expression for the separators between the terms of a unit.
# Terms multiply each other: 'Jy/beam km/s' is 'Jy/beam' times 'km/s'.
# A separator is one of:
# - a space, as in 'Jy/beam km/s';
# - a '*', as in 'Jy/beam*km/s' ('**' is a power, as in 'm**2');
# - a '.', as in 'Jy/beam.km/s', but not one between digits ('1.5').
_TERM_SEPARATOR = re.compile(r'\s+|(?<!\*)\*(?!\*)|(?<![0-9])\.|\.(?![0-9])')

# A regular expression for what splits the text of a unit into operands
# (see _split): a separator of terms, a '/' or a bracket.
_SPLITTER = re.compile(rf'{_TERM_SEPARATOR.pattern}|[/()]')


def parse_unit(text: str) -> astropy.units.UnitBase:
    """
    Return the unit that a text states.

    The text is read by the rules of the FITS standard (4.0, section
    4.3.1): from the left, each '/' divides by what follows it, and each
    space, '*' or '.' multiplies by it; brackets are read first, at any
    depth. For example, 'Jy/beam km/s' is (Jy/beam) (km/s); astropy
    alone reads it as Jy / (beam km s). The names of units are those
    that astropy reads, which include those of the standard.

    Names are case-sensitive, and nothing is corrected: 'JY/BEAM' is not
    a unit, and 'S' is the siemens. An empty text is dimensionless.

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
        If the text is not a unit.
    """
    try:
        return _expression(text)
    except (ValueError, astropy.units.UnitsError):
        raise ValueError(
            f"{text!r} is not a valid unit "
            f"(see the FITS standard, section 4.3)") from None


def parse_quantity(text: str) -> astropy.units.Quantity:
    """
    Return the quantity that a text states: a number and a unit.
    For example, '6562.8 Angstrom'.

    Parameters
    ----------
    text : str
        The text of the quantity.

    Returns
    -------
    astropy.units.Quantity
        The quantity.

    Raises
    ------
    ValueError
        If the text is not a number and a unit.
    """
    number, _, unit = text.strip().partition(' ')
    try:
        value = float(number)
    except ValueError:
        raise ValueError(
            f"{text!r} is not a number and a unit (e.g. "
            f"'6562.8 Angstrom')") from None
    return value * parse_unit(unit.strip())


def format_unit(unit: astropy.units.UnitBase) -> str:
    """
    Return a unit as text.

    Each term is written with its power, and there is no '/', which
    astropy reads by its own precedence: both parse_unit and astropy
    read the text back as the same unit.
    For example, 'Jy km beam-1 s-1'.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The unit.

    Returns
    -------
    str
        The text of the unit.
    """
    if isinstance(unit, astropy.units.FunctionUnitBase):
        # A unit of a function, e.g. a logarithm, is always the whole
        # unit (astropy combines it with no other): astropy writes it,
        # and parse_unit gives it to astropy to read
        return unit.to_string()
    # repr: the shortest text that reads back as the same scale
    terms = [] if unit.scale == 1 else [repr(float(unit.scale))]
    for base, power in zip(unit.bases, unit.powers):
        name = base.to_string()
        if power == 1:
            terms.append(name)
        elif power == int(power):
            terms.append(f'{name}{int(power)}')
        else:
            # The FITS standard allows a power that is not an integer
            # only in brackets: 'Hz(0.5)', not 'Hz0.5'
            terms.append(f'{name}({power})')
    return ' '.join(terms)


def format_quantity(quantity: astropy.units.Quantity) -> str:
    """
    Return a quantity as text: its number, a space and its unit, as
    format_unit writes it, which parse_quantity reads back as the same
    quantity.
    For example, '6562.8 Angstrom'.

    Parameters
    ----------
    quantity : astropy.units.Quantity
        The quantity, of a single number.

    Returns
    -------
    str
        The text of the quantity.
    """
    return f'{float(quantity.value)!r} {format_unit(quantity.unit)}'.rstrip()


def spatial_unit(unit: astropy.units.UnitBase) -> astropy.units.UnitBase:
    """
    Return what a unit of light is per, on the sky:

    - beam, for a unit per beam, e.g. Jy/beam;
    - sr, for a unit per solid angle, e.g. MJy/sr or erg/s/cm2/arcsec2;
    - pix, for a unit per pixel, e.g. Jy/pixel;
    - dimensionless, for a unit per nothing, e.g. erg/s/cm2/Angstrom.

    Light per nothing is that of a whole pixel or region, e.g. a spaxel.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The unit of the light.

    Returns
    -------
    astropy.units.UnitBase
        What it is per: beam, pix, sr or dimensionless_unscaled.

    Raises
    ------
    ValueError
        If the unit is none of these (e.g. Jy/arcsec or Jy/beam/pix).
    """
    # The powers of the beams, pixels and angles of the unit, a solid
    # angle (e.g. sr) counting as an angle squared
    beams = pixels = angles = 0
    for base, power in zip(unit.bases, unit.powers):
        if base == astropy.units.beam:
            beams += power
        elif base == astropy.units.pix:
            pixels += power
        elif base.physical_type == 'solid angle':
            angles += 2 * power
        elif base.physical_type == 'angle':
            angles += power
    # What a unit is per, by those powers: one of each, or none
    per = {
        (-1, 0, 0): astropy.units.beam,
        (0, 0, -2): astropy.units.sr,
        (0, -1, 0): astropy.units.pix,
        (0, 0, 0): astropy.units.dimensionless_unscaled}
    try:
        return per[beams, pixels, angles]
    except KeyError:
        raise ValueError(
            f"{format_unit(unit)!r} is not light per beam, per solid "
            f"angle, per pixel or per nothing") from None


def _expression(text: str) -> astropy.units.UnitBase:
    """
    Return the unit of a text of operands and operators, read from the
    left by the precedence of the FITS standard (4.0, section 4.3.1):
    a '/' divides by the operand after it, and a separator of terms
    (see _TERM_SEPARATOR) multiplies by it.
    For example, 'Jy/beam km/s' is Jy km / (beam s).

    An operand is either an expression in brackets (see _bracketed), or
    else a unit that astropy reads, e.g. 'km', 'cm2' or '10**(-17)'.
    astropy only ever reads operands, never a '/', so it never warns of
    a text with several '/', which the FITS standard allows.

    An empty text is dimensionless, and so is the operand before a '/'
    that starts the text: '/s' is 1/s.

    Raise ValueError if an operand is not a unit or is missing
    (e.g. 'Jy/' or 'Jy..km'), or if the brackets do not match, and
    astropy.units.UnitsError if astropy cannot combine the operands,
    e.g. a logarithm with any other unit.
    """
    # Remove the spaces around each '/', '*' and '.', and at the ends of
    # the text, as they do not separate terms.
    # For example, 'Jy/beam km / s' becomes 'Jy/beam km/s'.
    for operator in '/*.':
        text = operator.join(part.strip() for part in text.split(operator))
    unit = None
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
        # The first operand starts the unit: astropy multiplies no unit
        # of a function (e.g. a logarithm), even by dimensionless
        if unit is None:
            unit = value if operator == '*' else value ** -1
        elif operator == '/':
            unit = unit / value
        else:
            unit = unit * value
    return astropy.units.dimensionless_unscaled if unit is None else unit


def _split(text: str) -> list[tuple[str, str]]:
    """
    Return the operands of a text, split at the operators outside
    brackets, each with the operator before it: '/', or '*' for a
    separator of terms and for the first operand.
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
    expression in the brackets (see _expression), to the power after
    them, if any.
    For example, '(km/s)', '(km/s)2', '(km/s)**2' or '(km/s)^(-1/2)'.

    Such an operand is read here, not by astropy as any other operand
    is, because the brackets hold a whole expression, which astropy
    would read by its own precedence: '(Jy/beam km/s)' as
    Jy / (beam km s).

    Raise ValueError if the expression is empty or not a unit, or if
    the power is not a number.
    """
    # The index of the bracket that closes the first one (the brackets
    # of an operand match, see _split)
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
    Return the number of a power, in brackets or not.
    For example, '2' or '(1/2)'.

    Raise ValueError if it is not a number.
    """
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    try:
        return Fraction(text)
    except ZeroDivisionError:
        raise ValueError(f"{text!r} divides by 0") from None
