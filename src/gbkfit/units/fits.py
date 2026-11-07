"""
Units as text in the convention of the FITS standard (4.0, section
4.3), as in the headers of FITS files (BUNIT, CUNITn and TUNITn):
reading them by the precedence of the standard, and writing them.
"""

import re
from fractions import Fraction

import astropy.units


__all__ = [
    'format_unit',
    'parse_unit'
]


# A regular expression for the separators between the terms of a unit,
# in the convention of FITS.
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


def parse_unit(text: str) -> astropy.units.UnitBase:
    """
    Return the unit that a text of the FITS standard states.

    The text is read by the precedence of the standard (4.0, 4.3.1):
    from the left, each '/' divides by what follows it, and each space,
    '*' or '.' multiplies by it; brackets are read first, at any depth
    (see _expression). For example, 'Jy/beam km/s' is (Jy/beam) (km/s),
    as maps of moment 0 mean it; astropy alone reads it as
    Jy / (beam km s). The names of units are those that astropy reads,
    which include those of the standard.

    Names are case-sensitive, and nothing is corrected: 'JY/BEAM' is not
    a unit, and 'S' is the siemens. An empty text is dimensionless.

    Raises
    ------
    ValueError
        If the text is not a unit.
    """
    try:
        return _expression(text)
    except ValueError:
        raise ValueError(
            f"{text!r} is not a valid unit (see the FITS standard, section "
            f"4.3)") from None


def format_unit(unit: astropy.units.UnitBase) -> str:
    """
    Return a unit as text of the FITS standard, which parse_unit reads
    back as the same unit.
    For example, 'Jy km beam-1 s-1'.
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
            # The FITS standard allows a power that is not an integer
            # only in brackets: 'Hz(0.5)', not 'Hz0.5'
            terms.append(f'{name}({power})')
    return ' '.join(terms)


def _expression(text: str) -> astropy.units.UnitBase:
    """
    Return the unit of a text of operands and operators, read from the
    left by the precedence of the FITS standard (4.0, 4.3.1): a '/'
    divides by the operand after it, and a separator of terms
    (see _TERM_SEPARATOR) multiplies by it.
    For example, 'Jy/beam km/s' is Jy km / (beam s).

    An operand is either an expression in brackets (see _bracketed), or
    else a unit that astropy reads, e.g. 'km', 'cm2' or '10**(-17)'.
    astropy never reads a '/' between operands, so it never warns of a
    text with several '/' (the FITS standard discourages them);
    silencing that warning here would silence it in every other thread
    too.

    An empty text is dimensionless, and so is the operand before a '/'
    that starts the text: '/s' is 1/s.

    Raise ValueError if an operand is not a unit or is missing
    (e.g. 'Jy/' or 'Jy..km'), or if the brackets do not match.
    """
    # Remove the spaces around each '/', '*' and '.', and at the ends of
    # the text, as they do not separate terms.
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
