"""
Physical units: reading them from text and writing them as text, and
telling what a unit of light is per on the sky. The text of a unit is
in VOUnits, the recommendation of the IVOA, as in configurations and
outputs, or in the convention of the FITS standard, as in the headers
of FITS files.
"""

import re
from fractions import Fraction

import astropy.units


__all__ = [
    'format_unit',
    'parse_quantity',
    'parse_unit',
    'spatial_unit'
]


# The conventions of units as text: VOUnits, the recommendation of the
# IVOA (1.0), and that of the FITS standard (4.0, section 4.3)
_CONVENTIONS = ('vounits', 'fits')

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


def parse_unit(
        text: str, convention: str = 'vounits'
) -> astropy.units.UnitBase:
    """
    Return the unit that a text states, in a convention: 'vounits' (by
    default), as in configurations, or 'fits', as in the headers of FITS
    files.

    In 'vounits', astropy reads the text (its format 'vounit'). It has
    at most one '/': 'erg/s/cm2/Angstrom' is not a unit, but
    'erg.s-1.cm-2.Angstrom-1' and 'erg/(s.cm2.Angstrom)' are; and
    percent is the scale 0.01 ('1e-2').

    In 'fits', the text is read by the precedence of the FITS standard
    (4.0, 4.3.1): from the left, each '/' divides by what follows it,
    and each space, '*' or '.' multiplies by it; brackets are read
    first, at any depth (see _expression). For example, 'Jy/beam km/s'
    is (Jy/beam) (km/s), as maps of moment 0 mean it; astropy alone
    reads it as Jy / (beam km s). The names of units are those that
    astropy reads, which include those of the FITS standard.

    In both, names are case-sensitive, and nothing is corrected:
    'JY/BEAM' is not a unit, and 'S' is the siemens. An empty text is
    dimensionless.

    Parameters
    ----------
    text : str
        The text of the unit.
    convention : str, optional
        The convention of the text.

    Returns
    -------
    astropy.units.UnitBase
        The unit.

    Raises
    ------
    ValueError
        If the text is not a unit in the convention.
    """
    _check_convention(convention)
    if convention == 'fits':
        unit = _unit_of(text)
        if unit is None:
            raise ValueError(
                f"{text!r} is not a valid unit (see the FITS standard, "
                f"section 4.3)")
        return unit
    try:
        return astropy.units.Unit(text, format='vounit')
    except ValueError:
        raise ValueError(
            f"{text!r} is not a valid unit of VOUnits, the recommendation "
            f"of the IVOA") from None


def parse_quantity(text: str) -> astropy.units.Quantity:
    """
    Return the quantity that a text states: a number, a space, and a
    unit of VOUnits (see parse_unit).
    For example, '6562.8 Angstrom'.

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


def format_unit(
        unit: astropy.units.UnitBase, convention: str = 'vounits'
) -> str:
    """
    Return a unit as text in a convention: 'vounits' or 'fits', which
    parse_unit reads back as the same unit.
    For example, 'Jy.km.beam**-1.s**-1' in 'vounits', and
    'Jy km beam-1 s-1' in 'fits'.

    The text in 'vounits' reads back as the same unit in 'fits' too. A
    unit that VOUnits has no name for (e.g. the electron) is written in
    'fits', which only that convention reads.
    """
    _check_convention(convention)
    if convention == 'vounits':
        if unit.is_equivalent(astropy.units.dimensionless_unscaled):
            # (a scale, e.g. 0.01 for percent: astropy writes none)
            scale = unit.to(astropy.units.dimensionless_unscaled)
            return '' if scale == 1 else repr(float(scale))
        try:
            # (VOUnits deprecates the units of CGS, e.g. erg and
            # Angstrom, which astropy would warn of, or convert to SI)
            return unit.to_string('vounit', deprecations='silent')
        except ValueError:
            pass
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


def spatial_unit(unit: astropy.units.UnitBase) -> astropy.units.UnitBase:
    """
    Return what a unit of light is per, on the sky:

    - a beam (beam), e.g. Jy/beam;
    - a solid angle (sr), e.g. MJy/sr or erg/s/cm2/arcsec2;
    - a pixel, if the unit names it (pix), e.g. Jy/pixel;
    - nothing else (dimensionless): the light is that of a whole pixel
      or region, e.g. 1e-20 erg/s/cm2/Angstrom of a spaxel.

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
    # The sum of the powers of the angles of the unit, a solid angle
    # (e.g. sr) counting as an angle squared: -2 for a unit per solid
    # angle (per sr, or per arcsec2)
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


def _check_convention(convention: str) -> None:
    """Raise ValueError unless a convention is 'vounits' or 'fits'."""
    if convention not in _CONVENTIONS:
        raise ValueError(
            f"the convention of units must be one of {list(_CONVENTIONS)}; "
            f"it is {convention!r}")
