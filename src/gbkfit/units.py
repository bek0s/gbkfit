"""
Physical units: reading, writing and interpreting them.

The text of a unit, in FITS files, configurations and outputs alike, is
read by the rules of the FITS standard (4.0, section 4.3) for its
operators and brackets, in the arguments of functions too, and by
astropy for what they join: names of units and of functions, numbers
and powers. Nothing is corrected.

The operators are not left to astropy: its reader gives the products
after a '/' to the divisor, against the left-to-right rule of the
standard, so it misreads valid units, e.g. 'Jy/beam km/s' as
Jy / (beam km s) and 'Jy/beam*Hz' as Jy / (beam Hz), and it refuses
brackets that the standard recommends, as in '(Jy/beam) (km/s)'. Its
names are kept: those of the standard and more (e.g. 'Msun' or '%').

Units are written with negative powers and no '/' (see format_unit),
which both readers read alike. A text that is not a unit by these rules
(e.g. 'JY/BEAM') is refused, not guessed at: the unit of such data is
given instead, by the option unit.
"""

import re
from fractions import Fraction

import astropy.units


__all__ = [
    'AnyUnit',
    'format_quantity',
    'format_unit',
    'parse_quantity',
    'parse_unit',
    'spatial_unit'
]


# A unit of either kind that astropy has: a UnitBase, or a
# FunctionUnitBase for a unit of a function (e.g. mag(AB)), which is not
# a UnitBase
AnyUnit = astropy.units.UnitBase | astropy.units.FunctionUnitBase

# A regular expression for the tokens that _split looks for in the text
# of a unit: the operators, with any spaces around them, where it splits
# the text into operands, and the brackets and a number with a decimal
# point at the start of the text, where it does not
_TOKEN = re.compile(r"""
    (?P<number>\A[+-]?              # a number with a decimal point at
      (?:[0-9]+\.[0-9]*|\.[0-9]+))  #   the start, e.g. '1.E-17' or '.5'
    | \s*/\s*                       # a '/', which divides
    | \s*(?<!\*)\*(?!\*)\s*         # a '*', not in '**' (a power)
    | \s*(?<![0-9*^+-])\.\s*        # a '.', unless after a digit, '*',
    | \s*\.(?![0-9])\s*             #   '^' or sign, before a digit
    | \s+                           # spaces, which multiply
    | [()]                          # a bracket, which keeps what it
                                    #   holds in one operand
    """, re.VERBOSE)

# A regular expression for a power after brackets, as the FITS standard
# allows it (4.0, section 4.3.1): after an optional '**' or '^', an
# integer, with or without sign and brackets, or a decimal or a ratio of
# integers, with or without sign, in brackets; e.g. '2', '**-2',
# '^(+2)', '(1.5)' or '(-3/2)'
_POWER = re.compile(
    r'(?:\*\*|\^)?'
    r'(?P<number>[+-]?[0-9]+'
    r'|\([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+|[0-9]+/[0-9]+)\))')

# A regular expression for an operand that is a name and brackets: a
# function and its argument, e.g. 'dex(solMass/pc2)' or 'sqrt(Hz)', or
# a unit and its power, e.g. 'm(3/2)'
_CALL = re.compile(r'(?P<name>[^\W\d]\w*)\((?P<argument>.*)\)')

# What a unit of light is per (see spatial_unit), by the powers of the
# beams, the pixels and the angles of the unit; any other combination of
# them is an error
_PER = {
    (-1, 0, 0): astropy.units.beam,
    (0, 0, -2): astropy.units.sr,
    (0, -1, 0): astropy.units.pix,
    (0, 0, 0): astropy.units.dimensionless_unscaled}


class _FitsSyntaxError(ValueError):
    """
    Raised for the text of a unit that breaks a rule of the FITS
    standard (4.0, section 4.3.1).
    """


def parse_unit(text: str) -> AnyUnit:
    """
    Return the unit that a text states.

    Its operators and brackets are read by the rules of the FITS
    standard (4.0, section 4.3.1): from the left, each '/' divides by
    what follows it, and each space, '*' or '.' multiplies by it;
    brackets are read first, at any depth, and may take a power in the
    forms of the standard (e.g. '(km/s)2' or '(km/s)(1.5)'). For
    example, 'Jy/beam km/s' is (Jy/beam) (km/s); astropy alone reads it
    as Jy / (beam km s).

    What they join is read by astropy: names of units (those of the
    standard and more, e.g. 'Msun'), powers (e.g. 'm2' or 'm**(-2)'),
    and a number, which only comes first, as a scale (e.g.
    '1e-17 erg/s'). A function is one of astropy (e.g. 'dex(solMass)')
    or the square root of the standard ('sqrt(Hz)'), and its argument is
    read by the same rules as the whole text: 'mag(Jy/beam km/s)' is
    mag(Jy km / (beam s)). A unit of a function can only be the whole
    unit, with no power.

    Names are case-sensitive, and nothing is corrected: 'JY/BEAM' is not
    a unit, and 'S' is the siemens. An empty text is dimensionless.

    Parameters
    ----------
    text : str
        The text of the unit.

    Returns
    -------
    AnyUnit
        The unit.

    Raises
    ------
    ValueError
        If the text is not a unit; the message says why.
    """
    try:
        return _expression(text)
    except _FitsSyntaxError as e:
        reason = f"{e} (see the FITS standard, section 4.3.1)"
    except (ValueError, astropy.units.UnitsError) as e:
        reason = str(e)
    raise ValueError(f"{text!r} is not a valid unit: {reason}") from None


def parse_quantity(text: str) -> astropy.units.Quantity:
    """
    Return the quantity that a text states: a number and a unit.

    The text is split at its first space, or other whitespace, into a
    number and a unit, which parse_unit reads; a number alone is
    dimensionless. For example, '6562.8 Angstrom' or '0.5'.

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
        If it does not start with a number, or the rest is not a unit.
    """
    # Any whitespace counts as one space
    number, _, unit = ' '.join(text.split()).partition(' ')
    try:
        value = float(number)
    except ValueError:
        raise ValueError(
            f"{text!r} is not a number and a unit "
            f"(e.g. '6562.8 Angstrom')") from None
    return value * parse_unit(unit)


def format_unit(unit: AnyUnit) -> str:
    """
    Return a unit as text.

    The text is the scale of the unit, if any, then each term with its
    power. Division is written as a negative power, not a '/', so that
    parse_unit and astropy, which give '/' different precedence, read
    the text back as the same unit. A unit of a function is written as
    the function and its unit in brackets, and a dimensionless unit
    without a scale as the empty text.
    For example: '1e-17 erg s-1 cm-2', 'Jy km beam-1 s-1', 'dex(1)'.

    Parameters
    ----------
    unit : AnyUnit
        The unit.

    Returns
    -------
    str
        The text of the unit.
    """
    if isinstance(unit, astropy.units.FunctionUnitBase):
        # The name of the function, and its unit in brackets, e.g.
        # 'dex(solMass kpc-2)', or '1' for a dimensionless one, as in
        # 'dex(1)': a bare 'dex' reads back as a plain unit
        name = unit.function_unit.name
        physical = format_unit(unit.physical_unit) or '1'
        return f'{name}({physical})'
    # astropy holds a unit as a scale and bases to powers, e.g.
    # 1e-17 erg / (s cm2) as 1e-17, [erg, s, cm] and [1, -1, -2], which
    # are written in turn: '1e-17 erg s-1 cm-2'. The scale is written by
    # repr, the shortest text that reads back as the same number.
    terms = [] if unit.scale == 1 else [repr(float(unit.scale))]
    for base, power in zip(unit.bases, unit.powers):
        name = base.to_string()
        if power == 1:
            terms.append(name)
        elif power == int(power):
            terms.append(f'{name}{int(power)}')
        else:
            # The FITS standard allows a power that is not an integer
            # only in brackets, written after '**' so that it is not
            # read as the argument of a function: 'Hz**(0.5)'
            terms.append(f'{name}**({power})')
    return ' '.join(terms)


def format_quantity(quantity: astropy.units.Quantity) -> str:
    """
    Return a quantity as text.

    The text is its number and its unit, as format_unit writes it, and
    parse_quantity reads it back as the same quantity.
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
    number = repr(float(quantity.value))
    unit = format_unit(quantity.unit)
    # A dimensionless unit without a scale is written as the empty text
    return f'{number} {unit}' if unit else number


def spatial_unit(unit: AnyUnit) -> astropy.units.UnitBase:
    """
    Return what a unit of light is per, on the sky.

    It is one of:

    - beam, for a unit per beam, e.g. Jy/beam;
    - sr, for a unit per solid angle, e.g. MJy/sr or erg/s/cm2/arcsec2;
    - pix, for a unit per pixel, e.g. Jy/pixel;
    - dimensionless, for a unit per nothing, e.g. erg/s/cm2/Angstrom.

    Light per nothing is that of a whole pixel or region, e.g. a spaxel.

    Parameters
    ----------
    unit : AnyUnit
        The unit of the light.

    Returns
    -------
    astropy.units.UnitBase
        What it is per: beam, pix, sr or dimensionless_unscaled.

    Raises
    ------
    ValueError
        If the unit is none of these (e.g. Jy/arcsec or Jy/beam/pix), or
        is a logarithm (e.g. mag, mag(AB) or mag/arcsec2).
    """
    # The unit by what it is made of: a named unit by what it stands for
    # (e.g. the rayleigh, which is per sr), and a unit of a function by
    # its function (e.g. mag(AB) by mag)
    if isinstance(unit, astropy.units.FunctionUnitBase):
        decomposed = unit.function_unit.decompose()
    else:
        decomposed = unit.decompose()
    # A logarithm is not light per anything: mag, dB and dex all
    # decompose into dex
    if astropy.units.dex in decomposed.bases:
        raise ValueError(
            f"{format_unit(unit)!r} is a logarithm "
            f"(e.g. a magnitude), not light per anything")
    # The powers of the beams, pixels and angles of the unit, a solid
    # angle (e.g. sr) decomposing into an angle squared
    beams = pixels = angles = 0
    for base, power in zip(decomposed.bases, decomposed.powers):
        if base == astropy.units.beam:
            beams += power
        elif base == astropy.units.pix:
            pixels += power
        elif base.physical_type == 'angle':
            angles += power
    try:
        return _PER[beams, pixels, angles]
    except KeyError:
        raise ValueError(
            f"{format_unit(unit)!r} is not light per beam, per solid "
            f"angle, per pixel or per nothing") from None


def _expression(text: str) -> AnyUnit:
    """
    Return the unit of a text, read by the rules of parse_unit.

    Each operand is read on its own: one in brackets by _bracketed, a
    function by _function, and any other by _astropy_unit, so that
    astropy reads no operator, nor warns of several '/'. An operand
    before a '/' that starts the text is dimensionless: '/s' is 1/s.

    Raise ValueError, saying why, if an operand is not a unit or is
    missing (e.g. 'Jy/' or 'Jy..km'), if an operand other than the
    first has a number in it (e.g. 'Jy 30' or 'Jy/beam 2km'), if a unit
    of a function is not the whole unit, or if the brackets do not
    match.
    """
    text = text.strip()
    unit = None
    for index, (operator, operand) in enumerate(_split(text)):
        # An operand is empty only in an empty text, or before a '/'
        # that starts the text ('/s'); anywhere else, it is missing
        if not operand:
            if text and not (index == 0 and text.startswith('/')):
                raise _FitsSyntaxError("an operator has no operand")
            continue
        if operand.startswith('('):
            term = _bracketed(operand)
        else:
            term = _function(operand)
            if term is None:
                term = _astropy_unit(operand)
        if unit is not None and _has_number(operand, term):
            raise _FitsSyntaxError(f"the number in {operand!r} is not first")
        # A unit of a function (e.g. a logarithm) can only be the whole
        # unit: astropy refuses to combine one of a physical unit with
        # another unit, and makes a quantity of a number times one
        if unit is not None or operator == '/':
            if any(isinstance(x, astropy.units.FunctionUnitBase)
                   for x in (unit, term)):
                raise ValueError(
                    "a unit of a function (e.g. a logarithm) can only be "
                    "the whole unit")
        # The first operand starts the unit, rather than multiplying
        # dimensionless, which a unit of a function would refuse
        if unit is None:
            unit = term if operator == '*' else term ** -1
        elif operator == '/':
            unit = unit / term
        else:
            unit = unit * term
    return astropy.units.dimensionless_unscaled if unit is None else unit


def _split(text: str) -> list[tuple[str, str]]:
    """
    Return the operands of a text, split at the operators outside
    brackets, each with the operator before it: '/', or '*' for one
    that multiplies (see _TOKEN) and for the first operand.
    For example, 'Jy/(beam pix) km' gives
    [('*', 'Jy'), ('/', '(beam pix)'), ('*', 'km')].

    Raise ValueError if the brackets do not match.
    """
    operands = []
    operator, start, depth = '*', 0, 0
    for match in _TOKEN.finditer(text):
        token = match.group()
        # The '.' of a number is a decimal point, which does not split
        if match['number']:
            continue
        if token == '(':
            depth += 1
        elif token == ')':
            depth -= 1
            if depth < 0:
                raise _FitsSyntaxError("a ')' has no '('")
        elif depth == 0:
            operands.append((operator, text[start:match.start()]))
            operator = '/' if token.strip() == '/' else '*'
            start = match.end()
    if depth > 0:
        raise _FitsSyntaxError("a '(' has no ')'")
    operands.append((operator, text[start:]))
    return operands


def _bracketed(operand: str) -> AnyUnit:
    """
    Return the unit of an operand that starts with a bracket: the
    expression in the brackets (see _expression), to the power after
    them, if any.
    For example, '(km/s)', '(km/s)2', '(km/s)**2' or '(km/s)^(-1/2)'.

    Such an operand is read here, not by astropy: the brackets hold a
    whole expression, read by the rules of the FITS standard, and
    astropy refuses a '/' in brackets and a power after them, as in
    '(Jy/beam)' or '(km/s)2'.

    Raise ValueError if the expression is empty or not a unit, or if
    the power is not one that the FITS standard allows (see _power) or
    follows a unit of a function.
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
        raise _FitsSyntaxError(f"{operand!r} has nothing in its brackets")
    unit = _expression(inner)
    if not power:
        return unit
    if isinstance(unit, astropy.units.FunctionUnitBase):
        raise ValueError(
            "a unit of a function (e.g. a logarithm) takes no power")
    return unit ** _power(power)


def _function(operand: str) -> AnyUnit | None:
    """
    Return the unit of an operand that is a function and its argument
    in brackets, the argument read by the rules of parse_unit: the
    square root of the FITS standard ('sqrt(Hz)'), or a function of
    astropy (e.g. 'dex(solMass/pc2)' or 'mag(AB)'). Return None for any
    other operand, e.g. a unit and its power in brackets ('m(3/2)'),
    which _astropy_unit reads.
    """
    call = _CALL.fullmatch(operand)
    if call is None:
        return None
    name, argument = call['name'], call['argument']
    if name == 'sqrt':
        return _expression(argument) ** Fraction(1, 2)
    try:
        function = astropy.units.Unit(name)
    except ValueError:
        return None
    # The only units astropy can call, to make a unit of a function, are
    # its functions (e.g. dex, mag or dB)
    if not callable(function):
        return None
    return function(_expression(argument))


def _astropy_unit(operand: str) -> AnyUnit:
    """
    Return the unit of an operand that is neither in brackets nor a
    function, as astropy reads it: a name of a unit, a power after a
    name or a number, e.g. 'km', 'm2', 'm(3/2)' or '1e-17'.

    Raise ValueError if astropy reads no unit in it, or reads a
    structure of units.
    """
    try:
        unit = astropy.units.Unit(operand)
    except ValueError:
        raise ValueError(f"astropy reads no unit {operand!r}") from None
    # astropy reads a ',' as making a structure of units (e.g. 'km,s'),
    # which is not one unit
    if isinstance(unit, astropy.units.StructuredUnit):
        raise ValueError(f"astropy reads {operand!r} as a structure of units")
    return unit


def _has_number(operand: str, unit: AnyUnit) -> bool:
    """
    Return whether an operand, whose unit is given, has a number in it:
    one at its start, as in '30' or '2km', or a scale other than 1, as
    in 'sqrt(1e4 Hz)' or '(30 s)', which a named unit, prefixed or not,
    never has. A unit of a function, e.g. a logarithm, has no scale.
    """
    return operand[0] in '0123456789+-.' or (
        isinstance(unit, astropy.units.UnitBase) and unit.scale != 1)


def _power(text: str) -> Fraction:
    """
    Return the number of a power after brackets, as _POWER matches it:
    e.g. '2', '**-2', '^(1.5)' or '(3/2)'.

    Raise ValueError if the text is not such a power, or divides by 0.
    """
    match = _POWER.fullmatch(text)
    if not match:
        raise _FitsSyntaxError(
            f"the power {text!r} is neither an integer nor, in brackets, "
            f"a decimal or a ratio")
    try:
        return Fraction(match['number'].strip('()'))
    except ZeroDivisionError:
        raise _FitsSyntaxError(f"the power {text!r} divides by 0") from None
