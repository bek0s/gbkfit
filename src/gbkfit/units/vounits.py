"""
Units as text in VOUnits, the recommendation of the IVOA (1.0), as in
configurations and outputs: reading them, with astropy (its format
'vounit'), and writing them.
"""

import astropy.units

from . import fits


__all__ = [
    'format_unit',
    'parse_quantity',
    'parse_unit'
]


def parse_unit(text: str) -> astropy.units.UnitBase:
    """
    Return the unit that a text of VOUnits states.

    The text has at most one '/': 'erg/s/cm2/Angstrom' is not a unit,
    but 'erg.s-1.cm-2.Angstrom-1' and 'erg/(s.cm2.Angstrom)' are; and
    percent is the scale 0.01 ('1e-2').

    Names are case-sensitive, and nothing is corrected: 'JY/BEAM' is not
    a unit, and 'S' is the siemens. An empty text is dimensionless.

    Raises
    ------
    ValueError
        If the text is not a unit of VOUnits.
    """
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


def format_unit(unit: astropy.units.UnitBase) -> str:
    """
    Return a unit as text of VOUnits, which parse_unit reads back as the
    same unit, and so does fits.parse_unit.
    For example, 'Jy.km.beam**-1.s**-1'.

    A unit that VOUnits has no name for (e.g. the electron) is written
    as the FITS standard writes it, which only fits.parse_unit reads.
    """
    if unit.is_equivalent(astropy.units.dimensionless_unscaled):
        # (a scale, e.g. 0.01 for percent: astropy writes none)
        scale = unit.to(astropy.units.dimensionless_unscaled)
        return '' if scale == 1 else repr(float(scale))
    try:
        # (VOUnits deprecates the units of CGS, e.g. erg and Angstrom,
        # which astropy would warn of, or convert to SI)
        return unit.to_string('vounit', deprecations='silent')
    except ValueError:
        return fits.format_unit(unit)
