"""
Physical units of data and models: parsing them from the text of
configurations and FITS headers (BUNIT).
"""

import re
import warnings

import astropy.units

from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'format_unit',
    'parse_unit'
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
    (e.g. 'JY/BEAM.KM/S') are read in the case of their names.

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
        If the text is not a unit.
    """
    for candidate in (text, _uncapitalised(text)):
        try:
            return _product(candidate)
        except ValueError:
            pass
    raise ConfigError(
        f"{text!r} is not a unit (e.g. 'Jy/beam', 'K', 'MJy/sr', "
        f"'1e-17 erg/s/cm2/arcsec2')")


def format_unit(unit: astropy.units.UnitBase) -> str:
    """
    Return a unit as text that parse_unit reads as the same unit: its
    scale and its terms with their powers (e.g. 'Jy km beam-1 s-1').
    """
    # (repr: the shortest text of the scale that reads back exactly)
    terms = [] if unit.scale == 1 else [repr(float(unit.scale))]
    for base, power in zip(unit.bases, unit.powers):
        name = base.to_string()
        terms.append(name if power == 1 else f'{name}{power:g}')
    return ' '.join(terms)


def _product(text: str) -> astropy.units.UnitBase:
    """
    Return the product of the terms of a unit (see parse_unit). Raise
    ValueError if a term is not a unit.
    """
    terms = [term for term in _TERM_SEPARATOR.split(text.strip()) if term]
    if not terms:
        raise ValueError("no unit")
    unit = astropy.units.dimensionless_unscaled
    with warnings.catch_warnings():
        # (astropy warns of several '/' in a term, e.g. 'erg/s/cm2')
        warnings.simplefilter('ignore', astropy.units.UnitsWarning)
        for term in terms:
            unit = unit * astropy.units.Unit(term)
    return unit


def _uncapitalised(text: str) -> str:
    """Return a unit with the names of _CAPITALS in their case."""
    return re.sub(
        r'[A-Za-z]+',
        lambda match: _CAPITALS.get(match.group(0).upper(), match.group(0)),
        text)
