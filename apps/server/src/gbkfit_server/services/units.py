"""
Quantities as the Studio shows them: short text in the units of their
kind that read best.
"""

import astropy.units


__all__ = [
    'rest_text'
]


def rest_text(rest: astropy.units.Quantity) -> str:
    """
    Return a rest wavelength or frequency as short text: a wavelength in
    Angstrom (um from 1 um), a frequency in GHz (MHz below 1 GHz).
    """
    if rest.unit.physical_type == 'length':
        unit = 'Angstrom' if rest.to_value('um') < 1 else 'um'
    else:
        unit = 'GHz' if rest.to_value('GHz') >= 1 else 'MHz'
    return f"{rest.to_value(unit):.7g} {unit}"
