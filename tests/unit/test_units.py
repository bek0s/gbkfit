"""Tests for the physical units of data and models (unitutils)."""

import astropy.units as u
import pytest

from gbkfit.utils.parseutils import ConfigError
from gbkfit.utils.unitutils import format_unit, parse_unit


@pytest.mark.parametrize('text, unit', [
    ('Jy/beam', u.Jy / u.beam),
    ('JY/BEAM', u.Jy / u.beam),
    # (astropy alone reads these as Jy / (beam km s))
    ('Jy/beam km/s', u.Jy * u.km / u.s / u.beam),
    ('JY/BEAM.KM/S', u.Jy * u.km / u.s / u.beam),
    ('Jy beam-1 km s-1', u.Jy * u.km / u.s / u.beam),
    ('K km/s', u.K * u.km / u.s),
    ('MJy/sr', u.MJy / u.sr),
    ('10**(-20)*erg/s/cm**2/Angstrom',
     1e-20 * u.erg / u.s / u.cm ** 2 / u.Angstrom),
    ('1.5e-17 erg/s/cm2/arcsec2',
     1.5e-17 * u.erg / u.s / u.cm ** 2 / u.arcsec ** 2),
    ('Msun/pc2', u.Msun / u.pc ** 2)])
def test_units_are_read_as_they_are_meant(text, unit):
    parsed = parse_unit(text)
    assert parsed.is_equivalent(unit)
    assert (1 * parsed).to_value(unit) == pytest.approx(1, rel=1e-12)
    # (a unit written by format_unit reads back as itself)
    assert parse_unit(format_unit(parsed)) == parsed


def test_what_is_not_a_unit_is_an_error():
    with pytest.raises(ConfigError, match="is not a unit"):
        parse_unit('banana')
