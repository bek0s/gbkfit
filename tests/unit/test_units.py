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


def emission(info):
    from gbkfit.model.components.roles import emission_parser
    return emission_parser.load(info)


@pytest.mark.parametrize('info', [
    dict(type='lines', unit='1e-17 erg/s/cm2/arcsec2'),
    dict(type='lines', lines=['halpha', 'nii6583'], unit='Jy km/s/arcsec2'),
    dict(type='lines', unit='K km/s'),
    dict(type='lines', lines=['hi21'], unit='Msun/pc2'),
    dict(type='band', unit='MJy/sr'),
    dict(type='band', unit='1e-20 erg/s/cm2/Angstrom/arcsec2')])
def test_emission_takes_the_units_of_its_kind(info):
    loaded = emission(dict(info))
    assert loaded.unit() == parse_unit(info['unit'])
    assert emission(loaded.dump()).unit() == loaded.unit()


@pytest.mark.parametrize('info, message', [
    (dict(type='lines', unit='Jy/beam km/s'), "not \\(a unit per beam"),
    (dict(type='lines', unit='km/s'), "is not"),
    (dict(type='lines', unit='Msun/pc2'), "one of HI only"),
    (dict(type='lines', lines=['halpha'], unit='cm-2'), "one of HI only"),
    (dict(type='band', unit='K km/s'), "is not")])
def test_emission_refuses_units_of_other_kinds(info, message):
    with pytest.raises(ConfigError, match=message):
        emission(info)


def test_a_line_has_one_unit_in_a_galaxy():
    # (hi21 beside halpha, in a component of the unit of halpha)
    from gbkfit.model import galaxy_parser
    disk = dict(type='smdisk', rnodes=[0, 1, 2],
                density=dict(ptraits=dict(type='exponential')),
                kinematics=dict(vptraits=dict(type='tan_arctan'),
                                dptraits=dict(type='uniform')))
    with pytest.raises(ConfigError, match="the line 'hi21' has the unit"):
        galaxy_parser.load(dict(components=[
            disk | dict(name='hi', emission=dict(
                type='lines', lines=['hi21'], unit='Msun/pc2')),
            disk | dict(name='ionised', emission=dict(
                type='lines', lines=['halpha', 'hi21'], unit='K km/s'))]))
