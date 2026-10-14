"""Tests for the physical units of data and models (unitutils)."""

import astropy.units as u
import numpy as np
import pytest

from gbkfit.model.components._units import band_factor, line_factor
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


# The beam of the HI data of NGC 1313 (LVHIS), and pixels of 0.2 arcsec
BEAM = np.pi / (4 * np.log(2)) * 66 * 53.7 * u.arcsec ** 2
PIXEL = 0.04 * u.arcsec ** 2
HI = 1420.405751 * u.MHz


def test_hi_mass_is_its_moment_0_by_the_textbook():
    # N_HI = 1.105e24 S / (bmaj bmin) cm^-2 for S in Jy/beam km/s, and
    # 1.249e20 H atoms per cm^2 in 1 Msun/pc^2: the factor computed by
    # hand for NGC 1313 (to 0.1%, the rounding of those constants)
    factor = line_factor(
        parse_unit('Msun/pc2'), parse_unit('Jy/beam km/s'), HI, BEAM, PIXEL,
        True)
    assert factor == pytest.approx(1.249e20 * 66 * 53.7 / 1.105e24, rel=1e-3)


def test_lines_in_cubes_per_velocity():
    # A cube is the light per km/s: K km/s gives K; and an optical line
    # per arcsec^2 gives per Angstrom per spaxel (c / lambda km/s per
    # Angstrom, times the area of the pixel)
    assert line_factor(
        parse_unit('K km/s'), u.K, HI, None, PIXEL, False) == pytest.approx(1)
    halpha = 6564.61 * u.Angstrom
    factor = line_factor(
        parse_unit('1e-17 erg/s/cm2/arcsec2'),
        parse_unit('1e-20 erg/s/cm2/Angstrom'),
        halpha.to(u.Hz, u.spectral()), None, PIXEL, False)
    assert factor == pytest.approx(
        1e-17 * (299792.458 / 6564.61) * 0.04 / 1e-20, rel=1e-12)


def test_bands_per_beam_and_per_pixel():
    one_micron = (1 * u.um).to(u.Hz, u.spectral())
    assert band_factor(
        parse_unit('Jy/arcsec2'), parse_unit('Jy/beam'), HI, BEAM,
        PIXEL) == pytest.approx(BEAM.to_value(u.arcsec ** 2))
    assert band_factor(
        parse_unit('Jy/arcsec2'), u.Jy, one_micron, None,
        PIXEL) == pytest.approx(0.04)


def test_data_per_beam_need_a_beam():
    with pytest.raises(ConfigError, match="have no beam"):
        line_factor(
            parse_unit('K km/s'), parse_unit('Jy/beam'), HI, None, PIXEL,
            False)
