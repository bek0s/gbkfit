"""Tests for the physical units of data and models (unitutils)."""

import astropy.units as u
import numpy as np
import pytest

from gbkfit.model.components._units import (
    DataUnit, LineFactors, band_factor, flux_per_area, line_factors,
    mass_per_unit)
from gbkfit.model.components.lines import Band
from gbkfit.utils import gridutils
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


# The beam of the HI data of NGC 1313 (LVHIS) and pixels of 0.2 arcsec
# (arcsec^2), the rests of HI and Halpha, and the speed of light (km/s)
BEAM = np.pi / (4 * np.log(2)) * 66 * 53.7
PIXEL = 0.04
HI = 1420.405751768 * u.MHz
HALPHA = 6564.61 * u.Angstrom
C = 299792.458


def factors(source, target, rest=HI, rows=((0, 1, 1),), **options):
    """
    Return the factors of the lines of the given rows of a component of
    the source unit, in data of the target unit, on an axis of the rest.
    """
    spectral = gridutils.Grid((11,), gridutils.Coords(
        (10,), (5,), (0,), 0, rest), 0)
    return LineFactors(
        parse_unit(source), DataUnit(parse_unit(target), PIXEL, **options),
        np.array(rows, dtype=float), spectral)


def test_hi_mass_is_its_moment_0_by_the_textbook():
    # N_HI = 1.105e24 S / (bmaj bmin) cm^-2 for S in Jy/beam km/s, and
    # 1.249e20 H atoms per cm^2 in 1 Msun/pc^2: the factor computed by
    # hand for NGC 1313 (to 0.1%, the rounding of those constants) at the
    # rest frequency, which the textbook takes; at a systemic velocity,
    # the brightness temperature is that at the frequency of the line
    # there (~ nu^2)
    hi = factors('Msun/pc2', 'Jy/beam km/s', beam=BEAM, integrated=True)
    for vsys in (0, 470, 5000):
        assert hi(vsys) == pytest.approx(
            1.249e20 * 66 * 53.7 / 1.105e24 * (1 - vsys / C) ** 2, rel=1e-3)


def test_light_per_frequency_or_wavelength_is_per_km_s_of_the_axis():
    # The model holds light per km/s of the axis: an optical line of
    # energy is c / lambda_rest km/s per Angstrom of the axis (lambda_rest
    # of the axis, at any velocity), times the area of a spaxel; and on an
    # axis of radio velocities, nu_rest / c Hz per km/s
    halpha = factors(
        '1e-17 erg/s/cm2/arcsec2', '1e-20 erg/s/cm2/Angstrom', HALPHA)
    hi = factors('1e-17 erg/s/cm2/arcsec2', 'Jy/arcsec2')
    for vsys in (0, 50000):
        assert halpha(vsys) == pytest.approx(
            1e-17 * PIXEL / 1e-20 * C / 6564.61, rel=1e-12)
        assert hi(vsys) == pytest.approx(
            1e-17 * 1e-3 / 1e-26 * C / 1420.405751768e6, rel=1e-12)


def test_the_temperature_of_a_line_at_its_frequency():
    # K km/s per channel is K; a line of another rest spans scale km/s of
    # the axis per km/s of its own; and in Jy/beam, ~ nu^2 at the frequency
    # of the line at the systemic velocity
    assert factors('K km/s', 'K')(470) == pytest.approx(1, rel=1e-12)
    assert factors('K km/s', 'K', rows=[(C * 0.02, 1.02, 1)])(470) \
        == pytest.approx(1.02, rel=1e-12)
    jansky = factors('K km/s', 'Jy/beam', beam=BEAM)
    for vsys in (0, 3000):
        frequency = HI * (1 - vsys / C)
        expected = (1 * u.K).to(
            u.Jy / u.sr, u.brightness_temperature(frequency)) \
            * BEAM * u.arcsec ** 2
        assert jansky(vsys) == pytest.approx(expected.to_value(u.Jy), rel=1e-9)


def test_the_beam_of_each_channel_at_the_line():
    channels = factors(
        'Jy km/s/arcsec2', 'Jy/beam', beam=np.array([100.0, 200.0]),
        beam_spectral=[0, 100] * u.km / u.s)
    assert channels(50) == pytest.approx(150)


def test_an_axis_without_a_rest_takes_what_needs_no_frequency():
    assert factors('Jy km/s/arcsec2', 'Jy/beam', None, beam=BEAM)(470) \
        == pytest.approx(BEAM)
    with pytest.raises(ConfigError, match="needs its rest"):
        factors('K km/s', 'Jy/beam', None, beam=BEAM)


def test_bands_per_beam_and_per_pixel():
    jy = parse_unit('Jy/arcsec2')
    assert band_factor(jy, DataUnit(parse_unit('Jy/beam'), PIXEL, BEAM), None) \
        == pytest.approx(BEAM)
    assert band_factor(jy, DataUnit(u.Jy, PIXEL), Band('j', '1.2 um')) \
        == pytest.approx(PIXEL)
    with pytest.raises(ConfigError, match="needs its band"):
        band_factor(jy, DataUnit(u.K, PIXEL), None)


@pytest.mark.parametrize('target, message', [
    (DataUnit(parse_unit('Jy/beam'), PIXEL), "have no beam"),
    (DataUnit(None, PIXEL), "the data have no unit")])
def test_data_must_tell_their_unit_and_beam(target, message):
    with pytest.raises(ConfigError, match=message):
        line_factors(
            parse_unit('K km/s'), target, np.array([[0.0, 1.0, 1.0]]),
            gridutils.Grid((11,), gridutils.Coords(
                (10,), (5,), (0,), 0, HI), 0))


def test_regions_are_fluxes():
    assert flux_per_area(u.Jy) == u.Jy / u.arcsec ** 2
    with pytest.raises(ConfigError, match="are fluxes"):
        flux_per_area(parse_unit('Jy/beam'))


@pytest.mark.parametrize('info, expected', [
    # (HI: its own mass, 1.249e20 H atoms per cm^2 in 1 Msun/pc^2, so
    # 1.823e18 / 1.249e20 Msun/pc^2 per K km/s; others: 1 Msun/pc^2)
    (dict(type='lines', lines=['hi21'], unit='Msun/pc2'), 1e6),
    (dict(type='lines', lines=['hi21'], unit='K km/s'),
     1.823e18 / 1.249e20 * 1e6),
    (dict(type='lines', lines=['co10'], unit='K km/s'), 1e6),
    (dict(type='band', unit='Jy/arcsec2'), 1e6),
    (dict(type='lines'), None)])
def test_the_mass_of_a_unit_of_light(info, expected):
    # Msun/kpc^2 per unit of the density, for the mass that follows it
    # (to 0.2%: the textbook's 1.249e20 rounds 1.2477e20, of the masses
    # of the Sun and of the H atom today)
    mass = mass_per_unit(emission(info))
    assert mass == (None if expected is None
                    else pytest.approx(expected, rel=2e-3))


def test_data_per_pixel_named_so():
    # (e.g. BUNIT JY/PIXEL): the light of a pixel, as without the name
    assert factors('Jy km/s/arcsec2', 'JY/PIXEL')(470) \
        == pytest.approx(PIXEL, rel=1e-12)
