"""Tests for the physical units of data and models (unitutils)."""

import astropy.units as u
import numpy as np
import pytest

from gbkfit.model.components._units import (
    BandFactors, DataUnit, LineFactors, line_factors, mass_per_unit)
from gbkfit.model.components.lines import Band
from gbkfit.observation.observables._detail import flux_per_area
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
    ('Msun/pc2', u.Msun / u.pc ** 2),
    # (astropy's own forms, whose spaces are not between terms)
    ('Jy / beam', u.Jy / u.beam),
    ('Jy km / (beam s)', u.Jy * u.km / u.s / u.beam),
    ('MJy / sr', u.MJy / u.sr)])
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
    disk = dict(type='disk', rnodes=[0, 1, 2],
                density=dict(ptraits=dict(type='exponential')),
                kinematics=dict(type='traits',
                                vptraits=dict(type='arctan', direction='tan'),
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


def doppler(vsys):
    """The 1 + z of a systemic velocity (relativistic; see LinePlacement)."""
    return np.sqrt((1 + vsys / C) / (1 - vsys / C))


def factors(
        source, target, rest=HI, lines=(1.0,), redshift=0.0, convention=None,
        **options):
    """
    Return the factors of the lines of the given 1 + z on the axis (see
    Lines.factors) of a component of the source unit, of a galaxy at the
    redshift, in data of the target unit, on an axis of the rest and the
    convention.
    """
    spectral = gridutils.Grid((11,), gridutils.Coords(
        (10,), (5,), (0,), 0, rest, None, convention), 0)
    lines = np.array(lines, dtype=float)
    # (the frequency at which the galaxy emits each line: 1 + redshift
    # times that of the line on the axis, the rest frequency of the axis
    # over its 1 + z)
    frequencies = None
    if rest is not None:
        frequencies = rest.to_value(u.Hz, u.spectral()) / lines \
            * (1 + redshift)
    return LineFactors(
        parse_unit(source), DataUnit(parse_unit(target), PIXEL, **options),
        lines, spectral, frequencies)


def test_hi_mass_is_its_moment_0_by_the_textbook():
    # N_HI = 1.105e24 S / (bmaj bmin) cm^-2 for S in Jy/beam km/s, and
    # 1.249e20 H atoms per cm^2 in 1 Msun/pc^2: the factor computed by
    # hand for NGC 1313 (to 0.1%, the rounding of those constants) at the
    # rest frequency, which the textbook takes; at a systemic velocity,
    # the brightness temperature is that at the frequency of the line
    # there, and the data see the HI dimmed by (1 + z)^-3 (1 + z that of
    # vsys), its km/s spanning 1 - u / c = 1 / (1 + z) km/s of the radio
    # axis (u the radio velocity of the line): (1 + z)^-4 in all (as M_HI
    # = 2.356e5 D_L^2 int S du, u of the axis)
    hi = factors('Msun/pc2', 'Jy/beam km/s', beam=BEAM, integrated=True)
    for vsys in (0, 470, 5000):
        assert hi(vsys) == pytest.approx(
            1.249e20 * 66 * 53.7 / 1.105e24 * doppler(vsys) ** -4, rel=1e-3)


def test_light_per_frequency_or_wavelength_is_per_km_s_of_the_axis():
    # The model holds light per km/s of the axis: an optical line of
    # energy is c / lambda_rest km/s per Angstrom of the axis (lambda_rest
    # of the axis, at any velocity), times the area of a spaxel; and on an
    # axis of radio velocities, nu_rest / c Hz per km/s; the energy is
    # dimmed by (1 + z)^-4, 1 + z that of vsys on both axes
    halpha = factors(
        '1e-17 erg/s/cm2/arcsec2', '1e-20 erg/s/cm2/Angstrom', HALPHA)
    hi = factors('1e-17 erg/s/cm2/arcsec2', 'Jy/arcsec2')
    for vsys in (0, 50000):
        assert halpha(vsys) == pytest.approx(
            1e-17 * PIXEL / 1e-20 * C / 6564.61 / doppler(vsys) ** 4,
            rel=1e-12)
        assert hi(vsys) == pytest.approx(
            1e-17 * 1e-3 / 1e-26 * C / 1420.405751768e6 * doppler(vsys) ** -4,
            rel=1e-12)


def test_the_temperature_of_a_line_at_its_frequency():
    # K km/s per channel is K dimmed by 1 / (1 + z), 1 + z that of vsys,
    # times the km/s of the radio axis that a km/s of the galaxy spans at
    # the line (1 - u / c = 1 / (1 + z)), also for a line of another rest
    # (k = 1.02, which spans k / (1 + z)); in Jy/beam, the intensity at the
    # rest frequency dimmed by (1 + z)^-3
    assert factors('K km/s', 'K')(470) == pytest.approx(
        doppler(470) ** -2, rel=1e-12)
    assert factors('K km/s', 'K', lines=[1 / 1.02])(470) \
        == pytest.approx(1.02 * doppler(470) ** -2, rel=1e-12)
    jansky = factors('K km/s', 'Jy/beam', beam=BEAM)
    for vsys in (0, 3000):
        expected = (1 * u.K).to(u.Jy / u.sr, u.brightness_temperature(HI)) \
            * BEAM * u.arcsec ** 2 * doppler(vsys) ** -4
        assert jansky(vsys) == pytest.approx(expected.to_value(u.Jy), rel=1e-9)


def test_lines_at_a_redshift_by_the_textbook():
    # At z = 1.5, CO 2-1 of 1 K km/s (I_CO, as the galaxy emits it) on an
    # axis of radio velocities at its observed frequency: L' = I_CO area =
    # 3.25e7 S dv nu_obs^-2 D_L^2 (1 + z)^-3 (Solomon & Vanden Bout 2005,
    # ARA&A 43, 677; K km/s pc^2, Jy km/s, GHz, Mpc), and D_L = (1 + z)^2
    # D_A, so S dv in a beam = I_CO Omega 1e12 nu_obs^2 / (3.25e7 (1 + z))
    # (to 0.2%: 3.25e7 rounds c^2 / 2k); and Halpha of energy on an axis
    # at its rest (k = 1 + z) is dimmed by (1 + z)^-4
    co = factors(
        'K km/s', 'Jy/beam km/s', 230.538 / 2.5 * u.GHz, redshift=1.5,
        beam=BEAM, integrated=True)
    omega = (BEAM * u.arcsec ** 2).to_value(u.sr)
    assert co(0) == pytest.approx(
        omega * 1e12 * (230.538 / 2.5) ** 2 / (3.25e7 * 2.5), rel=2e-3)
    # (on an axis of log velocities, a km/s of the galaxy spans one km/s of
    # the axis, not 2.5: the line is 2.5 times narrower in its km/s, of
    # 1 / 2.5 the light per km/s, the same per Angstrom)
    for convention, stretch in (('optical', 2.5), ('log', 1)):
        halpha = factors(
            '1e-17 erg/s/cm2/arcsec2', '1e-20 erg/s/cm2/Angstrom', HALPHA,
            [2.5], redshift=1.5, convention=convention)
        assert halpha(0) == pytest.approx(
            1e-17 * PIXEL / 1e-20 * C / 6564.61 / 2.5 ** 4 * stretch / 2.5,
            rel=1e-12)


def test_the_beam_of_each_channel_at_the_line():
    channels = factors(
        'Jy km/s/arcsec2', 'Jy/beam', beam=np.array([100.0, 200.0]),
        beam_spectral=[0, 100] * u.km / u.s)
    # (the beam at the line, at the radio velocity of vsys)
    centre = C * (1 - 1 / doppler(50))
    assert channels(50) == pytest.approx(
        (100 + centre) * doppler(50) ** -4, rel=1e-12)


def test_light_with_a_unit_needs_the_rest_of_the_axis():
    # (the data see it dimmed by the redshift of each line)
    with pytest.raises(ConfigError, match="needs its rest"):
        factors('Jy km/s/arcsec2', 'Jy/beam', None, beam=BEAM)


def test_bands_per_beam_and_per_pixel():
    jy = parse_unit('Jy/arcsec2')
    assert BandFactors(
        jy, DataUnit(parse_unit('Jy/beam'), PIXEL, BEAM), None, 0.0)(0) \
        == pytest.approx(BEAM)
    assert BandFactors(
        jy, DataUnit(u.Jy, PIXEL), Band('j', '1.2 um'), 0.0)(0) \
        == pytest.approx(PIXEL)
    with pytest.raises(ConfigError, match="needs its band"):
        BandFactors(jy, DataUnit(u.K, PIXEL), None, 0.0)


def test_bands_dimmed_by_the_redshift():
    # Per Hz by (1 + z)^-3, per Angstrom by (1 + z)^-5 (the galaxy emits
    # at (1 + z) times the frequency of the band), 1 + z = (1 + redshift)
    # times that of vsys
    jy, per_a = (parse_unit(unit) for unit in (
        'Jy/arcsec2', '1e-20 erg/s/cm2/Angstrom/arcsec2'))
    band = Band('h', '1.65 um')
    one_plus_z = 2.5 * doppler(3000)
    assert BandFactors(jy, DataUnit(u.Jy, PIXEL), band, 1.5)(3000) \
        == pytest.approx(PIXEL / one_plus_z ** 3, rel=1e-12)
    assert BandFactors(per_a, DataUnit(per_a * u.arcsec ** 2, PIXEL), band,
                       1.5)(3000) \
        == pytest.approx(PIXEL / one_plus_z ** 5, rel=1e-12)


@pytest.mark.parametrize('target, message', [
    (DataUnit(parse_unit('Jy/beam'), PIXEL), "have no beam"),
    (DataUnit(None, PIXEL), "the data have no unit")])
def test_data_must_tell_their_unit_and_beam(target, message):
    with pytest.raises(ConfigError, match=message):
        line_factors(
            parse_unit('K km/s'), target, np.ones(1),
            gridutils.Grid((11,), gridutils.Coords(
                (10,), (5,), (0,), 0, HI), 0),
            np.array([HI.to_value(u.Hz)]))


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
        == pytest.approx(PIXEL * doppler(470) ** -4, rel=1e-12)
