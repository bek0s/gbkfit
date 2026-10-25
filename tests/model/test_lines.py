"""
Tests for the spectral lines of the spectral cubes: each channel holds
the mean of the line over the channel.
"""

import numpy as np
import pytest


C = 299792.458

# The 1 + z of a galaxy at the redshift 0 and vsys = 10000 km/s, a
# relativistic velocity
D = np.sqrt((1 + 10000 / C) / (1 - 10000 / C))


def line_flux(driver, evaluate_cases, dispersion):
    """
    The flux of the line of each spaxel (the sum over the channels times
    the channel width) of a rotating disk with the given dispersion.
    """
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[24, 24, 61], step=[1, 1, 10]),
        line_of_sight=dict(type='flat'),
        galaxy=dict(components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            density=dict(ptraits=dict(type='exponential')),
            kinematics=dict(
                vptraits=dict(type='arctan', direction='tan'),
                dptraits=dict(type='uniform')),
            emission=dict(type='lines'))]))
    # A systemic velocity between two channel centres
    properties = dict(
        vsys=3, xpos=0, ypos=0, posa=30, incl=45,
        pt_a=1, pt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=dispersion)
    data, _ = evaluate_cases([case], properties)
    scube = data[0]['spectra']['d']
    assert np.isfinite(scube).all()
    return scube.sum(axis=0) * 10


@pytest.mark.parametrize('dispersion', [0, 0.5, 3])
def test_narrow_lines_keep_their_flux(driver, evaluate_cases, dispersion):
    # Lines much narrower than a channel (10), and of width 0, have the
    # flux of a wide line
    expected = line_flux(driver, evaluate_cases, 20)
    actual = line_flux(driver, evaluate_cases, dispersion)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def test_the_catalogue_has_lines_and_those_of_the_user():
    from gbkfit.model import LineCatalogue
    from gbkfit.utils.parseutils import ConfigError
    catalogue = LineCatalogue(
        {'mgii2796': '2796.35 Angstrom', 'halpha': '6562.80 Angstrom'})
    # Rest wavelengths in vacuum, or rest frequencies
    rest = {name: catalogue.line(name).rest() for name in (
        'nii6583', 'hi21', 'mgii2796', 'halpha')}
    assert rest['nii6583'].to_value('Angstrom') == pytest.approx(6585.27)
    assert rest['hi21'].to_value('MHz') == pytest.approx(1420.405751768)
    assert rest['mgii2796'].to_value('Angstrom') == pytest.approx(2796.35)
    # The lines of the user replace those of the catalogue
    assert rest['halpha'].to_value('Angstrom') == pytest.approx(6562.80)
    with pytest.raises(ConfigError, match="there is no line 'oi6300'"):
        catalogue.line('oi6300')


def lines_config(lines, catalogue=None):
    """
    A configuration of a disk that emits the given lines (names of the
    catalogue, or lines), with the section 'lines' if catalogue.
    """
    config = dict(
        galaxies=[dict(components=[dict(
            type='smdisk', rnodes=[0, 2, 4],
            density=dict(ptraits=dict(type='exponential')),
            kinematics=dict(
                vptraits=dict(type='arctan', direction='tan'),
                dptraits=dict(type='uniform')),
            emission=dict(type='lines', lines=lines))])],
        observations=[dict(
            type='sky',
            observable=dict(
                type='pixel_spectra', size=[8, 8, 10],
                rest='6564.61 Angstrom'),
            line_of_sight=dict(type='flat'))])
    if catalogue is not None:
        config['lines'] = catalogue
    return config


def test_configurations_name_lines_of_the_catalogue():
    from modelutils import config_group_plan
    plan = config_group_plan(lines_config(
        ['halpha', 'mine', dict(name='other', rest='5000 Angstrom')],
        dict(mine='4000 Angstrom')))
    component, = plan.group().galaxies()[0].components()
    lines = component.emission().lines().lines()
    assert [line.name() for line in lines] == ['halpha', 'mine', 'other']
    assert [line.rest().to_value('Angstrom') for line in lines] \
        == pytest.approx([6564.61, 4000, 5000])


def test_configurations_name_known_lines():
    from modelutils import config_group_plan
    from gbkfit.utils.parseutils import ConfigError
    with pytest.raises(ConfigError) as error:
        config_group_plan(lines_config(['halpha', 'oi6300']))
    assert "there is no line 'oi6300' in the catalogue" in str(error.value)
    assert "emission.lines[1]" in str(error.value)


@pytest.mark.parametrize('medium, offset', [('air', 0), ('vacuum', 82.7)])
def test_lines_in_vacuum_on_an_axis_in_air(medium, offset):
    # The rest wavelengths of the lines are in vacuum: Halpha is at 6562.80
    # Angstrom in air, so on an axis in air of that rest it is at velocity
    # 0, and on an axis in vacuum it is 83 km/s above it (at 6564.61)
    from gbkfit.model import LineCatalogue
    from gbkfit.model.components.lines import LinePlacement, Lines
    from gbkfit.utils import gridutils
    spectral = gridutils.make_grid(
        (41,), 10, spectral_axis=0, rest='6562.80 Angstrom', medium=medium)
    lines = Lines([LineCatalogue().line('halpha')])
    (line_offset, scale), = LinePlacement(
        lines.factors(spectral, 0.0), spectral)(0.0)
    assert line_offset == pytest.approx(offset, abs=0.2)
    assert scale == pytest.approx(1, abs=3e-4)


def halpha_nii(rest, convention=None):
    """
    The offsets and scales of Halpha and [NII] 6583 of a galaxy at z = 1.5
    and the systemic velocity 0, on an axis of a rest (Angstrom) and a
    convention (see LinePlacement).
    """
    from gbkfit.model import LineCatalogue
    from gbkfit.model.components.lines import LinePlacement, Lines
    from gbkfit.utils import gridutils
    catalogue = LineCatalogue()
    spectral = gridutils.make_grid(
        (41,), 10, spectral_axis=0, rest=f'{rest} Angstrom',
        convention=convention)
    lines = Lines([catalogue.line('halpha'), catalogue.line('nii6583')])
    return LinePlacement(lines.factors(spectral, 1.5), spectral)(0.0)


@pytest.mark.parametrize('rest, k', [(6564.61 * 2.5, 1.0), (6564.61, 2.5)])
def test_the_redshift_places_the_lines(rest, k):
    # A galaxy at z = 1.5 on an axis of the rest of Halpha as observed (k =
    # 1) or in its frame (k = 1 + z): Halpha is at 0 or at c z (optical
    # velocities), [NII] 6583 is 943.6 km/s of the galaxy above it, and a
    # km/s of the galaxy spans k km/s of the axis
    (halpha, halpha_scale), (nii, nii_scale) = halpha_nii(rest)
    assert halpha == pytest.approx(C * (k - 1), abs=1e-6)
    assert halpha_scale == pytest.approx(k, rel=1e-12)
    assert (nii - halpha) / k == pytest.approx(
        C * (6585.27 / 6564.61 - 1), rel=1e-9)
    assert nii_scale == pytest.approx(k * 6585.27 / 6564.61, rel=1e-12)


@pytest.mark.parametrize('rest, k', [(6564.61 * 2.5, 1.0), (6564.61, 2.5)])
def test_the_redshift_places_the_lines_on_log_axes(rest, k):
    # The same on an axis uniform in ln w (log velocities, c ln D): Halpha
    # is at c ln k, [NII] 6583 at c ln(6585.27 / 6564.61) above it, and a
    # km/s of the galaxy spans one of the axis
    (halpha, halpha_scale), (nii, nii_scale) = halpha_nii(rest, 'log')
    assert halpha == pytest.approx(C * np.log(k), abs=1e-6)
    assert nii - halpha == pytest.approx(
        C * np.log(6585.27 / 6564.61), rel=1e-9)
    assert halpha_scale == pytest.approx(1, rel=1e-12)
    assert nii_scale == pytest.approx(1, rel=1e-12)


@pytest.mark.parametrize('rest, convention, centre, stretch', [
    ('1420.405751768 MHz', None, C * (1 - 1 / D), 1 / D),
    ('6564.61 Angstrom', None, C * (D - 1), D),
    ('6564.61 Angstrom', 'log', C * np.log(D), 1),
    (None, None, 10000, 1 - (10000 / C) ** 2)])
def test_the_systemic_velocity_is_one_on_every_axis(
        rest, convention, centre, stretch):
    # A galaxy at the redshift 0 and vsys = 10000 km/s, on an axis of the
    # rest of its line: the line is at the velocity of its 1 + z in the
    # convention of the axis, c (1 - 1 / D) for radio velocities, c (D -
    # 1) for optical ones, and 10000 km/s on an axis without a rest, of
    # relativistic velocities, and c ln D for log ones; a km/s of the
    # galaxy spans 1 - u / c, 1 + u / c, 1 - (u / c)^2 and 1 km/s of them
    # (1 / D, D, 1 - (vsys / c)^2 and 1)
    from gbkfit.model.components.lines import LinePlacement, Lines
    from gbkfit.utils import gridutils
    spectral = gridutils.make_grid(
        (41,), 10, spectral_axis=0, rest=rest, convention=convention)
    factors = Lines(None).factors(spectral, 0.0)
    (offset, scale), = LinePlacement(factors, spectral)(10000)
    assert offset + scale * 10000 == pytest.approx(centre, rel=1e-12)
    assert scale == pytest.approx(stretch, rel=1e-12)


def test_the_dispersion_is_in_the_frame_of_the_galaxy(driver, evaluate_cases):
    # A disk of HI of dispersion 30 km/s, without rotation, at vsys 0 and
    # 10000 km/s on a radio axis of the rest of HI: its line, at the radio
    # velocity u = c (1 - 1 / D), is 1 - u / c = 1 / D as wide on the axis
    # (to 1e-5: the channels add their width^2 / 12)
    def width(vsys):
        doppler = np.sqrt((1 + vsys / C) / (1 - vsys / C))
        centre = C * (1 - 1 / doppler)
        case = dict(
            driver=dict(type=driver.type()),
            observable=dict(
                type='pixel_spectra', size=[8, 8, 201], step=[1, 1, 2],
                rval=[0, 0, centre], rest='1420.405751768 MHz'),
            line_of_sight=dict(type='flat'),
            galaxy=dict(components=[dict(
                type='smdisk', loose=False, tilted=False,
                rnodes=list(range(0, 5)),
                density=dict(ptraits=dict(type='uniform')),
                kinematics=dict(
                    vptraits=dict(type='arctan', direction='tan'),
                    dptraits=dict(type='uniform')),
                emission=dict(type='lines', lines=['hi21']))]))
        properties = dict(
            vsys=vsys, xpos=0, ypos=0, posa=0, incl=45,
            pt_a=1, vpt_rt=2, vpt_vt=0, dpt_a=30)
        data, _ = evaluate_cases([case], properties)
        spectrum = data[0]['spectra']['d'].sum(axis=(1, 2))
        velocity = centre + 2 * (np.arange(201) - 100)
        mean = np.sum(spectrum * velocity) / spectrum.sum()
        return np.sqrt(np.sum(spectrum * (velocity - mean) ** 2)
                       / spectrum.sum())
    assert width(10000) / width(0) == pytest.approx(1 / D, rel=1e-4)
