"""
Tests for the spectral lines of the spectral cubes: each channel holds
the mean of the line over the channel.
"""

import numpy as np
import pytest


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
                vptraits=dict(type='tan_arctan'),
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
                vptraits=dict(type='tan_arctan'),
                dptraits=dict(type='uniform')),
            emission=dict(type='lines', lines=lines))])],
        observations=[dict(
            observable=dict(
                type='pixel_spectra', size=[8, 8, 10],
                rest='6564.61 Angstrom'),
            line_of_sight=dict(type='flat'))])
    if catalogue is not None:
        config['lines'] = catalogue
    return config


def test_configurations_name_lines_of_the_catalogue():
    from modelutils import config_group
    group = config_group(lines_config(
        ['halpha', 'mine', dict(name='other', rest='5000 Angstrom')],
        dict(mine='4000 Angstrom')))
    component, = group.galaxies()[0].components()
    lines = component.emission().lines().lines()
    assert [line.name() for line in lines] == ['halpha', 'mine', 'other']
    assert [line.rest().to_value('Angstrom') for line in lines] \
        == pytest.approx([6564.61, 4000, 5000])


def test_configurations_name_known_lines():
    from modelutils import config_group
    from gbkfit.utils.parseutils import ConfigError
    with pytest.raises(ConfigError) as error:
        config_group(lines_config(['halpha', 'oi6300']))
    assert "there is no line 'oi6300' in the catalogue" in str(error.value)
    assert "emission.lines[1]" in str(error.value)


@pytest.mark.parametrize('medium, offset', [('air', 0), ('vacuum', 82.7)])
def test_lines_in_vacuum_on_an_axis_in_air(medium, offset):
    # The rest wavelengths of the lines are in vacuum: Halpha is at 6562.80
    # Angstrom in air, so on an axis in air of that rest it is at velocity
    # 0, and on an axis in vacuum it is 83 km/s above it (at 6564.61)
    from gbkfit.model import LineCatalogue
    from gbkfit.model.components.lines import Lines
    from gbkfit.utils import gridutils
    spectral = gridutils.make_grid(
        (41,), 10, spectral_axis=0, rest='6562.80 Angstrom', medium=medium)
    lines = Lines([LineCatalogue().line('halpha')])
    (line_offset, scale, _), = lines.values(spectral, 0.0)
    assert line_offset == pytest.approx(offset, abs=0.2)
    assert scale == pytest.approx(1, abs=3e-4)


@pytest.mark.parametrize('rest, k', [(6564.61 * 2.5, 1.0), (6564.61, 2.5)])
def test_the_redshift_places_the_lines(rest, k):
    # A galaxy at z = 1.5 on an axis of the rest of Halpha as observed (k =
    # 1) or in its frame (k = 1 + z): Halpha is at 0 or at c z (optical
    # velocities), [NII] 6583 is 943.6 km/s of the galaxy above it, and a
    # km/s of the galaxy spans k km/s of the axis
    from gbkfit.model import LineCatalogue
    from gbkfit.model.components.lines import Lines
    from gbkfit.utils import gridutils
    catalogue = LineCatalogue()
    spectral = gridutils.make_grid(
        (41,), 10, spectral_axis=0, rest=f'{rest} Angstrom')
    (halpha, halpha_scale, _), (nii, nii_scale, _) = Lines(
        [catalogue.line('halpha'), catalogue.line('nii6583')]).values(
        spectral, 1.5)
    c = 299792.458
    assert halpha == pytest.approx(c * (k - 1), abs=1e-6)
    assert halpha_scale == pytest.approx(k, rel=1e-12)
    assert (nii - halpha) / k == pytest.approx(
        c * (6585.27 / 6564.61 - 1), rel=1e-9)
    assert nii_scale == pytest.approx(k * 6585.27 / 6564.61, rel=1e-12)
