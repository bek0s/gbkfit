"""
Tests for the emission lines of spectral components: each line is at its
place on the spectral axis, whose velocities refer to a rest wavelength
(optical velocities) or frequency (radio velocities).
"""

import copy

import astropy.constants
import astropy.units as u
import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group


C = astropy.constants.c.to_value('km/s')

COMPONENT = dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    bptraits=dict(type='exponential'),
    vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))

PROPERTIES = dict(
    vsys=100, xpos=0.3, ypos=-0.6, posa=50, incl=60,
    bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=20)

INSTRUMENT = dict(
    psf=dict(type='gauss', sigma=1.5), lsf=dict(type='gauss', sigma=15))


def scube(rest, **options):
    return dict(
        type='pixel_spectra', size=[32, 41, 301], step=[1, 1, 10],
        rval=[0, 0, 500], rest=rest, **INSTRUMENT) | options


def evaluate(driver, model, observable, properties):
    group = observation_group([dict(
        driver=dict(type=driver.type()), observable=copy.deepcopy(observable),
        model=copy.deepcopy(model))])
    params = gbkfit.params.EvaluationParams(group.pdescs(), properties)
    return group.model_h(params.evaluate())[0]['spectra']['d'].copy()


def kinematics(component, model_type='kinematics_2d'):
    return dict(type=model_type, components=[component])


def scaled(properties, ratio, offset):
    """The properties of a line at offset + ratio * velocity."""
    return properties | dict(
        vsys=offset + ratio * properties['vsys'],
        vpt_vt=ratio * properties['vpt_vt'],
        dpt_a=ratio * properties['dpt_a'])


@pytest.mark.parametrize('rest, lines, convention', [
    ('6562.8 Angstrom',
     [dict(name='ha', rest='6562.8 Angstrom'),
      dict(name='nii6583', rest='6583.45 Angstrom')], 'optical'),
    ('330.587965 GHz',
     [dict(name='co13', rest='330.587965 GHz'),
      dict(name='c18o', rest='329.330553 GHz')], 'radio'),
    # (rests given as frequencies on a wavelength axis, and the reverse)
    ('6562.8 Angstrom',
     [dict(name='ha', rest='6562.8 Angstrom'),
      dict(name='nii6583', rest='455.37287896 THz')], 'optical'),
    ('330.587965 GHz',
     [dict(name='co13', rest='330.587965 GHz'),
      dict(name='c18o', rest='0.91030867 mm')], 'radio')])
def test_lines_are_at_their_places(driver, rest, lines, convention):
    # The second line is the first at the velocities offset + k v, with
    # the flux ratio, where k is the ratio of the rests: of the
    # wavelengths with optical velocities, of the frequencies with radio
    # velocities
    single = evaluate(
        driver, kinematics(COMPONENT), scube(rest), PROPERTIES)
    double = evaluate(
        driver, kinematics(COMPONENT | dict(lines=lines)), scube(rest),
        PROPERTIES | {f"{lines[1]['name']}_ratio": 0.4})
    k = (u.Quantity(lines[1]['rest']).to_value(
        u.Quantity(rest).unit, u.spectral()) / u.Quantity(rest).value)
    offset = C * (k - 1) if convention == 'optical' else C * (1 - k)
    second = evaluate(
        driver, kinematics(COMPONENT), scube(rest),
        scaled(PROPERTIES, k, offset))
    np.testing.assert_allclose(
        double, single + 0.4 * second, rtol=1e-4, atol=1e-6 * single.max())
    # The lines are apart: the second is at about offset km/s
    assert abs(offset) > 600


def test_lines_of_a_monte_carlo_disk_share_the_clouds(driver):
    # With the same clouds, the flux of the cube is (1 + ratio) times that
    # of one line
    component = dict(COMPONENT, type='mcdisk', cflux=1e-2,
                     bhtraits=dict(type='sech2'))
    properties = PROPERTIES | dict(bht_s=1)
    lines = [dict(name='ha', rest='6562.8 Angstrom'),
             dict(name='nii6583', rest='6583.45 Angstrom')]
    rest = '6562.8 Angstrom'
    single = evaluate(
        driver, kinematics(component, 'kinematics_3d'), scube(rest),
        properties)
    double = evaluate(
        driver, kinematics(component | dict(lines=lines), 'kinematics_3d'),
        scube(rest), properties | dict(nii6583_ratio=0.25))
    np.testing.assert_allclose(double.sum(), 1.25 * single.sum(), rtol=1e-5)


def test_one_line_without_a_rest_is_at_the_velocity_of_the_axis(driver):
    # Without lines, the line is at the velocity of the axis, whatever the
    # rest of the axis
    without = evaluate(
        driver, kinematics(COMPONENT), scube(None), PROPERTIES)
    with_rest = evaluate(
        driver, kinematics(COMPONENT), scube('6583.45 Angstrom'), PROPERTIES)
    np.testing.assert_array_equal(without, with_rest)


def test_lines_need_the_rest_of_the_spectral_axis(driver):
    lines = [dict(name='ha', rest='6562.8 Angstrom')]
    with pytest.raises(RuntimeError, match="need the rest"):
        evaluate(
            driver, kinematics(COMPONENT | dict(lines=lines)), scube(None),
            PROPERTIES)


def test_line_parameters_and_options():
    from gbkfit.model import model_parser
    lines = [dict(name='ha', rest='6562.8 Angstrom'),
             dict(name='nii6583', rest='6583.45 Angstrom'),
             dict(name='nii6548', rest='6548.05 Angstrom')]
    model = model_parser.load(kinematics(COMPONENT | dict(lines=lines)))
    assert {'nii6583_ratio', 'nii6548_ratio'} <= set(model.pdescs())
    assert 'ha_ratio' not in model.pdescs()
    # The lines survive a round trip through the configuration
    info = model_parser.dump(model)
    assert [line['name'] for line in info['components'][0]['lines']] == [
        'ha', 'nii6583', 'nii6548']
    assert model_parser.dump(model_parser.load(info)) == info
    # Without lines, there are no lines in the configuration
    info = model_parser.dump(model_parser.load(kinematics(COMPONENT)))
    assert 'lines' not in info['components'][0]


@pytest.mark.parametrize('lines, message', [
    ([], "at least one line"),
    ([dict(name='ha', rest='6562.8 Angstrom'),
      dict(name='ha', rest='6563 Angstrom')], "different names"),
    ([dict(name='ha', rest='6562.8 Angstrom'),
      dict(name='vpt', rest='6563 Angstrom')], None),
    ([dict(name='ha', rest='6562.8')], "positive wavelength or frequency")])
def test_invalid_lines(lines, message):
    from gbkfit.model import model_parser
    info = kinematics(COMPONENT | dict(lines=lines))
    if message is None:
        # A line may be named after a prefix of parameters
        model_parser.load(info)
        return
    with pytest.raises(Exception, match=message):
        model_parser.load(info)


def evaluate_observation(driver, model, observable, properties, **options):
    """The model of an observation with options (e.g. its lines)."""
    from gbkfit.model import model_parser
    from gbkfit.observation import ObservationGroup, observation_parser
    observation = observation_parser.load(copy.deepcopy(dict(
        driver=dict(type=driver.type()), observable=observable) | options))
    group = ObservationGroup(
        [model_parser.load(copy.deepcopy(model))], [observation])
    params = gbkfit.params.EvaluationParams(group.pdescs(), properties)
    data = group.model_h(params.evaluate())[0]
    return {key: value['d'].copy() for key, value in data.items()}


LINES = [dict(name='ha', rest='6562.8 Angstrom'),
         dict(name='nii6583', rest='6583.45 Angstrom')]


def test_observations_see_their_lines(driver):
    # The first line alone is the model of one line; the second alone has
    # its flux ratio; together they are the model of both
    model = kinematics(COMPONENT | dict(lines=LINES))
    observable = dict(
        type='pixel_spectra', size=[32, 41, 301], step=[1, 1, 10],
        rval=[0, 0, 500], rest='6562.8 Angstrom')
    properties = PROPERTIES | dict(nii6583_ratio=0.4)
    both = evaluate_observation(
        driver, model, observable, properties)['spectra']
    ha = evaluate_observation(
        driver, model, observable, properties, lines=['ha'])['spectra']
    nii = evaluate_observation(
        driver, model, observable, properties, lines=['nii6583'])['spectra']
    single = evaluate_observation(
        driver, kinematics(COMPONENT), observable, PROPERTIES)['spectra']
    np.testing.assert_allclose(ha, single, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        ha + nii, both, rtol=1e-5, atol=1e-6 * both.max())


def test_moment_maps_of_one_line(driver):
    # On a spectral axis wide enough for both lines, the moment maps of
    # the first line are those of a model of that line alone
    model = kinematics(COMPONENT | dict(lines=LINES))
    observable = dict(
        type='pixel_moments', size=[32, 41], spec_size=301, spec_step=10,
        spec_rval=500, spec_rest='6562.8 Angstrom', mask_cutoff=1e-3)
    properties = PROPERTIES | dict(nii6583_ratio=0.4)
    ha = evaluate_observation(
        driver, model, observable, properties, lines=['ha'])
    both = evaluate_observation(driver, model, observable, properties)
    single = evaluate_observation(
        driver, kinematics(COMPONENT), observable, PROPERTIES)
    np.testing.assert_allclose(
        ha['moment1'], single['moment1'], rtol=1e-5, atol=1e-3)
    good = np.isfinite(both['moment1'])
    assert np.abs(both['moment1'] - single['moment1'])[good].max() > 100


@pytest.mark.parametrize('lines, message', [
    (['hb'], "no component has the lines \\['hb'\\]"),
    (['ha', 'hb'], "no component has the lines \\['hb'\\]")])
def test_unknown_lines_are_errors(driver, lines, message):
    model = kinematics(COMPONENT | dict(lines=LINES))
    with pytest.raises(Exception, match=message):
        evaluate_observation(
            driver, model, scube('6562.8 Angstrom'), PROPERTIES,
            lines=lines)


def test_a_component_must_have_a_selected_line(driver):
    model = dict(type='kinematics_2d', components=[
        COMPONENT | dict(name='gas', lines=LINES),
        COMPONENT | dict(name='co', lines=[
            dict(name='co32', rest='345.79599 GHz')])])
    properties = {f'{name}_{key}': value for name in ('gas', 'co')
                  for key, value in PROPERTIES.items()}
    with pytest.raises(Exception, match="has none of the selected lines"):
        evaluate_observation(
            driver, model, scube('6562.8 Angstrom'), properties,
            lines=['ha'])
    # Leaving the component out works
    evaluate_observation(
        driver, model, scube('6562.8 Angstrom'),
        properties | dict(gas_nii6583_ratio=0.3),
        lines=['ha'], components=['gas'])


def test_line_selection_round_trip(driver):
    from gbkfit.observation import observation_parser
    info = dict(driver=dict(type=driver.type()),
                observable=scube('6562.8 Angstrom'), lines=['ha'])
    dumped = observation_parser.dump(observation_parser.load(dict(info)))
    assert dumped['lines'] == ['ha']
