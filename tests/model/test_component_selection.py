"""
Tests for the components an observation sees: e.g. the tracer of the
line of each observation of a galaxy of several tracers.
"""

import copy

import gbkfit.params
import numpy as np
import pytest

from gbkfit.model import galaxy_parser
from gbkfit.observation import ObservationGroup, observation_parser


FLAT = dict(type='flat')
GRID = dict(type='grid')

THIN = dict(ptraits=dict(type='exponential'))
THICK = THIN | dict(htraits=dict(type='sech2', zmax=8))


def disk(name, density=THIN):
    """A rotating disk of the given density that emits a line."""
    return dict(
        type='smdisk', name=name, loose=False, tilted=False,
        rnodes=list(range(0, 14)),
        density=density,
        kinematics=dict(
            vptraits=dict(type='tan_arctan'),
            dptraits=dict(type='uniform')),
        emission=dict(type='lines'))


GALAXY = dict(components=[disk('ha'), disk('co')])

PROPERTIES = dict(
    ha_vsys=0, ha_xpos=0.3, ha_ypos=-0.6, ha_posa=50, ha_incl=60,
    ha_pt_a=1, ha_pt_s=4, ha_vpt_rt=2, ha_vpt_vt=150, ha_dpt_a=20,
    co_vsys=0, co_xpos=0.3, co_ypos=-0.6, co_posa=50, co_incl=60,
    co_pt_a=2, co_pt_s=2, co_vpt_rt=2, co_vpt_vt=150, co_dpt_a=8)

SCUBE = dict(type='pixel_spectra', size=[32, 41, 51], step=[1, 1, 10])


def observation(driver, components=None, **options):
    info = dict(observable=SCUBE, line_of_sight=FLAT)
    if components is not None:
        info['components'] = components
    return observation_parser.load(
        copy.deepcopy(info | options), drivers={None: driver})


def evaluate(galaxy, observations, properties):
    group = ObservationGroup(
        [galaxy_parser.load(copy.deepcopy(galaxy))], observations)
    params = gbkfit.params.ParamSpace(group.pdescs(), properties)
    return [data['spectra']['d'].copy()
            for data in group.model_h(params.evaluate())]


def test_an_observation_sees_its_components(driver):
    # The model of one component of a galaxy is that of a galaxy of that
    # component alone
    [ha] = evaluate(GALAXY, [observation(driver, ['ha'])], PROPERTIES)
    alone = dict(GALAXY, components=[disk('ha')])
    [expected] = evaluate(alone, [observation(driver)], {
        k: v for k, v in PROPERTIES.items() if k.startswith('ha_')})
    np.testing.assert_array_equal(ha, expected)


def test_observations_of_different_components_add_up(driver):
    ha, co, both = evaluate(GALAXY, [
        observation(driver, ['ha'], name='muse'),
        observation(driver, ['co'], name='alma'),
        observation(driver, name='all')], PROPERTIES)
    np.testing.assert_allclose(ha + co, both, rtol=1e-5, atol=1e-7)
    assert ha.sum() > 0 and co.sum() > 0


def test_components_with_opacity_always_absorb(driver):
    # The light of each component goes through the opacity of the galaxy:
    # that of its dust, which no observation selects (it emits nothing)
    emitters = [disk('ha', THICK), disk('co', THICK)]
    dust = dict(
        type='smdisk', name='dust', loose=False, tilted=False,
        rnodes=list(range(0, 14)), density=THICK, opacity=dict(type='dust'))
    galaxy = dict(components=emitters + [dust])
    properties = PROPERTIES | dict(
        ha_ht_s=1, co_ht_s=0.5, dust_xpos=0.3, dust_ypos=-0.6,
        dust_posa=50, dust_incl=60, dust_pt_a=0.3, dust_pt_s=4,
        dust_ht_s=0.5)
    ha, co, both = evaluate(galaxy, [
        observation(driver, ['ha'], name='muse', line_of_sight=GRID),
        observation(driver, ['co'], name='alma', line_of_sight=GRID),
        observation(driver, name='all', line_of_sight=GRID)], properties)
    np.testing.assert_allclose(
        ha + co, both, rtol=1e-4, atol=1e-6 * both.max())
    clear = evaluate(
        dict(components=emitters),
        [observation(driver, ['ha'], line_of_sight=GRID)],
        {k: v for k, v in properties.items() if not k.startswith('dust')})
    assert ha.sum() < 0.99 * clear[0].sum()


@pytest.mark.parametrize('components, message', [
    (['ha', 'hi'], "no components named \\['hi'\\]"),
    (['ha', 'ha'], "repeat"),
    ([], "at least one")])
def test_invalid_selections(driver, components, message):
    with pytest.raises(Exception, match=message):
        evaluate(GALAXY, [observation(driver, components)], PROPERTIES)


def test_selection_round_trip(driver):
    dumped = observation_parser.dump(observation(driver, ['co']))
    assert dumped['components'] == ['co']
    assert 'components' not in observation_parser.dump(observation(driver))
    assert observation_parser.load(dumped).components() == ('co',)
