"""
Tests for the components an observation sees: e.g. the tracer of the
line of each observation of a galaxy of several tracers.
"""

import copy

import gbkfit.params
import numpy as np
import pytest

from gbkfit.model import gmodel_parser
from gbkfit.observation import ObservationGroup, observation_parser


def disk(name, **traits):
    return dict(
        type='smdisk', name=name, loose=False, tilted=False,
        rnodes=list(range(0, 14)),
        bptraits=dict(type='exponential'),
        vptraits=dict(type='tan_arctan'),
        dptraits=dict(type='uniform')) | traits


GMODEL = dict(type='kinematics_2d', components=[
    disk('ha'), disk('co')])

PROPERTIES = dict(
    ha_vsys=0, ha_xpos=0.3, ha_ypos=-0.6, ha_posa=50, ha_incl=60,
    ha_bpt_a=1, ha_bpt_s=4, ha_vpt_rt=2, ha_vpt_vt=150, ha_dpt_a=20,
    co_vsys=0, co_xpos=0.3, co_ypos=-0.6, co_posa=50, co_incl=60,
    co_bpt_a=2, co_bpt_s=2, co_vpt_rt=2, co_vpt_vt=150, co_dpt_a=8)

SCUBE = dict(type='pixel_spectra', size=[32, 41, 51], step=[1, 1, 10])


def observation(driver, components=None, **options):
    info = dict(driver=dict(type=driver.type()), observable=SCUBE)
    if components is not None:
        info['components'] = components
    return observation_parser.load(copy.deepcopy(info | options))


def evaluate(gmodel, observations, properties):
    group = ObservationGroup(
        [gmodel_parser.load(copy.deepcopy(gmodel))], observations)
    params = gbkfit.params.EvaluationParams(group.pdescs(), properties)
    return [data['spectra']['d'].copy()
            for data in group.model_h(params.evaluate())]


def test_an_observation_sees_its_components(driver):
    # The model of one component of a gmodel is that of a gmodel of that
    # component alone
    [ha] = evaluate(GMODEL, [observation(driver, ['ha'])], PROPERTIES)
    alone = dict(GMODEL, components=[disk('ha')])
    [expected] = evaluate(alone, [observation(driver)], {
        k: v for k, v in PROPERTIES.items() if k.startswith('ha_')})
    np.testing.assert_array_equal(ha, expected)


def test_observations_of_different_components_add_up(driver):
    ha, co, both = evaluate(GMODEL, [
        observation(driver, ['ha'], name='muse'),
        observation(driver, ['co'], name='alma'),
        observation(driver, name='all')], PROPERTIES)
    np.testing.assert_allclose(ha + co, both, rtol=1e-5, atol=1e-7)
    assert ha.sum() > 0 and co.sum() > 0


def test_opacity_components_always_absorb(driver):
    # The light of each component goes through the opacity of the gmodel
    gmodel = dict(
        type='kinematics_3d',
        components=[
            disk('ha', bhtraits=dict(type='sech2')),
            disk('co', bhtraits=dict(type='sech2'))],
        opacity_components=[dict(
            type='smdisk', name='dust', loose=False, tilted=False,
            rnodes=list(range(0, 14)),
            optraits=dict(type='exponential'),
            ohtraits=dict(type='sech2'))])
    properties = PROPERTIES | dict(
        ha_bht_s=1, co_bht_s=0.5, dust_xpos=0.3, dust_ypos=-0.6,
        dust_posa=50, dust_incl=60, dust_opt_a=0.3, dust_opt_s=4,
        dust_oht_s=0.5)
    ha, co, both = evaluate(gmodel, [
        observation(driver, ['ha'], name='muse'),
        observation(driver, ['co'], name='alma'),
        observation(driver, name='all')], properties)
    np.testing.assert_allclose(
        ha + co, both, rtol=1e-4, atol=1e-6 * both.max())
    clear = evaluate(
        dict(gmodel, opacity_components=[]),
        [observation(driver, ['ha'])],
        {k: v for k, v in properties.items() if not k.startswith('dust')})
    assert ha.sum() < 0.99 * clear[0].sum()


@pytest.mark.parametrize('components, message', [
    (['ha', 'hi'], "no components named \\['hi'\\]"),
    (['ha', 'ha'], "repeat"),
    ([], "at least one")])
def test_invalid_selections(driver, components, message):
    with pytest.raises(Exception, match=message):
        evaluate(GMODEL, [observation(driver, components)], PROPERTIES)


def test_selection_round_trip(driver):
    dumped = observation_parser.dump(observation(driver, ['co']))
    assert dumped['components'] == ['co']
    assert 'components' not in observation_parser.dump(observation(driver))
    assert observation_parser.load(dumped).selection().components == ('co',)
