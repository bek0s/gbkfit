"""
Tests for the evaluation contract of models: evaluating a model must
not modify its inputs, and must not depend on previous evaluations.
"""

import copy

import gbkfit.model
import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group


def nodewise_relative_model(driver):
    """A model with a node-wise rotation curve in relative mode."""
    return dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='scube', size=[32, 32, 41], step=[1, 1, 10]),
        gmodel=dict(type='kinematics_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'),
            vptraits=dict(
                type='nw_tan_uniform',
                nwmode=dict(type='relative1', origin=0)),
            dptraits=dict(type='uniform'))]))


def test_evaluation_does_not_modify_params(driver):
    model_group = observation_group([nodewise_relative_model(driver)])
    params = gbkfit.params.EvaluationParams(model_group.pdescs(), dict(
        vsys=0, xpos=0, ypos=0, posa=30, incl=45,
        bpt_a=1, bpt_s=4, dpt_a=10,
        vpt_vt=[100] + [10] * 11))
    values = params.evaluate()
    values_before = copy.deepcopy(values)
    first = copy.deepcopy(model_group.model_h(values)[0]['scube']['d'])
    second = model_group.model_h(values)[0]['scube']['d']
    np.testing.assert_array_equal(values['vpt_vt'], values_before['vpt_vt'])
    np.testing.assert_array_equal(first, second)


# A model of each type of two-dimensional gmodel: the gmodel, its data
# model, its data key, and its parameter properties
GMODELS_2D = dict(
    kinematics_2d=(
        dict(type='kinematics_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'),
            vptraits=dict(type='tan_arctan'),
            dptraits=dict(type='uniform'))]),
        'scube', (32, 32, 41), 'scube',
        dict(vsys=0, xpos=0, ypos=0, posa=30, incl=45,
             bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=100, dpt_a=10)),
    intensity_2d=(
        dict(type='intensity_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'))]),
        'image', (32, 32), 'image',
        dict(xpos=0, ypos=0, posa=30, incl=45, bpt_a=1, bpt_s=4)))


@pytest.mark.parametrize('gmodel_type', GMODELS_2D)
def test_one_gmodel_observed_on_grids_with_different_steps(
        driver, gmodel_type):
    # Observations of one gmodel on grids with different steps each get
    # the model on their own grid
    from gbkfit.model import gmodel_parser
    from gbkfit.observation import (
        Observation, ObservationGroup, observable_parser)
    gmodel_info, observable_type, size, key, properties = \
        GMODELS_2D[gmodel_type]

    def observation(step):
        steps = (step, step, 10)[:len(size)]
        return Observation(driver, observable_parser.load(dict(
            type=observable_type, size=list(size), step=list(steps))))

    def evaluate(group, i):
        params = gbkfit.params.EvaluationParams(group.pdescs(), properties)
        return group.model_h(params.evaluate())[i][key]['d'].copy()

    shared = ObservationGroup(
        [gmodel_parser.load(gmodel_info)], [observation(1), observation(0.5)])
    alone = ObservationGroup(
        [gmodel_parser.load(gmodel_info)], [observation(0.5)])
    np.testing.assert_array_equal(evaluate(shared, 1), evaluate(alone, 0))


def test_unsupported_dtype_fails_when_planning(driver):
    # An observation in a dtype the drivers do not support fails when its
    # plan is made, before any evaluation, every time
    from gbkfit.observation import (
        Observation, ObservationGroup, observable_parser)
    observation = Observation(
        driver, observable_parser.load(dict(type='image', size=[8, 8])),
        dtype='float64')
    gmodel = gbkfit.model.gmodel_parser.load(dict(
        type='intensity_2d', components=dict(
            type='smdisk', loose=False, tilted=False, rnodes=[0, 2, 4],
            bptraits=dict(type='uniform'))))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="does not support dtype"):
            ObservationGroup([gmodel], [observation])


@pytest.mark.parametrize('dmodel, gmodel', [
    (dict(type='image', size=[8, 8]), 'kinematics_2d'),
    (dict(type='scube', size=[8, 8, 8]), 'intensity_2d')])
def test_incompatible_observations_are_rejected_before_evaluation(
        dmodel, gmodel):
    # An image observable needs an image gmodel, and the others a spectral
    # cube gmodel; the group is rejected before any evaluation
    component = dict(
        type='smdisk', loose=False, tilted=False, rnodes=[0, 1, 2],
        bptraits=dict(type='exponential'))
    if gmodel.startswith('kinematics'):
        component |= dict(
            vptraits=dict(type='tan_arctan'), dptraits=dict(type='uniform'))
    model = dict(
        driver=dict(type='host'), dmodel=dmodel,
        gmodel=dict(type=gmodel, components=[component]))
    with pytest.raises(Exception, match="is not compatible with"):
        observation_group([model])
