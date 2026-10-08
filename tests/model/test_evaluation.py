"""
Tests for the evaluation contract of models: evaluating a model must
not modify its inputs, and must not depend on previous evaluations.
"""

import copy

import gbkfit.model
import gbkfit.params
import numpy as np


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
    model_group = gbkfit.model.ModelGroup(gbkfit.model.model_parser.load(
        [nodewise_relative_model(driver)]))
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
