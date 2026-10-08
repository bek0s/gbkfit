"""
Tests for the evaluation contract of models: evaluating a model must
not modify its inputs, and must not depend on previous evaluations.
"""

import copy

import gbkfit.model
import gbkfit.params
import numpy as np
import pytest


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
        'DModelSCube', (32, 32, 41), 'scube',
        dict(vsys=0, xpos=0, ypos=0, posa=30, incl=45,
             bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=100, dpt_a=10)),
    intensity_2d=(
        dict(type='intensity_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'))]),
        'DModelImage', (32, 32), 'image',
        dict(xpos=0, ypos=0, posa=30, incl=45, bpt_a=1, bpt_s=4)))


@pytest.mark.parametrize('gmodel_type', GMODELS_2D)
def test_gmodel_shared_by_grids_with_different_steps(driver, gmodel_type):
    # A gmodel evaluated on a grid must not keep using the step of the
    # grid it was evaluated on before
    from gbkfit.model import Model, dmodels, gmodel_parser
    gmodel_info, dmodel_name, size, key, properties = GMODELS_2D[gmodel_type]

    def evaluate(gmodel, step):
        steps = (step, step, 10)[:len(size)]
        dmodel = getattr(dmodels, dmodel_name)(size=size, step=steps)
        model_group = gbkfit.model.ModelGroup(
            [Model(driver, dmodel, gmodel)])
        params = gbkfit.params.EvaluationParams(
            model_group.pdescs(), properties)
        return model_group.model_h(params.evaluate())[0][key]['d'].copy()

    shared = gmodel_parser.load(gmodel_info)
    evaluate(shared, step=1)
    np.testing.assert_array_equal(
        evaluate(shared, step=0.5),
        evaluate(gmodel_parser.load(gmodel_info), step=0.5))
