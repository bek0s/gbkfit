"""
Tests for the sampling of the node-wise parameters of traits: at the
radial nodes of their disk (the default), which interpolates them to the
subnodes, or at the subnodes themselves ('subrings'), so that an
expression of the subnodes ('subrnodes') is not limited by the nodes.
"""

import numpy as np
import pytest

import gbkfit.model
from gbkfit.math import interpolation
from gbkfit.model import gmodel_parser
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.params.space import ParamSpace


PROPERTIES = dict(
    vsys=0, xpos=0, ypos=0, posa=30, incl=45, bpt_a=1, bpt_s=4, dpt_a=10)


def disk(type_, rnodes, vptraits, **options):
    disk_options = dict(cflux=1e-3, bhtraits=dict(type='sech2')) \
        if type_ == 'mcdisk' else {}
    return dict(
        type=type_, loose=False, tilted=False, rnodes=rnodes,
        bptraits=dict(type='exponential'), vptraits=vptraits,
        dptraits=dict(type='uniform'), **disk_options, **options)


def model(component):
    gmodel_type = 'kinematics_3d' if component['type'] == 'mcdisk' \
        else 'kinematics_2d'
    return dict(
        driver=dict(type='host'),
        dmodel=dict(type='scube', size=[32, 32, 41], step=[1, 1, 10]),
        gmodel=dict(type=gmodel_type, components=[component]))


def group(component):
    return gbkfit.model.ModelGroup(
        gbkfit.model.model_parser.load([model(component)]))


@pytest.mark.parametrize('type_', ['smdisk', 'mcdisk'])
def test_values_at_the_subnodes_are_used_as_they_are(evaluate_models, type_):
    # A curve given at the nodes gives the same model as its values
    # interpolated to the subnodes, given at the subnodes
    rnodes = [0, 4, 10, 20]
    curve = [0, 150, 180, 170]
    at_rnodes = disk(type_, rnodes, dict(type='nw_tan_uniform'))
    at_subnodes = disk(
        type_, rnodes, dict(type='nw_tan_uniform', sampling='subrings'))
    subrnodes = group(at_subnodes).constants()['subrnodes']
    interpolated = interpolation.InterpolatorLinear(rnodes, curve)(subrnodes)
    expected, _ = evaluate_models(
        [model(at_rnodes)], PROPERTIES | dict(bht_s=1, vpt_vt=curve))
    actual, _ = evaluate_models(
        [model(at_subnodes)],
        PROPERTIES | dict(bht_s=1, vpt_vt=interpolated.tolist()))
    # The host adds the clouds of a thick disk in any order (float32)
    rtol = 1e-4 if type_ == 'mcdisk' else 0
    np.testing.assert_allclose(
        actual[0]['scube']['d'], expected[0]['scube']['d'], rtol=rtol)


def test_node_wise_parameters_have_a_value_for_each_subnode():
    component = disk(
        'smdisk', [0, 20], dict(type='nw_tan_harmonic', order=1,
                                sampling='subrings'),
        rstep=0.5)
    model_group = group(component)
    nsubrnodes = len(model_group.constants()['subrnodes'])
    # The edges of the disk and the centres of 40 rings
    assert nsubrnodes == 42
    assert model_group.pdescs()['vpt_a'].size() == nsubrnodes
    assert model_group.pdescs()['vpt_p'].size() == nsubrnodes


def test_flaring_at_the_subnodes():
    gmodel = gmodel_parser.load(dict(
        type='intensity_3d', components=[dict(
            type='smdisk', loose=False, tilted=False, rnodes=[0, 20],
            rstep=0.5, bptraits=dict(type='exponential'),
            bhtraits=dict(type='sech2', rnodes=True, sampling='subrings'))]))
    assert gmodel.pdescs()['bht_s'].size() == 42


def evaluate_curve(component, free):
    # Evaluate the velocity field of the component with vmax and rt free,
    # as a fitter does
    model_group = group(component)
    pdescs = model_group.pdescs() | dict(
        vmax=ParamScalarDesc('vmax'), rt=ParamScalarDesc('rt'))
    space = ParamSpace(pdescs, PROPERTIES | dict(
        vmax=dict(val=100), rt=dict(val=1),
        vpt_vt='vmax * 2 / np.pi * np.arctan(subrnodes / rt)'),
        constants=model_group.constants())
    extra = {}
    model_group.model_h(space.evaluate(free), extra)
    return extra['model0_gmodel_component0_vdata'].data


@pytest.mark.parametrize('vmax, rt', [(150, 2), (220, 4)])
def test_an_expression_of_the_subnodes_follows_the_curve(
        evaluate_models, vmax, rt):
    # With two nodes, a curve given at the subnodes is the analytical one,
    # up to its linear interpolation between the subnodes
    rnodes = [0, 20]
    analytical = disk(
        'smdisk', rnodes, dict(type='tan_arctan'), rstep=0.1)
    at_subnodes = disk(
        'smdisk', rnodes, dict(type='nw_tan_uniform', sampling='subrings'),
        rstep=0.1)
    _, extra = evaluate_models(
        [model(analytical)], PROPERTIES | dict(vpt_vt=vmax, vpt_rt=rt))
    expected = extra['model0_gmodel_component0_vdata'].data
    actual = evaluate_curve(at_subnodes, dict(vmax=vmax, rt=rt))
    np.testing.assert_allclose(actual, expected, atol=0.1, equal_nan=True)


def test_sampling_is_dumped():
    gmodel = gmodel_parser.load(dict(
        type='kinematics_2d', components=[disk(
            'smdisk', [0, 20],
            dict(type='nw_tan_uniform', sampling='subrings'))]))
    info = gmodel_parser.dump(gmodel)
    assert info['components'][0]['vptraits'] == [dict(
        type='nw_tan_uniform', sampling='subrings')]
    gmodel = gmodel_parser.load(info)
    assert gmodel_parser.dump(gmodel) == info
    # The default is not dumped
    gmodel = gmodel_parser.load(dict(
        type='kinematics_2d', components=[disk(
            'smdisk', [0, 20], dict(type='nw_tan_uniform'))]))
    info = gmodel_parser.dump(gmodel)
    assert info['components'][0]['vptraits'] == [dict(type='nw_tan_uniform')]


@pytest.mark.parametrize('traits, message', [
    (dict(vptraits=dict(type='nw_tan_uniform', sampling='nodes')),
     r"sampling must be one of \['rnodes', 'subrings'\]; it is 'nodes'"),
    (dict(bhtraits=dict(type='sech2', sampling='subrings')),
     "sampling is 'subrings', but rnodes is False: the trait has no "
     "node-wise parameters")])
def test_invalid_sampling_is_rejected(traits, message):
    component = disk(
        'smdisk', [0, 20], dict(type='tan_arctan'),
        bhtraits=dict(type='sech2')) | traits
    with pytest.raises(Exception, match=message):
        gmodel_parser.load(dict(type='kinematics_3d', components=[component]))
