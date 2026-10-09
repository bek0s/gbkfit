"""
Tests for the constants of the models, which parameter expressions can
use: the radial nodes and subnodes of each disk.
"""

import numpy as np
from modelutils import observation_group



RNODES = list(range(0, 21, 2))

PROPERTIES = dict(
    vsys=0, xpos=0, ypos=0, posa=30, incl=45, bpt_a=1, bpt_s=4, dpt_a=10)


def disk(**options):
    return dict(
        type='smdisk', loose=False, tilted=False, rnodes=RNODES,
        bptraits=dict(type='exponential'), dptraits=dict(type='uniform'),
    ) | options


def kinematics_2d(components):
    return dict(
        driver=dict(type='host'),
        dmodel=dict(type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
        gmodel=dict(type='kinematics_2d', components=components))


def test_rotation_curve_from_radial_nodes(evaluate_models):
    # A node-wise rotation curve given by an expression of the radial
    # nodes is the same as the one given by its values
    model = kinematics_2d([disk(vptraits=dict(type='nw_tan_uniform'))])
    curve = 200 * np.arctan(np.array(RNODES) / 3)
    expression, _ = evaluate_models(
        [model], PROPERTIES | dict(vpt_vt='200 * np.arctan(rnodes / 3)'))
    values, _ = evaluate_models(
        [model], PROPERTIES | dict(vpt_vt=curve.tolist()))
    np.testing.assert_array_equal(
        expression[0]['spectra']['d'], values[0]['spectra']['d'])


def test_names_of_constants():
    # The constants are named like the parameters of their components,
    # opacity components and gmodels (by name, when there are several)
    vptraits = dict(type='tan_arctan')
    model0 = dict(
        name='g0',
        driver=dict(type='host'),
        dmodel=dict(type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
        gmodel=dict(
            type='kinematics_3d',
            components=[
                disk(vptraits=vptraits, bhtraits=dict(type='sech2')),
                disk(vptraits=vptraits, bhtraits=dict(type='sech2'),
                     rnodes=[0, 5, 10])],
            opacity_components=[dict(
                type='smdisk', loose=False, tilted=False, rnodes=[0, 1],
                optraits=dict(type='exponential'),
                ohtraits=dict(type='sech2'))]))
    model1 = kinematics_2d([disk(vptraits=vptraits, rnodes=[0, 3, 6])]) \
        | dict(name='g1')
    constants = observation_group([model0, model1]).constants()
    assert list(constants) == [
        'g0_rnodes', 'g0_subrnodes', 'g0_cmp1_rnodes', 'g0_cmp1_subrnodes',
        'g0_ocmp_rnodes', 'g0_ocmp_subrnodes', 'g1_rnodes', 'g1_subrnodes']
    assert constants['g0_rnodes'] == tuple(RNODES)
    assert constants['g0_cmp1_rnodes'] == (0, 5, 10)
    assert constants['g0_ocmp_rnodes'] == (0, 1)
    assert constants['g1_rnodes'] == (0, 3, 6)
