"""
Tests for the constants of the galaxies, which parameter expressions can
use: the radial nodes and subnodes of each disk.
"""

import numpy as np
from modelutils import observation_group


RNODES = list(range(0, 21, 2))

PROPERTIES = dict(
    vsys=0, xpos=0, ypos=0, posa=30, incl=45, pt_a=1, pt_s=4, dpt_a=10)

EXPONENTIAL = dict(type='exponential')
SECH2 = dict(type='sech2', zmax=8)


def disk(vptraits, density=None, **options):
    """
    A smooth disk that emits lines, with the given velocity traits and
    density (thin by default).
    """
    if density is None:
        density = dict(ptraits=EXPONENTIAL)
    return dict(
        type='smdisk', loose=False, tilted=False, rnodes=RNODES,
        density=density,
        kinematics=dict(vptraits=vptraits, dptraits=dict(type='uniform')),
        emission=dict(type='lines')) | options


def spectral_cube(components, line_of_sight):
    return dict(
        driver=dict(type='host'),
        observable=dict(
            type='pixel_spectra', size=[32, 32, 41], step=[1, 1, 10]),
        line_of_sight=line_of_sight,
        galaxy=dict(components=components))


def test_rotation_curve_from_radial_nodes(evaluate_cases):
    # A node-wise rotation curve given by an expression of the radial
    # nodes is the same as the one given by its values
    case = spectral_cube(
        [disk(dict(type='nw_tan_uniform'))], dict(type='flat'))
    curve = 200 * np.arctan(np.array(RNODES) / 3)
    expression, _ = evaluate_cases(
        [case], PROPERTIES | dict(vpt_vt='200 * np.arctan(rnodes / 3)'))
    values, _ = evaluate_cases(
        [case], PROPERTIES | dict(vpt_vt=curve.tolist()))
    np.testing.assert_array_equal(
        expression[0]['spectra']['d'], values[0]['spectra']['d'])


def test_names_of_constants():
    # The constants are named like the parameters: prefixed by the name
    # of their component, when its galaxy has several (dust too), and by
    # that of their galaxy, when there are several
    vptraits = dict(type='tan_arctan')
    thick = dict(ptraits=EXPONENTIAL, htraits=SECH2)
    galaxy0 = spectral_cube(
        [disk(vptraits, thick, name='gas'),
         disk(vptraits, thick, name='ring', rnodes=[0, 5, 10]),
         dict(type='smdisk', name='dust', loose=False, tilted=False,
              rnodes=[0, 1], density=thick, opacity=dict(type='dust'))],
        dict(type='grid')) | dict(name='g0')
    galaxy1 = spectral_cube(
        [disk(vptraits, rnodes=[0, 3, 6])], dict(type='flat')) \
        | dict(name='g1')
    constants = observation_group([galaxy0, galaxy1]).constants()
    assert list(constants) == [
        'g0_gas_rnodes', 'g0_gas_subrnodes', 'g0_ring_rnodes',
        'g0_ring_subrnodes', 'g0_dust_rnodes', 'g0_dust_subrnodes',
        'g1_rnodes', 'g1_subrnodes']
    assert constants['g0_gas_rnodes'] == tuple(RNODES)
    assert constants['g0_ring_rnodes'] == (0, 5, 10)
    assert constants['g0_dust_rnodes'] == (0, 1)
    assert constants['g1_rnodes'] == (0, 3, 6)
