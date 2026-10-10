"""
Tests for the geometries that components share: their parameters, and
their errors.
"""

import numpy as np
import pytest

from gbkfit.model import model_parser


RNODES = [0, 2, 4, 6, 8, 10]

TRAITS = dict(
    bptraits=dict(type='exponential'), vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))

OBSERVABLE = dict(type='pixel_spectra', size=[24, 24, 31], step=[1, 1, 10])

# The properties of a warped disk, and of the traits of two disks
GEOMETRY = dict(
    vsys=10, xpos=0.5, ypos=-0.5, posa=[30, 32, 34, 36, 38, 40],
    incl=[60, 58, 56, 54, 52, 50])
TRAIT_PROPERTIES = dict(
    hi_bpt_a=1, hi_bpt_s=4, hi_vpt_rt=2, hi_vpt_vt=150, hi_dpt_a=20,
    ha_bpt_a=2, ha_bpt_s=2, ha_vpt_rt=3, ha_vpt_vt=120, ha_dpt_a=40)


def disk(name, **options):
    return dict(type='smdisk', name=name, **TRAITS, **options)


def test_a_shared_geometry_is_a_tied_one(driver, evaluate_cases):
    # Two disks with a shared warp, and the same disks with the geometry
    # of the second tied to that of the first, give the same spectra
    shared = dict(type='kinematics_2d', geometries=[dict(
        name='disk', tilted=True, rnodes=RNODES)], components=[
        disk('hi', geometry='disk'), disk('ha', geometry='disk')])
    tied = dict(type='kinematics_2d', components=[
        disk('hi', loose=False, tilted=True, rnodes=RNODES),
        disk('ha', loose=False, tilted=True, rnodes=RNODES)])
    case = dict(driver=dict(type=driver.type()), observable=OBSERVABLE)
    data_shared, _ = evaluate_cases(
        [case | dict(model=shared)], TRAIT_PROPERTIES | {
            f'disk_{name}': value for name, value in GEOMETRY.items()})
    data_tied, _ = evaluate_cases(
        [case | dict(model=tied)], TRAIT_PROPERTIES | {
            f'hi_{name}': value for name, value in GEOMETRY.items()} | {
            f'ha_{name}': f'hi_{name}' for name in GEOMETRY})
    np.testing.assert_allclose(
        data_shared[0]['spectra']['d'], data_tied[0]['spectra']['d'],
        rtol=1e-6, atol=1e-9)


def test_a_geometry_shares_the_parameters_it_is_given():
    # A disk and a point share their centre and systemic velocity; the
    # disk keeps its own orientation
    model = model_parser.load(dict(
        type='kinematics_2d',
        geometries=[dict(name='centre', params=['xpos', 'ypos', 'vsys'])],
        components=[
            disk('gas', geometry='centre', rnodes=RNODES),
            dict(type='point', name='nucleus', geometry='centre')]))
    pdescs = set(model.pdescs())
    assert {'centre_xpos', 'centre_ypos', 'centre_vsys', 'gas_posa',
            'gas_incl', 'nucleus_flux', 'nucleus_disp'} <= pdescs
    assert not {'gas_xpos', 'gas_vsys', 'nucleus_xpos'} & pdescs
    # The dump names the geometry, and loads to the same model
    info = model_parser.dump(model)
    assert info['components'][1]['geometry'] == 'centre'
    assert model_parser.dump(model_parser.load(info)) == info


WARP = dict(name='disk', tilted=True, rnodes=RNODES)


@pytest.mark.parametrize('geometries, components, message', [
    ([], [disk('hi', geometry='disk', rnodes=RNODES)],
     "unknown geometry 'disk'"),
    ([WARP, dict(name='other')], [disk('hi', geometry='disk')],
     "no component uses the geometries ['other']"),
    ([WARP, WARP], [disk('hi', geometry='disk')],
     "the geometries must have different names"),
    ([WARP], [disk('hi', geometry='disk', tilted=True)],
     "the warps of a component are those of its geometry 'disk'"),
    ([WARP], [disk('hi', geometry='disk', rnodes=RNODES)],
     "the rings of a component are those of its warped geometry"),
    ([dict(name='disk', loose=True, rnodes=RNODES)],
     [dict(type='point', name='nucleus', geometry='disk')],
     "a point has one centre"),
    ([dict(name='centre', params=['posa'])],
     [dict(type='point', name='nucleus', geometry='centre')],
     "has its parameters ['posa']"),
    ([dict(name='disk', rnodes=RNODES)], [disk('hi', geometry='disk')],
     "the rings of a geometry are those of its warps"),
    ([dict(WARP, name='hi')], [disk('hi', geometry='hi')],
     "the geometries and the components must have different names")])
def test_geometry_errors(geometries, components, message):
    with pytest.raises(Exception) as error:
        model_parser.load(dict(
            type='kinematics_2d', geometries=geometries,
            components=components))
    assert message in str(error.value)
