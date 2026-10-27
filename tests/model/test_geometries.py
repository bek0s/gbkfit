"""
Tests for the geometries that components share: their parameters, and
their errors.
"""

import numpy as np
import pytest

from gbkfit.model import galaxy_parser


RNODES = [0, 2, 4, 6, 8, 10]

LINES = dict(type='lines')

PARTS = dict(
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        type='traits',
        vptraits=dict(type='arctan', direction='tan'), dptraits=dict(type='uniform')),
    emission=LINES)

OBSERVABLE = dict(
    type='pixel_spectra', size=[24, 24], step=[1, 1],
    spectral=dict(size=31, step=10, convention='relativistic'))

# The properties of a warped disk, and of the traits of two disks
GEOMETRY = dict(
    vsys=10, xpos=0.5, ypos=-0.5, posa=[30, 32, 34, 36, 38, 40],
    incl=[60, 58, 56, 54, 52, 50])
TRAIT_PROPERTIES = dict(
    hi_pt_a=1, hi_pt_s=4, hi_vpt_rt=2, hi_vpt_vt=150, hi_dpt_a=20,
    ha_pt_a=2, ha_pt_s=2, ha_vpt_rt=3, ha_vpt_vt=120, ha_dpt_a=40)


def disk(name, **options):
    return dict(type='disk', name=name, **PARTS, **options)


def point(name, geometry):
    return dict(type='point', name=name, geometry=geometry, emission=LINES)


def test_a_shared_geometry_is_a_tied_one(driver, evaluate_cases):
    # Two disks with a shared warp, and the same disks with the geometry
    # of the second tied to that of the first, give the same spectra
    shared = dict(geometries=[dict(
        name='disk', tilted=True, rnodes=RNODES)], components=[
        disk('hi', geometry='disk'), disk('ha', geometry='disk')])
    tied = dict(components=[
        disk('hi', loose=False, tilted=True, rnodes=RNODES),
        disk('ha', loose=False, tilted=True, rnodes=RNODES)])
    case = dict(
        driver=dict(type=driver.type()), observable=OBSERVABLE,
        line_of_sight=dict(type='flat'))
    data_shared, _ = evaluate_cases(
        [case | dict(galaxy=shared)], TRAIT_PROPERTIES | {
            f'disk_{name}': value for name, value in GEOMETRY.items()})
    data_tied, _ = evaluate_cases(
        [case | dict(galaxy=tied)], TRAIT_PROPERTIES | {
            f'hi_{name}': value for name, value in GEOMETRY.items()} | {
            f'ha_{name}': f'hi_{name}' for name in GEOMETRY})
    np.testing.assert_allclose(
        data_shared[0]['spectra']['d'], data_tied[0]['spectra']['d'],
        rtol=1e-6, atol=1e-9)


def test_a_geometry_shares_the_parameters_it_is_given():
    # A disk and a point share their centre and systemic velocity; the
    # disk keeps its own orientation
    galaxy = galaxy_parser.load(dict(
        geometries=[dict(name='centre', params=['xpos', 'ypos', 'vsys'])],
        components=[
            disk('gas', geometry='centre', rnodes=RNODES),
            point('nucleus', 'centre')]))
    pdescs = set(galaxy.pdescs())
    assert {'centre_xpos', 'centre_ypos', 'centre_vsys', 'gas_posa',
            'gas_incl', 'nucleus_flux', 'nucleus_disp'} <= pdescs
    assert not {'gas_xpos', 'gas_vsys', 'nucleus_xpos'} & pdescs
    # The dump names the geometry, and loads to the same galaxy
    info = galaxy_parser.dump(galaxy)
    assert info['components'][1]['geometry'] == 'centre'
    assert galaxy_parser.dump(galaxy_parser.load(info)) == info


WARP = dict(name='disk', tilted=True, rnodes=RNODES)


@pytest.mark.parametrize('geometries, components, message', [
    ([], [disk('hi', geometry='disk', rnodes=RNODES)],
     "unknown geometry 'disk'"),
    ([WARP, dict(name='other')], [disk('hi', geometry='disk')],
     "no component uses the geometries ['other']"),
    ([WARP, WARP], [disk('hi', geometry='disk')],
     "the geometries must have different names"),
    ([WARP], [disk('hi', geometry='disk', tilted=False)],
     "the warps of a component are those of the geometry 'disk', which "
     "differ"),
    ([WARP], [disk('hi', geometry='disk', rnodes=RNODES)],
     "the rings of a component are those of the geometry 'disk', which is "
     "warped"),
    ([dict(name='disk', loose=True, rnodes=RNODES)],
     [point('nucleus', 'disk')],
     "a point has one centre"),
    ([dict(name='centre', params=['posa'])],
     [point('nucleus', 'centre')],
     "has its parameters ['posa']"),
    ([dict(name='disk', rnodes=RNODES)], [disk('hi', geometry='disk')],
     "the rings of a geometry are those of its warps"),
    ([dict(WARP, name='hi')], [disk('hi', geometry='hi')],
     "the geometries and the components must have different names")])
def test_geometry_errors(geometries, components, message):
    with pytest.raises(Exception) as error:
        galaxy_parser.load(dict(geometries=geometries, components=components))
    assert message in str(error.value)


# A gas disk, a biconical outflow along its axis and a nuclear point
GAS = dict(type='disk', name='gas', rnodes=RNODES, **PARTS)
OUTFLOW = dict(
    type='cone', name='outflow', rnodes=[1, 3, 6],
    density=dict(
        ptraits=dict(type='exponential'),
        htraits=dict(type='gauss', zmax=1)),
    kinematics=dict(
        type='traits', vptraits=dict(type='uniform', direction='out'),
        dptraits=dict(type='uniform')),
    emission=LINES)
NUCLEUS = dict(type='point', name='nucleus', emission=LINES)


def test_the_components_share_the_geometry_of_their_galaxy():
    # (a point its centre and systemic velocity, which has no plane); one
    # of its own keeps its own parameters, named after it
    shared = galaxy_parser.load(dict(components=[GAS, OUTFLOW, NUCLEUS]))
    geometry = {'xpos', 'ypos', 'vsys', 'posa', 'incl'}
    assert geometry <= set(shared.pdescs())
    assert not any(name.endswith(('_xpos', '_posa', '_vsys'))
                   for name in shared.pdescs())
    own = galaxy_parser.load(dict(components=[
        GAS, OUTFLOW | dict(geometry='own'), NUCLEUS]))
    assert set(own.pdescs()) - set(shared.pdescs()) == {
        f'outflow_{name}' for name in geometry}
    assert galaxy_parser.dump(own)['components'][1]['geometry'] == 'own'
    assert 'geometry' not in galaxy_parser.dump(own)['components'][0]


def test_a_warped_galaxy_geometry_is_shared_by_its_disks():
    # (whose rings are its rings)
    warp = dict(tilted=True, rnodes=RNODES)
    gas = {key: value for key, value in GAS.items() if key != 'rnodes'}
    galaxy = galaxy_parser.load(dict(
        geometry=warp, components=[gas, NUCLEUS]))
    assert galaxy.pdescs()['posa'].size() == len(RNODES)
    assert galaxy.constants()['rnodes'] == tuple(RNODES)
    assert galaxy_parser.dump(galaxy)['geometry'] == warp | dict(loose=False)
    with pytest.raises(Exception, match="give it geometry 'own'"):
        galaxy_parser.load(dict(geometry=warp, components=[gas, OUTFLOW]))


def test_a_warped_galaxy_geometry_is_given_to_its_components():
    # (in Python: a component whose geometry is 'galaxy' does not know its
    # rings)
    from gbkfit.model import Galaxy, Geometry, component_parser
    warp = Geometry(tilted=True, rnodes=RNODES)
    disk = component_parser.load(dict(GAS))
    with pytest.raises(Exception, match="give it to its component 'gas'"):
        Galaxy(disk, geometry=warp)
