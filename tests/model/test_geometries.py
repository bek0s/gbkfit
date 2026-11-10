"""
Tests for the geometries of components: each has its own, and those that
share one tie their parameters.
"""

import numpy as np
import pytest

from gbkfit.config import ConfigError
from gbkfit.model import galaxy_parser


RNODES = [0, 2, 4, 6, 8, 10]

LINES = dict(type='lines')

PARTS = dict(
    rptraits=dict(type='exponential'),
    vptraits=dict(type='arctan', direction='tan'),
    dptraits=dict(type='uniform'),
    emission=LINES)

OBSERVABLE = dict(
    type='pixel_spectra', size=[24, 24], step=[1, 1],
    spectral=dict(size=31, step=10, convention='relativistic'))

# The properties of a warped disk, and of the traits of two disks
GEOMETRY = dict(
    vsys=10, xpos=0.5, ypos=-0.5, posa=[30, 32, 34, 36, 38, 40],
    incl=[60, 58, 56, 54, 52, 50])
TRAIT_PROPERTIES = dict(
    hi_rpt_a=1, hi_rpt_s=4, hi_vpt_rt=2, hi_vpt_vt=150, hi_dpt_a=20,
    ha_rpt_a=2, ha_rpt_s=2, ha_vpt_rt=3, ha_vpt_vt=120, ha_dpt_a=40)


def disk(name, **options):
    return dict(type='thin_disk', name=name, **PARTS, **options)


def test_a_tied_geometry_is_the_geometry_it_is_tied_to(
        driver, evaluate_cases):
    # Two warped disks of the same geometric values, and the same disks
    # with the geometry of the second tied to that of the first (vectors
    # included), give the same spectra
    galaxy = dict(components=[
        disk('hi', loose=False, tilted=True, rnodes=RNODES),
        disk('ha', loose=False, tilted=True, rnodes=RNODES)])
    case = dict(
        driver=dict(type=driver.type()), observable=OBSERVABLE,
        line_of_sight=dict(type='flat'), galaxy=galaxy)
    data_given, _ = evaluate_cases([case], TRAIT_PROPERTIES | {
        f'{cmp}_{name}': value for name, value in GEOMETRY.items()
        for cmp in ('hi', 'ha')})
    data_tied, _ = evaluate_cases([case], TRAIT_PROPERTIES | {
        f'hi_{name}': value for name, value in GEOMETRY.items()} | {
        f'ha_{name}': f'hi_{name}' for name in GEOMETRY})
    np.testing.assert_allclose(
        data_given[0]['spectra']['d'], data_tied[0]['spectra']['d'],
        rtol=1e-6, atol=1e-9)


# A gas disk, a biconical outflow along its axis and a nuclear point
GAS = dict(type='thin_disk', name='gas', rnodes=RNODES, **PARTS)
OUTFLOW = dict(
    type='cone', name='outflow', rnodes=[0, 1, 3, 6],
    rptraits=dict(type='exponential'),
    rhtraits=dict(type='gauss', zmax=1),
    vptraits=dict(type='uniform', direction='out'),
    dptraits=dict(type='uniform'),
    emission=LINES)
NUCLEUS = dict(type='point', name='nucleus', emission=LINES)


def test_each_component_has_its_own_geometric_parameters():
    # A point has a centre and a systemic velocity, and no plane
    galaxy = galaxy_parser.load(dict(components=[GAS, OUTFLOW, NUCLEUS]))
    pdescs = set(galaxy.pdescs())
    geometry = {'xpos', 'ypos', 'vsys', 'posa', 'incl'}
    assert {f'{cmp}_{name}' for cmp in ('gas', 'outflow')
            for name in geometry} <= pdescs
    assert {'nucleus_xpos', 'nucleus_ypos', 'nucleus_vsys'} <= pdescs
    assert not geometry & pdescs
    assert not {'nucleus_posa', 'nucleus_incl'} & pdescs
    # One component without a name has them plainly
    alone = galaxy_parser.load(dict(components=[
        {key: value for key, value in GAS.items() if key != 'name'}]))
    assert geometry <= set(alone.pdescs())


def test_a_warped_disk_has_its_warp_on_its_rings():
    galaxy = galaxy_parser.load(dict(components=[
        GAS | dict(tilted=True), NUCLEUS]))
    assert galaxy.pdescs()['gas_posa'].size() == len(RNODES)
    assert galaxy.pdescs()['gas_xpos'].size() == 1
    assert galaxy.constants()['gas_rnodes'] == tuple(RNODES)
    dumped = galaxy_parser.dump(galaxy)['components'][0]
    assert dumped['loose'] is False and dumped['tilted'] is True


@pytest.mark.parametrize('galaxy, unknown', [
    (dict(geometry=dict(tilted=True), components=[GAS]), 'geometry'),
    (dict(geometries=[dict(name='disk')], components=[GAS]), 'geometries'),
    (dict(components=[GAS | dict(geometry='own')]), 'geometry')])
def test_shared_geometries_are_gone(galaxy, unknown):
    # A configuration of version 3 is migrated (see v4.OwnGeometries)
    with pytest.raises(ConfigError, match=f"unknown options.*{unknown}"):
        galaxy_parser.load(galaxy)
