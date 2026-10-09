"""
Tests for the height traits: of velocity and dispersion (e.g. lags), and
the Moffat profile of brightness; and for the radial range selection
trait (e.g. truncations).
"""

import copy

import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group


DISK = dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    bptraits=dict(type='exponential'),
    bhtraits=dict(type='sech2'),
    vptraits=dict(type='tan_arctan'),
    dptraits=dict(type='uniform'))

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=75,
    bpt_a=1, bpt_s=4, bht_s=1, vpt_rt=2, vpt_vt=150, dpt_a=20)

SCUBE = dict(type='scube', size=[32, 41, 51], step=[1, 1, 20])


def evaluate(driver, component, properties):
    group = observation_group([dict(
        driver=dict(type=driver.type()), dmodel=copy.deepcopy(SCUBE),
        gmodel=dict(type='kinematics_3d', components=[component]))])
    params = gbkfit.params.EvaluationParams(group.pdescs(), properties)
    return group.model_h(params.evaluate())[0]['scube']['d'].copy()


def with_second(key, polar, height, polar_values, height_values):
    """
    The disk with a second velocity or dispersion trait (key 'v' or 'd')
    and its height trait, and the properties of the parameters (those of
    the second traits are prefixed vpt1_ and vht1_, or dpt1_ and dht1_).
    """
    component = DISK | {
        f'{key}ptraits': [DISK[f'{key}ptraits'], polar],
        f'{key}htraits': [dict(type='one'), height]}
    return component, PROPERTIES | polar_values | height_values


@pytest.mark.parametrize('key, polar, values', [
    ('v', dict(type='tan_uniform'), dict(vpt1_vt=-30)),
    ('d', dict(type='uniform'), dict(dpt1_a=15))])
def test_linear_height_trait_above_the_disk_changes_nothing(
        driver, key, polar, values):
    # A lag (or a dispersion gradient) that starts above the disk (its
    # sech2 profile is truncated at a few scale heights) changes nothing
    plain = evaluate(driver, DISK, PROPERTIES)
    component, properties = with_second(
        key, polar, dict(type='linear'), values, {f'{key}ht1_z0': 1e6})
    np.testing.assert_allclose(
        evaluate(driver, component, properties), plain, rtol=1e-5,
        atol=1e-7 * plain.max())


@pytest.mark.parametrize('key, polar, values', [
    ('v', dict(type='tan_uniform'), dict(vpt1_vt=-30)),
    ('d', dict(type='uniform'), dict(dpt1_a=15))])
def test_linear_height_trait_changes_the_model(driver, key, polar, values):
    # A lag from the mid-plane changes the model, and is equal to the
    # same lag from a height of 0 given node-wise
    plain = evaluate(driver, DISK, PROPERTIES)
    component, properties = with_second(
        key, polar, dict(type='linear'), values, {f'{key}ht1_z0': 0})
    lagged = evaluate(driver, component, properties)
    assert np.abs(lagged - plain).max() > 1e-3 * plain.max()
    component, properties = with_second(
        key, polar, dict(type='linear', rnodes=True), values,
        {f'{key}ht1_z0': [0] * len(DISK['rnodes'])})
    np.testing.assert_allclose(
        evaluate(driver, component, properties), lagged, rtol=1e-5,
        atol=1e-7 * plain.max())
    # A flux-conserving change: the total flux is the same
    np.testing.assert_allclose(lagged.sum(), plain.sum(), rtol=1e-4)


@pytest.mark.parametrize('trait', ['exponential', 'gauss'])
def test_falling_height_traits_of_a_large_scale_are_one(driver, trait):
    plain = evaluate(driver, DISK, PROPERTIES)
    component = DISK | dict(vhtraits=dict(type=trait))
    properties = PROPERTIES | dict(vht_s=1e8)
    np.testing.assert_allclose(
        evaluate(driver, component, properties), plain, rtol=1e-5,
        atol=1e-7 * plain.max())


@pytest.mark.parametrize('trunc', [0, 2], ids=['full', 'truncated'])
def test_moffat_height_of_shape_1_is_lorentz(driver, trunc):
    # (1 + (z / s)^2)^-1 is the Lorentz profile: the same pdf, and the
    # same cdf, which normalises a truncated profile
    lorentz = evaluate(
        driver, DISK | dict(bhtraits=dict(type='lorentz', trunc=trunc)),
        PROPERTIES)
    moffat = evaluate(
        driver, DISK | dict(bhtraits=dict(type='moffat', trunc=trunc)),
        PROPERTIES | dict(bht_b=1))
    np.testing.assert_allclose(
        moffat, lorentz, rtol=1e-5, atol=1e-6 * lorentz.max())


def test_radial_range_truncates_the_disk(driver):
    plain = evaluate(driver, DISK, PROPERTIES)
    component = DISK | dict(sptraits=dict(type='rrange'))
    # A range that covers the disk changes nothing
    covering = evaluate(
        driver, component, PROPERTIES | dict(spt_rmin=0, spt_rmax=1e6))
    np.testing.assert_allclose(
        covering, plain, rtol=1e-5, atol=1e-7 * plain.max())
    # An inner hole and an outer edge remove light
    ring = evaluate(
        driver, component, PROPERTIES | dict(spt_rmin=2, spt_rmax=6))
    assert 0.2 * plain.sum() < ring.sum() < 0.9 * plain.sum()
    inner = evaluate(
        driver, component, PROPERTIES | dict(spt_rmin=0, spt_rmax=2))
    outer = evaluate(
        driver, component, PROPERTIES | dict(spt_rmin=6, spt_rmax=1e6))
    np.testing.assert_allclose(
        inner + ring + outer, plain, rtol=1e-5, atol=1e-7 * plain.max())


@pytest.mark.parametrize('height, values, factor', [
    (dict(type='linear'), dict(dht1_z0=1),
     lambda z: np.maximum(0, np.abs(z) - 1)),
    (dict(type='exponential'), dict(dht1_s=2),
     lambda z: np.exp(-np.abs(z) / 2)),
    (dict(type='gauss'), dict(dht1_s=2),
     lambda z: np.exp(-0.5 * (z / 2) ** 2))])
def test_height_traits_at_each_height(driver, height, values, factor):
    # A face-on thick disk, whose voxels are at the height of their z:
    # its dispersion is 20, plus 5 times the factor of the height trait
    # (the velocity height traits have the same factors)
    component, properties = with_second(
        'd', dict(type='uniform'), height, dict(dpt1_a=5), values)
    group = observation_group([dict(
        driver=dict(type=driver.type()), dmodel=copy.deepcopy(SCUBE),
        gmodel=dict(type='kinematics_3d', components=[component]))])
    params = gbkfit.params.EvaluationParams(
        group.pdescs(), properties | dict(incl=0))
    extra = {}
    group.model_h(params.evaluate(), extra)
    ddata = extra['observation0_gmodel_component0_ddata']
    coords = ddata.coords
    nz = ddata.data.shape[0]
    z = coords.rval[2] + (np.arange(nz) - coords.rpix[2]) * coords.step[2]
    # (the spaxel nearest the centre of the disk, at pixel (15.8, 19.4))
    centre = ddata.data[:, 19, 16]
    np.testing.assert_allclose(centre, 20 + 5 * factor(z), rtol=1e-5)
