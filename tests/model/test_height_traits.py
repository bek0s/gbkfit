"""
Tests for the height traits: of velocity and dispersion (e.g. lags), and
of density (the Moffat profile, and zmax, the height above the midplane
that holds the density); for the vertical distortion that lifts the
midplane; and for the radial range selection trait (e.g. truncations).
"""

import copy
import logging

import gbkfit.params
import numpy as np
import pytest
from gbkfit.utils.parseutils import ConfigError
from modelutils import observation_group


DISK = dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    density=dict(
        ptraits=dict(type='exponential'),
        htraits=dict(type='sech2', zmax=8)),
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))

PROPERTIES = dict(
    vsys=0, xpos=0.3, ypos=-0.6, posa=50, incl=75,
    pt_a=1, pt_s=4, ht_s=1, vpt_rt=2, vpt_vt=150, dpt_a=20)

SCUBE = dict(type='pixel_spectra', size=[32, 41, 51], step=[1, 1, 20])


def observe(driver, component):
    """A spectral cube of a galaxy of the component, on a grid."""
    return observation_group([dict(
        driver=dict(type=driver.type()), observable=copy.deepcopy(SCUBE),
        line_of_sight=dict(type='grid'),
        galaxy=dict(components=[component]))])


def evaluate(driver, component, properties):
    group = observe(driver, component)
    params = gbkfit.params.ParamSpace(group.pdescs(), properties)
    return group.model_h(params.evaluate())[0]['spectra']['d'].copy()


def with_density(**traits):
    """The disk with the given density traits instead of its own."""
    return DISK | dict(density=DISK['density'] | traits)


def with_kinematics(**traits):
    """The disk with the given kinematics traits instead of its own."""
    return DISK | dict(kinematics=DISK['kinematics'] | traits)


def with_second(key, polar, height, polar_values, height_values):
    """
    The disk with a second velocity or dispersion trait (key 'v' or 'd')
    and its height trait, and the properties of the parameters (those of
    the second traits are prefixed vpt1_ and vht1_, or dpt1_ and dht1_).
    """
    first = DISK['kinematics'][f'{key}ptraits']
    component = with_kinematics(**{
        f'{key}ptraits': [first, polar],
        f'{key}htraits': [dict(type='one'), height]})
    return component, PROPERTIES | polar_values | height_values


@pytest.mark.parametrize('key, polar, values', [
    ('v', dict(type='uniform', direction='tan'), dict(vpt1_vt=-30)),
    ('d', dict(type='uniform'), dict(dpt1_a=15))])
def test_linear_height_trait_above_the_disk_changes_nothing(
        driver, key, polar, values):
    # A lag (or a dispersion gradient) that starts above the disk (its
    # sech2 profile is truncated at zmax) changes nothing
    plain = evaluate(driver, DISK, PROPERTIES)
    component, properties = with_second(
        key, polar, dict(type='linear'), values, {f'{key}ht1_z0': 1e6})
    np.testing.assert_allclose(
        evaluate(driver, component, properties), plain, rtol=1e-5,
        atol=1e-7 * plain.max())


@pytest.mark.parametrize('key, polar, values', [
    ('v', dict(type='uniform', direction='tan'), dict(vpt1_vt=-30)),
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
    component = with_kinematics(vhtraits=dict(type=trait))
    properties = PROPERTIES | dict(vht_s=1e8)
    np.testing.assert_allclose(
        evaluate(driver, component, properties), plain, rtol=1e-5,
        atol=1e-7 * plain.max())


@pytest.mark.parametrize('zmax', [30, 2], ids=['wide', 'truncated'])
def test_moffat_height_of_shape_1_is_lorentz(driver, zmax):
    # (1 + (z / s)^2)^-1 is the Lorentz profile: the same pdf, and the
    # same cdf, which normalises the profile truncated at zmax
    lorentz = evaluate(
        driver, with_density(htraits=dict(type='lorentz', zmax=zmax)),
        PROPERTIES)
    moffat = evaluate(
        driver, with_density(htraits=dict(type='moffat', zmax=zmax)),
        PROPERTIES | dict(ht_b=1))
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



def test_selections_keep_what_every_trait_selects(driver):
    # A ring and the receding half of the disk select the half of the
    # ring: half the light of the ring of a symmetric disk
    component = DISK | dict(sptraits=[
        dict(type='rrange'), dict(type='azrange')])
    ring = dict(spt_rmin=2, spt_rmax=6)
    half = dict(spt1_p=0, spt1_s=180)
    half_ring = evaluate(driver, component, PROPERTIES | ring | half)
    whole_ring = evaluate(
        driver, component, PROPERTIES | ring | dict(spt1_p=0, spt1_s=360))
    assert half_ring.sum() == pytest.approx(0.5 * whole_ring.sum(), rel=0.05)

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
    group = observe(driver, component)
    params = gbkfit.params.ParamSpace(
        group.pdescs(), properties | dict(incl=0))
    extra = {}
    group.model_h(params.evaluate(), extra)
    ddata = extra['observation0_model_component0_ddata']
    coords = ddata.coords
    nz = ddata.data.shape[0]
    z = coords.rval[2] + (np.arange(nz) - coords.rpix[2]) * coords.step[2]
    # (the spaxel nearest the centre of the disk, at pixel (15.8, 19.4),
    # where it has density: within zmax of its plane)
    centre = ddata.data[:, 19, 16]
    within = np.abs(z) < DISK['density']['htraits']['zmax']
    np.testing.assert_allclose(
        centre[within], 20 + 5 * factor(z[within]), rtol=1e-5)


def observe_depth(driver, component, depth):
    """
    A spectral cube of a galaxy of the component, on a grid of the given
    depth axis (the options of Depth).
    """
    return observation_group([dict(
        driver=dict(type=driver.type()), observable=copy.deepcopy(SCUBE),
        line_of_sight=dict(type='grid', depth=depth),
        galaxy=dict(components=[component]))])


def evaluate_with(group, properties, out_extra=None):
    """The spectral cube of a group of one observation."""
    params = gbkfit.params.ParamSpace(group.pdescs(), properties)
    return group.model_h(
        params.evaluate(), out_extra)[0]['spectra']['d'].copy()


def flux(group, properties, out_extra=None):
    """The total of the spectral cube of a group of one observation."""
    return evaluate_with(group, properties, out_extra).sum()


def test_the_depth_axis_holds_the_density_up_to_zmax(driver, caplog):
    # A fat disk at an inclination of 45 degrees: its rim, up to zmax (8)
    # above its plane, reaches sqrt(R^2 + zmax^2) along the line of sight,
    # beyond R (13). The default depth axis holds it all: the flux of a
    # longer one (of voxels at the same depths); a shorter one loses some,
    # and a warning says so
    properties = PROPERTIES | dict(incl=45, ht_s=3, pt_s=8)
    default = flux(observe_depth(driver, DISK, {}), properties)
    longer = flux(observe_depth(driver, DISK, dict(extent=61)), properties)
    np.testing.assert_allclose(longer, default, rtol=1e-5)
    caplog.set_level(logging.WARNING)
    shorter = flux(observe_depth(driver, DISK, dict(extent=25)), properties)
    assert shorter < (1 - 1e-4) * default
    assert 'shorter than the disks need' in caplog.text


@pytest.mark.parametrize('scale, warns', [(1, False), (4, True)])
def test_a_profile_wide_for_its_zmax_warns(driver, caplog, scale, warns):
    # zmax (8) cuts 2e-7 of a sech2 profile of scale height 1, and 3.6% of
    # one of 4: zmax then shapes the density (with extra outputs)
    caplog.set_level(logging.WARNING)
    flux(observe(driver, DISK), PROPERTIES | dict(ht_s=scale), {})
    assert ('zmax shapes the density' in caplog.text) == warns


@pytest.mark.parametrize('trait, values', [
    ('uniform', {}), ('exponential', {}), ('gauss', {}),
    ('ggauss', dict(ht_b=1.5)), ('lorentz', {}), ('moffat', dict(ht_b=2.5)),
    ('sech2', {})])
def test_the_cdf_of_each_density_height_trait(driver, trait, values):
    # The cdf (with which the share of the profile within zmax is checked)
    # is the integral of the profile the kernels evaluate, which they
    # renormalise within zmax (here 40)
    from gbkfit.model import galaxy_parser
    galaxy = galaxy_parser.load(dict(components=[
        with_density(htraits=dict(type=trait, zmax=40))]))
    disk = galaxy.components()[0].density_disk()
    plan = disk.plan(driver, 0, np.dtype(np.float32))
    z = np.linspace(-40, 40, 160001)
    params = PROPERTIES | dict(ht_s=1.5) | values
    pdf = plan.evaluate_height_traits(params, 'rht', 5.0, z)[0]
    integral = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2)]) \
        * (z[1] - z[0])
    trait_ = disk.traits('rht')[0]
    args = [params['ht_s']] + ([params['ht_b']] if 'ht_b' in values else [])
    cdf = trait_.cdf(z, *args)
    expected = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    # (the trapezoids are off by about 2e-4 at the steps of uniform)
    np.testing.assert_allclose(integral, expected, atol=1e-3)


def mean_depth(bdata):
    """The mean depth of the light of the extra output of a disk."""
    coords = bdata.coords
    z = coords.rval[2] \
        + (np.arange(bdata.data.shape[0]) - coords.rpix[2]) * coords.step[2]
    light = bdata.data.sum(axis=(1, 2))
    return (z * light).sum() / light.sum()


@pytest.mark.parametrize('disk_type', ['smdisk', 'mcdisk'])
def test_a_lifted_midplane_carries_its_light(driver, caplog, disk_type):
    # A face-on disk whose midplane a vertical distortion lifts by 12:
    # beyond the zmax of its density (8, above the midplane) but within
    # its own (20). The light moves with the midplane, and the depth axis
    # holds it: each column keeps its light (exactly for the clouds; the
    # smooth disk samples its profile at the centres of the voxels)
    component = DISK | dict(
        type=disk_type, zptraits=dict(type='nw_uniform', zmax=20))
    if disk_type == 'mcdisk':
        component |= dict(cflux=1e-3)
    group = observe(driver, component)
    properties = PROPERTIES | dict(incl=0)
    flat = flux(group, properties | dict(zpt_a=[0] * 14))
    extra = {}
    lifted = flux(group, properties | dict(zpt_a=[12] * 14), extra)
    np.testing.assert_allclose(lifted, flat, rtol=1e-5)
    bdata = extra['observation0_model_component0_bdata']
    assert mean_depth(bdata) == pytest.approx(12, abs=0.05)
    # zmax is the valid range of the amplitude: a lift of 30 is not a
    # valid value. The kernels limit the displacement to zmax all the same
    # (e.g. where an interpolation of the amplitude overshoots it): given
    # past the parameters, a lift of 30 stops at 20, and a warning says so
    with pytest.raises(ConfigError, match="outside its valid range"):
        gbkfit.params.ParamSpace(
            group.pdescs(), properties | dict(zpt_a=[30] * 14))
    at_zmax = evaluate_with(group, properties | dict(zpt_a=[20] * 14))
    values = gbkfit.params.ParamSpace(
        group.pdescs(), properties | dict(zpt_a=[20] * 14)).evaluate()
    caplog.set_level(logging.WARNING)
    limited = group.model_h(
        values | dict(zpt_a=np.full(14, 30.0)), {})[0]['spectra']['d']
    assert 'zmax shapes the disk' in caplog.text
    np.testing.assert_allclose(
        limited, at_zmax, rtol=1e-5, atol=1e-7 * at_zmax.max())
