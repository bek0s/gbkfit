"""
Tests for the spatial weights of the data of observations: functions of
the azimuth of each pixel in the plane of a disk.
"""

import numpy as np
import pytest
from modelutils import config_group_plan

from gbkfit.model import Extras
from gbkfit.observation import WeightAzimuthPower, WeightAzimuthRange
from gbkfit.utils.parseutils import ConfigError


def test_weight_functions():
    theta = np.radians([0, 45, 80, 90, 100, 180, 270])
    np.testing.assert_allclose(
        WeightAzimuthPower(2).weight(theta), np.cos(theta) ** 2, atol=1e-15)
    np.testing.assert_allclose(
        WeightAzimuthPower(1, axis='major').weight(theta),
        np.abs(np.sin(theta)), atol=1e-15)
    # No weight within 20 degrees of the minor axis, on either side
    np.testing.assert_array_equal(
        WeightAzimuthRange(20).weight(theta), [1, 1, 0, 0, 0, 1, 0])


def config(driver, observable, weights, component=None):
    """
    A configuration of a spectral cube of a thin disk 'gas' of a flat
    rotation curve, with the given weights.
    """
    gas = component or dict(
        type='disk', name='gas', rnodes=list(range(0, 12)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=dict(type='uniform', direction='tan'),
            dptraits=dict(type='uniform')),
        emission=dict(type='lines'))
    return dict(
        driver=dict(type=driver.type()),
        galaxies=[dict(components=[gas])],
        observations=[dict(
            type='sky',
            observable=observable, line_of_sight=dict(type='flat'),
            weights=weights)])


PROPERTIES = dict(
    vsys=0, xpos=1, ypos=-2, posa=30, incl=60,
    gas_pt_a=1, gas_pt_s=4, gas_vpt_vt=100, gas_dpt_a=20)


def test_weights_follow_the_plane_of_the_disk(driver):
    # Weights |cos(theta)| of the azimuth in the plane of the disk: those of
    # the line-of-sight velocity of a flat rotation curve, v sin(incl)
    # |cos(theta)|, as the disk kernels make it
    import gbkfit.params
    plan = config_group_plan(config(
        driver,
        dict(type='pixel_spectra', size=[24, 24, 21], step=[1, 1, 20]),
        dict(component='gas', traits=dict(type='azimuth_power', power=1))))
    params = gbkfit.params.ParamSpace(
        plan.group().pdescs(), PROPERTIES).evaluate()
    extra = Extras()
    weights = plan.model_h(params, extra)[0]['spectra']['w']
    velocity = extra.get('vdata', component='gas').data
    disk = np.isfinite(velocity)
    assert disk.sum() > 100
    np.testing.assert_allclose(
        weights[0][disk],
        np.abs(velocity[disk]) / (100 * np.sin(np.radians(60))),
        atol=1e-5)
    # The same in every channel
    np.testing.assert_array_equal(weights, weights[:1].repeat(21, axis=0))


@pytest.mark.parametrize('observable, component, message', [
    (dict(type='pixel_moments', size=[24, 24], method='gaussian_fit',
          spec_convention='relativistic'), None,
     "gaussian_fit does not support weights"),
    (dict(type='pixel_spectra', size=[24, 24, 21]),
     dict(type='point', name='gas', emission=dict(type='lines')),
     "weights need the plane of a disk")])
def test_weights_are_checked(driver, observable, component, message):
    weights = dict(
        component='gas', traits=dict(type='azimuth_range', half_width=20))
    with pytest.raises(ConfigError, match=message):
        config_group_plan(config(driver, observable, weights, component))
