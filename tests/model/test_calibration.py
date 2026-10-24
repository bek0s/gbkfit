"""
Tests for the calibration of observations: a parameter that scales the
flux of the model of an observation.
"""

import gbkfit.params
import numpy as np
import pytest
from modelutils import config_group

from gbkfit.utils.parseutils import ConfigError


def config(driver, observables, calibration):
    """
    A configuration of observations, of the given observables, of a thin
    disk 'gas' of a flat rotation curve, named 'first' and 'second'; the
    first with the given calibration option.
    """
    gas = dict(
        type='smdisk', name='gas', rnodes=list(range(0, 12)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=dict(type='uniform', direction='tan'),
            dptraits=dict(type='uniform')),
        emission=dict(type='lines'))
    observations = [
        dict(type='sky', observable=observable,
             line_of_sight=dict(type='flat'), name=name)
        for observable, name in zip(observables, ['first', 'second'])]
    observations[0]['calibration'] = calibration
    return dict(
        driver=dict(type=driver.type()),
        galaxies=[dict(components=[gas])],
        observations=observations)


PROPERTIES = dict(
    gas_vsys=0, gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60,
    gas_pt_a=1, gas_pt_s=4, gas_vpt_vt=100, gas_dpt_a=20)


def test_calibration_scales_the_flux(driver):
    # The spectra and moment 0 of a calibrated observation are times its
    # calibration; the velocities (moment 1) and the other observations
    # are not
    observables = [
        dict(type='pixel_moments', size=[24, 24], orders=[0, 1],
             spec_convention='relativistic'),
        dict(type='pixel_spectra', size=[24, 24, 21], step=[1, 1, 20])]

    def evaluate(calibration):
        group = config_group(config(driver, observables, calibration))
        properties = PROPERTIES | (
            dict(first_calibration=2) if calibration else {})
        params = gbkfit.params.ParamSpace(
            group.pdescs(), properties).evaluate()
        data = group.model_h(params)
        return (
            data[0]['moment0']['d'].copy(), data[0]['moment1']['d'].copy(),
            data[1]['spectra']['d'].copy())
    moment0, moment1, spectra = evaluate(False)
    calibrated = evaluate(True)
    np.testing.assert_allclose(calibrated[0], 2 * moment0, rtol=1e-6)
    np.testing.assert_allclose(
        calibrated[1], moment1, rtol=1e-5, atol=1e-3)
    np.testing.assert_array_equal(calibrated[2], spectra)


def test_calibration_needs_a_name(driver):
    info = config(
        driver, [dict(type='pixel_spectra', size=[8, 8, 10])], True)
    del info['observations'][0]['name']
    with pytest.raises(ConfigError, match="a calibration needs a name"):
        config_group(info)
