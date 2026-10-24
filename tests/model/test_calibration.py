"""
Tests for the flux scale 'one' of observations: a parameter that scales
the flux of the model of an observation (e.g. its calibration).
"""

import gbkfit.params
import numpy as np
import pytest
from modelutils import config_group

from gbkfit.dataset import Data, DatasetPixelMoments
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    LikelihoodGaussian, ObservationGroup, ObservationSky, PixelMoments,
    line_of_sight_parser)
from gbkfit.utils.parseutils import ConfigError


GAS = dict(
    type='smdisk', name='gas', rnodes=list(range(0, 12)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='uniform', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))


def config(driver, observables, flux_scale):
    """
    A configuration of observations, of the given observables, of a thin
    disk 'gas' of a flat rotation curve, named 'first' and 'second'; the
    first with the given flux scale option.
    """
    observations = [
        dict(type='sky', observable=observable,
             line_of_sight=dict(type='flat'), name=name)
        for observable, name in zip(observables, ['first', 'second'])]
    observations[0]['flux_scale'] = flux_scale
    return dict(
        driver=dict(type=driver.type()),
        galaxies=[dict(components=[GAS])],
        observations=observations)


PROPERTIES = dict(
    gas_vsys=0, gas_xpos=0, gas_ypos=0, gas_posa=30, gas_incl=60,
    gas_pt_a=1, gas_pt_s=4, gas_vpt_vt=100, gas_dpt_a=20)


OBSERVABLES = [
    dict(type='pixel_moments', size=[24, 24], orders=[0, 1],
         spec_convention='relativistic'),
    dict(type='pixel_spectra', size=[24, 24, 21], step=[1, 1, 20])]


def evaluate(driver, flux_scale):
    """
    The moments 0 and 1 of the first observation (see config) and the
    spectra of the second, of a flux scale of 2 or none.
    """
    group = config_group(config(driver, OBSERVABLES, flux_scale))
    properties = PROPERTIES | (
        dict(first_flux_scale=2) if flux_scale else {})
    params = gbkfit.params.ParamSpace(group.pdescs(), properties).evaluate()
    data = group.model_h(params)
    return (
        data[0]['moment0']['d'].copy(), data[0]['moment1']['d'].copy(),
        data[1]['spectra']['d'].copy())


def test_a_flux_scale_scales_the_flux(driver):
    # Moment 0 of a scaled observation is times its scale; the velocities
    # (moment 1) and the other observations are not
    moment0, moment1, spectra = evaluate(driver, None)
    scaled = evaluate(driver, 'one')
    np.testing.assert_allclose(scaled[0], 2 * moment0, rtol=1e-6)
    np.testing.assert_allclose(scaled[1], moment1, rtol=1e-5, atol=1e-3)
    np.testing.assert_array_equal(scaled[2], spectra)


def test_a_solved_flux_scale_is_that_of_the_data(driver):
    # Data of moment 0 twice the model, and of its moment 1: the solved
    # scale is 2 (but for the pull of its prior), of moment 0 only
    def group(moment0, moment1):
        dataset = DatasetPixelMoments({
            0: Data(moment0, error=np.full_like(moment0, 1e-3)),
            1: Data(moment1, error=np.full_like(moment1, 1.0))})
        return ObservationGroup(
            [galaxy_parser.load(dict(components=[GAS]))], [ObservationSky(
                PixelMoments.from_data(
                    dataset, spec_size=21, spec_step=20,
                    spec_convention='relativistic'),
                line_of_sight_parser.load(dict(type='flat')), driver,
                data=dataset, flux_scale='one', name='first',
                likelihood=LikelihoodGaussian(missing_residual=3))])
    blank = np.zeros((24, 24))
    model = group(blank, blank).model_h(gbkfit.params.ParamSpace(
        group(blank, blank).pdescs(),
        PROPERTIES | dict(first_flux_scale=1)).evaluate())[0]
    moment0, moment1 = model['moment0']['d'], model['moment1']['d']
    scaled = group(2 * moment0, np.nan_to_num(moment1))
    params = gbkfit.params.ParamSpace(scaled.pdescs(), PROPERTIES | dict(
        first_flux_scale=dict(solve=dict(
            type='gaussian', mean=1, sigma=0.5, minimum=0))))
    objective = Objective(scaled, params.solves())
    objective.chi2(params.evaluate())
    assert objective.solved()['first_flux_scale'] \
        == pytest.approx(2, rel=1e-6)


def test_a_flux_scale_needs_a_name(driver):
    info = config(
        driver, [dict(type='pixel_spectra', size=[8, 8, 10])], 'one')
    del info['observations'][0]['name']
    with pytest.raises(ConfigError, match="a flux scale needs a name"):
        config_group(info)
