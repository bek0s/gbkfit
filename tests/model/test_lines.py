"""
Tests for the spectral lines of the spectral cubes: each channel holds
the mean of the line over the channel.
"""

import numpy as np
import pytest


def line_flux(driver, evaluate_models, dispersion):
    """
    The flux of the line of each spaxel (the sum over the channels times
    the channel width) of a rotating disk with the given dispersion.
    """
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='scube', size=[24, 24, 61], step=[1, 1, 10]),
        gmodel=dict(type='kinematics_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'),
            vptraits=dict(type='tan_arctan'),
            dptraits=dict(type='uniform'))]))
    # A systemic velocity between two channel centres
    properties = dict(
        vsys=3, xpos=0, ypos=0, posa=30, incl=45,
        bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=150, dpt_a=dispersion)
    data, _ = evaluate_models([model], properties)
    scube = data[0]['scube']['d']
    assert np.isfinite(scube).all()
    return scube.sum(axis=0) * 10


@pytest.mark.parametrize('dispersion', [0, 0.5, 3])
def test_narrow_lines_keep_their_flux(driver, evaluate_models, dispersion):
    # Lines much narrower than a channel (10), and of width 0, have the
    # flux of a wide line
    expected = line_flux(driver, evaluate_models, 20)
    actual = line_flux(driver, evaluate_models, dispersion)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
