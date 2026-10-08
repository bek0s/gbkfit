"""
Tests for opacity components: they absorb the light of the components
behind them along the line of sight.
"""

import copy

import gbkfit.model
import gbkfit.params
import numpy as np
import pytest


DISK = dict(
    type='smdisk', loose=False, tilted=False, rnodes=list(range(0, 12)))

PROPERTIES = dict(
    vsys=0, xpos=0, ypos=0, posa=30, incl=60,
    bpt_a=1, bpt_s=4, bht_s=1, vpt_rt=2, vpt_vt=150, dpt_a=20,
    ocmp_xpos=0, ocmp_ypos=0, ocmp_posa=30, ocmp_incl=60,
    ocmp_opt_s=4, ocmp_oht_a=1, ocmp_oht_s=1)


def kinematics_3d_model(driver, with_opacity):
    gmodel = dict(type='kinematics_3d', components=[dict(
        **DISK,
        bptraits=dict(type='exponential'),
        bhtraits=dict(type='sech2'),
        vptraits=dict(type='tan_arctan'),
        dptraits=dict(type='uniform'))])
    if with_opacity:
        gmodel['opacity_components'] = [dict(
            **DISK,
            optraits=dict(type='exponential'),
            ohtraits=dict(type='sech2'))]
    return dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='scube', size=[32, 32, 41], step=[1, 1, 10]),
        gmodel=gmodel)


def evaluate(driver, with_opacity, opacity, times=1):
    """Evaluate a thick disk (with or without opacity) several times."""
    model_group = gbkfit.model.ModelGroup(gbkfit.model.model_parser.load(
        [kinematics_3d_model(driver, with_opacity)]))
    properties = dict(PROPERTIES, ocmp_opt_a=opacity)
    if not with_opacity:
        properties = {
            k: v for k, v in properties.items() if not k.startswith('ocmp')}
    params = gbkfit.params.EvaluationParams(model_group.pdescs(), properties)
    values = params.evaluate()
    return [
        copy.deepcopy(model_group.model_h(values)[0]['scube']['d'])
        for _ in range(times)]


def assert_same_model(actual, desired):
    """
    Compare two model cubes to float32 precision. Thick disks are not
    bitwise reproducible on the host, because the threads add to the
    cube in a different order every time.
    """
    np.testing.assert_allclose(
        actual, desired, rtol=1e-5, atol=1e-6 * np.abs(desired).max())


def test_zero_opacity_has_no_effect(driver):
    without, = evaluate(driver, with_opacity=False, opacity=0)
    with_zero, = evaluate(driver, with_opacity=True, opacity=0)
    assert_same_model(with_zero, without)


def test_opacity_absorbs_flux(driver, request):
    if driver.type() == 'cuda':
        request.applymarker(pytest.mark.xfail(
            strict=True, reason="known bug: the cuda thick disk only "
            "evaluates the z=0 slice, which no opacity is in front of"))
    without, = evaluate(driver, with_opacity=False, opacity=0)
    absorbed, = evaluate(driver, with_opacity=True, opacity=0.05)
    assert absorbed.sum() < 0.99 * without.sum()


def test_opacity_does_not_accumulate(driver):
    first, second = evaluate(
        driver, with_opacity=True, opacity=0.05, times=2)
    assert_same_model(second, first)
