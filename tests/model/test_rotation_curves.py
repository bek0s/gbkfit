"""
The parametric rotation curves (velocity polar traits): the velocity of
the model on the major axis of a thin disk is the curve times sin(incl).
"""

import numpy as np
import pytest


def nfw(r, rt, vt):
    u = r / rt
    f = (np.log1p(u) - u / (1 + u)) / u
    return vt * np.sqrt(f / 0.21621659550187317)


CURVES = dict(
    tan_arctan=(
        dict(rt=3.0, vt=180.0),
        lambda r, rt, vt: vt * 2 / np.pi * np.arctan(r / rt)),
    tan_boissier=(
        dict(rt=3.0, vt=180.0),
        lambda r, rt, vt: vt * (1 - np.exp(-r / rt))),
    tan_epinat=(
        dict(rt=3.0, vt=180.0, a=1.5, g=0.8),
        lambda r, rt, vt, a, g: vt * (r / rt) ** g / (1 + (r / rt) ** a)),
    tan_lramp=(
        dict(rt=3.0, vt=180.0),
        lambda r, rt, vt: vt * np.minimum(r / rt, 1)),
    tan_tanh=(
        dict(rt=3.0, vt=180.0),
        lambda r, rt, vt: vt * np.tanh(r / rt)),
    tan_polyex=(
        dict(rt=3.0, vt=180.0, a=0.05),
        lambda r, rt, vt, a: vt * (1 - np.exp(-r / rt)) * (1 + a * r / rt)),
    # (as the kernel has it: Courteau's form, of r / rt in the first factor)
    tan_rix=(
        dict(rt=3.0, vt=180.0, b=0.2, g=2.0),
        lambda r, rt, vt, b, g:
            vt * (1 + r / rt) ** b * (1 + (r / rt) ** -g) ** (-1 / g)),
    tan_courteau=(
        dict(rt=3.0, vt=180.0, b=0.4, g=2.0),
        lambda r, rt, vt, b, g:
            vt * (1 + rt / r) ** b / (1 + (rt / r) ** g) ** (1 / g)),
    tan_brandt=(
        dict(rt=4.0, vt=150.0, n=1.5),
        lambda r, rt, vt, n:
            vt * (r / rt) / (1 / 3 + 2 / 3 * (r / rt) ** n) ** (3 / (2 * n))),
    tan_iso=(
        dict(rt=2.0, vt=160.0),
        lambda r, rt, vt: vt * np.sqrt(1 - rt / r * np.arctan(r / rt))),
    tan_nfw=(dict(rt=3.0, vt=200.0), nfw))


@pytest.mark.parametrize('trait', list(CURVES))
def test_rotation_curves(driver, trait, evaluate_models):
    values, curve = CURVES[trait]
    incl = 60
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='pixel_spectra', size=[33, 41, 81], step=[0.5, 0.5, 10]),
        gmodel=dict(type='kinematics_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 14)),
            bptraits=dict(type='exponential'),
            vptraits=dict(type=trait),
            dptraits=dict(type='uniform'))]))
    properties = dict(
        vsys=0, xpos=0, ypos=0, posa=0, incl=incl, bpt_a=1, bpt_s=4,
        dpt_a=20) | {f'vpt_{k}': v for k, v in values.items()}
    _, extra = evaluate_models([model], properties)
    velocity = extra['observation0_model_component0_vdata'].data
    # The major axis (posa 0) is the y axis, through x = 0 (column 16);
    # the receding side (positive velocities) is north
    j = np.arange(41)
    y = (j - 20) * 0.5
    north = y > 0
    expected = curve(y[north], **values) * np.sin(np.radians(incl))
    np.testing.assert_allclose(
        velocity[north, 16], expected, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(
        velocity[::-1][north, 16], -expected, rtol=1e-4, atol=1e-3)
    if trait == 'tan_nfw':
        # The maximum is vt, at 2.163 rt
        r = np.linspace(0.1, 30, 30000)
        assert nfw(r, 3.0, 200.0).max() == pytest.approx(200, rel=1e-6)


@pytest.mark.parametrize('trait', ['tan_iso', 'tan_nfw'])
def test_rotation_curves_near_the_centre(driver, trait, evaluate_models):
    # Near the centre, 1 - atan(u) / u (pseudo-isothermal) and ln(1 + u) -
    # u / (1 + u) (NFW) lose their precision in float32, and the first
    # could fall below 0 (a NaN velocity). A pixel 1e-4 core radii from
    # the centre, on the major axis, has the velocity of the curve there.
    values, curve = CURVES[trait]
    values = values | dict(rt=3.0)
    incl = 60
    radius = 1e-4 * values['rt']
    model = dict(
        driver=dict(type=driver.type()),
        dmodel=dict(type='pixel_spectra', size=[9, 9, 41], step=[1, 1, 10]),
        gmodel=dict(type='kinematics_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 6)),
            bptraits=dict(type='exponential'),
            vptraits=dict(type=trait),
            dptraits=dict(type='uniform'))]))
    # The centre of the disk is radius south of pixel (4, 4)
    properties = dict(
        vsys=0, xpos=0, ypos=-radius, posa=0, incl=incl, bpt_a=1,
        bpt_s=4, dpt_a=20) | {f'vpt_{k}': v for k, v in values.items()}
    data, extra = evaluate_models([model], properties)
    assert np.isfinite(data[0]['spectra']['d']).all()
    velocity = extra['observation0_model_component0_vdata'].data
    expected = curve(radius, **values) * np.sin(np.radians(incl))
    assert velocity[4, 4] == pytest.approx(expected, rel=1e-3)
