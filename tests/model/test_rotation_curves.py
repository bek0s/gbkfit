"""
The parametric rotation curves (velocity polar traits): the velocity of
a thin disk on its major axis is the curve times sin(incl).
"""

import numpy as np
import pytest


def nfw(r, rt, vt):
    u = r / rt
    f = (np.log1p(u) - u / (1 + u)) / u
    return vt * np.sqrt(f / 0.21621659550187317)


CURVES = dict(
    arctan=(
        dict(rt=3.0, vt=180.0),
        lambda r, rt, vt: vt * 2 / np.pi * np.arctan(r / rt)),
    boissier=(
        dict(rt=3.0, vt=180.0),
        lambda r, rt, vt: vt * (1 - np.exp(-r / rt))),
    epinat=(
        dict(rt=3.0, vt=180.0, a=1.5, g=0.8),
        lambda r, rt, vt, a, g: vt * (r / rt) ** g / (1 + (r / rt) ** a)),
    lramp=(
        dict(rt=3.0, vt=180.0),
        lambda r, rt, vt: vt * np.minimum(r / rt, 1)),
    tanh=(
        dict(rt=3.0, vt=180.0),
        lambda r, rt, vt: vt * np.tanh(r / rt)),
    polyex=(
        dict(rt=3.0, vt=180.0, a=0.05),
        lambda r, rt, vt, a: vt * (1 - np.exp(-r / rt)) * (1 + a * r / rt)),
    # (Rix et al. 1997, eq. 8)
    rix=(
        dict(rt=3.0, vt=180.0, b=0.2, g=2.0),
        lambda r, rt, vt, b, g:
            vt * (1 + r / rt) ** b * (1 + (r / rt) ** -g) ** (-1 / g)),
    courteau=(
        dict(rt=3.0, vt=180.0, b=0.4, g=2.0),
        lambda r, rt, vt, b, g:
            vt * (1 + rt / r) ** b / (1 + (rt / r) ** g) ** (1 / g)),
    brandt=(
        dict(rt=4.0, vt=150.0, n=1.5),
        lambda r, rt, vt, n:
            vt * (r / rt) / (1 / 3 + 2 / 3 * (r / rt) ** n) ** (3 / (2 * n))),
    iso=(
        dict(rt=2.0, vt=160.0),
        lambda r, rt, vt: vt * np.sqrt(1 - rt / r * np.arctan(r / rt))),
    nfw=(dict(rt=3.0, vt=200.0), nfw))


def thin_disk(trait, rnodes, **options):
    """
    A thin exponential disk that emits lines, whose velocity polar trait
    is the given one.
    """
    return dict(
        type='smdisk', rnodes=rnodes, **options,
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(vptraits=trait, dptraits=dict(type='uniform')),
        emission=dict(type='lines'))


@pytest.mark.parametrize('trait', list(CURVES))
def test_rotation_curves(driver, trait, evaluate_cases):
    values, curve = CURVES[trait]
    incl = 60
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[33, 41, 81], step=[0.5, 0.5, 10]),
        line_of_sight=dict(type='flat'),
        galaxy=dict(components=[thin_disk(
            dict(type=trait, direction='tan'), list(range(0, 14)),
            loose=False, tilted=False)]))
    properties = dict(
        vsys=0, xpos=0, ypos=0, posa=0, incl=incl, pt_a=1, pt_s=4,
        dpt_a=20) | {f'vpt_{k}': v for k, v in values.items()}
    _, extra = evaluate_cases([case], properties)
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
    if trait == 'nfw':
        # The maximum is vt, at 2.163 rt
        r = np.linspace(0.1, 30, 30000)
        assert nfw(r, 3.0, 200.0).max() == pytest.approx(200, rel=1e-6)


@pytest.mark.parametrize('trait', ['iso', 'nfw'])
def test_rotation_curves_near_the_centre(driver, trait, evaluate_cases):
    # Near the centre, 1 - atan(u) / u (pseudo-isothermal) and ln(1 + u) -
    # u / (1 + u) (NFW) lose their precision in float32, and the first
    # could fall below 0 (a NaN velocity). A pixel 1e-4 core radii from
    # the centre, on the major axis, has the velocity of the curve there.
    values, curve = CURVES[trait]
    values = values | dict(rt=3.0)
    incl = 60
    radius = 1e-4 * values['rt']
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[9, 9, 41], step=[1, 1, 10]),
        line_of_sight=dict(type='flat'),
        galaxy=dict(components=[thin_disk(
            dict(type=trait, direction='tan'), list(range(0, 6)),
            loose=False, tilted=False)]))
    # The centre of the disk is radius south of pixel (4, 4)
    properties = dict(
        vsys=0, xpos=0, ypos=-radius, posa=0, incl=incl, pt_a=1,
        pt_s=4, dpt_a=20) | {f'vpt_{k}': v for k, v in values.items()}
    data, extra = evaluate_cases([case], properties)
    assert np.isfinite(data[0]['spectra']['d']).all()
    velocity = extra['observation0_model_component0_vdata'].data
    expected = curve(radius, **values) * np.sin(np.radians(incl))
    assert velocity[4, 4] == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize('trait, values', [
    (dict(type='uniform', direction='tan'), dict(vt=150)),
    # (a speed that is not 0 at the centre)
    (dict(type='epinat', direction='tan'), dict(rt=3, vt=150, a=1, g=0)),
    (dict(type='uniform', direction='rad'), dict(vr=150))])
def test_the_centre_does_not_move(driver, evaluate_cases, trait, values):
    # The azimuth of the centre is undefined, and so are the tangential
    # and radial directions there: its velocity is the systemic one, also
    # with speeds that are not 0 there
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[33, 33, 41], step=[1, 1, 10]),
        line_of_sight=dict(type='flat'),
        galaxy=dict(components=[thin_disk(trait, list(range(0, 12)))]))
    _, extra = evaluate_cases([case], dict(
        vsys=0, xpos=0, ypos=0, posa=0, incl=60, pt_a=1, pt_s=4,
        dpt_a=20) | {f'vpt_{k}': v for k, v in values.items()})
    velocity = extra['observation0_model_component0_vdata'].data
    # (the centre is the pixel 16, 16; the pixels on its diagonals move)
    assert velocity[16, 16] == 0
    assert np.all(velocity[[15, 15, 17, 17], [15, 17, 15, 17]] != 0)
