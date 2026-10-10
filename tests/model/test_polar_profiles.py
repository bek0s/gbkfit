"""
The radial profiles of the polar traits of brightness and dispersion: a
face-on thin disk has them at the radii of its pixels.
"""

import numpy as np
import pytest


# The profiles, of a = 2 and s = 3 (and b = 1.5), at the radius r
PROFILES = dict(
    uniform=lambda r: 2 + 0 * r,
    exponential=lambda r: 2 * np.exp(-r / 3),
    gauss=lambda r: 2 * np.exp(-0.5 * (r / 3) ** 2),
    ggauss=lambda r: 2 * np.exp(-(r / 3) ** 1.5),
    lorentz=lambda r: 2 / (1 + (r / 3) ** 2),
    moffat=lambda r: 2 / (1 + (r / 3) ** 2) ** 1.5,
    sech2=lambda r: 2 / np.cosh(r / 3) ** 2)

VALUES = dict(a=2.0, s=3.0, b=1.5)


def face_on_disk(driver, evaluate_cases, key, trait):
    """
    The image (key 'b') or the dispersion map (key 'd') of a face-on thin
    disk with a polar trait of the given type, and the radius of each
    pixel (pixels of 0.5, centred on the disk).
    """
    from gbkfit.model.components.disks import traits
    parser = dict(b=traits.bpt_parser, d=traits.dpt_parser)[key]
    names = [p.name() for p in parser.load(dict(type=trait)).params_sm()]
    traits_ = dict(
        bptraits=dict(type=trait if key == 'b' else 'uniform'),
        vptraits=dict(type='tan_uniform'),
        dptraits=dict(type=trait if key == 'd' else 'uniform'))
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[33, 33, 61], step=[0.5, 0.5, 5]),
        model=dict(type='kinematics_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 10)), **traits_)]))
    properties = dict(
        vsys=0, xpos=0, ypos=0, posa=0, incl=0, vpt_vt=0,
        bpt_a=1, dpt_a=10) | {f'{key}pt_{n}': VALUES[n] for n in names}
    _, extra = evaluate_cases([case], properties)
    output = dict(b='bdata', d='ddata')[key]
    data = extra[f'observation0_model_component0_{output}'].data
    y, x = (np.indices((33, 33)) - 16) * 0.5
    return data.reshape(33, 33), np.hypot(x, y)


@pytest.mark.parametrize('key', ['b', 'd'])
@pytest.mark.parametrize('trait', list(PROFILES))
def test_polar_profiles(driver, evaluate_cases, key, trait):
    data, radius = face_on_disk(driver, evaluate_cases, key, trait)
    # (inside the last node, away from its edge)
    inside = radius < 8.5
    np.testing.assert_allclose(
        data[inside], PROFILES[trait](radius[inside]), rtol=1e-5)
