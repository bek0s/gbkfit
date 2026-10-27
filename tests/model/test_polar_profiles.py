"""
The radial profiles of the polar traits of density and dispersion: a
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
    from gbkfit.model.components import traits
    parser = dict(b=traits.rpt_parser, d=traits.dpt_parser)[key]
    prefix = dict(b='pt', d='dpt')[key]
    names = list(parser.load(dict(type=trait)).params_sm())
    density = dict(ptraits=dict(type=trait if key == 'b' else 'uniform'))
    kinematics = dict(
        type='traits', vptraits=dict(type='uniform', direction='tan'),
        dptraits=dict(type=trait if key == 'd' else 'uniform'))
    case = dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[33, 33], step=[0.5, 0.5],
            spectral=dict(size=61, step=5, convention='relativistic')),
        line_of_sight=dict(type='flat'),
        galaxy=dict(components=[dict(
            type='disk', loose=False, tilted=False,
            rnodes=list(range(0, 10)), density=density,
            kinematics=kinematics, emission=dict(type='lines'))]))
    properties = dict(
        vsys=0, xpos=0, ypos=0, posa=0, incl=0, vpt_vt=0,
        pt_a=1, dpt_a=10) | {f'{prefix}_{n}': VALUES[n] for n in names}
    _, extra = evaluate_cases([case], properties)
    output = dict(b='bdata', d='ddata')[key]
    data = extra.get(output, component='component0').data
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


def map_component(ptraits):
    """A thin disk of the given density polar traits."""
    from gbkfit.model import component_parser
    return component_parser.load(dict(
        type='disk', loose=False, tilted=False, rnodes=list(range(0, 10)),
        density=dict(ptraits=ptraits),
        kinematics=dict(
            type='traits',
            vptraits=dict(type='uniform', direction='tan'), dptraits=dict(type='uniform')),
        emission=dict(type='lines')))


def test_map_is_its_cells():
    # The density of each cell at its centre (row by row, x along the
    # azimuth 0), and 0 beyond the grid
    from gbkfit.driver.drivers.host import DriverHost
    disk = map_component(dict(type='map', size=[3, 2], step=2.0)) \
        .density_disk()
    params = dict(
        xpos=0, ypos=0, posa=0, incl=0, vsys=0, vpt_vt=0, dpt_a=1,
        pt_a=np.arange(1.0, 7.0))
    x = np.array([-2, 0, 2, -2, 0, 2, 3.5, 0])
    y = np.array([-1, -1, -1, 1, 1, 1, 0, 2.5])
    values = disk.plan(DriverHost(), 0, np.float64).evaluate_polar_traits(
        params, 'rpt', np.hypot(x, y), np.rad2deg(np.arctan2(y, x)))[0]
    np.testing.assert_array_equal(values, [1, 2, 3, 4, 5, 6, 0, 0])


def test_map_lies_where_blobs_lie(driver, evaluate_cases):
    # Seen inclined, a map of one lit cell and a round blob at the centre
    # of that cell put their light at the same place on the sky (the
    # centroids of their images): the cell (4, 1) of a 6 by 4 map of 1
    # arcsec has its centre at x = 1.5, y = -0.5 in the plane of the disk
    x, y = 1.5, -0.5
    lit = np.zeros(24)
    lit[1 * 6 + 4] = 1
    cases = [
        (dict(type='map', size=[6, 4], step=1.0), dict(pt_a=lit)),
        (dict(type='mixture_gauss', nblobs=1), dict(
            pt_r=np.hypot(x, y), pt_t=np.rad2deg(np.arctan2(y, x)), pt_a=1,
            pt_s=0.3, pt_q=1, pt_p=0))]
    yy, xx = (np.indices((121, 121)) - 60) * 0.05
    centroids = []
    for ptraits, values in cases:
        case = dict(
            driver=dict(type=driver.type()),
            observable=dict(
                type='pixel_spectra', size=[121, 121], step=[0.05, 0.05],
                spectral=dict(size=61, step=5, convention='relativistic')),
            line_of_sight=dict(type='flat'),
            galaxy=dict(components=[map_component(ptraits).dump()]))
        properties = dict(
            vsys=0, xpos=0, ypos=0, posa=30, incl=60, vpt_vt=0,
            dpt_a=10) | values
        _, extra = evaluate_cases([case], properties)
        image = extra.get('bdata', component='component0').data \
            .reshape(121, 121)
        centroids.append(
            np.array([np.sum(xx * image), np.sum(yy * image)])
            / np.sum(image))
    np.testing.assert_allclose(*centroids, atol=0.02)


@pytest.mark.parametrize('options, message', [
    (dict(size=[3], step=1.0), "size must be"),
    (dict(size=[3, 0], step=1.0), "size must be"),
    (dict(size=[3, 2.5], step=1.0), "size must be"),
    (dict(size=[3, 2], step=0), "step must be")])
def test_map_options_are_checked(options, message):
    from gbkfit.model.components import traits
    from gbkfit.utils.parseutils import ConfigError
    with pytest.raises(ConfigError, match=message):
        traits.RPTraitMap(**options)
