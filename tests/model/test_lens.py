"""
Tests for gravitational lensing (the foreground lens of an observation):
the light of each point of the image comes from the source plane, at its
position less the deflection there.
"""

import copy

import gbkfit.params
import numpy as np
import pytest
from modelutils import observation_group

from gbkfit.observation import (
    Foreground, LensDeflectionMap, LensSIE, foreground_parser)
from gbkfit.utils import fitsutils, gridutils


DISK = dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    density=dict(ptraits=dict(type='exponential')))

GALAXY = dict(components=[DISK | dict(emission=dict(type='band'))])

PROPERTIES = dict(xpos=0.0, ypos=0.0, posa=50, incl=60, pt_a=1, pt_s=2)

IMAGE = dict(type='pixel_brightness', size=[32, 41], step=[1, 1])


def evaluate(driver, foreground=None, properties=PROPERTIES, observable=IMAGE,
             galaxy=GALAXY):
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=galaxy,
        line_of_sight=dict(type='flat'),
        observable=copy.deepcopy(observable))])
    observation = group.observations()[0]
    if foreground is not None:
        from gbkfit.observation import Observation, ObservationGroup
        observation = Observation(
            observation.observable(), observation.line_of_sight(),
            observation.driver(), foreground=foreground,
            instrument=observation.instrument(), name='obs')
        group = ObservationGroup(group.galaxies(), [observation])
    params = gbkfit.params.ParamSpace(group.pdescs(), properties)
    data = group.model_h(params.evaluate())[0]
    return next(iter(data.values()))['d'].copy()


def deflection_lens(alpha_x, alpha_y, source_size, source_step, **coords):
    return Foreground(LensDeflectionMap(
        alpha_x, alpha_y, source_size, source_step, **coords))


# A smooth disk, whose bilinear interpolation on a source grid of half
# the pixels of the image is within 1% of its peak
SMOOTH_GALAXY = dict(components=[
    GALAXY['components'][0] | dict(density=dict(ptraits=dict(type='gauss')))])

SMOOTH_PROPERTIES = PROPERTIES | dict(pt_s=4)


def test_no_deflection_is_no_lens(driver):
    # On a source grid equal to the grid of the image, the light of each
    # pixel is that of its source pixel
    plain = evaluate(driver)
    zero = np.zeros((41, 32))
    lensed = evaluate(driver, deflection_lens(zero, zero, (32, 41), (1, 1)))
    np.testing.assert_allclose(lensed, plain, rtol=1e-6, atol=1e-7)


def test_no_deflection_is_no_lens_on_a_rotated_image(driver):
    # The source grid is aligned with the sky, and each pixel of a rotated
    # image takes the light of the source at its position on the sky
    image = IMAGE | dict(rota=30)
    plain = evaluate(driver, None, SMOOTH_PROPERTIES, image, SMOOTH_GALAXY)
    zero = np.zeros((4, 4))
    lens = deflection_lens(zero, zero, (120, 120), (0.5, 0.5), step=100)
    lensed = evaluate(driver, lens, SMOOTH_PROPERTIES, image, SMOOTH_GALAXY)
    np.testing.assert_allclose(lensed, plain, atol=0.01 * plain.max())


def test_deflection_of_a_rotated_map():
    # At the centres of the pixels of a map, the deflection is that of the
    # pixel; between them, interpolated bilinearly; beyond them, that of
    # the nearest edge
    j, i = np.mgrid[0:4, 0:5]
    alpha_x = 1.0 * i
    alpha_y = 10.0 * j
    lens = LensDeflectionMap(alpha_x, alpha_y, (8, 8), (1, 1), rota=90)
    # (a grid with the pixels of the maps, and more beyond them)
    grid = gridutils.make_grid((20, 20), rpix=(2, 1.5), rota=90)
    deflection_x, deflection_y = lens.deflection(grid, {})
    np.testing.assert_allclose(deflection_x[:4, :5], alpha_x)
    np.testing.assert_allclose(deflection_y[:4, :5], alpha_y)
    np.testing.assert_allclose(
        [deflection_x[13, 14], deflection_y[13, 14]], [4, 30])
    # (a pixel between the pixels 1 and 2 of x, at the pixel 1 of y)
    middle = gridutils.make_grid((1, 1), rpix=(0.5, 0.5), rota=90)
    np.testing.assert_allclose(lens.deflection(middle, {}), [[[1.5]], [[10]]])


@pytest.mark.parametrize('observable', [
    IMAGE, dict(type='pixel_spectra', size=[32, 41, 31], step=[1, 1, 20])])
def test_constant_deflection_moves_the_source(driver, observable):
    # With a constant deflection alpha, the image is the source moved by
    # alpha: the galaxy with its centre moved by alpha
    galaxy = GALAXY
    properties = PROPERTIES
    if observable['type'] == 'pixel_spectra':
        galaxy = dict(components=[DISK | dict(
            kinematics=dict(
                vptraits=dict(type='tan_arctan'),
                dptraits=dict(type='uniform')),
            emission=dict(type='lines'))])
        properties = PROPERTIES | dict(
            vsys=0, vpt_rt=2, vpt_vt=150, dpt_a=20)
    lens = deflection_lens(
        np.full((41, 32), 2.0), np.full((41, 32), -3.0), (60, 61), (1, 1))
    lensed = evaluate(driver, lens, properties, observable, galaxy)
    moved = evaluate(
        driver, None, properties | dict(xpos=2.0, ypos=-3.0), observable,
        galaxy)
    np.testing.assert_allclose(
        lensed, moved, rtol=1e-5, atol=1e-6 * moved.max())


def test_deflection_between_source_pixels(driver):
    # A deflection of fractions of the pixels of the source: the light
    # comes from between them, interpolated bilinearly
    lens = deflection_lens(
        np.full((41, 32), 2.5), np.full((41, 32), -3.25), (120, 122),
        (0.5, 0.5))
    lensed = evaluate(driver, lens, SMOOTH_PROPERTIES, IMAGE, SMOOTH_GALAXY)
    moved = evaluate(
        driver, None, SMOOTH_PROPERTIES | dict(xpos=2.5, ypos=-3.25), IMAGE,
        SMOOTH_GALAXY)
    np.testing.assert_allclose(lensed, moved, atol=0.01 * moved.max())


def test_point_mass_makes_an_einstein_ring(driver):
    # A round source behind a point mass, both at the centre: the image is
    # a ring at the Einstein radius, brighter in total than the source
    # (lensing conserves surface brightness and magnifies)
    j, i = np.mgrid[0:161, 0:161]
    x = (i - 80) * 0.25
    y = (j - 80) * 0.25
    radius2 = np.maximum(x * x + y * y, 1e-6)
    einstein = 6.0
    lens = deflection_lens(
        einstein ** 2 * x / radius2, einstein ** 2 * y / radius2,
        (80, 80), (0.25, 0.25), step=0.25)
    properties = PROPERTIES | dict(incl=0, pt_s=0.8)
    image = dict(type='pixel_brightness', size=[64, 64], step=[0.5, 0.5])
    source = evaluate(driver, None, properties, image)
    lensed = evaluate(driver, lens, properties, image)
    grid = gridutils.make_grid((64, 64), 0.5)
    sky_x, sky_y = gridutils.sky_positions(grid)
    radius = np.hypot(sky_x, sky_y)
    ring = np.abs(radius - einstein) < 0.5
    assert lensed[ring].mean() > 20 * lensed[radius < 3].mean()
    assert lensed.sum() > 3 * source.sum()


def test_the_deflection_of_an_sie_is_that_of_its_convergence():
    # div alpha = 2 kappa, with kappa = theta_e / (2 sqrt(q u^2 + v^2 / q))
    # along its axes; and q = 1 is the isothermal sphere, theta_e r / |r|
    # (alpha at a point moved by h is alpha with the lens moved by -h)
    lens = LensSIE((8, 8), (1, 1))
    params = dict(theta_e=1.3, q=0.6, pa=30, x=0.2, y=-0.1)
    h = 1e-4
    grid = gridutils.make_grid((5, 5), 0.37)

    def alpha(dx, dy):
        return lens.deflection(
            grid, params | dict(x=0.2 - dx, y=-0.1 - dy))
    divergence = (alpha(h, 0)[0] - alpha(-h, 0)[0]
                  + alpha(0, h)[1] - alpha(0, -h)[1]) / (2 * h)
    x, y = gridutils.sky_positions(grid)
    pa = np.radians(30)
    u = -np.sin(pa) * (x - 0.2) + np.cos(pa) * (y + 0.1)
    v = np.cos(pa) * (x - 0.2) + np.sin(pa) * (y + 0.1)
    kappa = 1.3 / (2 * np.sqrt(0.6 * u ** 2 + v ** 2 / 0.6))
    np.testing.assert_allclose(divergence, 2 * kappa, rtol=1e-5)
    sphere = lens.deflection(grid, params | dict(q=1))
    r = np.hypot(x - 0.2, y + 0.1)
    np.testing.assert_allclose(
        sphere, [1.3 * (x - 0.2) / r, 1.3 * (y + 0.1) / r])


def test_an_sie_of_no_mass_and_no_shear_is_no_lens(driver):
    lens = Foreground(LensSIE((32, 41), (1, 1), shear=True))
    params = dict(theta_e=0, q=0.7, pa=20, x=0, y=0, gamma=0, gamma_pa=0)
    lensed = evaluate(driver, lens, PROPERTIES | {
        f'obs_lens_{name}': value for name, value in params.items()})
    np.testing.assert_allclose(lensed, evaluate(driver), atol=1e-6)


def test_an_isothermal_sphere_makes_an_einstein_ring(driver):
    # (as the point mass above: a round source behind it, both at the
    # centre, makes a ring at theta_e)
    einstein = 6.0
    lens = Foreground(LensSIE((80, 80), (0.25, 0.25)))
    properties = PROPERTIES | dict(incl=0, pt_s=0.8) | {
        f'obs_lens_{name}': value for name, value in dict(
            theta_e=einstein, q=1, pa=0, x=0, y=0).items()}
    image = dict(type='pixel_brightness', size=[64, 64], step=[0.5, 0.5])
    lensed = evaluate(driver, lens, properties, image)
    sky_x, sky_y = gridutils.sky_positions(gridutils.make_grid((64, 64), 0.5))
    radius = np.hypot(sky_x, sky_y)
    ring = np.abs(radius - einstein) < 0.5
    assert lensed[ring].mean() > 20 * lensed[radius < 3].mean()

def test_a_scaled_map_scales_its_deflection():
    # (e.g. by D_LS / D_S, of a source at another redshift than the map's)
    lens = LensDeflectionMap(
        np.full((5, 5), 2.0), np.full((5, 5), -1.0), (8, 8), (1, 1),
        scaled=True)
    assert set(lens.pdescs()) == {'scale'}
    grid = gridutils.make_grid((3, 3), 1)
    np.testing.assert_allclose(
        lens.deflection(grid, dict(scale=0.5)),
        [np.full((3, 3), 1.0), np.full((3, 3), -0.5)])

def test_deflection_maps_from_files(tmp_path):
    coords = gridutils.Coords((0.5, 0.5), (9.5, 7.5), (150.0, 2.0), 0)
    fitsutils.write_data(str(tmp_path / 'ax.fits'), np.ones((16, 20)), coords)
    fitsutils.write_data(
        str(tmp_path / 'ay.fits'), np.full((16, 20), 2.0), coords)
    foreground = foreground_parser.load(dict(lens=dict(
        type='deflection_map', alpha_x=str(tmp_path / 'ax.fits'),
        alpha_y=str(tmp_path / 'ay.fits'), source_size=[10, 10],
        source_step=[0.5, 0.5])))
    lens = foreground.lens()
    grid = gridutils.make_grid((3, 2), 0.5, rval=(150.0, 2.0))
    deflection = lens.deflection(grid, {})
    np.testing.assert_allclose(
        deflection, [np.ones((2, 3)), np.full((2, 3), 2)])
    dumped = foreground_parser.dump(foreground, prefix=str(tmp_path / 'd_'))
    again = foreground_parser.load(copy.deepcopy(dumped))
    assert again.lens().source_grid() == lens.source_grid()
    # The world coordinates given replace those of the files
    stretched = foreground_parser.load(dict(lens=dict(
        type='deflection_map', alpha_x=str(tmp_path / 'ax.fits'),
        alpha_y=str(tmp_path / 'ay.fits'), source_size=[10, 10],
        source_step=[0.5, 0.5], step=[1, 1], rpix=[0, 0])))
    assert stretched.lens()._grid.coords.step == (1, 1)
    assert stretched.lens()._grid.coords.rpix == (0, 0)


def test_deflection_maps_must_match():
    with pytest.raises(RuntimeError, match="one shape"):
        LensDeflectionMap(np.zeros((4, 4)), np.zeros((4, 5)), (8, 8), (1, 1))
