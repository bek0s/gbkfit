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
    Foreground, Lens, LensDeflectionMap, LensEPL, LensMultipole, LensNFW,
    LensSIE, LensShear, foreground_parser)
from gbkfit.utils import fitsutils, gridutils
from gbkfit.utils.parseutils import ConfigError


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
    return Foreground(Lens(
        source_size, source_step,
        LensDeflectionMap(alpha_x, alpha_y, **coords)))


def lens_params(**params):
    return {f'obs_lens_{name}': value for name, value in params.items()}


def on_axes(grid, params):
    # The positions of the pixels of a grid along the major and minor axes
    # of a mass (see lens_masses)
    x, y = gridutils.sky_positions(grid)
    x = x - params['x']
    y = y - params['y']
    pa = np.radians(params['pa'])
    return (-np.sin(pa) * x + np.cos(pa) * y,
            np.cos(pa) * x + np.sin(pa) * y)


def divergence(mass, grid, params, h=1e-4):
    # div alpha, by central differences: alpha at a point moved by h is
    # alpha with the mass moved by -h
    def alpha(dx, dy):
        return mass.deflection(
            grid, params | dict(x=params['x'] - dx, y=params['y'] - dy))
    return (alpha(h, 0)[0] - alpha(-h, 0)[0]
            + alpha(0, h)[1] - alpha(0, -h)[1]) / (2 * h)


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
    lens = LensDeflectionMap(alpha_x, alpha_y, rota=90)
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
                vptraits=dict(type='arctan', direction='tan'),
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


# The masses: a grid around their centre, and their elliptical radius
GRID = gridutils.make_grid((5, 5), 0.37)

ELLIPSE = dict(q=0.6, pa=30, x=0.2, y=-0.1)


def elliptical_radius(grid, params):
    u, v = on_axes(grid, params)
    return np.sqrt(params['q'] * u ** 2 + v ** 2 / params['q'])


def test_the_deflection_of_an_sie_is_that_of_its_convergence():
    # div alpha = 2 kappa, with kappa = theta_e / (2 xi); and q = 1 is the
    # isothermal sphere, theta_e r / |r|
    params = ELLIPSE | dict(theta_e=1.3)
    kappa = 1.3 / (2 * elliptical_radius(GRID, params))
    np.testing.assert_allclose(
        divergence(LensSIE(), GRID, params), 2 * kappa, rtol=1e-5)
    sphere = LensSIE().deflection(GRID, params | dict(q=1))
    x, y = gridutils.sky_positions(GRID)
    r = np.hypot(x - 0.2, y + 0.1)
    np.testing.assert_allclose(
        sphere, [1.3 * (x - 0.2) / r, 1.3 * (y + 0.1) / r])


def test_the_deflection_of_an_epl_is_that_of_its_convergence():
    # kappa = (2 - t) / 2 (theta_e / xi)^t, t = slope - 1; and slope 2 is
    # the SIE
    params = ELLIPSE | dict(theta_e=1.3, slope=2.3)
    kappa = 0.7 / 2 * (1.3 / elliptical_radius(GRID, params)) ** 1.3
    np.testing.assert_allclose(
        divergence(LensEPL(), GRID, params), 2 * kappa, rtol=1e-5)
    np.testing.assert_allclose(
        LensEPL().deflection(GRID, params | dict(slope=2)),
        LensSIE().deflection(GRID, params), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('q', [1, 0.5])
def test_the_deflection_of_an_nfw_halo_is_that_of_its_convergence(q):
    # kappa = 2 kappa_s (1 - F(x)) / (x^2 - 1), x = xi / r_s; on a grid
    # within and beyond r_s
    grid = gridutils.make_grid((7, 7), 0.5)
    params = ELLIPSE | dict(q=q, kappa_s=0.3, r_s=0.8)
    x = elliptical_radius(grid, params) / 0.8
    root = np.sqrt(np.abs(x ** 2 - 1))
    within = x < 1
    f = np.arctan(root) / root
    f[within] = np.arctanh(root[within]) / root[within]
    kappa = 2 * 0.3 * (1 - f) / (x ** 2 - 1)
    np.testing.assert_allclose(
        divergence(LensNFW(), grid, params), 2 * kappa, rtol=1e-5)


def test_an_nfw_halo_of_nearly_one_axis_ratio_is_round():
    # (the integral over the ellipses, of q < 1, and the closed form)
    params = ELLIPSE | dict(kappa_s=0.3, r_s=0.8)
    np.testing.assert_allclose(
        LensNFW().deflection(GRID, params | dict(q=1 - 1e-9)),
        LensNFW().deflection(GRID, params | dict(q=1)), rtol=1e-8)


@pytest.mark.parametrize('order', [1, 3, 4])
def test_the_deflection_of_a_multipole_is_that_of_its_convergence(order):
    # kappa = theta_e a / (2 r) cos(m (phi - phi_m)), with phi_m the angle
    # of the position angle
    params = dict(theta_e=1.3, a=0.05, pa=30, x=0.2, y=-0.1)
    x, y = gridutils.sky_positions(GRID)
    phi = np.arctan2(y + 0.1, x - 0.2)
    kappa = 1.3 * 0.05 / (2 * np.hypot(x - 0.2, y + 0.1)) * np.cos(
        order * (phi - np.radians(30 + 90)))
    np.testing.assert_allclose(
        divergence(LensMultipole(order), GRID, params), 2 * kappa,
        rtol=1e-5, atol=1e-6)


def test_a_shear_stretches_along_its_position_angle():
    # alpha = gamma r along the position angle, and -gamma r across it
    pa = np.radians(30)
    along = np.array([-np.sin(pa), np.cos(pa)])
    across = np.array([np.cos(pa), np.sin(pa)])
    for direction, sign in ((along, 1), (across, -1)):
        # (a grid of one pixel at 2 direction)
        point = gridutils.make_grid((1, 1), rpix=tuple(-2 * direction))
        alpha = LensShear().deflection(point, dict(gamma=0.1, pa=30))
        np.testing.assert_allclose(
            np.ravel(alpha), sign * 0.1 * 2 * direction, atol=1e-15)


def test_the_deflections_of_the_masses_of_a_lens_add_up():
    lens = Lens((8, 8), (1, 1), [
        LensSIE(name='main'), LensShear(name='shear')])
    assert list(lens.pdescs()) == [
        'main_theta_e', 'main_q', 'main_pa', 'main_x', 'main_y',
        'shear_gamma', 'shear_pa']
    sie = ELLIPSE | dict(theta_e=1.3)
    shear = dict(gamma=0.1, pa=70)
    total = lens.deflection(GRID, {
        **{f'main_{name}': value for name, value in sie.items()},
        **{f'shear_{name}': value for name, value in shear.items()}})
    np.testing.assert_allclose(total, np.add(
        LensSIE().deflection(GRID, sie), LensShear().deflection(GRID, shear)))
    # Without names, the first mass has none and the others their index
    unnamed = Lens((8, 8), (1, 1), [LensSIE(), LensShear()])
    assert 'theta_e' in unnamed.pdescs()
    assert 'mass1_gamma' in unnamed.pdescs()
    with pytest.raises(ConfigError, match="must have a name"):
        Lens((8, 8), (1, 1), [LensSIE(name='main'), LensShear()])


def test_masses_of_no_mass_and_no_shear_are_no_lens(driver):
    lens = Foreground(Lens((32, 41), (1, 1), [
        LensSIE(name='sie'), LensEPL(name='epl'), LensShear(name='shear')]))
    params = lens_params(
        sie_theta_e=0, sie_q=0.7, sie_pa=20, sie_x=0, sie_y=0,
        epl_theta_e=0, epl_slope=2.2, epl_q=0.7, epl_pa=20, epl_x=0,
        epl_y=0, shear_gamma=0, shear_pa=0)
    lensed = evaluate(driver, lens, PROPERTIES | params)
    np.testing.assert_allclose(lensed, evaluate(driver), atol=1e-6)


def test_an_isothermal_sphere_makes_an_einstein_ring(driver):
    # (as the point mass above: a round source behind it, both at the
    # centre, makes a ring at theta_e)
    einstein = 6.0
    lens = Foreground(Lens((80, 80), (0.25, 0.25), LensSIE()))
    properties = PROPERTIES | dict(incl=0, pt_s=0.8) | lens_params(
        theta_e=einstein, q=1, pa=0, x=0, y=0)
    image = dict(type='pixel_brightness', size=[64, 64], step=[0.5, 0.5])
    lensed = evaluate(driver, lens, properties, image)
    sky_x, sky_y = gridutils.sky_positions(gridutils.make_grid((64, 64), 0.5))
    radius = np.hypot(sky_x, sky_y)
    ring = np.abs(radius - einstein) < 0.5
    assert lensed[ring].mean() > 20 * lensed[radius < 3].mean()

def test_a_scaled_map_scales_its_deflection():
    # (e.g. by D_LS / D_S, of a source at another redshift than the map's)
    lens = LensDeflectionMap(
        np.full((5, 5), 2.0), np.full((5, 5), -1.0), scaled=True)
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
    def load(**options):
        return foreground_parser.load(dict(lens=dict(
            source_size=[10, 10], source_step=[0.5, 0.5],
            masses=dict(
                type='deflection_map', alpha_x=str(tmp_path / 'ax.fits'),
                alpha_y=str(tmp_path / 'ay.fits'), **options))))
    foreground = load()
    lens = foreground.lens()
    grid = gridutils.make_grid((3, 2), 0.5, rval=(150.0, 2.0))
    deflection = lens.deflection(grid, {})
    np.testing.assert_allclose(
        deflection, [np.ones((2, 3)), np.full((2, 3), 2)])
    dumped = foreground_parser.dump(foreground, prefix=str(tmp_path / 'd_'))
    again = foreground_parser.load(copy.deepcopy(dumped))
    assert again.lens().source_grid() == lens.source_grid()
    # The world coordinates given replace those of the files
    stretched = load(step=[1, 1], rpix=[0, 0]).lens().masses()[0]
    assert stretched._grid.coords.step == (1, 1)
    assert stretched._grid.coords.rpix == (0, 0)


def test_lenses_from_configurations():
    foreground = foreground_parser.load(dict(lens=dict(
        source_size=[10, 10], source_step=[0.5, 0.5], masses=[
            dict(type='epl', name='main'),
            dict(type='multipole', order=4, name='m4'),
            dict(type='nfw', name='halo'),
            dict(type='shear', name='shear')])))
    lens = foreground.lens()
    assert [mass.type() for mass in lens.masses()] == [
        'epl', 'multipole', 'nfw', 'shear']
    assert lens.masses()[1].order() == 4
    again = foreground_parser.load(foreground_parser.dump(foreground))
    assert again.lens().pdescs().keys() == lens.pdescs().keys()
    with pytest.raises(ConfigError, match="order"):
        foreground_parser.load(dict(lens=dict(
            source_size=[10, 10], source_step=[0.5, 0.5],
            masses=dict(type='multipole', order=0))))


def test_deflection_maps_must_match():
    with pytest.raises(RuntimeError, match="one shape"):
        LensDeflectionMap(np.zeros((4, 4)), np.zeros((4, 5)))
