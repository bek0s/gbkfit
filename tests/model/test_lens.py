"""
Tests for gravitational lensing (the foreground lens of an observation):
the light of each point of the image comes from the source plane, at its
position less the deflection there.
"""

import copy

from gbkfit.config import ConfigError
from gbkfit.grid import Coords, make_grid, sky_positions
from gbkfit.io import write_data
import gbkfit.params
import numpy as np
import pytest
from astropy.io import fits
from modelutils import observation_group

from gbkfit.observation import (
    Foreground, Lens, LensDeflectionMap, LensEPL, LensMultipole, LensNFW,
    LensSIE, LensShear, foreground_parser, lens_mass_parser)


DISK = dict(
    type='thin_disk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    rptraits=dict(type='exponential'))

GALAXY = dict(components=[DISK | dict(emission=dict(type='band'))])

PROPERTIES = dict(xpos=0.0, ypos=0.0, posa=50, incl=60, rpt_a=1, rpt_s=2)

IMAGE = dict(type='pixel_brightness', size=[32, 41], step=[1, 1])


def evaluate(driver, foreground=None, properties=PROPERTIES, observable=IMAGE,
             galaxy=GALAXY):
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=galaxy,
        line_of_sight=dict(type='flat'),
        observable=copy.deepcopy(observable))])
    observation = group.observations()[0]
    if foreground is not None:
        from gbkfit.observation import ObservationGroup, ObservationSky
        observation = ObservationSky(
            observation.observable(), observation.line_of_sight(),
            foreground=foreground, instrument=observation.instrument(),
            name='obs')
        group = ObservationGroup(group.galaxies(), [observation])
    params = gbkfit.params.ParamSpace.from_properties(
        group.pdescs(), properties)
    data = group.plan(driver).model_h(params.evaluate())[0]
    return next(iter(data.values())).data.copy()


def deflection_lens(alpha_x, alpha_y, source_size, source_step, **coords):
    return Foreground(Lens(
        source_size, source_step,
        LensDeflectionMap(alpha_x, alpha_y, **coords)))


def lens_params(**params):
    return {f'obs_lens_{name}': value for name, value in params.items()}


def on_axes(grid, params):
    # The positions of the pixels of a grid along the major and minor axes
    # of a mass (see gbkfit.observation.lensing.masses)
    x, y = sky_positions(grid)
    x = x - params['xpos']
    y = y - params['ypos']
    pa = np.radians(params['posa'])
    return (-np.sin(pa) * x + np.cos(pa) * y,
            np.cos(pa) * x + np.sin(pa) * y)


def divergence(mass, grid, params, h=1e-4):
    # div alpha, by central differences: alpha at a point moved by h is
    # alpha with the mass moved by -h
    def alpha(dx, dy):
        return mass.deflection(
            grid, params | dict(
                xpos=params['xpos'] - dx, ypos=params['ypos'] - dy))
    return (alpha(h, 0)[0] - alpha(-h, 0)[0]
            + alpha(0, h)[1] - alpha(0, -h)[1]) / (2 * h)


# A smooth disk, whose bilinear interpolation on a source grid of half
# the pixels of the image is within 1% of its peak
SMOOTH_GALAXY = dict(components=[
    GALAXY['components'][0] | dict(rptraits=dict(type='gauss'))])

SMOOTH_PROPERTIES = PROPERTIES | dict(rpt_s=4)


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
    grid = make_grid((20, 20), rpix=(2, 1.5), rota=90)
    deflection_x, deflection_y = lens.deflection(grid, {})
    np.testing.assert_allclose(deflection_x[:4, :5], alpha_x)
    np.testing.assert_allclose(deflection_y[:4, :5], alpha_y)
    np.testing.assert_allclose(
        [deflection_x[13, 14], deflection_y[13, 14]], [4, 30])
    # (a pixel between the pixels 1 and 2 of x, at the pixel 1 of y)
    middle = make_grid((1, 1), rpix=(0.5, 0.5), rota=90)
    np.testing.assert_allclose(lens.deflection(middle, {}), [[[1.5]], [[10]]])


@pytest.mark.parametrize('observable', [
    IMAGE, dict(
        type='pixel_spectra', size=[32, 41], step=[1, 1],
        spectral=dict(size=31, step=20, convention='relativistic'))])
def test_constant_deflection_moves_the_source(driver, observable):
    # With a constant deflection alpha, the image is the source moved by
    # alpha: the galaxy with its centre moved by alpha
    galaxy = GALAXY
    properties = PROPERTIES
    if observable['type'] == 'pixel_spectra':
        galaxy = dict(components=[DISK | dict(
            vptraits=dict(type='arctan', direction='tan'),
            dptraits=dict(type='uniform'),
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
    properties = PROPERTIES | dict(incl=0, rpt_s=0.8)
    image = dict(type='pixel_brightness', size=[64, 64], step=[0.5, 0.5])
    source = evaluate(driver, None, properties, image)
    lensed = evaluate(driver, lens, properties, image)
    grid = make_grid((64, 64), 0.5)
    sky_x, sky_y = sky_positions(grid)
    radius = np.hypot(sky_x, sky_y)
    ring = np.abs(radius - einstein) < 0.5
    assert lensed[ring].mean() > 20 * lensed[radius < 3].mean()
    assert lensed.sum() > 3 * source.sum()


# The masses: a grid around their centre, and their elliptical radius
GRID = make_grid((5, 5), 0.37)

ELLIPSE = dict(q=0.6, posa=30, xpos=0.2, ypos=-0.1)


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
    x, y = sky_positions(GRID)
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
    grid = make_grid((7, 7), 0.5)
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
    params = dict(theta_e=1.3, a=0.05, posa=30, xpos=0.2, ypos=-0.1)
    x, y = sky_positions(GRID)
    phi = np.arctan2(y + 0.1, x - 0.2)
    kappa = 1.3 * 0.05 / (2 * np.hypot(x - 0.2, y + 0.1)) * np.cos(
        order * (phi - np.radians(30 + 90)))
    np.testing.assert_allclose(
        divergence(LensMultipole(order), GRID, params), 2 * kappa,
        rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('order', [1, 3, 4])
def test_the_position_angle_of_a_multipole_repeats_every_360_over_m(order):
    # (its period was 360, which gave samplers m modes of one deflection:
    # audit 2026-11-11)
    mass = LensMultipole(order)
    assert mass.pdescs()['posa'].period() == 360 / order
    params = dict(theta_e=1.3, a=0.05, posa=30, xpos=0.2, ypos=-0.1)
    np.testing.assert_allclose(
        mass.deflection(GRID, params | dict(posa=30 + 360 / order)),
        mass.deflection(GRID, params), atol=1e-12)


def test_a_shear_stretches_along_its_position_angle():
    # alpha = gamma r along the position angle, and -gamma r across it
    pa = np.radians(30)
    along = np.array([-np.sin(pa), np.cos(pa)])
    across = np.array([np.cos(pa), np.sin(pa)])
    for direction, sign in ((along, 1), (across, -1)):
        # (a grid of one pixel at 2 direction)
        point = make_grid((1, 1), rpix=tuple(-2 * direction))
        alpha = LensShear().deflection(point, dict(gamma=0.1, posa=30))
        np.testing.assert_allclose(
            np.ravel(alpha), sign * 0.1 * 2 * direction, atol=1e-15)


def test_the_deflections_of_the_masses_of_a_lens_add_up():
    lens = Lens((8, 8), (1, 1), [
        LensSIE(name='main'), LensShear(name='shear')])
    assert list(lens.pdescs()) == [
        'main_theta_e', 'main_q', 'main_posa', 'main_xpos', 'main_ypos',
        'shear_gamma', 'shear_posa']
    sie = ELLIPSE | dict(theta_e=1.3)
    shear = dict(gamma=0.1, posa=70)
    total = lens.deflection(GRID, {
        **{f'main_{name}': value for name, value in sie.items()},
        **{f'shear_{name}': value for name, value in shear.items()}})
    np.testing.assert_allclose(total, np.add(
        LensSIE().deflection(GRID, sie), LensShear().deflection(GRID, shear)))
    # Without names, the first mass has none and the others their index
    unnamed = Lens((8, 8), (1, 1), [LensSIE(), LensShear()])
    assert 'theta_e' in unnamed.pdescs()
    assert 'mass1_gamma' in unnamed.pdescs()
    with pytest.raises(ValueError, match="must have a name"):
        Lens((8, 8), (1, 1), [LensSIE(name='main'), LensShear()])


def test_masses_of_no_mass_and_no_shear_are_no_lens(driver):
    lens = Foreground(Lens((32, 41), (1, 1), [
        LensSIE(name='sie'), LensEPL(name='epl'), LensShear(name='shear')]))
    params = lens_params(
        sie_theta_e=0, sie_q=0.7, sie_posa=20, sie_xpos=0, sie_ypos=0,
        epl_theta_e=0, epl_slope=2.2, epl_q=0.7, epl_posa=20, epl_xpos=0,
        epl_ypos=0, shear_gamma=0, shear_posa=0)
    lensed = evaluate(driver, lens, PROPERTIES | params)
    np.testing.assert_allclose(lensed, evaluate(driver), atol=1e-6)


def test_an_isothermal_sphere_makes_an_einstein_ring(driver):
    # (as the point mass above: a round source behind it, both at the
    # centre, makes a ring at theta_e)
    einstein = 6.0
    lens = Foreground(Lens((80, 80), (0.25, 0.25), LensSIE()))
    properties = PROPERTIES | dict(incl=0, rpt_s=0.8) | lens_params(
        theta_e=einstein, q=1, posa=0, xpos=0, ypos=0)
    image = dict(type='pixel_brightness', size=[64, 64], step=[0.5, 0.5])
    lensed = evaluate(driver, lens, properties, image)
    sky_x, sky_y = sky_positions(make_grid((64, 64), 0.5))
    radius = np.hypot(sky_x, sky_y)
    ring = np.abs(radius - einstein) < 0.5
    assert lensed[ring].mean() > 20 * lensed[radius < 3].mean()

def test_the_frame_of_an_observation_moves_its_lens(driver):
    # An offset of the frame moves the centres of the masses of its lens
    # as those of the components: as moving both by it
    from gbkfit.observation import ObservationGroup, ObservationSky
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=GALAXY,
        line_of_sight=dict(type='flat'), observable=copy.deepcopy(IMAGE))])
    plain = group.observations()[0]
    reference = ObservationSky(
        plain.observable(), plain.line_of_sight(), name='reference')
    observation = ObservationSky(
        plain.observable(), plain.line_of_sight(),
        foreground=Foreground(Lens((40, 40), (0.5, 0.5), LensSIE())),
        name='obs', offsets=dict(position=True))
    plan = ObservationGroup(
        group.galaxies(), [reference, observation]).plan(driver)
    lens = dict(theta_e=3.0, q=0.8, posa=20, ypos=0.3)
    images = []
    for x, offset in ((1.5, 0.0), (0.0, 1.5)):
        properties = PROPERTIES | dict(xpos=x, obs_dx=offset, obs_dy=0) \
            | lens_params(**lens, xpos=x)
        params = gbkfit.params.ParamSpace.from_properties(
            plan.group().pdescs(), properties)
        images.append(plan.model_h(params.evaluate())[1][
            'brightness'].data.copy())
    np.testing.assert_allclose(images[1], images[0], rtol=1e-5, atol=1e-6)


def sis_maps(size=100, step=0.5, theta_e=3.0):
    """
    The maps of the deflection of a singular isothermal sphere at the
    centre of a grid of pixels of step arcsec, along x and y.
    """
    x, y = sky_positions(make_grid((size, size), step))
    radius = np.hypot(x, y)
    return theta_e * x / radius, theta_e * y / radius


def test_the_frame_of_an_observation_moves_its_shifted_maps(driver):
    # An offset of the frame moves the shifted deflection maps of its lens
    # as the centres of its analytic masses (above): as moving both the
    # galaxy and the maps by it (the maps stayed: audit 2026-11-11 d#4)
    from gbkfit.observation import ObservationGroup, ObservationSky
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=GALAXY,
        line_of_sight=dict(type='flat'), observable=copy.deepcopy(IMAGE))])
    plain = group.observations()[0]
    reference = ObservationSky(
        plain.observable(), plain.line_of_sight(), name='reference')
    maps = LensDeflectionMap(*sis_maps(), step=0.5, shifted=True)
    observation = ObservationSky(
        plain.observable(), plain.line_of_sight(),
        foreground=Foreground(Lens((40, 40), (0.5, 0.5), maps)),
        name='obs', offsets=dict(position=True))
    plan = ObservationGroup(
        group.galaxies(), [reference, observation]).plan(driver)
    images = []
    for x, offset in ((1.5, 0.0), (0.0, 1.5)):
        properties = PROPERTIES | dict(xpos=x, obs_dx=offset, obs_dy=0) \
            | lens_params(xpos=x, ypos=0.3)
        params = gbkfit.params.ParamSpace.from_properties(
            plan.group().pdescs(), properties)
        images.append(plan.model_h(params.evaluate())[1][
            'brightness'].data.copy())
    np.testing.assert_allclose(images[1], images[0], rtol=1e-5, atol=1e-6)
    # (and the lens is seen)
    unlensed = evaluate(driver, None, PROPERTIES | dict(xpos=1.5))
    assert not np.allclose(images[0], unlensed, atol=0.1 * unlensed.max())


@pytest.mark.parametrize('shifted', [False, True])
def test_the_alignment_of_frames_leaves_maps_where_they_are(
        driver, shifted):
    # A cube whose reference pixel is 2 arcsec west of that of the first
    # sees the galaxy at x - 2 in its frame (see test_frames), but the
    # deflection maps of its lens where their RA and Dec put them, shifted
    # or not: as the cube alone, with the galaxy at x - 2
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from gbkfit.observation import ObservationGroup, ObservationSky
    origin = (150.0, 2.0)
    first = SkyCoord(*origin, unit='deg')
    west = first.spherical_offsets_by(-2 * u.arcsec, 0 * u.arcsec)
    east, north = west.spherical_offsets_to(first)
    dx, dy = -east.to_value('arcsec'), north.to_value('arcsec')

    def observation(name, rval, lens=None):
        observable = IMAGE | dict(rval=list(rval))
        plain = observation_group([dict(
            driver=dict(type=driver.type()), galaxy=GALAXY,
            line_of_sight=dict(type='flat'),
            observable=observable)]).observations()[0]
        return ObservationSky(
            plain.observable(), plain.line_of_sight(), name=name,
            foreground=lens)
    maps = LensDeflectionMap(
        *sis_maps(), step=0.5, rval=origin, shifted=shifted)
    lens = Foreground(Lens((40, 40), (0.5, 0.5), maps))
    cube = observation('obs', (west.ra.deg, west.dec.deg), lens)
    galaxies = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=GALAXY,
        line_of_sight=dict(type='flat'),
        observable=copy.deepcopy(IMAGE))]).galaxies()
    shift = lens_params(xpos=0, ypos=0) if shifted else {}
    images = []
    for observations, (x, y) in (
            ([observation('reference', origin), cube], (0, 0)),
            ([cube], (dx, dy))):
        group = ObservationGroup(galaxies, observations)
        params = gbkfit.params.ParamSpace.from_properties(
            group.pdescs(), PROPERTIES | dict(xpos=x, ypos=y) | shift)
        images.append(group.plan(driver).model_h(params.evaluate())[-1][
            'brightness'].data.copy())
    assert dx == pytest.approx(-2)
    np.testing.assert_allclose(images[0], images[1], rtol=1e-6, atol=1e-6)


def test_a_lens_is_placed_again_only_for_other_values(driver, monkeypatch):
    # Its deflection, costly (seconds for an elliptical NFW halo on
    # 400x400 pixels), is worked out again only when the values of its
    # parameters change, not those of the galaxies alone (it was at every
    # evaluation: audit 2026-11-11)
    from gbkfit.observation import ObservationGroup, ObservationSky
    calls = []
    deflection = LensSIE.deflection

    def counted(self, grid, params):
        calls.append(params)
        return deflection(self, grid, params)
    monkeypatch.setattr(LensSIE, 'deflection', counted)
    group = observation_group([dict(
        driver=dict(type=driver.type()), galaxy=GALAXY,
        line_of_sight=dict(type='flat'), observable=copy.deepcopy(IMAGE))])
    plain = group.observations()[0]
    foreground = Foreground(Lens((40, 40), (0.5, 0.5), LensSIE()))
    observation = ObservationSky(
        plain.observable(), plain.line_of_sight(), foreground=foreground,
        name='obs')
    plan = ObservationGroup(group.galaxies(), [observation]).plan(driver)
    cases = [
        PROPERTIES | lens_params(
            theta_e=theta_e, q=0.8, posa=20, xpos=0, ypos=0)
        | dict(rpt_s=rpt_s)
        for rpt_s, theta_e in ((2, 3.0), (3, 3.0), (3, 4.0))]
    images = []
    for properties in cases:
        params = gbkfit.params.ParamSpace.from_properties(
            plan.group().pdescs(), properties)
        images.append(plan.model_h(params.evaluate())[0][
            'brightness'].data.copy())
    assert len(calls) == 2
    for image, properties in zip(images, cases):
        np.testing.assert_allclose(
            image, evaluate(driver, foreground, properties), rtol=1e-6,
            atol=1e-7)


def test_the_source_plane_is_where_the_image_is_on_the_sky():
    # Its centre, the origin of the frame of the model, is at the RA and
    # Dec of the reference pixel of the image, as the extra outputs of the
    # model on it say (it was at 0, 0: audit 2026-11-11)
    from gbkfit.driver.drivers.host import DriverHost
    lens = Lens((10, 10), (0.5, 0.5), LensSIE())
    image = make_grid((8, 8), 1.0, rval=(150.0, 2.0))
    plan = lens.plan(
        DriverHost(), image, np.dtype(np.float32),
        {name: name for name in lens.pdescs()})
    assert plan.source_grid(image).coords.rval == image.coords.rval


def test_a_lens_refuses_pixel_weights_as_the_observation_is_made():
    # (not first when it is planned on a driver, which describe and check
    # do not do: audit 2026-10-31 #51)
    from gbkfit.observation import ObservationSky, weights_parser
    plain = observation_group([dict(
        driver=dict(type='host'), galaxy=GALAXY,
        line_of_sight=dict(type='flat'), observable=copy.deepcopy(IMAGE))
    ]).observations()[0]
    with pytest.raises(ValueError, match="a lens does not support"):
        ObservationSky(
            plain.observable(), plain.line_of_sight(),
            foreground=Foreground(Lens((40, 40), (0.5, 0.5), LensSIE())),
            pixel_weights=weights_parser.load(dict(component='disk', traits=[
                dict(type='azimuth_power', power=1)])),
            name='obs')


def test_a_scaled_map_scales_its_deflection():
    # (e.g. by D_LS / D_S, of a source at another redshift than the map's)
    lens = LensDeflectionMap(
        np.full((5, 5), 2.0), np.full((5, 5), -1.0), scaled=True)
    assert set(lens.pdescs()) == {'scale'}
    grid = make_grid((3, 3), 1)
    np.testing.assert_allclose(
        lens.deflection(grid, dict(scale=0.5)),
        [np.full((3, 3), 1.0), np.full((3, 3), -0.5)])

def test_deflection_maps_from_files(tmp_path):
    coords = Coords((0.5, 0.5), (9.5, 7.5), (150.0, 2.0), 0)
    write_data(str(tmp_path / 'ax.fits'), np.ones((16, 20)), coords)
    write_data(
        str(tmp_path / 'ay.fits'), np.full((16, 20), 2.0), coords)
    def load(**options):
        return foreground_parser.load(dict(lens=dict(
            source_size=[10, 10], source_step=[0.5, 0.5],
            masses=dict(
                type='deflection_map', alpha_x=str(tmp_path / 'ax.fits'),
                alpha_y=str(tmp_path / 'ay.fits'), **options))))
    foreground = load()
    lens = foreground.lens()
    grid = make_grid((3, 2), 0.5, rval=(150.0, 2.0))
    deflection = lens.deflection(grid, {})
    np.testing.assert_allclose(
        deflection, [np.ones((2, 3)), np.full((2, 3), 2)])
    dumped = foreground_parser.dump(foreground, prefix=str(tmp_path / 'd_'))
    again = foreground_parser.load(copy.deepcopy(dumped))
    assert again.lens().source_grid() == lens.source_grid()
    # The world coordinates given replace those of the files
    stretched = load(step=[1, 1], rpix=[0, 0]).lens().masses()[0]
    assert stretched.grid().coords.step == (1, 1)
    assert stretched.grid().coords.rpix == (0, 0)
    # (without parameters, unless shifted or scaled)
    assert lens.pdescs() == {}
    assert set(load(shifted=True).lens().pdescs()) == {'xpos', 'ypos'}


def write_sis_maps(path, axes, dec_first=False, size=20):
    """
    Write the deflection of a singular isothermal sphere of theta_e 1 at
    RA 150, Dec 2 to files of maps, along their own pixel axes, of pixels
    whose axes are the columns of axes: arcsec along x and y of the frame
    of the model (+x west, +y north) of a step along the first and the
    second axis of the files; with Dec first, if given. Return the names
    of the files of x and y.
    """
    axes = np.asarray(axes, dtype=float)
    # (the reference pixel between pixels, off the centre of the sphere)
    centre = (size - 1) / 2
    j, i = np.mgrid[0:size, 0:size]
    position = np.tensordot(axes, np.stack([i - centre, j - centre]), 1)
    alpha = position / np.hypot(*position)
    units = axes / np.hypot(*axes)
    along = [np.tensordot(units[:, k], alpha, 1) for k in range(2)]
    # (the rows of CD are east and north, in degrees; Dec first, swapped)
    cd = np.array([-axes[0], axes[1]]) / 3600
    world = [('RA---TAN', 150.0), ('DEC--TAN', 2.0)]
    if dec_first:
        cd, world = cd[::-1], world[::-1]
    header = fits.Header()
    for a, (ctype, crval) in enumerate(world, 1):
        header.update({
            f'CTYPE{a}': ctype, f'CRVAL{a}': crval, f'CRPIX{a}': centre + 1,
            f'CUNIT{a}': 'deg', f'CD{a}_1': cd[a - 1, 0],
            f'CD{a}_2': cd[a - 1, 1]})
    names = [str(path / f'{key}.fits') for key in ('ax', 'ay')]
    for name, data in zip(names, along):
        fits.writeto(name, data, header)
    return names


def turned(angle, mirrored=False, step=0.5):
    """
    The axes of pixels (see write_sis_maps) of a grid of rotation angle,
    with x mirrored (east to the right of north), if given.
    """
    angle = np.radians(angle)
    axes = step * np.array([
        [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return axes * [[-1], [1]] if mirrored else axes


@pytest.mark.parametrize('axes, dec_first', [
    (turned(0), False),
    # East to the right (CDELT1 > 0)
    (turned(0, mirrored=True), False),
    # Dec first: the first axis north, the second west
    (turned(0)[:, ::-1], True),
    (turned(30), False),
    (turned(30, mirrored=True)[:, ::-1], True)])
def test_deflection_maps_from_files_turn_their_vectors_with_them(
        tmp_path, axes, dec_first):
    # The files give the deflection along their own axes, which the maps
    # are read in the orientation of the model with: a mirrored sphere is
    # the sphere, as LensSIE (the maps of x were negated, swapped or
    # turned: audit 2026-11-11 d#1), and so are the maps dumped
    alpha_x, alpha_y = write_sis_maps(tmp_path, axes, dec_first)
    mass = LensDeflectionMap.from_files(alpha_x, alpha_y)
    # (at the pixels of the maps, as they are)
    grid = mass.grid()
    sphere = LensSIE().deflection(
        grid, dict(theta_e=1, q=1, posa=0, xpos=0, ypos=0))
    np.testing.assert_allclose(
        mass.deflection(grid, {}), sphere, rtol=0, atol=1e-8)
    dumped = lens_mass_parser.dump(mass, prefix=str(tmp_path / 'd_'))
    np.testing.assert_allclose(
        lens_mass_parser.load(dumped).deflection(grid, {}), sphere,
        rtol=0, atol=1e-8)


def test_a_shifted_map_is_sampled_at_the_shifted_positions():
    # Its deflection at a point is that of the maps at the point less the
    # shift: at the pixels of a grid moved by it
    j, i = np.mgrid[0:12, 0:10]
    alpha_x, alpha_y = 1.0 * i + 0.1 * j ** 2, np.sin(0.3 * i * j)
    shifted = LensDeflectionMap(
        alpha_x, alpha_y, step=0.5, rota=20, shifted=True)
    plain = LensDeflectionMap(alpha_x, alpha_y, step=0.5, rota=20)
    assert set(shifted.pdescs()) == {'xpos', 'ypos'}
    assert shifted.shifted() and not plain.shifted()
    # (a shift of 2 and -1 pixels of 0.5 arcsec)
    grid = make_grid((6, 5), 0.5, rpix=(2.5, 2))
    moved = make_grid((6, 5), 0.5, rpix=(4.5, 1))
    np.testing.assert_allclose(
        shifted.deflection(grid, dict(xpos=1, ypos=-0.5)),
        plain.deflection(moved, {}), rtol=1e-12, atol=1e-12)
    info = lens_mass_parser.dump(shifted, prefix='')
    assert info['shifted'] is True and 'scaled' not in info


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
    with pytest.raises(ValueError, match="one shape"):
        LensDeflectionMap(np.zeros((4, 4)), np.zeros((4, 5)))
