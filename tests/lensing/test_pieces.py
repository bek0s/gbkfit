"""
Tests of the pieces of lenses (gbkfit.lensing.pieces): each deflects
the light at points of the plane of its lens, as its convergence says
(div alpha = 2 kappa, curl alpha = 0).
"""

import copy

import numpy as np
import pytest

from gbkfit.config import ConfigError
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.grid import Coords, make_grid, sky_positions
from gbkfit.io import write_data
from gbkfit.lensing import (
    LensDeflectionMap, LensEPL, LensNFW, LensSIE, LensShear, LensSheet,
    Normalisation, SkyPoints, lens_piece_parser)
from gbkfit.params import ModelEvaluationError


# Points around the centre of the shapes (off it), and the shape of an
# ellipse
ORIGIN = (150.0, 2.0)

POINTS = SkyPoints(*sky_positions(make_grid((7, 6), 0.37)), ORIGIN)

ELLIPSE = dict(q=0.6, posa=30, xpos=0.2, ypos=-0.1)


def deflect(piece, points=POINTS, **params):
    return piece.deflection(points, params, DriverHost())


def moved(points, dx, dy):
    return SkyPoints(points.x + dx, points.y + dy, points.origin)


def derivatives(piece, params, points=POINTS, h=1e-5):
    """
    The divergence and the curl of the deflection of a piece, by central
    differences.
    """
    def alpha(dx, dy):
        return deflect(piece, moved(points, dx, dy), **params)
    ahead_x, behind_x = alpha(h, 0), alpha(-h, 0)
    ahead_y, behind_y = alpha(0, h), alpha(0, -h)
    divergence = (ahead_x[0] - behind_x[0] + ahead_y[1] - behind_y[1]) \
        / (2 * h)
    curl = (ahead_x[1] - behind_x[1] - ahead_y[0] + behind_y[0]) / (2 * h)
    return divergence, curl


def elliptical_radius(params, points=POINTS):
    x, y = points.x - params['xpos'], points.y - params['ypos']
    pa = np.radians(params['posa'])
    u = -np.sin(pa) * x + np.cos(pa) * y
    v = np.cos(pa) * x + np.sin(pa) * y
    return np.sqrt(params['q'] * u ** 2 + v ** 2 / params['q'])


def multipole_kappa(order, theta_e, t, a, posa, xpos, ypos, points=POINTS):
    x, y = points.x - xpos, points.y - ypos
    phi = np.arctan2(y, x)
    return a / 2 * (theta_e / np.hypot(x, y)) ** t * np.cos(
        order * (phi - np.radians(posa + 90)))


def test_the_normalisation_of_each_kind(tmp_path):
    # Maps are at D_ls / D_s = 1, or quoted for one source if made so
    quoted = [LensSIE(), LensEPL(), LensNFW(), LensShear(), LensSheet()]
    assert all(
        piece.normalisation() is Normalisation.QUOTED for piece in quoted)
    maps = np.zeros((2, 2)), np.zeros((2, 2))
    assert LensDeflectionMap(*maps).normalisation() is Normalisation.UNIT
    made_for_one_source = LensDeflectionMap(*maps, quoted=True)
    assert made_for_one_source.normalisation() is Normalisation.QUOTED
    dumped = lens_piece_parser.dump(
        made_for_one_source, prefix=str(tmp_path / 'd_'))
    assert dumped['quoted'] is True
    assert lens_piece_parser.load(dumped).normalisation() \
        is Normalisation.QUOTED


def test_the_deflection_of_an_sie_is_that_of_its_convergence():
    # div alpha = 2 kappa, with kappa = theta_e / (2 xi); and q = 1 is the
    # isothermal sphere, theta_e r / |r|
    params = ELLIPSE | dict(theta_e=1.3)
    divergence, curl = derivatives(LensSIE(), params)
    np.testing.assert_allclose(
        divergence, 1.3 / elliptical_radius(params), rtol=1e-6)
    np.testing.assert_allclose(curl, 0, atol=1e-6)
    x, y = POINTS.x - 0.2, POINTS.y + 0.1
    r = np.hypot(x, y)
    np.testing.assert_allclose(
        deflect(LensSIE(), **params | dict(q=1)), [1.3 * x / r, 1.3 * y / r])


def test_the_deflection_of_an_epl_is_that_of_its_convergence():
    # kappa = (2 - t) / 2 (theta_e / xi)^t, t = slope - 1; slope 2 is the
    # SIE
    params = ELLIPSE | dict(theta_e=1.3, slope=2.3)
    divergence, _ = derivatives(LensEPL(), params)
    kappa = 0.7 / 2 * (1.3 / elliptical_radius(params)) ** 1.3
    np.testing.assert_allclose(divergence, 2 * kappa, rtol=1e-6)
    np.testing.assert_allclose(
        deflect(LensEPL(), **params | dict(slope=2)),
        deflect(LensSIE(), **ELLIPSE | dict(theta_e=1.3)),
        rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('q', [1, 0.5])
def test_the_deflection_of_an_nfw_halo_is_that_of_its_convergence(q):
    # kappa = 2 kappa_s (1 - F(x)) / (x^2 - 1), x = xi / r_s; within and
    # beyond r_s
    params = ELLIPSE | dict(q=q, kappa_s=0.3, r_s=0.8)
    x = elliptical_radius(params) / 0.8
    root = np.sqrt(np.abs(x ** 2 - 1))
    within = x < 1
    f = np.arctan(root) / root
    f[within] = np.arctanh(root[within]) / root[within]
    divergence, _ = derivatives(LensNFW(), params)
    np.testing.assert_allclose(
        divergence, 4 * 0.3 * (1 - f) / (x ** 2 - 1), rtol=1e-5)


@pytest.mark.parametrize('order', [1, 3, 4])
@pytest.mark.parametrize('slope', [2, 1.7, 2.4])
def test_the_multipoles_of_a_shape_are_those_of_their_convergence(
        order, slope):
    # kappa = a / 2 (theta_e / r)^t cos(m (phi - phi_m)), t = slope - 1,
    # about the centre of the shape, added to its deflection
    shape = dict(theta_e=1.3, slope=slope) | ELLIPSE
    multipole = {f'a{order}': 0.05, f'posa{order}': 30}
    divergence, curl = derivatives(
        LensEPL(multipoles=[order]), shape | multipole)
    plain, _ = derivatives(LensEPL(), shape)
    kappa = multipole_kappa(order, 1.3, slope - 1, 0.05, 30, 0.2, -0.1)
    np.testing.assert_allclose(divergence - plain, 2 * kappa, atol=1e-6)
    np.testing.assert_allclose(curl, 0, atol=1e-6)


@pytest.mark.parametrize('order', [1, 3, 4])
def test_the_multipoles_of_an_sie_are_those_of_an_isothermal_epl(order):
    params = ELLIPSE | {
        'theta_e': 1.3, f'a{order}': 0.05, f'posa{order}': 30}
    np.testing.assert_allclose(
        deflect(LensEPL(multipoles=[order]), **params | dict(slope=2)),
        deflect(LensSIE(multipoles=[order]), **params),
        rtol=1e-12, atol=1e-14)


def test_the_multipole_of_order_1_is_continuous_at_the_isothermal_slope():
    # (its potential is a log at slope 2 exactly, a power elsewhere)
    params = ELLIPSE | dict(theta_e=1.3, a1=0.05, posa1=30)
    exact = deflect(LensEPL(multipoles=[1]), **params | dict(slope=2))
    for slope in (2 - 1e-7, 2 + 1e-7):
        np.testing.assert_allclose(
            deflect(LensEPL(multipoles=[1]), **params | dict(slope=slope)),
            exact, atol=1e-6)


@pytest.mark.parametrize('order', [1, 3, 4])
def test_the_position_angle_of_a_multipole_repeats_every_360_over_m(order):
    piece = LensSIE(multipoles=[order])
    assert piece.pdescs()[f'posa{order}'].period() == 360 / order
    params = ELLIPSE | {
        'theta_e': 1.3, f'a{order}': 0.05, f'posa{order}': 30}
    np.testing.assert_allclose(
        deflect(piece, **params | {f'posa{order}': 30 + 360 / order}),
        deflect(piece, **params), atol=1e-12)


def test_a_multipole_of_order_2_about_a_uniform_sheet_has_no_deflection():
    # (no power of r is a solution there: slope 1 is a uniform sheet)
    params = ELLIPSE | dict(theta_e=1.3, slope=1, a2=0.05, posa2=0)
    with pytest.raises(ModelEvaluationError, match="order 2"):
        deflect(LensEPL(multipoles=[2]), **params)


def test_the_options_of_multipoles():
    piece = lens_piece_parser.load(dict(type='sie', multipoles=[4, 3]))
    assert piece.multipoles() == (4, 3)
    assert list(piece.pdescs())[-4:] == ['a4', 'posa4', 'a3', 'posa3']
    assert lens_piece_parser.dump(piece) == dict(
        type='sie', multipoles=[4, 3])
    assert lens_piece_parser.dump(LensEPL()) == dict(type='epl')
    with pytest.raises(ConfigError):
        lens_piece_parser.load(dict(type='epl', multipoles=[0]))
    with pytest.raises(ValueError, match="repeat"):
        LensSIE(multipoles=[4, 4])


def test_a_shear_stretches_along_its_position_angle():
    # alpha = gamma r along the position angle, and -gamma r across it,
    # 0 at the origin of the frame
    pa = np.radians(30)
    along = np.array([-np.sin(pa), np.cos(pa)])
    across = np.array([np.cos(pa), np.sin(pa)])
    for direction, sign in ((along, 1), (across, -1)):
        point = SkyPoints(*(2 * direction[:, None]), ORIGIN)
        alpha = deflect(LensShear(), point, gamma=0.1, posa=30)
        np.testing.assert_allclose(
            np.ravel(alpha), sign * 0.1 * 2 * direction, atol=1e-15)


def test_a_sheet_deflects_by_its_convergence_times_the_position():
    divergence, curl = derivatives(LensSheet(), dict(kappa=0.07))
    np.testing.assert_allclose(divergence, 2 * 0.07)
    np.testing.assert_allclose(curl, 0, atol=1e-9)
    np.testing.assert_allclose(
        deflect(LensSheet(), kappa=0.07), [0.07 * POINTS.x, 0.07 * POINTS.y])


# Deflection maps: their pixels, and points within them
def map_points(grid):
    """Points within the span of the centres of the pixels of a grid."""
    x, y = sky_positions(make_grid(
        (5, 4), 0.3, rval=grid.coords.rval, rota=13))
    return SkyPoints(x, y, grid.coords.rval)


def test_a_map_deflects_by_its_bilinear_interpolation():
    # A deflection linear in the pixels is its own interpolation
    j, i = np.mgrid[0:12, 0:10]
    piece = LensDeflectionMap(0.5 * i - 0.2 * j, 0.3 * j + 1, step=0.5)
    x, y = sky_positions(piece.grid())
    pixels = SkyPoints(x, y, piece.grid().coords.rval)
    np.testing.assert_allclose(
        deflect(piece, pixels), [0.5 * i - 0.2 * j, 0.3 * j + 1],
        atol=1e-12)
    points = map_points(piece.grid())
    i, j = (points.x + 2.25) / 0.5, (points.y + 2.75) / 0.5
    np.testing.assert_allclose(
        deflect(piece, points), [0.5 * i - 0.2 * j, 0.3 * j + 1],
        rtol=1e-6)


def test_a_map_far_from_the_origin_of_the_frame_is_turned_into_it():
    # Maps 2 arcmin east of the origin of the frame, at Dec 80 deg: the
    # meridians there turn by about 3e-3 rad, 0.06 arcsec of a deflection
    # of 20 arcsec. The deflection, as astropy moves the points of the
    # frame into the projection of the maps, deflects them there and
    # moves the source back
    from astropy.wcs import WCS

    def tangent(ra, dec):
        # (pixels of 1 arcsec, x to the west, 0 at the reference)
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [1, 1]
        wcs.wcs.cdelt = [-1 / 3600, 1 / 3600]
        return wcs
    origin = (150.0, 80.0)
    reference = (150.0 + 2 / 60 / np.cos(np.radians(80.0)), 80.0)
    frame, maps = tangent(*origin), tangent(*reference)
    alpha = (20.0, -5.0)
    piece = LensDeflectionMap(
        np.full((21, 21), alpha[0]), np.full((21, 21), alpha[1]), step=1.0,
        rval=reference)
    centre = frame.wcs_world2pix([reference], 0)[0]
    x, y = np.meshgrid(
        centre[0] + np.linspace(-5, 5, 4), centre[1] + np.linspace(-5, 5, 3))
    sky = frame.wcs_pix2world(np.stack([x.ravel(), y.ravel()], -1), 0)
    on_maps = maps.wcs_world2pix(sky, 0) - alpha
    source = frame.wcs_world2pix(maps.wcs_pix2world(on_maps, 0), 0)
    expected = np.stack([x.ravel(), y.ravel()], -1) - source
    deflection = deflect(piece, SkyPoints(x, y, origin))
    np.testing.assert_allclose(
        np.stack([d.ravel() for d in deflection], -1), expected, atol=1e-8)
    # (that it is turned, and by about that much)
    assert np.abs(expected - alpha).max() > 0.03


def test_a_map_has_no_deflection_off_its_pixels():
    piece = LensDeflectionMap(
        np.ones((4, 5)), np.ones((4, 5)), step=1, rval=ORIGIN)
    # (the span of the centres of its pixels: -2 to 2 along x, -1.5 to
    # 1.5 along y)
    inside = SkyPoints(np.array([-2.0, 2.0]), np.array([-1.5, 1.5]), ORIGIN)
    deflect(piece, inside)
    outside = SkyPoints(np.array([0.0, 2.2]), np.array([0.0, 0.0]), ORIGIN)
    with pytest.raises(ModelEvaluationError, match="1 of the 2 points"):
        deflect(piece, outside)


def test_a_map_is_placed_by_its_world_coordinates():
    # Points in a frame of another origin, 2 arcsec east of the maps' (+x
    # is to the west); the gnomonic projections about the two origins
    # differ by about 1e-7 arcsec there
    j, i = np.mgrid[0:12, 0:10]
    alpha_x, alpha_y = 0.5 * i - 0.2 * j, np.sin(0.3 * i * j)
    piece = LensDeflectionMap(alpha_x, alpha_y, step=0.5, rval=ORIGIN)
    here = map_points(piece.grid())
    east = (ORIGIN[0] + 2 / 3600 / np.cos(np.radians(ORIGIN[1])), ORIGIN[1])
    there = SkyPoints(here.x + 2, here.y, east)
    np.testing.assert_allclose(
        deflect(piece, there), deflect(piece, here), atol=1e-5)


def test_a_shifted_map_is_sampled_at_the_shifted_points():
    j, i = np.mgrid[0:12, 0:10]
    alpha_x, alpha_y = 1.0 * i + 0.1 * j ** 2, np.sin(0.3 * i * j)
    shifted = LensDeflectionMap(
        alpha_x, alpha_y, step=0.5, rota=20, shifted=True)
    plain = LensDeflectionMap(alpha_x, alpha_y, step=0.5, rota=20)
    assert set(shifted.pdescs()) == {'xpos', 'ypos'}
    assert plain.pdescs() == {}
    points = map_points(plain.grid())
    np.testing.assert_allclose(
        deflect(shifted, moved(points, 1, -0.5), xpos=1, ypos=-0.5),
        deflect(plain, points), rtol=1e-12, atol=1e-12)


def test_maps_must_be_two_finite_images_of_one_shape():
    with pytest.raises(ValueError, match="one shape"):
        LensDeflectionMap(np.zeros((4, 4)), np.zeros((4, 5)))
    with pytest.raises(ValueError, match="two pixels"):
        LensDeflectionMap(np.zeros((1, 4)), np.zeros((1, 4)))
    with pytest.raises(ValueError, match="finite"):
        LensDeflectionMap(np.full((3, 3), np.nan), np.zeros((3, 3)))


def write_maps(path, alpha_x, alpha_y, coords):
    names = [str(path / f'{key}.fits') for key in ('ax', 'ay')]
    for name, data in zip(names, (alpha_x, alpha_y)):
        write_data(name, data, coords)
    return names


def test_maps_from_files_in_arcsec_or_in_pixels(tmp_path):
    coords = Coords((0.5, 0.25), (9.5, 7.5), ORIGIN, 0)
    j, i = np.mgrid[0:16, 0:20]
    alpha_x, alpha_y = write_maps(tmp_path, 0.1 * i, 0.2 * j + 1, coords)
    arcsec = lens_piece_parser.load(dict(
        type='deflection_map', alpha_x=alpha_x, alpha_y=alpha_y))
    pixel = lens_piece_parser.load(dict(
        type='deflection_map', alpha_x=alpha_x, alpha_y=alpha_y,
        unit='pixel'))
    points = map_points(arcsec.grid())
    expected = np.array(deflect(arcsec, points))
    np.testing.assert_allclose(
        deflect(pixel, points), expected * [[[0.5]], [[0.25]]])
    # Dumped, as read in arcsec
    dumped = lens_piece_parser.dump(
        pixel, prefix=str(tmp_path / 'd_'))
    assert 'unit' not in dumped
    np.testing.assert_allclose(
        deflect(lens_piece_parser.load(copy.deepcopy(dumped)), points),
        deflect(pixel, points))
    # The world coordinates given replace those of the files
    stretched = lens_piece_parser.load(dict(
        type='deflection_map', alpha_x=alpha_x, alpha_y=alpha_y,
        step=[1, 1], rpix=[0, 0]))
    assert stretched.grid().coords.step == (1, 1)
    assert stretched.grid().coords.rpix == (0, 0)
