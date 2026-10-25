"""
Tests for the spheroids (Spheroid): their light in images against the
exact projections of their profiles, the mean density near the centre
of a cusp, and their options.
"""

import numpy as np
import pytest
import scipy.special

from gbkfit.model import Spheroid, component_parser, galaxy_parser
from gbkfit.model.mass import _numerics
from gbkfit.utils.parseutils import ConfigError


BAND = dict(type='band')

GEOMETRY = dict(xpos=1.5, ypos=-1, posa=30, incl=60)

# The pixels of the images (see image), in arcsec
STEP = 0.5


def image(driver, spheroid, dtype='float32', size=64, flat=False):
    """
    A case of an image of size by size pixels of STEP arcsec of a
    spheroid, with a depth axis of voxels of 0.25 arcsec, or without one
    (flat).
    """
    line_of_sight = dict(type='flat') if flat \
        else dict(type='grid', depth=dict(step=0.25))
    return dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_brightness', size=[size, size], step=[STEP, STEP],
            dtype=dtype),
        line_of_sight=line_of_sight,
        galaxy=dict(components=[spheroid]))


def sky_frame(xpos, ypos, posa, incl, size=64):
    """
    The position of each pixel of an image of size by size pixels (see
    image) along the major and the minor axis of the projection of a
    spheroid of the given centre and axis (see the cones' cone_frame), and
    the axis ratio of the projection of a spheroid of the axis ratio q, as
    a function of q: q'^2 = cos^2(incl) + q^2 sin^2(incl).
    """
    pixels = (np.arange(size) - (size - 1) / 2) * STEP
    y, x = np.meshgrid(pixels, pixels, indexing='ij')
    p = np.radians(posa)
    x, y = x - xpos, y - ypos
    major = -x * np.sin(p) + y * np.cos(p)
    minor = -x * np.cos(p) - y * np.sin(p)
    i = np.radians(incl)
    return major, minor, lambda q: np.sqrt(
        np.cos(i) ** 2 + q ** 2 * np.sin(i) ** 2)


def sersic_light(n, ie, re):
    """The total light of a Sersic profile."""
    b = _numerics.sersic_b(n)
    return 2 * np.pi * n * re ** 2 * ie * np.exp(b) * b ** (-2 * n) \
        * scipy.special.gamma(2 * n)


def test_a_flattened_sersic_spheroid_projects_exactly(driver, evaluate_cases):
    # Seen at the inclination i, the density rho_sph(m) / q of axis ratio q
    # projects to Sigma_sph(m') / q', m'^2 = x'^2 + y'^2 / q'^2: the Sersic
    # profile of the sphere, of the axis ratio q'. Away from the centre
    # (where the profile peaks within a pixel) and from rmax (7 re, which
    # cuts the lines of sight; n = 1, of little light beyond it). The total
    # light is that of the profile, up to the sampling of the pixels at
    # their centres (about 0.1% for re of 6 pixels).
    spheroid = dict(
        type='spheroid', profile=dict(type='sersic'),
        shape=dict(type='flattened'), rmax=21, emission=BAND)
    properties = GEOMETRY | dict(ie=2, re=3, n=1, q=0.6)
    data, _ = evaluate_cases(
        [image(driver, spheroid, size=96)], properties)
    major, minor, projected = sky_frame(**GEOMETRY, size=96)
    qp = projected(0.6)
    m = np.sqrt(major ** 2 + (minor / qp) ** 2)
    b = _numerics.sersic_b(1)
    expected = 2 * np.exp(-b * (m / 3 - 1)) / qp
    model = data[0]['brightness']['d']
    within = (m > 3) & (m < 9)
    np.testing.assert_allclose(model[within], expected[within], rtol=2e-3)
    flux = model.sum() * STEP ** 2
    assert flux == pytest.approx(sersic_light(1, 2, 3), rel=2e-3)


def test_an_mge_spheroid_projects_exactly(driver, evaluate_cases):
    # Gaussian k of axis ratio q_k projects to i0_k / q'_k exp(-(x'^2 +
    # y'^2 / q'_k^2) / (2 sigma_k^2)), of the light 2 pi i0_k sigma_k^2.
    # The voxels within 3 voxel diagonals (2.25 arcsec) of the centre hold
    # their mean density, and the lines of sight beyond it their density
    # at their centres: the light differs where the two meet, by the
    # curvature of the profile times the square of the voxel (0.2% here).
    spheroid = dict(
        type='spheroid', profile=dict(type='mge', ngauss=2), rmax=14,
        emission=BAND)
    i0, sigma, q = [3, 1], [0.8, 2.5], [0.9, 0.4]
    properties = GEOMETRY | dict(i0=i0, sigma=sigma, q=q)
    data, _ = evaluate_cases([image(driver, spheroid)], properties)
    major, minor, projected = sky_frame(**GEOMETRY)
    expected = sum(
        a / projected(qk) * np.exp(
            -(major ** 2 + (minor / projected(qk)) ** 2) / (2 * s ** 2))
        for a, s, qk in zip(i0, sigma, q))
    model = data[0]['brightness']['d']
    away = np.hypot(major, minor) > 2.25
    np.testing.assert_allclose(model[away], expected[away], atol=1e-5)
    flux = model.sum() * STEP ** 2
    total = sum(2 * np.pi * a * s ** 2 for a, s in zip(i0, sigma))
    assert flux == pytest.approx(total, rel=3e-3)


@pytest.mark.parametrize('profile, properties', [
    (dict(type='sersic'), dict(ie=2, re=3, n=4, q=0.6)),
    (dict(type='mge', ngauss=2), dict(i0=[3, 1], sigma=[0.8, 2.5],
                                      q=[0.9, 0.4])),
])
def test_without_a_depth_axis_a_spheroid_is_its_exact_projection(
        driver, evaluate_cases, caplog, profile, properties):
    # Sigma_sph(m') / q' of each term (see the tests above), to any radius:
    # rmax cuts nothing, and warns of nothing. The pixels within 3 pixel
    # diagonals of the centre hold their mean (where n = 4 peaks).
    shape = dict(shape=dict(type='flattened')) \
        if profile['type'] == 'sersic' else {}
    spheroid = dict(
        type='spheroid', profile=profile, rmax=1, emission=BAND) | shape
    data, _ = evaluate_cases(
        [image(driver, spheroid, 'float64', flat=True)],
        GEOMETRY | properties)
    major, minor, projected = sky_frame(**GEOMETRY)
    if profile['type'] == 'sersic':
        qp = projected(0.6)
        b = _numerics.sersic_b(4)
        m = np.sqrt(major ** 2 + (minor / qp) ** 2)
        expected = 2 * np.exp(-b * ((m / 3) ** (1 / 4) - 1)) / qp
    else:
        expected = sum(
            a / projected(qk) * np.exp(
                -(major ** 2 + (minor / projected(qk)) ** 2) / (2 * s ** 2))
            for a, s, qk in zip(
                properties['i0'], properties['sigma'], properties['q']))
    away = np.hypot(major, minor) > 3 * np.hypot(STEP, STEP)
    np.testing.assert_allclose(
        data[0]['brightness']['d'][away], expected[away], rtol=1e-9)
    assert 'its rmax cuts' not in caplog.text


def test_with_and_without_a_depth_axis_a_spheroid_agrees(
        driver, evaluate_cases):
    # Its voxels, summed along the line of sight, and its projection
    spheroid = dict(
        type='spheroid', profile=dict(type='sersic'),
        shape=dict(type='flattened'), rmax=21, emission=BAND)
    properties = GEOMETRY | dict(ie=2, re=3, n=1, q=0.6)
    (grid,), _ = evaluate_cases(
        [image(driver, spheroid, size=96)], properties)
    (flat,), _ = evaluate_cases(
        [image(driver, spheroid, size=96, flat=True)], properties)
    grid, flat = grid['brightness']['d'], flat['brightness']['d']
    bright = flat > 1e-2 * flat.max()
    np.testing.assert_allclose(grid[bright], flat[bright], rtol=5e-3)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_the_cusp_of_a_sersic_sphere_keeps_its_light(
        driver, evaluate_cases, dtype):
    # The density of n = 4 is infinite at the centre: the voxels there hold
    # its mean along the line of sight, and the image is finite, with the
    # light of the sphere within rmax: the total times the share of the
    # deprojected profile within rmax / re, up to the sampling of the
    # pixels at their centres (with 3 by 3 pixels in each: about 0.3%).
    # The centre of the sphere is the centre of a voxel (where the density
    # at the centre of the voxel is infinite).
    spheroid = dict(
        type='spheroid', profile=dict(type='sersic'), rmax=12,
        emission=BAND)
    properties = dict(xpos=0, ypos=0, ie=1, re=2, n=4)
    case = image(driver, spheroid, dtype)
    case['observable'] |= dict(oversampling=[3, 3])
    data, _ = evaluate_cases([case], properties)
    model = data[0]['brightness']['d']
    assert np.isfinite(model).all()
    flux = model.sum() * STEP ** 2
    share = float(_numerics.sersic_tables(4).mass(12 / 2))
    assert flux == pytest.approx(sersic_light(4, 1, 2) * share, rel=4e-3)


@pytest.mark.parametrize('q', [0.5, 1.6])
def test_rmax_cuts_each_term_at_its_spheroidal_radius(
        driver, evaluate_cases, q):
    # A Gaussian of sigma 3 arcsec cut at m = rmax / max(1, q) (rmax 6):
    # the light left is the total times one minus the share that the
    # warning reports (see LightMGE.cut), up to the voxels along the edge
    spheroid = dict(
        type='spheroid', profile=dict(type='mge', ngauss=1), rmax=6,
        emission=BAND)
    properties = GEOMETRY | dict(i0=[1], sigma=[3], q=[q])
    data, _ = evaluate_cases([image(driver, spheroid)], properties)
    flux = data[0]['brightness']['d'].sum() * STEP ** 2
    _, cut = component_parser.load(dict(spheroid)).extent_cut(properties)
    assert 0.1 < cut < 0.9
    assert flux == pytest.approx(2 * np.pi * 9 * (1 - cut), rel=1e-2)


@pytest.mark.parametrize('rmax, warns', [(4, True), (12, False)])
def test_a_warning_says_when_rmax_cuts_the_light(
        driver, evaluate_cases, caplog, rmax, warns):
    # A sphere of n = 1 has 25% of its light beyond 2 re (rmax 4 for re of
    # 2 arcsec), and 0.1% beyond 6 re (rmax 12)
    spheroid = dict(
        type='spheroid', profile=dict(type='sersic'), rmax=rmax,
        emission=BAND)
    evaluate_cases(
        [image(driver, spheroid)], dict(xpos=0, ypos=0, ie=1, re=2, n=1))
    assert ('its rmax cuts' in caplog.text) == warns


@pytest.mark.parametrize('profile, properties', [
    (dict(type='sersic'), dict(ie=1, re=2, n=2, q=0)),
    (dict(type='sersic'), dict(ie=1, re=0, n=2, q=0.5)),
    (dict(type='mge', ngauss=2), dict(i0=[1, 2], sigma=[1, 2], q=[0, 0.9])),
])
def test_a_term_at_a_bound_of_zero_is_seen(
        driver, evaluate_cases, profile, properties):
    # A scale or an axis ratio of 0 (the bound of its parameter) makes the
    # light NaN, rather than an error or a term that vanishes
    shape = dict(shape=dict(type='flattened')) \
        if profile['type'] == 'sersic' else {}
    spheroid = dict(
        type='spheroid', profile=profile, rmax=8, emission=BAND) | shape
    data, _ = evaluate_cases(
        [image(driver, spheroid)], GEOMETRY | properties)
    assert np.isnan(data[0]['brightness']['d']).any()


def test_a_sphere_has_no_axis():
    spheroid = Spheroid.load(dict(
        profile=dict(type='sersic'), rmax=10, emission=BAND))
    assert set(spheroid.pdescs()) == {'xpos', 'ypos', 'ie', 're', 'n'}
    flattened = Spheroid.load(dict(
        profile=dict(type='sersic'), shape=dict(type='flattened'),
        rmax=10, emission=BAND))
    assert set(flattened.pdescs()) == {
        'xpos', 'ypos', 'posa', 'incl', 'ie', 're', 'n', 'q'}


def test_a_spheroid_round_trips():
    info = dict(
        type='spheroid', name='bulge', profile=dict(type='mge', ngauss=3),
        rmax=10, emission=BAND | dict(bands=['sdss_r']))
    galaxy = galaxy_parser.load(dict(components=[info]))
    dumped = galaxy.dump()['components'][0]
    assert dumped['profile'] == dict(type='mge', ngauss=3)
    assert dumped['rmax'] == 10
    assert galaxy_parser.load(dict(components=[dumped])).dump() \
        == galaxy.dump()


@pytest.mark.parametrize('options, message', [
    (dict(rmax=0), "rmax must be"),
    (dict(rmax=float('inf')), "rmax must be"),
    (dict(emission=dict(type='lines')), "emits in bands"),
    (dict(profile=dict(type='mge', ngauss=2),
          shape=dict(type='flattened')), "takes no shape"),
    (dict(profile=dict(type='mge', ngauss=0)), "ngauss must be"),
])
def test_invalid_spheroids_are_refused(options, message):
    info = dict(profile=dict(type='sersic'), rmax=10, emission=BAND) \
        | options
    with pytest.raises(ConfigError, match=message):
        Spheroid.load(info)
