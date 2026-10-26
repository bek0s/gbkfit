"""
Tests for the spheroids (Spheroid): their light in images against the
exact projections of their profiles, the mean density near the centre
of a cusp, and their options; the mass that follows their light
(mass.MassLightSpheroid); their Jeans kinematics (JeansKinematics),
against exact and independent solutions, and in spectral cubes of stars;
and the kinematics of their traits (TraitsKinematics).
"""

import functools

import numpy as np
import pytest
import scipy.integrate
import scipy.optimize
import scipy.special

from gbkfit.model import (
    MassSpheroid, ProfileSersic, ShapeFlattened, Spheroid, component_parser,
    galaxy_parser)
from gbkfit.model.mass import G, _numerics
from gbkfit.utils import fitsutils, gridutils
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
    # (stars do not flow out)
    (dict(kinematics=dict(
        type='traits', vptraits=dict(type='uniform', direction='out'),
        dptraits=dict(type='uniform'))), "takes velocity traits along"),
    (dict(kinematics=dict(
        type='traits', vptraits=dict(type='uniform', direction='tan'),
        dptraits=dict(type='uniform'), rnodes=[0, 5])),
     "go from 0 to its rmax"),
])
def test_invalid_spheroids_are_refused(options, message):
    info = dict(profile=dict(type='sersic'), rmax=10, emission=BAND) \
        | options
    with pytest.raises(ConfigError, match=message):
        Spheroid.load(info)


# A galaxy of a bulge (a spheroid) whose mass follows its light, at a
# distance
BULGE = dict(
    type='spheroid', name='bulge', rmax=10, emission=BAND,
    profile=dict(type='sersic'), shape=dict(type='flattened'))
STARS = dict(type='light_spheroid', name='stars', component='bulge')
DISTANCE = dict(type='direct', cosmology='Planck18')
BULGE_PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, bulge_ie=2, bulge_re=1.5, bulge_n=3,
    bulge_q=0.6, mass_stars_upsilon=4e6, distance=8)


def bulge_mass(bulge, properties):
    """
    The mass of a galaxy of the given bulge whose mass follows its light
    (STARS): the mass at the given values, and its component.
    """
    galaxy = galaxy_parser.load(dict(
        components=[bulge], mass=[STARS], distance=DISTANCE))
    mass = galaxy.plan_stage().mass(properties)
    return mass, mass.model.components()[0]


def test_the_mass_of_a_sersic_spheroid_is_that_of_its_light():
    # upsilon (Msun/kpc^2 per unit, without a unit) times its light, 2 pi
    # n re^2 ie e^b b^-2n Gamma(2n) arcsec^2, in kpc: the mass spheroid of
    # that Sersic profile and shape, to any radius
    mass, stars = bulge_mass(BULGE, BULGE_PROPERTIES)
    kpc = mass.kpc_per_arcsec
    upsilon = BULGE_PROPERTIES['mass_stars_upsilon']
    light = sersic_light(3, 2, 1.5)
    same = MassSpheroid('same', ProfileSersic(), ShapeFlattened())
    values = dict(
        log_m=np.log10(upsilon * light * kpc ** 2), re=1.5 * kpc, n=3,
        q=0.6)
    R = np.array([0.01, 0.05, 0.2, 1.0])
    z = np.array([0.0, 0.03, 0.1, 0.5])
    own = dict(upsilon=upsilon)
    np.testing.assert_allclose(
        stars.vcirc2(R, own, mass.context),
        same.vcirc2(R, values, mass.context), rtol=1e-12)
    for mine, theirs in zip(
            stars.forces(R, z, own, mass.context),
            same.forces(R, z, values, mass.context)):
        np.testing.assert_allclose(mine, theirs, rtol=1e-12)
    total = stars.enclosed_mass(np.array([1e4]), own, mass.context)[0]
    assert total == pytest.approx(upsilon * light * kpc ** 2, rel=1e-6)


def test_the_mass_of_an_mge_is_that_of_its_gaussians():
    # A round Gaussian of the mass M of its light: G M(<r) / r, with the
    # mass within r from the density by quadrature, and its potential
    # -G M erf(r / (sqrt(2) sigma)) / r; and a flattened one of the same
    # mass in all
    bulge = BULGE | dict(profile=dict(type='mge', ngauss=2))
    del bulge['shape']
    properties = {
        name: value for name, value in BULGE_PROPERTIES.items()
        if not name.startswith('bulge_')} | dict(
        bulge_i0=[3, 1], bulge_sigma=[0.5, 2], bulge_q=[1, 0.5])
    mass, stars = bulge_mass(bulge, properties)
    kpc = mass.kpc_per_arcsec
    upsilon = properties['mass_stars_upsilon']
    own = dict(upsilon=upsilon)
    masses = [upsilon * 2 * np.pi * i0 * (s * kpc) ** 2
              for i0, s in ((3, 0.5), (1, 2))]
    shape, profile, values = mass.context.density('bulge').terms()[0]
    r = np.array([0.005, 0.02, 0.1])
    within = [scipy.integrate.quad(
        lambda x: 4 * np.pi * x ** 2 * float(profile.density(
            np.array(x), values, mass.context)), 0, radius)[0]
        for radius in r]
    np.testing.assert_allclose(
        upsilon * shape.vcirc2(r, profile, values, mass.context),
        G * upsilon * np.array(within) / r, rtol=1e-8)
    sigma = 0.5 * kpc
    np.testing.assert_allclose(
        upsilon * profile.potential(r, values, mass.context),
        -G * masses[0] * scipy.special.erf(r / (np.sqrt(2) * sigma)) / r,
        rtol=1e-12)
    total = stars.enclosed_mass(np.array([1e3]), own, mass.context)[0]
    assert total == pytest.approx(sum(masses), rel=1e-6)


def test_a_disk_rotates_in_the_mass_of_the_light_of_a_spheroid(driver):
    # The same rotation as in the mass spheroid of the same profile
    from gbkfit.observation import (
        ObservationGroup, ObservationSky, PixelSpectra, line_of_sight_parser)
    from gbkfit.params import ParamSpace
    disk = dict(
        type='disk', name='gas', rnodes=list(range(0, 13, 2)),
        density=dict(ptraits=dict(type='exponential')),
        kinematics=dict(
            vptraits=dict(type='mass'), dptraits=dict(type='uniform')),
        emission=dict(type='lines'))
    disk_properties = dict(vsys=0, gas_pt_a=1, gas_pt_s=4, gas_dpt_a=15)

    def evaluate(mass, properties):
        galaxy = galaxy_parser.load(dict(
            components=[BULGE, disk], mass=[mass], distance=DISTANCE))
        group = ObservationGroup([galaxy], [ObservationSky(
            PixelSpectra(size=(24, 24, 41), step=(1, 1, 10)),
            line_of_sight_parser.load(dict(type='flat')))])
        params = ParamSpace(
            group.pdescs(), light | disk_properties | properties)
        return group.plan(driver).model_h(params.evaluate())[0][
            'spectra']['d']
    upsilon = BULGE_PROPERTIES['mass_stars_upsilon']
    light = {name: value for name, value in BULGE_PROPERTIES.items()
             if name != 'mass_stars_upsilon'}
    mass, _ = bulge_mass(BULGE, BULGE_PROPERTIES)
    total = sersic_light(3, 2, 1.5) * mass.kpc_per_arcsec ** 2
    followed = evaluate(STARS, dict(mass_stars_upsilon=upsilon))
    same = evaluate(
        dict(type='spheroid', name='same', profile=dict(type='sersic'),
             shape=dict(type='flattened')),
        dict(mass_same_log_m=np.log10(upsilon * total),
             mass_same_re=1.5 * mass.kpc_per_arcsec, mass_same_n=3,
             mass_same_q=0.6))
    np.testing.assert_allclose(
        followed, same, rtol=1e-5, atol=1e-6 * same.max())


@pytest.mark.parametrize('mass, message', [
    (dict(type='light_spheroid', name='stars', component='gas'),
     "follows the density of a spheroid, but 'gas' is a disk: the mass "
     "that follows it is of the type 'disk'"),
    (dict(type='disk', name='stars', surface=dict(
        type='component', component='bulge'), vertical=dict(type='sech2')),
     "follows the density of a disk, but 'bulge' is a spheroid: the mass "
     "that follows it is of the type 'light_spheroid'"),
])
def test_a_mass_follows_components_of_its_kind(mass, message):
    disk = dict(
        type='disk', name='gas', rnodes=[0, 2, 4],
        density=dict(ptraits=dict(type='exponential')), emission=BAND)
    with pytest.raises(ConfigError, match=message):
        galaxy_parser.load(dict(
            components=[BULGE, disk], mass=[mass], distance=DISTANCE))


# A round Gaussian of stars of dispersion S arcsec, of Jeans kinematics,
# around a point mass of 10^LOG_M Msun (its own mass left out): isotropic
# (beta_z = 0) and without rotation, the exact sigma^2(r) = G M [1 / r -
# sqrt(pi / 2) / s exp(r^2 / (2 s^2)) erfc(r / (sqrt(2) s))], s in kpc
S = 10
LOG_M = 10
STARS_EMISSION = dict(type='stars')
JEANS_SPHEROID = dict(
    type='spheroid', name='stars', rmax=40, emission=STARS_EMISSION,
    profile=dict(type='mge', ngauss=1), kinematics=dict(type='jeans'))
JEANS_PROPERTIES = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=1000, stars_i0=[1],
    stars_sigma=[S], stars_q=[1], stars_beta_z=0, stars_kappa=0,
    mass_bh_log_m=LOG_M, distance=10)


def jeans_galaxy(spheroid=JEANS_SPHEROID):
    """A galaxy of a spheroid of Jeans kinematics around a point mass."""
    return galaxy_parser.load(dict(
        components=[spheroid],
        mass=[dict(type='point', name='bh')], distance=DISTANCE))


def gaussian_sigma(r, s):
    """The exact sigma (km/s) at r (kpc) of a Gaussian (s, kpc) above."""
    return np.sqrt(G * 10 ** LOG_M * (
        1 / r - np.sqrt(np.pi / 2) / s * np.exp(r ** 2 / (2 * s ** 2))
        * scipy.special.erfc(r / (np.sqrt(2) * s))))


def table_points(tables):
    """The radii and heights (arcsec) of the nodes of Jeans tables."""
    shape = tables.sigma_z.shape
    R = np.exp(tables.lnr_zero + tables.lnr_step * np.arange(shape[0]))
    z = np.exp(tables.lnz_zero + tables.lnz_step * np.arange(shape[1]))
    return np.meshgrid(R, z, indexing='ij')


def test_the_jeans_tables_of_a_gaussian_around_a_point_mass_are_exact():
    galaxy = jeans_galaxy()
    state = galaxy.plan_stage().evaluate(JEANS_PROPERTIES)
    tables = state.states[0]
    kpc = state.mass.kpc_per_arcsec
    R, z = table_points(tables)
    r = np.hypot(R, z) * kpc
    exact = gaussian_sigma(r, S * kpc)
    within = r < 3 * S * kpc
    for sigma in (tables.sigma_z, tables.sigma_r, tables.sigma_phi):
        np.testing.assert_allclose(sigma[within], exact[within], rtol=3e-3)
    assert not tables.vphi.any()


def test_the_jeans_tables_of_a_rotating_flattened_spheroid():
    # The Jeans equations at a few points, by adaptive quadrature: nu
    # sigma_z^2 = int_z^inf nu G M z' / r'^3 dz', and R d(nu sigma_z^2)/dR
    # by central differences of it
    q, beta_z, kappa = 0.6, 0.3, 0.8
    properties = JEANS_PROPERTIES | dict(
        stars_q=[q], stars_beta_z=beta_z, stars_kappa=kappa)
    state = jeans_galaxy().plan_stage().evaluate(properties)
    tables = state.states[0]
    k = state.mass.kpc_per_arcsec
    gm = G * 10 ** LOG_M
    s = S * k

    def nu(R, z):
        return np.exp(-(R ** 2 + z ** 2 / q ** 2) / (2 * s ** 2))

    def nu_sz2(R, z):
        return scipy.integrate.quad(
            lambda x: nu(R, x) * gm * x / np.hypot(R, x) ** 3, z, np.inf,
            epsrel=1e-12, limit=200)[0]
    # (where R << z, near the point mass, it cannot rotate so: there its
    # rotation is 0, and sigma_phi^2 = <v_phi^2>)
    R, z = table_points(tables)
    b = 1 / (1 - beta_z)
    points = (20, 40), (40, 10), (40, 40), (50, 30), (55, 52)
    expected, still = [], []
    for i, j in points:
        Rk, zk = R[i, j] * k, z[i, j] * k
        h = 1e-4 * Rk
        I = nu_sz2(Rk, zk)
        dI = Rk * (nu_sz2(Rk + h, zk) - nu_sz2(Rk - h, zk)) / (2 * h)
        vphi2 = (b * (dI + I) + Rk * nu(Rk, zk) * gm * Rk
                 / np.hypot(Rk, zk) ** 3) / nu(Rk, zk)
        sz2 = I / nu(Rk, zk)
        rotation = vphi2 - b * sz2
        vphi = kappa * np.sqrt(max(rotation, 0.0))
        still.append(rotation < 0)
        expected.append((
            vphi, np.sqrt(b * sz2), np.sqrt(vphi2 - vphi ** 2),
            np.sqrt(sz2)))
    index = tuple(np.transpose(points))
    model = np.transpose([
        tables.vphi[index], tables.sigma_r[index],
        tables.sigma_phi[index], tables.sigma_z[index]])
    assert any(still)
    np.testing.assert_allclose(model, expected, rtol=3e-3)


def stars_cube(driver, spheroid, line_of_sight='grid'):
    """
    A case of a spectral cube of the stars of a spheroid (lines: [stars]),
    of 32 by 32 pixels of 2 arcsec and 181 channels of 10 km/s, with a
    depth axis of voxels of 1 arcsec.
    """
    los = dict(type='grid', depth=dict(step=1)) \
        if line_of_sight == 'grid' else dict(type=line_of_sight)
    return dict(
        driver=dict(type=driver.type()),
        observable=dict(
            type='pixel_spectra', size=[32, 32, 181], step=[2, 2, 10],
            dtype='float64'),
        line_of_sight=los, lines=['stars'],
        galaxy=dict(components=[spheroid], mass=[dict(
            type='point', name='bh')], distance=DISTANCE))


def test_the_stars_of_a_cube_have_the_moments_of_their_jeans_equations(
        driver, evaluate_cases):
    # Each line of sight of the round Gaussian around a point mass
    # (isotropic, no rotation): its mean velocity vsys, and its dispersion
    # that of the light-weighted mean of sigma^2 along it (int nu sigma^2
    # dz / int nu dz, with nu sigma^2 = G M [nu / r - sqrt(pi / 2) / s
    # erfc(r / (sqrt(2) s))]), up to the channels (their width^2 / 12)
    # and the voxels
    properties = JEANS_PROPERTIES | dict(vsys=0)
    spheroid = JEANS_SPHEROID | dict(geometry='galaxy')
    data, _ = evaluate_cases([stars_cube(driver, spheroid)], properties)
    cube = data[0]['spectra']['d']
    velocity = (np.arange(181) - 90) * 10.0
    pixels = (np.arange(32) - 15.5) * 2
    galaxy = jeans_galaxy()
    kpc = galaxy.plan_stage().mass(properties).kpc_per_arcsec
    s, gm = S * kpc, G * 10 ** LOG_M
    for ix in (19, 21, 24, 27):
        spectrum = cube[:, 15, ix]
        total = spectrum.sum()
        mean = (spectrum * velocity).sum() / total
        variance = (spectrum * (velocity - mean) ** 2).sum() / total \
            - 10.0 ** 2 / 12
        R = np.hypot(pixels[ix], pixels[15]) * kpc

        def nu(z):
            return np.exp(-(R ** 2 + z ** 2) / (2 * s ** 2))

        def nu_sigma2(z):
            r = np.hypot(R, z)
            return gm * (nu(z) / r - np.sqrt(np.pi / 2) / s
                         * scipy.special.erfc(r / (np.sqrt(2) * s)))
        expected = scipy.integrate.quad(nu_sigma2, 0, np.inf)[0] \
            / scipy.integrate.quad(nu, 0, np.inf)[0]
        assert mean == pytest.approx(0, abs=0.05)
        assert np.sqrt(variance) == pytest.approx(
            np.sqrt(expected), rel=5e-3)


@pytest.mark.parametrize('change, message', [
    # (a cube sees the stars only when its lines name them)
    (dict(lines=None), "it sees none"),
    # (in a cube, their velocities change along the line of sight)
    (dict(line_of_sight='flat'), "needs a line of sight with a depth axis"),
    # (a cube needs their kinematics)
    (dict(kinematics=None), "with kinematics only"),
])
def test_what_sees_the_stars_of_a_spheroid(driver, evaluate_cases,
                                           change, message):
    spheroid = JEANS_SPHEROID | dict(geometry='galaxy')
    if 'kinematics' in change:
        del spheroid['kinematics']
    case = stars_cube(
        driver, spheroid, change.get('line_of_sight', 'grid'))
    if 'lines' in change:
        del case['lines']
    properties = {
        name: value for name, value in JEANS_PROPERTIES.items()
        if 'kinematics' not in change
        or name not in ('vsys', 'stars_beta_z', 'stars_kappa')}
    with pytest.raises(ConfigError, match=message):
        evaluate_cases([case], properties)


def test_an_image_sees_the_stars_of_a_spheroid(driver, evaluate_cases):
    # (as band light: its projection, without a depth axis)
    spheroid = JEANS_SPHEROID | dict(geometry='galaxy')
    case = image(driver, spheroid, 'float64', flat=True)
    case['galaxy'] |= dict(
        mass=[dict(type='point', name='bh')], distance=DISTANCE)
    data, _ = evaluate_cases([case], JEANS_PROPERTIES)
    # (the light of the Gaussian within the image, 64 pixels wide)
    flux = data[0]['brightness']['d'].sum() * STEP ** 2
    within = scipy.special.erf(64 * STEP / 2 / (np.sqrt(2) * S)) ** 2
    assert flux == pytest.approx(2 * np.pi * S ** 2 * within, rel=1e-3)


def test_no_line_is_named_stars():
    with pytest.raises(ConfigError, match="no line can be named 'stars'"):
        galaxy_parser.load(dict(components=[dict(
            type='disk', rnodes=[0, 1, 2],
            density=dict(ptraits=dict(type='exponential')),
            emission=dict(type='lines', lines=[
                dict(name='stars', rest='656.28 nm')]))]))


# A round Gaussian of stars of the kinematics of traits: a uniform rotation
# of 100 km/s around the axis that falls with the height (exponentially,
# of the scale height 4 arcsec), and the dispersions 30, 50 and 70 km/s
# along R, phi and z
TRAITS_SPHEROID = dict(
    type='spheroid', name='stars', rmax=30, emission=STARS_EMISSION,
    profile=dict(type='mge', ngauss=1), kinematics=dict(
        type='traits', vptraits=dict(type='uniform', direction='tan'),
        vhtraits=dict(type='exponential'), dptraits=[
            dict(type='uniform', direction=direction)
            for direction in ('rad', 'tan', 'ver')]))
TRAITS_PROPERTIES = GEOMETRY | dict(
    vsys=20, stars_i0=[1], stars_sigma=[8], stars_q=[1], stars_vpt_vt=100,
    stars_vht_s=4, stars_dpt_a=30, stars_dpt1_a=50, stars_dpt2_a=70)


def spheroid_frame(grid_data, xpos, ypos, posa, incl):
    """
    The position of each voxel of an extra output on a 3d grid in the
    frame of a spheroid of the given centre and axis (see the cones'
    cone_frame): x along the major axis, y towards the far side, and z
    along the axis.
    """
    coords = grid_data.coords
    depth, y, x = np.meshgrid(*(
        coords.rval[axis] * (axis == 2)
        + (np.arange(size) - coords.rpix[axis]) * coords.step[axis]
        for axis, size in zip((2, 1, 0), grid_data.data.shape)),
        indexing='ij')
    p, i = np.radians(posa), np.radians(incl)
    x, y = x - xpos, y - ypos
    major = -x * np.sin(p) + y * np.cos(p)
    far = -x * np.cos(p) - y * np.sin(p)
    return (
        major,
        far * np.cos(i) + depth * np.sin(i),
        -far * np.sin(i) + depth * np.cos(i))


def test_the_stars_move_as_their_traits(driver, evaluate_cases):
    # At each voxel, vsys plus the rotation on the line of sight, 100
    # exp(-|z| / 4) cos(theta) sin(incl), and the dispersion of the
    # velocity ellipsoid on it; but in the voxels near the centre, which
    # hold the moments of their parts (within 3 voxel diagonals)
    case = stars_cube(driver, TRAITS_SPHEROID | dict(geometry='galaxy'))
    case['galaxy'] = dict(components=[case['galaxy']['components'][0]])
    _, extra = evaluate_cases([case], TRAITS_PROPERTIES)
    vdata = extra.get('vdata', component='stars')
    ddata = extra.get('ddata', component='stars')
    x, y, z = spheroid_frame(vdata, **GEOMETRY)
    theta = np.arctan2(y, x)
    incl = np.radians(GEOMETRY['incl'])
    velocity = 20 + 100 * np.exp(-np.abs(z) / 4) * np.cos(theta) \
        * np.sin(incl)
    dispersion = np.sqrt(
        (30 * np.sin(theta) * np.sin(incl)) ** 2
        + (50 * np.cos(theta) * np.sin(incl)) ** 2
        + (70 * np.cos(incl)) ** 2)
    on = np.isfinite(vdata.data) & (np.sqrt(x ** 2 + y ** 2 + z ** 2) > 9)
    assert on.sum() > 1000
    np.testing.assert_allclose(vdata.data[on], velocity[on], atol=1e-3)
    np.testing.assert_allclose(ddata.data[on], dispersion[on], atol=1e-3)


def test_a_spheroid_of_traits_round_trips():
    # (it moves without mass: its galaxy has none)
    galaxy = galaxy_parser.load(dict(components=[TRAITS_SPHEROID]))
    assert set(galaxy.pdescs()) == set(TRAITS_PROPERTIES)
    assert galaxy_parser.load(galaxy.dump()).dump() == galaxy.dump()


def test_the_stars_of_bins_have_the_gauss_hermite_moments_of_their_traits(
        driver, evaluate_cases, tmp_path):
    # The kinematics of the stars in Voronoi bins as pPXF gives them:
    # stars of a uniform dispersion of 80 km/s and no rotation have a
    # Gaussian distribution of velocities everywhere, so each bin has V the
    # velocity c ln D of vsys = 1000 km/s (D its relativistic Doppler
    # factor; the convention log, 1.7 km/s from the optical c (D - 1)),
    # sigma 80 km/s (with the channels of 10 km/s, sqrt(80^2 + 10^2 / 12)),
    # and h3 = h4 = 0
    pixels = (np.arange(32) - 15.5) * 2
    x, y = np.meshgrid(pixels, pixels)
    index = np.where(
        np.hypot(x, y) < 20, (x > 0) + 2 * (y > 0) + 4 * (np.hypot(x, y) > 8),
        -1)
    fitsutils.write_data(
        str(tmp_path / 'bins.fits'), index,
        gridutils.Coords((2, 2), (15.5, 15.5), (0, 0), 0))
    spheroid = TRAITS_SPHEROID | dict(geometry='galaxy', kinematics=dict(
        type='traits', vptraits=dict(type='uniform', direction='tan'),
        dptraits=dict(type='uniform')))
    case = stars_cube(driver, spheroid)
    case['galaxy'] = dict(components=[spheroid])
    case['observable'] = dict(
        type='region_moments',
        regions=dict(type='bins', file=str(tmp_path / 'bins.fits')),
        orders=[0, 1, 2, 3, 4], method='gauss_hermite', spec_size=81,
        spec_step=10, spec_rval=1000, spec_rest='5000 Angstrom',
        spec_convention='log', psf=dict(type='gauss', sigma=3),
        dtype='float64')
    properties = GEOMETRY | dict(
        vsys=1000, stars_i0=[1], stars_sigma=[8], stars_q=[1],
        stars_vpt_vt=0, stars_dpt_a=80)
    data, _ = evaluate_cases([case], properties)
    moments = data[0]
    c = 299792.458
    assert moments['moment1']['d'].shape == (8,)
    np.testing.assert_allclose(
        moments['moment1']['d'], c * np.arctanh(1000 / c), atol=0.02)
    np.testing.assert_allclose(
        moments['moment2']['d'], np.sqrt(80 ** 2 + 10 ** 2 / 12), rtol=1e-3)
    np.testing.assert_allclose(moments['moment3']['d'], 0, atol=1e-3)
    np.testing.assert_allclose(moments['moment4']['d'], 0, atol=1e-3)


# The exact mock of the Schwarzschild proof of concept (model.orbits): a
# sphere of Hernquist light of the scale radius 1 kpc, of a mass-to-light
# ratio of 4 and a black hole of 5e8 Msun, its stars isotropic, of the
# distribution function of Eddington's formula, at 10 Mpc (its kpc per
# arcsec HERNQUIST_KPC)
HERNQUIST_A = 1.0
HERNQUIST_LIGHT = 1e10
HERNQUIST_UPSILON = 4.0
HERNQUIST_BH = 5e8
HERNQUIST_DISTANCE = 10
HERNQUIST_KPC = HERNQUIST_DISTANCE * 1e3 * np.pi / 648000


@functools.cache
def hernquist_mock():
    """The exact mock (its tables of its distribution function made once)."""
    from gbkfit.model.orbits.mocks import MockGalaxy
    from gbkfit.model.orbits.profiles import Hernquist
    return MockGalaxy(
        Hernquist(HERNQUIST_LIGHT, HERNQUIST_A), HERNQUIST_UPSILON,
        HERNQUIST_BH)


def hernquist_mge(a, ngauss=60):
    """
    The dispersions (arcsec) and the peak surface brightnesses of an MGE
    of the Hernquist profile of the scale radius a (arcsec), of unit
    light: Gaussians of dispersions geometric from 1e-5 a to 1e3 a, of the
    amplitudes of least squares of the relative error, within 3e-4 from
    1e-3 a to 30 a.
    """
    sigma = np.geomspace(1e-5 * a, 1e3 * a, ngauss)
    r = np.geomspace(3e-6 * a, 3e3 * a, 800)
    density = a / (2 * np.pi * r * (r + a) ** 3)
    design = np.exp(-r[:, None] ** 2 / (2 * sigma ** 2)) / density[:, None]
    scale = np.abs(design).max(axis=0)
    amplitudes = scipy.optimize.lsq_linear(
        design / scale, np.ones_like(r), bounds=(0, np.inf), tol=1e-14,
        max_iter=20000).x / scale
    return sigma, amplitudes * np.sqrt(2 * np.pi) * sigma


def hernquist_galaxy():
    """
    The spheroid of the stars of the exact mock, of isotropic Jeans
    kinematics (beta_z = 0, kappa = 0) and an MGE of its light, in the
    gravity of its Hernquist mass (a Zhao profile) and its black hole; its
    galaxy and the values of its parameters.
    """
    a = HERNQUIST_A / HERNQUIST_KPC
    sigma, i0 = hernquist_mge(a)
    spheroid = dict(
        type='spheroid', name='stars', geometry='galaxy', rmax=30 * a,
        emission=STARS_EMISSION, profile=dict(type='mge', ngauss=len(i0)),
        kinematics=dict(type='jeans'))
    galaxy = dict(
        components=[spheroid],
        mass=[dict(type='spheroid', name='light',
                   profile=dict(type='zhao', parameters='density_radius')),
              dict(type='point', name='bh')],
        distance=DISTANCE)
    properties = dict(
        xpos=0, ypos=0, posa=0, incl=60, vsys=0,
        distance=HERNQUIST_DISTANCE, stars_i0=list(i0),
        stars_sigma=list(sigma), stars_q=[1.0] * len(i0), stars_beta_z=0,
        stars_kappa=0,
        mass_light_log_rho=np.log10(
            HERNQUIST_UPSILON * HERNQUIST_LIGHT
            / (2 * np.pi * HERNQUIST_A ** 3)),
        mass_light_rs=HERNQUIST_A, mass_light_alpha=1.0,
        mass_light_beta=4.0, mass_light_gamma=1.0,
        mass_bh_log_m=np.log10(HERNQUIST_BH))
    return galaxy, properties


def test_the_jeans_tables_of_the_exact_mock_are_its_dispersion():
    # The dispersion of its distribution function (isotropic: along every
    # direction), at every 8th radius and height of the tables from 2e-3
    # to 5 kpc (to the MGE and the tables)
    galaxy, properties = hernquist_galaxy()
    tables = galaxy_parser.load(galaxy).plan_stage().evaluate(
        properties).states[0]
    R, z = table_points(tables)
    r = np.hypot(R, z)[::8, ::8] * HERNQUIST_KPC
    within = (r > 2e-3 * HERNQUIST_A) & (r < 5 * HERNQUIST_A)
    assert within.sum() > 30
    exact = hernquist_mock().dispersion(r[within])
    for sigma in (tables.sigma_r, tables.sigma_phi, tables.sigma_z):
        np.testing.assert_allclose(
            sigma[::8, ::8][within], exact, rtol=2e-4)


def test_the_stars_of_a_cube_have_the_second_moments_of_the_exact_mock(
        driver, evaluate_cases):
    # The second moment of the spectrum of each line of sight (but those
    # of the centre) is that of the distribution of the velocities of the
    # mock on it, to the voxels. Its Gauss-Hermite series is not: Jeans
    # kinematics take a Gaussian distribution of velocities at each point
    # (here h4 is about 0.01, where the mock's is about -0.02, and the
    # dispersion of the series 5% below the mock's)
    galaxy, properties = hernquist_galaxy()
    case = stars_cube(driver, galaxy['components'][0])
    case['galaxy'] = galaxy
    case['line_of_sight'] = dict(type='grid', depth=dict(step=2))
    data, _ = evaluate_cases([case], properties)
    cube = data[0]['spectra']['d']
    velocity = (np.arange(181) - 90) * 10.0
    exact = np.linspace(-1500, 1500, 3001)
    pixels = (np.arange(32) - 15.5) * 2
    for iy, ix in ((15, 18), (15, 23), (15, 31), (25, 25), (31, 31)):
        spectrum = cube[:, iy, ix]
        second = (spectrum * velocity ** 2).sum() / spectrum.sum() \
            - 10.0 ** 2 / 12
        losvd = hernquist_mock().losvd(
            [np.hypot(pixels[ix], pixels[iy]) * HERNQUIST_KPC], exact)[0]
        expected = np.trapezoid(losvd * exact ** 2, exact) \
            / np.trapezoid(losvd, exact)
        assert np.sqrt(second) == pytest.approx(np.sqrt(expected), rel=6e-4)
