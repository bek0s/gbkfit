"""
Lenses of the masses of galaxies (see LensGalaxyMass), end to end: the
mass of an NFW halo against the NFW lens of its convergence, seen by an
observation that does not see the galaxy; the rays of a galaxy of a
disk, a spheroid and a halo against its deflection; the parts of the
stage that a step evaluates again (the mass, its field and its sky); the
deflections kept layer by layer between evaluations; the rings of a
warped disk, and its bands of mass where the kernels draw its rings; and
the refusals when the stage is planned.
"""

import math

import numpy as np
import pytest

import gbkfit.lensing._mass_deflections as mass_deflections
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.grid import Grid, sky_positions
from gbkfit.lensing import (
    Lens, LensGalaxyMass, LensNFW, LensShear, SkyPoints)
from gbkfit.lensing._layers import K
from gbkfit.lensing._mass_deflections import MassDeflections
from gbkfit.model import Extras, galaxy_parser
from gbkfit.observation import ObservationGroup, ObservationPixelBrightness
from gbkfit.params import ParamSpace


# The redshifts of the lens galaxy and of the source behind it
LENS_Z, SOURCE_Z = 0.35, 1.2

# The image: 61 pixels of 0.04 arcsec
SIZE, STEP = 61, 0.04

# The arcsec of a radian
ARCSEC = math.degrees(1) * 3600


def disk(name, **options):
    """A thin exponential disk that emits in a band."""
    return dict(
        type='thin_disk', name=name, loose=False, tilted=False,
        rnodes=[0, 0.5, 1, 1.5, 2, 3], rptraits=dict(type='exponential'),
        light=dict(type='continuum')) | options


# A gas disk that the mass moves: it places the mass on the sky
GAS = disk(
    'gas', light=dict(type='lines'), vptraits=dict(type='mass'),
    dptraits=dict(type='uniform'))

GAS_PARAMS = dict(
    gas_xpos=0.1, gas_ypos=-0.05, gas_posa=30, gas_incl=60, gas_vsys=0,
    gas_rpt_a=1, gas_rpt_s=0.5, gas_dpt_a=10)

# An NFW halo of a scale density (Msun/kpc^3) and radius (kpc)
HALO = dict(
    type='halo', name='halo',
    profile=dict(type='nfw', parameterisation='density_radius'))

RHO_S, R_S = 3e6, 20.0

# (the mass at the centre of the components of the galaxies here)
HALO_PARAMS = dict(
    mass_xpos=0.1, mass_ypos=-0.05, mass_halo_log_rho=math.log10(RHO_S),
    mass_halo_rs=R_S)


def deflector(components=(GAS,), mass=(HALO,), **options):
    """The galaxy whose mass bends the light of the arc."""
    return galaxy_parser.load(dict(
        name='deflector', redshift=LENS_Z, components=list(components),
        mass=list(mass), distance=dict(type='cosmological')) | options)


ARC = galaxy_parser.load(dict(
    name='arc', redshift=SOURCE_Z, components=[disk('disk')]))

ARC_PARAMS = dict(
    arc_disk_xpos=0.05, arc_disk_ypos=0.02, arc_disk_posa=20,
    arc_disk_incl=40, arc_disk_rpt_a=1, arc_disk_rpt_s=0.15)


def image(name):
    """An image, float64 (of what its group says it sees)."""
    return ObservationPixelBrightness(
        size=(SIZE, SIZE), step=STEP, name=name, dtype='float64')


def prefixed(prefix, values):
    """The values with their names prefixed."""
    return {f'{prefix}_{name}': value for name, value in values.items()}


def values_of(group, properties):
    """The values of the parameters of a group (all fixed)."""
    return ParamSpace.from_properties(group.pdescs(), properties).evaluate()


def galaxy_values_of(galaxy, properties):
    """
    The values of the parameters of a galaxy (all fixed), with those of
    where its mass is on the sky (see Galaxy.sky_pdescs).
    """
    return ParamSpace.from_properties(
        galaxy.pdescs() | galaxy.sky_pdescs(), properties).evaluate()


def evaluate(group, values, driver):
    """
    Return copies of the model images of the observations of a group,
    planned at the values, and the plan.
    """
    plan = group.plan(driver, reference=values)
    images = [model['brightness'].data.copy()
              for model in plan.model_h(values)]
    return images, plan


@pytest.mark.parametrize('components', [(GAS,), ()])
def test_the_mass_of_a_halo_lenses_as_the_nfw_lens(driver, components):
    # The image of the arc alone, bent by the mass of an NFW halo of the
    # deflector (which it does not see), is that of the NFW lens of the
    # convergence kappa_s = rho_s r_s / Sigma_cr, quoted for the arc, at
    # the centre of the mass: of a galaxy of a gas disk that the mass
    # moves, or of the mass alone (a dark lens)
    group = ObservationGroup(
        [deflector(components), ARC], image('alma'),
        Lens('lens', LENS_Z, LensGalaxyMass('deflector')), seen=[['arc']])
    own = GAS_PARAMS if components else {}
    values = values_of(group, prefixed('deflector', own | HALO_PARAMS)
                       | ARC_PARAMS)
    [lensed], _ = evaluate(group, values, driver)
    cosmology = group.cosmology()
    d_l = 1e3 * cosmology.angular_diameter_distance(LENS_Z)
    d_s = 1e3 * cosmology.angular_diameter_distance(SOURCE_Z)
    d_ls = 1e3 * cosmology.angular_diameter_distance_between(
        LENS_Z, SOURCE_Z)
    kpc_per_arcsec = d_l / ARCSEC
    # (the deflector of the NFW lens is not its mass: no place of it)
    halo = {k: v for k, v in HALO_PARAMS.items()
            if k not in ('mass_xpos', 'mass_ypos')}
    nfw = ObservationGroup(
        [deflector(), ARC], image('alma'),
        Lens('lens', LENS_Z, LensNFW(), source_redshift=SOURCE_Z),
        seen=[['arc']])
    [expected], _ = evaluate(nfw, values_of(nfw, prefixed(
        'deflector', GAS_PARAMS | halo) | ARC_PARAMS | dict(
        lens_nfw_kappa_s=math.pi * K * RHO_S * R_S * d_l * d_ls / d_s,
        lens_nfw_r_s=R_S / kpc_per_arcsec, lens_nfw_q=1, lens_nfw_posa=0,
        lens_nfw_xpos=HALO_PARAMS['mass_xpos'],
        lens_nfw_ypos=HALO_PARAMS['mass_ypos'])), driver)
    assert lensed.max() > 0.1
    np.testing.assert_allclose(lensed, expected, rtol=0, atol=1e-6)


# A galaxy of stars in a disk and a flattened spheroid, each followed by
# its mass, and an NFW halo
BULGE = dict(
    type='spheroid', name='bulge', profile=dict(type='sersic'),
    shape=dict(type='flattened'), rmax=4, light=dict(type='continuum'))

STARS = disk('stars')

GALAXY_MASS = [
    dict(type='traced_disk', name='stars', component='stars'),
    dict(type='traced_spheroid', name='bulge', component='bulge'), HALO]

GALAXY_PARAMS = dict(
    stars_xpos=0.1, stars_ypos=-0.05, stars_posa=30, stars_incl=60,
    stars_rpt_a=1, stars_rpt_s=0.5, bulge_xpos=0.1, bulge_ypos=-0.05,
    bulge_posa=30, bulge_incl=60, bulge_ie=1, bulge_re=0.3, bulge_n=2,
    bulge_q=0.6, mass_stars_upsilon=3e9, mass_bulge_upsilon=1e9) \
    | HALO_PARAMS


def test_the_components_of_a_lens_are_at_its_mass_and_in_its_plane():
    # The components that the mass of a lens galaxy moves or follows are at
    # the centre of its mass on the sky (mass_xpos, mass_ypos), and those
    # that it follows in one plane (the disk and the axis of the bulge):
    # else an evaluation fails, naming the tie
    from gbkfit.params import ModelEvaluationError
    galaxy = deflector([STARS, BULGE], GALAXY_MASS)
    stage = galaxy.plan_stage(sky_mass=True)
    values = galaxy_values_of(galaxy, GALAXY_PARAMS)
    stage.evaluate(values)
    with pytest.raises(ModelEvaluationError, match="different centres"):
        stage.evaluate(values | dict(mass_xpos=0.3))
    with pytest.raises(ModelEvaluationError, match="lies in one plane"):
        stage.evaluate(values | dict(bulge_incl=50))
    # (without the sky, of the stars and the bulge alone)
    plain = galaxy.plan_stage()
    plain.evaluate(values)
    with pytest.raises(ModelEvaluationError, match="lies in one plane"):
        plain.evaluate(values | dict(bulge_incl=50))


def galaxy_group():
    """
    The galaxy of stars (see GALAXY_MASS) in front of the arc, a lens of
    its mass and a shear, an image of both and an image of the arc alone.
    """
    lens = Lens(
        'lens', LENS_Z, [LensGalaxyMass('deflector'), LensShear()],
        source_redshift=2.0)
    return ObservationGroup(
        [deflector([STARS, BULGE], GALAXY_MASS), ARC],
        [image('hst'), image('alma')], lens, seen=[None, ['arc']])


def galaxy_values(group):
    """The values of the parameters of the group of galaxy_group."""
    return values_of(group, prefixed('deflector', GALAXY_PARAMS) | ARC_PARAMS
                     | dict(lens_shear_gamma=0.05, lens_shear_posa=10))


def test_the_rays_of_a_galaxy_of_stars_and_a_halo(driver):
    # The rays of the arc in the image of it alone are bent by the mass of
    # the deflector laid out on the sky (with the shear): theta - f alpha
    group = galaxy_group()
    values = galaxy_values(group)
    plan = group.plan(driver, reference=values)
    extras = Extras()
    plan.model_h(values, extras)
    rays = {x.name: x.value for x in extras
            if x.ids.get('observation') == 'alma' and x.name in (
                'ray_x', 'ray_y')}
    ray_x = rays['ray_x']
    x, y = sky_positions(Grid(
        ray_x.data.shape[::-1], ray_x.coords, ray_x.spectral_axis))
    galaxy = group.galaxies()[0]
    own = {name: values[full] for name, full in group.mappings()[0].items()}
    sky = galaxy.plan_stage(
        group.sky_origin(), driver, group.cosmology(), True).sky_mass(own)
    points = SkyPoints(x, y, group.sky_origin())
    alpha_x, alpha_y = MassDeflections(driver).deflection(sky, points)
    shear = LensShear().deflection(
        points, dict(gamma=0.05, posa=10), driver)
    factor = group.tracer().factor(0, 1)
    # (of the shear, its second piece)
    quoted = group.lenses()[0].quoted_factors(LENS_Z, group.cosmology())[1]
    np.testing.assert_allclose(
        rays['ray_x'].data,
        x - factor * (alpha_x + quoted * shear[0]), rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        rays['ray_y'].data,
        y - factor * (alpha_y + quoted * shear[1]), rtol=0, atol=1e-10)


def test_deflections_are_kept_layer_by_layer(monkeypatch):
    # The layers of the mass of the deflector are deflected again only
    # where their unit shapes change: not for a step in a lens parameter,
    # in the light of the arc (a fixed mass is deflected once), or in a
    # mass-to-light ratio (which scales its layer), but for its halo or
    # the orientation of its stars and its bulge (tied: its mass lies in
    # one plane)
    calls = []
    unit_deflection = mass_deflections.unit_deflection

    def counted(layer, *args):
        calls.append(type(layer).__name__)
        return unit_deflection(layer, *args)
    monkeypatch.setattr(mass_deflections, 'unit_deflection', counted)
    group = galaxy_group()
    values = galaxy_values(group)
    driver = DriverHost()
    plan = group.plan(driver, reference=values)
    plan.model_h(values)
    assert calls
    sky = plan.stage_caches()['galaxy.deflector.sky']
    for names, again in (
            (['lens_shear_gamma'], []), (['arc_disk_rpt_s'], []),
            (['deflector_mass_stars_upsilon'], []),
            (['deflector_mass_halo_rs'], ['RoundLayer'] * 2),
            (['deflector_stars_posa', 'deflector_bulge_posa'],
             ['BandLayer'] * 2 + ['EllipticalLayer'] * 2)):
        name = names[0]
        calls.clear()
        stepped = values | {n: 1.1 * values[n] for n in names}
        version = sky.version()
        images = [model['brightness'].data.copy()
                  for model in plan.model_h(stepped)]
        assert sorted(calls) == again, name
        # (the sky is laid out again only for the parameters of the
        # deflector that move its mass)
        assert (sky.version() != version) == name.startswith('deflector'), \
            name
        # (a plan of no last evaluation, of grids at the same reference)
        fresh = group.plan(driver, reference=values).model_h(stepped)
        for image_, expected in zip(images, fresh):
            np.testing.assert_allclose(
                image_, expected['brightness'].data, rtol=1e-12, err_msg=name)
        plan.model_h(values)


# A deflector of a gas disk that its mass follows and moves, and a disk of
# stars that its mass moves alone, with an NFW halo
GAS_DISK = GAS | dict(name='gasdisk')

STARS_DISK = GAS | dict(name='stars')

KINEMATIC_MASS = [
    dict(type='traced_disk', name='gasdisk', component='gasdisk'), HALO]

DISK_VALUES = {name[4:]: value for name, value in GAS_PARAMS.items()}

KINEMATIC_PARAMS = prefixed('gasdisk', DISK_VALUES) | prefixed(
    'stars', DISK_VALUES | dict(posa=50)) | dict(
    mass_gasdisk_upsilon=1e9) | HALO_PARAMS


def test_steps_that_move_no_mass_render_no_lensed_observation(monkeypatch):
    # A step in the systemic velocity or the dispersion of the disk that
    # the mass follows builds the mass again, and lays it out on the sky
    # again, but as it was: the observation of the arc alone is not
    # evaluated again, and no layer is deflected again; nor for a step in
    # the orientation of the disk that the mass only moves (no part of the
    # mass lies in its plane). A step in the mass is.
    calls = []
    unit_deflection = mass_deflections.unit_deflection

    def counted(layer, *args):
        calls.append(type(layer).__name__)
        return unit_deflection(layer, *args)
    monkeypatch.setattr(mass_deflections, 'unit_deflection', counted)
    group = ObservationGroup(
        [deflector([GAS_DISK, STARS_DISK], KINEMATIC_MASS),
         ARC], image('alma'),
        Lens('lens', LENS_Z, LensGalaxyMass('deflector')), seen=[['arc']])
    values = values_of(
        group, prefixed('deflector', KINEMATIC_PARAMS) | ARC_PARAMS)
    plan = group.plan(DriverHost(), reference=values)
    plan.model_h(values)
    caches = plan.stage_caches()
    sky, alma = caches['galaxy.deflector.sky'], caches['observation.alma']
    for name, moves in (
            ('deflector_gasdisk_vsys', False),
            ('deflector_gasdisk_dpt_a', False),
            ('deflector_stars_posa', False),
            ('deflector_mass_gasdisk_upsilon', True)):
        calls.clear()
        versions = sky.version(), alma.version()
        mass = caches['galaxy.deflector.mass'].version()
        plan.model_h(values | {name: values[name] + 1.0})
        assert (sky.version(), alma.version() != versions[1]) == (
            versions[0] + moves, moves), name
        assert calls == [], name
        if name.startswith('deflector_gasdisk'):
            assert caches['galaxy.deflector.mass'].version() != mass, name
        plan.model_h(values)


def test_a_step_evaluates_again_only_the_parts_it_moves():
    # The mass on the sky takes the mass, not its field: a step in the
    # gravity lays out neither the mass nor its sky again; one in the
    # orientation or the dispersion of the gas, which the mass moves but
    # does not follow, none of them; one in the halo all three; one in the
    # centre of the mass (and of the gas, tied to it) its sky alone
    galaxy = deflector(gravity=dict(type='hu_sawicki'))
    stage = galaxy.plan_stage(sky_mass=True)
    values = galaxy_values_of(galaxy, GAS_PARAMS | HALO_PARAMS | dict(
        gravity_log_fr0=-6.0))
    first = stage.evaluate(values)
    assert first.sky is not None
    caches = stage.caches()
    for name, again in (
            ('gravity_log_fr0', {'field'}),
            ('gas_posa', set()),
            ('gas_dpt_a', set()),
            ('mass_halo_rs', {'mass', 'field', 'sky'})):
        versions = {
            part: caches[part].version() for part in ('mass', 'field', 'sky')}
        state = stage.evaluate(values | {name: 1.01 * values[name]})
        assert {part for part, version in versions.items()
                if caches[part].version() != version} == again, name
        assert (state.sky is first.sky) == ('sky' not in again), name
        stage.evaluate(values)
        first = stage.evaluate(values)
    versions = {
        part: caches[part].version() for part in ('mass', 'field', 'sky')}
    moved = stage.evaluate(values | dict(mass_xpos=0.2, gas_xpos=0.2))
    assert {part for part, version in versions.items()
            if caches[part].version() != version} == {'sky'}
    assert moved.sky is not first.sky
    # A disk of mass in the plane of the mass, its own: a step in it moves
    # the disk on the sky, one in the orientation of the gas does not
    galaxy = deflector(mass=[HALO, dict(
        type='disk', name='disk', surface=dict(type='exponential'),
        vertical=dict(type='thin'))])
    stage = galaxy.plan_stage(sky_mass=True)
    values = galaxy_values_of(galaxy, GAS_PARAMS | HALO_PARAMS | dict(
        mass_posa=30, mass_incl=60, mass_disk_log_m=10.5, mass_disk_rd=2.0))
    first = stage.evaluate(values)
    version = stage.caches()['sky'].version()
    assert stage.evaluate(values | dict(gas_posa=40)).sky is first.sky
    assert stage.evaluate(values | dict(mass_posa=40)).sky is not first.sky
    assert stage.caches()['sky'].version() == version + 1


# A thick disk that tilts, lifts (order 0) and warps (order 1)
WARPED = dict(
    type='thick_disk', name='gas', loose=False, tilted=True,
    rnodes=[0, 1, 2, 3], rptraits=dict(type='exponential'),
    rhtraits=dict(type='sech2', zmax=2),
    zptraits=[dict(type='nw_uniform', zmax=2),
              dict(type='nw_harmonic', order=1, zmax=2)],
    light=dict(type='continuum'))

# (linear in the radius, as the interpolations of the disk give them at
# any radius)
WARPED_PARAMS = dict(
    xpos=0.1, ypos=-0.05, posa=[30, 33, 36, 39], incl=[60, 61, 62, 63],
    rpt_a=1, rpt_s=1, rht_s=0.1, zpt_a=[0, 0.05, 0.1, 0.15],
    zpt2_a=[0, 0.1, 0.2, 0.3], zpt2_p=[0, 10, 20, 30])


class Placements:
    """A placement of a mass on the sky (see SkyPlacements), by name."""

    def __init__(self):
        self.placed = {}

    def add_rings(self, name, rings):
        self.placed[name] = rings

    def add_axis(self, name, axis):
        self.placed[name] = axis


def ring_placement(stage, values, kpc_per_arcsec):
    """The rings that the stage of a disk places (see Placed.place)."""
    placements = Placements()
    stage.place('gas', values, kpc_per_arcsec, placements)
    return placements.placed['gas']


def test_the_rings_of_a_warped_disk_as_it_draws_them():
    # At its nodes, its rings have the geometry of the nodes, the lift of
    # its distortion of order 0 and the tilt of that of order 1, a cos(theta
    # - p) = tilt_x x + tilt_y y in its plane (a / R cos p, a / R sin p);
    # between them, those of their interpolations (here linear)
    galaxy = galaxy_parser.load(dict(
        name='deflector', redshift=LENS_Z, components=[WARPED],
        mass=[dict(type='traced_disk', name='gas', component='gas')],
        distance=dict(type='cosmological')))
    [component] = galaxy.components()
    stage = component.plan_stage(
        DriverHost(), None, galaxy.links().of(component), 'gas')
    assert component.rings_gap() is None
    values = galaxy_values_of(
        galaxy, prefixed('gas', WARPED_PARAMS) | dict(mass_gas_upsilon=1))
    own = {name: values[f'gas_{name}'] for name in component.pdescs()}
    kpc = 5.0
    radii = np.array([1.0, 1.7, 2.0, 3.0])
    rings = ring_placement(stage, own, kpc).rings(radii * kpc)
    np.testing.assert_allclose(rings.x, 0.1 * kpc)
    np.testing.assert_allclose(rings.y, -0.05 * kpc)
    np.testing.assert_allclose(rings.posa, 30 + 3 * radii, rtol=1e-12)
    np.testing.assert_allclose(rings.incl, 60 + radii, rtol=1e-12)
    np.testing.assert_allclose(rings.lift, 0.05 * radii * kpc, rtol=1e-12)
    a = 0.1 * radii
    p = np.radians(10 * radii)
    np.testing.assert_allclose(rings.tilt_x, a * np.cos(p) / radii,
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(rings.tilt_y, a * np.sin(p) / radii,
                               rtol=1e-10, atol=1e-14)
    # (no tilt at the centre, a ring of no radius)
    centre = ring_placement(stage, own, kpc).rings(np.zeros(1))
    assert centre.tilt_x[0] == 0 and centre.tilt_y[0] == 0


def test_a_warped_disk_samples_no_profiles_unless_it_places_a_mass():
    # Its plan of the vertical distortions is made for its rings on the
    # sky alone, at its first placement: a disk that places no mass on
    # the sky samples no profiles, warped or not (which decides where it
    # is lit, see DiskStage._lit)
    moved = WARPED | dict(
        vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
        light=dict(type='lines'))
    galaxy = galaxy_parser.load(dict(
        name='deflector', redshift=LENS_Z, components=[moved],
        mass=[HALO], distance=dict(type='cosmological')))
    [component] = galaxy.components()
    stage = component.plan_stage(
        DriverHost(), None, galaxy.links().of(component), 'gas')
    assert stage._plan is None and stage._rings_plan is None
    values = galaxy_values_of(galaxy, prefixed(
        'gas', WARPED_PARAMS | dict(vsys=0, dpt_a=10)) | HALO_PARAMS)
    own = {name: values[f'gas_{name}'] for name in component.pdescs()}
    ring_placement(stage, own, 5.0).rings(np.ones(1))
    assert stage._plan is None and stage._rings_plan is not None


def test_the_bands_of_a_warped_disk_are_where_it_draws_its_rings():
    # The points of each ring of the disk, lifted and tilted by its
    # distortions, are where the kernels draw them (see disk_to_sky): on
    # the ellipse of the band of the mass on that ring
    from gbkfit.model.components._common import disk_to_sky
    from gbkfit.model.mass.sky import SkyPlacement, ring_geometry
    galaxy = galaxy_parser.load(dict(
        name='deflector', redshift=LENS_Z, components=[WARPED],
        mass=[dict(type='traced_disk', name='gas', component='gas')],
        distance=dict(type='cosmological')))
    [component] = galaxy.components()
    stage = component.plan_stage(
        DriverHost(), None, galaxy.links().of(component), 'gas')
    values = galaxy_values_of(
        galaxy, prefixed('gas', WARPED_PARAMS) | dict(mass_gas_upsilon=1))
    own = {name: values[f'gas_{name}'] for name in component.pdescs()}
    radii = np.array([0.6, 1.7, 2.5])
    rings = ring_placement(stage, own, 1.0)
    centres, posa, stretch, _ = ring_geometry(
        SkyPlacement((0.0, 0.0), disks=dict(gas=rings)), 'gas')(radii)
    geometry = rings.rings(radii)
    theta = np.radians(np.arange(0, 360, 15))[:, None]
    x_d, y_d = radii * np.cos(theta), radii * np.sin(theta)
    z = geometry.lift + geometry.tilt_x * x_d + geometry.tilt_y * y_d
    sky_x, sky_y, _ = disk_to_sky(
        x_d, y_d, z, geometry.x, geometry.y, geometry.posa, geometry.incl)
    p = np.radians(posa)
    dx, dy = sky_x - centres[:, 0], sky_y - centres[:, 1]
    u = -dx * np.sin(p) + dy * np.cos(p)
    v = -dx * np.cos(p) - dy * np.sin(p)
    np.testing.assert_allclose(
        (u / (stretch[:, 0] * radii)) ** 2
        + (v / (stretch[:, 1] * radii)) ** 2, 1, rtol=1e-10)


def test_distortions_of_order_two_place_no_mass():
    # The rings of a disk of a distortion of order 2 are not ellipses on
    # the sky: a galaxy whose mass follows them cannot lens (refused with
    # the group, as gbkfit.describe sees it); one whose mass only moves the
    # disk can
    bowl = WARPED | dict(zptraits=[
        dict(type='nw_uniform', zmax=2),
        dict(type='nw_harmonic', order=2, zmax=2)])
    lens = Lens('lens', LENS_Z, LensGalaxyMass('deflector'))
    galaxy = galaxy_parser.load(dict(
        name='deflector', redshift=LENS_Z, components=[bowl],
        mass=[dict(type='traced_disk', name='gas', component='gas')],
        distance=dict(type='cosmological')))
    with pytest.raises(ValueError, match="not ellipses on the sky"):
        ObservationGroup([galaxy, ARC], image('alma'), lens, seen=[['arc']])
    moved = bowl | dict(
        vptraits=dict(type='mass'), dptraits=dict(type='uniform'),
        light=dict(type='lines'))
    galaxy = galaxy_parser.load(dict(
        name='deflector', redshift=LENS_Z, components=[moved],
        mass=[HALO], distance=dict(type='cosmological')))
    group = ObservationGroup(
        [galaxy, ARC], image('alma'), lens, seen=[['arc']])
    values = values_of(group, prefixed('deflector', prefixed(
        'gas', WARPED_PARAMS | dict(vsys=0, dpt_a=10))
        | HALO_PARAMS) | ARC_PARAMS)
    [image_], _ = evaluate(group, values, DriverHost())
    assert image_.max() > 0.1
