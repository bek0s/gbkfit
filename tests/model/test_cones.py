"""
Tests for the cones (Cone): their light, the velocity of each voxel,
their walls and sides, and the dust that dims them.
"""

import numpy as np
import pytest

from gbkfit.config import ConfigError
from gbkfit.model import galaxy_parser


UNIFORM = dict(type='uniform')
BAND = dict(type='band')
LINES = dict(type='lines')

# A hollow cone: its walls (smooth, within 1.5 arcsec of their mid-surface
# at 30 degrees from the axis) touch neither its axis nor the plane of its
# apex. Its density is 1 between the shells 4 and 10, linear to 0 at 3.5
# and 10.5, and 0 within 3.5 and beyond 10.5 (its tail is that of 0).
HOLLOW_RNODES = [0, 3.5, 4, 10, 10.5]
HOLLOW_A = [0, 0, 1, 1, 0]
HOLLOW = dict(
    type='cone', rnodes=HOLLOW_RNODES,
    density=dict(
        ptraits=dict(type='nw_uniform'),
        htraits=dict(type='gauss', zmax=1.5)))
HOLLOW_PROPERTIES = dict(wall=30, pt_a=HOLLOW_A, ht_s=0.5)

GEOMETRY = dict(xpos=1.5, ypos=-1, posa=30, incl=60)


def case(driver, cone, observable, line_of_sight='grid', components=()):
    """
    A galaxy of a cone and other components, seen as the observable, with
    a depth axis of voxels of 0.25 arcsec.
    """
    return dict(
        driver=dict(type=driver.type()),
        observable=observable,
        line_of_sight=dict(type=line_of_sight, depth=dict(step=0.25)),
        galaxy=dict(components=[cone, *components]))


def image(driver, cone, **options):
    """A case of an image of 48 by 48 pixels of a cone (see case)."""
    observable = dict(type='pixel_brightness', size=[48, 48], step=[0.5, 0.5])
    return case(driver, cone, observable, **options)


def cone_frame(grid_data, xpos, ypos, posa, incl):
    """
    The position of each voxel of an extra output on a 3d grid in the
    frame of a cone of the given apex (xpos, ypos) and axis (see traits):
    x along the major axis of the disk of the axis, on the side of posa
    (from north, +y, towards -x), y towards the far side, and z along the
    axis, away from the viewer for inclinations below 90 degrees.
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


@pytest.mark.parametrize('sides, nsides', [
    ('both', 2), ('positive', 1), ('negative', 1)])
def test_a_cone_has_the_light_of_its_walls_on_its_sides(
        driver, evaluate_cases, sides, nsides):
    # The density is the light per unit area of the mid-surface, and the
    # height trait its profile across the walls: a side has the light 2
    # pi sin(wall) times the integral of r a(r) (a its density at the
    # shell r) of the area of its mid-surface, exactly for walls that
    # touch neither the axis nor the plane of the apex (up to the
    # sampling of the voxels). The positive side is along +z of the frame
    # of the cone.
    cone = HOLLOW | dict(sides=sides, emission=BAND)
    data, extra = evaluate_cases(
        [image(driver, cone)], GEOMETRY | HOLLOW_PROPERTIES)
    flux = data[0]['brightness']['d'].sum() * 0.5 ** 2
    r = np.linspace(0, 11, 110001)
    moment = np.trapezoid(r * np.interp(r, HOLLOW_RNODES, HOLLOW_A), r)
    expected = nsides * 2 * np.pi * moment * np.sin(np.radians(30))
    assert flux == pytest.approx(expected, rel=5e-3)
    bdata = extra.get('bdata', component='component0')
    z = cone_frame(bdata, **GEOMETRY)[2]
    lit = bdata.data > 0
    assert lit.sum() > 1000
    assert sides != 'positive' or (z[lit] > 0).all()
    assert sides != 'negative' or (z[lit] < 0).all()


@pytest.mark.parametrize('direction, speed', [('tan', 'vt'), ('out', 'vo')])
def test_the_velocity_of_each_voxel(
        driver, evaluate_cases, direction, speed):
    # vsys plus the speed of 100 km/s around the axis (tan) or out of the
    # apex (out), projected on the line of sight: in the frame of the
    # cone, whose line of sight is (0, sin(incl), cos(incl)), 100 x
    # sin(incl) / R (R the distance from the axis) or 100 (y sin(incl) + z
    # cos(incl)) / r (r the distance from the apex)
    cone = HOLLOW | dict(
        kinematics=dict(
            type='traits', vptraits=dict(type='uniform', direction=direction),
            dptraits=UNIFORM),
        emission=LINES)
    observable = dict(
        type='pixel_spectra', size=[48, 48], step=[0.5, 0.5],
        spectral=dict(size=41, step=10, convention='relativistic'))
    _, extra = evaluate_cases(
        [case(driver, cone, observable)],
        GEOMETRY | HOLLOW_PROPERTIES | dict(vsys=20, dpt_a=10) | {
            f'vpt_{speed}': 100})
    vdata = extra.get('vdata', component='component0')
    x, y, z = cone_frame(vdata, **GEOMETRY)
    incl = np.radians(GEOMETRY['incl'])
    if direction == 'tan':
        speed_los = x * np.sin(incl) / np.hypot(x, y)
    else:
        speed_los = (y * np.sin(incl) + z * np.cos(incl)) / np.sqrt(
            x ** 2 + y ** 2 + z ** 2)
    on_cone = np.isfinite(vdata.data)
    assert on_cone.sum() > 1000
    np.testing.assert_allclose(
        vdata.data[on_cone], 20 + 100 * speed_los[on_cone], atol=1e-3)


def test_a_uniform_wall_is_a_hard_cone(driver, evaluate_cases):
    # The hard walls of Barolo and DysmalPy: the angles theta from the axis
    # between theta_in = 20 and theta_out = 40 degrees. The distance of a
    # point from the mid-surface at wall = 30 degrees is r sin(theta -
    # wall), so the walls are within r sin(10 degrees) of it: a uniform
    # height trait of that half-width at each node, which is linear in r
    # between them as the cone interpolates it (of the hollow cone, whose
    # density is 0 within 3.5: the half-width there is that at 3.5).
    cone = HOLLOW | dict(emission=BAND, density=HOLLOW['density'] | dict(
        htraits=dict(type='uniform', zmax=2, nodewise=True)))
    _, extra = evaluate_cases([image(driver, cone)], GEOMETRY | dict(
        wall=30, pt_a=HOLLOW_A, ht_s=list(np.multiply(
            np.maximum(HOLLOW_RNODES, 3.5), np.sin(np.radians(10))))))
    bdata = extra.get('bdata', component='component0')
    x, y, z = cone_frame(bdata, **GEOMETRY)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.degrees(np.arctan2(np.hypot(x, y), np.abs(z)))
    inside = (r > 3.5) & (r < 10.5) & (theta > 20) & (theta < 40)
    # (but where the voxels are on the edges, to rounding, and within a
    # subring, 0.25 arcsec, of the shells 3.5 and 10.5, where the density,
    # linear between the subnodes, is not quite 0)
    clear = (np.abs(theta - 20) > 0.1) & (np.abs(theta - 40) > 0.1) \
        & (np.abs(r - 3.5) > 0.25) & (np.abs(r - 10.5) > 0.25)
    assert inside.sum() > 1000
    np.testing.assert_array_equal((bdata.data > 0)[clear], inside[clear])


@pytest.mark.parametrize('line_of_sight', ['grid', 'transfer'])
def test_the_dust_of_the_disk_dims_the_far_cone(
        driver, evaluate_cases, line_of_sight):
    # A face-on cone out of the plane of a thin layer of dust (that of the
    # disk the cone flows out of): its far side (positive) is behind all
    # the dust of each line of sight, its near side in front of it
    dust = dict(
        type='thick_disk', name='dust', loose=False, tilted=False,
        rnodes=list(range(0, 13)), opacity=dict(type='dust'),
        density=dict(
            ptraits=dict(type='exponential'),
            htraits=dict(type='sech2', zmax=1)))
    properties = dict(
        cone_xpos=0, cone_ypos=0, cone_posa=0, cone_incl=0, cone_wall=30,
        cone_pt_a=HOLLOW_A, cone_ht_s=0.5, dust_xpos=0, dust_ypos=0,
        dust_posa=0,
        dust_incl=0, dust_pt_a=0.5, dust_pt_s=4, dust_ht_s=0.2)
    for sides, fraction in (('positive', 1), ('negative', 0)):
        cone = HOLLOW | dict(name='cone', sides=sides, emission=BAND)
        group = [image(
            driver, cone, line_of_sight=line_of_sight, components=[dust])]
        clear, _ = evaluate_cases(group, properties | dict(dust_pt_a=0))
        dimmed, extra = evaluate_cases(group, properties)
        tau = extra.get('total_odata').data.sum(axis=0)
        clear, dimmed = (
            data[0]['brightness']['d'] for data in (clear, dimmed))
        visible = clear > 1e-3 * clear.max()
        assert tau[visible].min() > 0.05
        np.testing.assert_allclose(
            dimmed[visible] / clear[visible],
            np.exp(-fraction * tau[visible]), rtol=1e-5)


KINEMATICS = dict(
    type='traits',
    vptraits=dict(type='uniform', direction='out'), dptraits=UNIFORM)


@pytest.mark.parametrize('options, message', [
    (dict(kinematics=KINEMATICS | dict(vptraits=dict(type='mass'))),
     "a cone does not take velocity polar trait mass"),
    (dict(kinematics=KINEMATICS | dict(
        vptraits=dict(type='uniform', direction='rad'))),
     r"a cone takes velocity traits along \['tan', 'out'\]"),
    (dict(density=dict(
        ptraits=dict(type='mixture_gauss', nblobs=1),
        htraits=dict(type='gauss', zmax=1))),
     "a cone does not take density polar trait mixture_gauss"),
    (dict(density=dict(ptraits=UNIFORM)),
     "missing required options: 'htraits'"),
    (dict(opacity=dict(type='line', coefficients=UNIFORM)),
     "the opacity of a cone is dust"),
    (dict(sides='up'), "'sides' must be one of")])
def test_what_a_cone_does_not_take(options, message):
    # What disks take, but cones do not
    cone = HOLLOW | dict(kinematics=KINEMATICS, emission=LINES) | options
    with pytest.raises((ConfigError, ValueError), match=message):
        galaxy_parser.load(dict(components=[cone]))
