"""
Tests of the flows of the plane of a disk: along a bar (the velocity
traits bar and nw_bar), in two wedges (wedge) and of a spiral density
wave (spiral), seen on the line of sight as their formulas give; and of
the density of a spiral density wave (the density trait spiral).
"""

import numpy as np
import pytest

from gbkfit.model import component_parser, galaxy_parser
from gbkfit.model.components.disks._profiles import DiskDensity
from gbkfit.model.components.traits import RPTraitSpiral, VPTraitSpiral
from gbkfit.utils.parseutils import ConfigError


RNODES = list(range(0, 13))
INCL = 60

# Points of the plane of the disk: radii (arcsec) and azimuths (degrees)
_rng = np.random.default_rng(1)
R = _rng.uniform(0.1, 11.9, 400)
THETA = _rng.uniform(-180, 180, 400)


def disk_config(vptraits, ptraits=None):
    """
    A disk of the nodes RNODES, of the velocity traits and the density
    traits (by default, an exponential).
    """
    return dict(
        type='smdisk', loose=False, tilted=False, rnodes=RNODES,
        density=dict(ptraits=ptraits or dict(type='exponential')),
        kinematics=dict(vptraits=vptraits, dptraits=dict(type='uniform')),
        emission=dict(type='lines'))


def disk(vptraits, ptraits=None):
    return component_parser.load(
        disk_config(vptraits, ptraits)).density_disk()


# The values of the parameters of the disk but those of its velocity
# traits
DISK_VALUES = dict(
    xpos=0, ypos=0, posa=0, incl=INCL, vsys=0, pt_a=1, pt_s=3, dpt_a=20)


def seen(driver, vptrait, values):
    """
    The line-of-sight velocity of a velocity trait of a disk at the points
    (R, THETA), at the inclination INCL, for the values of its parameters
    (without their prefix).
    """
    params = DISK_VALUES \
        | {f'vpt_{name}': value for name, value in values.items()}
    plan = disk(vptrait).plan(driver, 0, np.float64)
    return plan.evaluate_polar_traits(params, 'vpt', R, THETA, INCL)[0]


def bar_flow(speed, p, s):
    """
    The line-of-sight velocity of the flow of the speed (at each point)
    along the axis of a bar at the azimuth p, within the strip of the
    width s: v sign(x) sin(p) sin(incl) where |y| < s / 2.
    """
    x = R * np.cos(np.deg2rad(THETA - p))
    y = R * np.sin(np.deg2rad(THETA - p))
    return np.where(
        np.abs(y) < s / 2,
        speed * np.sign(x) * np.sin(np.deg2rad(p)) * np.sin(np.deg2rad(INCL)),
        0)


def test_a_bar_flows_along_its_axis(driver):
    # (inwards, along an axis 30 degrees from the major axis)
    values = dict(v=-90, p=30, s=4)
    np.testing.assert_allclose(
        seen(driver, dict(type='bar'), values),
        bar_flow(-90, 30, 4), atol=1e-9)


def test_the_speed_of_nw_bar_is_that_at_the_distance_along_it(driver):
    # A speed that falls to 0 at 8 arcsec along the bar (its length), at
    # the subnodes and linear between them, as the disk interpolates it
    trait = dict(type='nw_bar')
    speeds = np.clip(80 - 10 * np.array(RNODES, dtype=float), 0, None)
    subrnodes = disk(trait).subrnodes()
    distance = np.abs(R * np.cos(np.deg2rad(THETA - 120)))
    speed = np.interp(
        distance, subrnodes, np.interp(subrnodes, RNODES, speeds))
    np.testing.assert_allclose(
        seen(driver, trait, dict(p=120, s=3, v=speeds)),
        bar_flow(speed, 120, 3), atol=1e-9)


def test_a_wedge_has_its_speed_within_its_two_wedges(driver):
    # A radial flow within 30 degrees of the azimuths 70 and 250 degrees
    within = np.abs(np.cos(np.deg2rad(THETA - 70))) \
        >= np.cos(np.deg2rad(30))
    radial = 50 * np.sin(np.deg2rad(THETA)) * np.sin(np.deg2rad(INCL))
    np.testing.assert_allclose(
        seen(driver, dict(type='wedge', direction='rad'),
             dict(vr=50, p=70, s=60)),
        np.where(within, radial, 0), atol=1e-9)


def test_cones_take_no_flows_of_the_plane_of_a_disk():
    for trait in (dict(type='bar'), dict(type='wedge', direction='tan')):
        with pytest.raises(ConfigError, match="a cone does not take"):
            component_parser.load(dict(
                type='smcone', rnodes=RNODES,
                density=dict(
                    ptraits=dict(type='uniform'),
                    htraits=dict(type='gauss', zmax=1.5)),
                kinematics=dict(
                    vptraits=trait, dptraits=dict(type='uniform')),
                emission=dict(type='lines')))


def davies(r, v, m, c, e, o, sense):
    """
    The tables of a spiral density wave of m arms at the radii r (see
    VPTraitSpiral), in the flat rotation v: by Davies et al. (2009, A3,
    A10 and A11), with kappa^2 = 2 Omega^2, the sign sense of k Omega
    (-1: trailing) and f by the trapezoid rule; 0 at the centre.
    """
    r = np.asarray(r, dtype=float)
    omega = np.divide(v, r, out=np.zeros_like(r), where=r > 0)
    k = np.where(
        r > 0, sense * np.sqrt(m ** 2 * (omega - o) ** 2 - 2 * omega ** 2) / c,
        0)
    nonzero = k != 0
    return dict(
        ar=np.divide(
            -e * m * (omega - o), k, out=np.zeros_like(r), where=nonzero),
        at=np.divide(
            e * 2 * omega ** 2, 2 * k * omega, out=np.zeros_like(r),
            where=nonzero),
        f=np.concatenate(([0], np.cumsum(np.diff(r) * (k[1:] + k[:-1]) / 2))))


def test_the_tables_of_a_spiral_wave_are_those_of_davies():
    # (in the example of Davies et al.: 3 arms, c = 50, e = 0.1, flat
    # rotation of 100 km/s: v_R = 5.7 cos(phi) and v_t = -1.9 sin(phi) for
    # trailing arms, k = -sqrt(7) Omega / c)
    r = np.linspace(0, 12, 25)
    v = np.full_like(r, 100.0)
    values = dict(c=50, e=0.1, o=0)
    for winding, sense in (('trailing', -1), ('leading', 1)):
        tables = VPTraitSpiral(3, winding).wave(r, v, 0 * v, None, values)
        expected = davies(r, 100, 3, 50, 0.1, 0, sense)
        for name in ('ar', 'at', 'f'):
            np.testing.assert_allclose(tables[name], expected[name])
    assert tables['ar'][1:] == pytest.approx(-0.1 * 50 * 3 / np.sqrt(7))
    # One arm in a flat rotation has no wave (m^2 Omega^2 < kappa^2)
    tables = VPTraitSpiral(1).wave(r, v, 0 * v, None, values)
    assert not any(np.any(table) for table in tables.values())


def test_a_spiral_wave_in_the_rotation_of_its_disk(driver):
    # A wave of 3 trailing arms in the flat rotation of its disk, of a
    # pattern speed below that of the disk: its tables at the subnodes,
    # which the gravity of the galaxy gives, interpolated linearly
    vptraits = [dict(type='uniform', direction='tan'),
                dict(type='spiral', order=3)]
    values = DISK_VALUES | dict(
        vpt_vt=100, vpt1_p=40, vpt1_c=50, vpt1_e=0.1, vpt1_o=2)
    galaxy = galaxy_parser.load(dict(components=[disk_config(vptraits)]))
    given = galaxy.plan_gravity().evaluate(values)[0]
    subrnodes = disk(vptraits).subrnodes()
    tables = davies(subrnodes, 100, 3, 50, 0.1, 2, -1)
    ar, at, f = (np.interp(R, subrnodes, tables[name])
                 for name in ('ar', 'at', 'f'))
    phi = 3 * np.deg2rad(THETA - 40) - f
    theta = np.deg2rad(THETA)
    expected = (ar * np.cos(phi) * np.sin(theta)
                + at * np.sin(phi) * np.cos(theta)) * np.sin(np.deg2rad(INCL))
    plan = disk(vptraits).plan(driver, 0, np.float64)
    np.testing.assert_allclose(
        plan.evaluate_polar_traits(
            values | given, 'vpt', R, THETA, INCL)[1],
        expected, rtol=1e-9, atol=1e-9)


# A disk of an exponential density and a flat rotation, with the density
# of a wave of 3 trailing arms
WAVE_VPTRAITS = dict(type='uniform', direction='tan')
WAVE_PTRAITS = [dict(type='exponential'), dict(type='spiral', order=3)]
WAVE_VALUES = DISK_VALUES | dict(
    vpt_vt=100, pt1_p=40, pt1_c=50, pt1_e=0.3, pt1_o=2)


def test_the_density_of_a_spiral_wave(driver):
    # e Sigma_0 cos(phi), Sigma_0 the exponential: its tables at the
    # subnodes (no wave at the centre), which the gravity of the galaxy
    # gives, interpolated linearly
    galaxy = galaxy_parser.load(dict(
        components=[disk_config(WAVE_VPTRAITS, WAVE_PTRAITS)]))
    given = galaxy.plan_gravity().evaluate(WAVE_VALUES)[0]
    wave_disk = disk(WAVE_VPTRAITS, WAVE_PTRAITS)
    subrnodes = np.asarray(wave_disk.subrnodes())
    density = np.where(subrnodes > 0, 0.3 * np.exp(-subrnodes / 3), 0)
    f = davies(subrnodes, 100, 3, 50, 0.3, 2, -1)['f']
    phi = 3 * np.deg2rad(THETA - 40) - np.interp(R, subrnodes, f)
    plan = wave_disk.plan(driver, 0, np.float64)
    np.testing.assert_allclose(
        plan.evaluate_polar_traits(WAVE_VALUES | given, 'rpt', R, THETA)[1],
        np.interp(R, subrnodes, density) * np.cos(phi), rtol=1e-9,
        atol=1e-12)


def test_the_gravity_takes_no_density_of_a_spiral_wave(driver):
    # (its azimuthal mean is 0, whatever its tables)
    wave_disk = disk(WAVE_VPTRAITS, WAVE_PTRAITS)
    density = DiskDensity(
        wave_disk, wave_disk.plan(driver, 0, np.float64), WAVE_VALUES, 1.0)
    radii = np.array([0.5, 3.0, 7.5])
    np.testing.assert_allclose(
        density.surface_densities(radii),
        [np.exp(-radii / 3), np.zeros(3)], atol=1e-12)


def test_a_spiral_wave_needs_the_rotation_of_its_disk():
    for ptraits, vptraits in (
            (None, [dict(type='spiral', order=2),
                    dict(type='uniform', direction='rad')]),
            (WAVE_PTRAITS, dict(type='uniform', direction='rad'))):
        with pytest.raises(ConfigError, match="give the disk tangential"):
            disk(vptraits, ptraits)


def test_a_given_wavenumber_winds_the_wave_as_given():
    # That of the dispersion relation (in a flat rotation, trailing),
    # given, makes the same wave
    r = np.linspace(0.5, 12, 24)
    v = np.full_like(r, 100.0)
    omega = 100 / r
    k = -np.sqrt(9 * (omega - 2) ** 2 - 2 * omega ** 2) / 50
    dispersion = VPTraitSpiral(3).wave(
        r, v, 0 * v, None, dict(c=50, e=0.1, o=2))
    given = VPTraitSpiral(3, wavenumber='given').wave(
        r, v, 0 * v, None, dict(e=0.1, o=2, k=k))
    for name in ('ar', 'at', 'f'):
        np.testing.assert_allclose(given[name], dispersion[name])


def test_the_arms_of_an_image_by_a_given_wavenumber(driver):
    # Two trailing arms of the pitch angle 20 degrees (a logarithmic
    # spiral), in the light of a disk without kinematics: k = -2 / (R
    # tan(20 degrees)), and f its integral by the trapezoid rule
    component = dict(
        type='smdisk', loose=False, tilted=False, rnodes=RNODES,
        density=dict(ptraits=[
            dict(type='exponential'),
            dict(type='spiral', order=2, wavenumber='given')]),
        emission=dict(type='band'))
    image_disk = component_parser.load(component).density_disk()
    subrnodes = np.asarray(image_disk.subrnodes())
    k = -2 / (np.where(subrnodes > 0, subrnodes, np.inf)
              * np.tan(np.deg2rad(20)))
    values = dict(
        xpos=0, ypos=0, posa=0, incl=INCL, pt_a=1, pt_s=3, pt1_p=40,
        pt1_e=0.3, pt1_k=k)
    galaxy = galaxy_parser.load(dict(components=[component]))
    given = galaxy.plan_gravity().evaluate(values)[0]
    f = np.concatenate(
        ([0], np.cumsum(np.diff(subrnodes) * (k[1:] + k[:-1]) / 2)))
    density = np.where(subrnodes > 0, 0.3 * np.exp(-subrnodes / 3), 0)
    phi = 2 * np.deg2rad(THETA - 40) - np.interp(R, subrnodes, f)
    plan = image_disk.plan(driver, 0, np.float64)
    np.testing.assert_allclose(
        plan.evaluate_polar_traits(values | given, 'rpt', R, THETA)[1],
        np.interp(R, subrnodes, density) * np.cos(phi), rtol=1e-9,
        atol=1e-12)


def test_the_parameters_of_the_wavenumbers():
    dispersion = dict(p=None, c=None, e=None, o=None)
    assert VPTraitSpiral(2).params_sm().keys() == dispersion.keys()
    assert RPTraitSpiral(2).params_sm().keys() == dispersion.keys()
    assert list(VPTraitSpiral(2, wavenumber='given').params_sm()) \
        == ['p', 'e', 'o']
    assert list(RPTraitSpiral(2, wavenumber='given').params_sm()) \
        == ['p', 'e']
    assert list(RPTraitSpiral(2, wavenumber='given').params_rnw(3)) \
        == ['a', 'f', 'k']
    assert RPTraitSpiral(2, wavenumber='given').dump() == dict(
        type='spiral', order=2, wavenumber='given')
    with pytest.raises(ConfigError, match="winds the arms by its sign"):
        VPTraitSpiral(2, winding='leading', wavenumber='given')
