"""
Tests of the paths of light through lens planes (gbkfit.lensing.Tracer):
the rays of an image bent by the lenses in front of each plane,
recursively, by D_ls / D_s of the redshifts.
"""

import numpy as np
import pytest
from astropy.cosmology import Planck18

from gbkfit.cosmology import DEFAULT, CosmologyFlatLambdaCDM
from gbkfit.driver.drivers.host import DriverHost
from gbkfit.lensing import (
    Lens, LensDeflectionMap, LensSIE, Plane, SkyPoints, Tracer,
    plane_layout, stretch)


ORIGIN = (150.0, 2.0)

# (the driver of the deflections: the host)
HOST = DriverHost()


def distance(z1, z2=None):
    """The angular diameter distance (Mpc) of astropy, of Planck18."""
    if z2 is None:
        return Planck18.angular_diameter_distance(z1).value
    return Planck18.angular_diameter_distance(z1, z2).value


def ratio(z_lens, z_source):
    """D_ls / D_s."""
    return distance(z_lens, z_source) / distance(z_source)


def circle(radius, centre=(0.0, 0.0), n=16):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return SkyPoints(
        centre[0] + radius * np.cos(angle),
        centre[1] + radius * np.sin(angle), ORIGIN)


def sis(name, redshift, source_redshift):
    return Lens(name, redshift, LensSIE(), source_redshift=source_redshift)


def sphere(theta_e, centre=(0.0, 0.0)):
    """The parameters of an isothermal sphere: an SIE of q = 1."""
    return dict(theta_e=theta_e, q=1, posa=0, xpos=centre[0], ypos=centre[1])


def lens_values(values):
    """
    The parameters of a lens of one SIE: those of the SIE (values),
    prefixed by the name of the piece, after its type.
    """
    return {f'sie_{name}': value for name, value in values.items()}


def sis_values(theta_e, centre=(0.0, 0.0)):
    return lens_values(sphere(theta_e, centre))


def tracer_of(galaxies, lenses):
    planes = plane_layout(
        galaxies, [(lens.name(), lens.redshift()) for lens in lenses])
    return Tracer(planes, lenses, DEFAULT)


def test_the_distance_between_two_redshifts():
    assert DEFAULT.angular_diameter_distance_between(0.5, 2.0) \
        == pytest.approx(distance(0.5, 2.0), rel=1e-12)
    assert DEFAULT.angular_diameter_distance_between(0.5, 0.5) == 0
    flat = CosmologyFlatLambdaCDM(70, 0.3)
    # (a flat universe: D_12 = (D_M2 - D_M1) / (1 + z2))
    cosmology = flat.astropy_cosmology()
    expected = (cosmology.comoving_transverse_distance(2.0)
                - cosmology.comoving_transverse_distance(0.5)).value / 3
    assert flat.angular_diameter_distance_between(0.5, 2.0) \
        == pytest.approx(expected, rel=1e-10)
    with pytest.raises(ValueError, match="beyond"):
        DEFAULT.angular_diameter_distance_between(2.0, 0.5)


def test_an_isothermal_sphere_makes_an_einstein_ring():
    # A source at the redshift its Einstein radius is quoted for: the ring
    # of that radius about its centre is the image of the centre
    tracer = tracer_of(
        [('deflector', 0.35), ('arc', 2.1)], [sis('lens', 0.35, 2.1)])
    centre = (0.1, -0.2)
    rays = tracer.trace(circle(1.2, centre), [sis_values(1.2, centre)], HOST)
    assert len(rays) == 2
    np.testing.assert_allclose(rays[1].x, 0.1, atol=1e-12)
    np.testing.assert_allclose(rays[1].y, -0.2, atol=1e-12)
    # (and the deflector, of the plane of the lens, is not lensed)
    np.testing.assert_array_equal(rays[0].x, circle(1.2, (0.1, -0.2)).x)
    assert not tracer.lensed(0) and tracer.lensed(1)


def test_one_lens_of_its_quoted_source_is_todays_model():
    # theta_s = theta - alpha, of the strengths quoted
    lens = Lens('lens', 0.35, LensSIE(), source_redshift=2.1)
    tracer = tracer_of([('arc', 2.1)], [lens])
    points = SkyPoints(*np.mgrid[-2:2:7j, -2:2:5j], ORIGIN)
    values = dict(theta_e=1.3, q=0.6, posa=30, xpos=0.2, ypos=-0.1)
    alpha = LensSIE().deflection(points, values, HOST)
    ray = tracer.trace(points, [lens_values(values)], HOST)[-1]
    np.testing.assert_allclose(ray.x, points.x - alpha[0], atol=1e-12)
    np.testing.assert_allclose(ray.y, points.y - alpha[1], atol=1e-12)


@pytest.mark.parametrize('z_source', [0.8, 1.5, 3.5])
def test_the_deflection_scales_by_d_ls_over_d_s(z_source):
    # The Einstein radius of a source at another redshift: theta_e times
    # the ratio of the D_ls / D_s of the two
    tracer = tracer_of([('arc', z_source)], [sis('lens', 0.35, 2.1)])
    radius = 1.2 * ratio(0.35, z_source) / ratio(0.35, 2.1)
    ray = tracer.trace(circle(radius), [sis_values(1.2)], HOST)[-1]
    np.testing.assert_allclose(np.hypot(ray.x, ray.y), 0, atol=1e-12)


def test_maps_are_of_a_unit_ratio():
    # A uniform deflection map: the source moves by D_ls / D_s of it
    deflection = LensDeflectionMap(
        np.full((5, 5), 0.4), np.full((5, 5), -0.2), step=1, rval=ORIGIN)
    tracer = tracer_of([('arc', 1.5)], [Lens('cluster', 0.4, deflection)])
    points = SkyPoints(np.array([0.5, -1.0]), np.array([0.0, 1.0]), ORIGIN)
    ray = tracer.trace(points, [{}], HOST)[-1]
    f = ratio(0.4, 1.5)
    np.testing.assert_allclose(ray.x, points.x - f * 0.4, atol=1e-12)
    np.testing.assert_allclose(ray.y, points.y + f * 0.2, atol=1e-12)


def test_maps_made_for_one_source_are_quoted_for_it():
    # A source at the redshift of the map: moved by the map itself; at
    # another, by the ratio of the D_ls / D_s of the two
    deflection = LensDeflectionMap(
        np.full((5, 5), 0.4), np.full((5, 5), -0.2), step=1, rval=ORIGIN,
        quoted=True)
    points = SkyPoints(np.array([0.5, -1.0]), np.array([0.0, 1.0]), ORIGIN)
    for z_source in (1.5, 3.0):
        tracer = tracer_of([('arc', z_source)], [
            Lens('cluster', 0.4, deflection, source_redshift=1.5)])
        ray = tracer.trace(points, [{}], HOST)[-1]
        f = ratio(0.4, z_source) / ratio(0.4, 1.5)
        np.testing.assert_allclose(ray.x, points.x - f * 0.4, atol=1e-12)
        np.testing.assert_allclose(ray.y, points.y + f * 0.2, atol=1e-12)


def test_the_lenses_of_one_plane_add_up():
    lenses = [sis('a', 0.4, 2.0), sis('b', 0.401, 2.0)]
    tracer = tracer_of([('arc', 2.0)], lenses)
    assert len(tracer.planes()) == 2
    points = SkyPoints(*np.mgrid[-2:2:5j, -2:2:4j], ORIGIN)
    values = [sphere(0.8, (0.3, 0)), sphere(0.5, (-0.4, 0.2))]
    both = tracer.trace(points, list(map(lens_values, values)), HOST)[-1]
    # (each at the distance of the plane, their mean redshift)
    z = 0.4005
    expected_x, expected_y = points.x.copy(), points.y.copy()
    for lens, value in zip(lenses, values):
        quoted = distance(2.0) / distance(z, 2.0)
        alpha = LensSIE().deflection(points, value, HOST)
        expected_x -= ratio(z, 2.0) * quoted * alpha[0]
        expected_y -= ratio(z, 2.0) * quoted * alpha[1]
    np.testing.assert_allclose(both.x, expected_x, atol=1e-12)
    np.testing.assert_allclose(both.y, expected_y, atol=1e-12)


def two_planes():
    """
    Two isothermal spheres in planes in a row (at 0.3 and 1.0) and a
    source behind them (at 2.5), each quoted for the next: the tracer,
    and the deflection (alpha-hat) of each lens per arcsec of theta_e.
    """
    lenses = [sis('near', 0.3, 1.0), sis('far', 1.0, 2.5)]
    tracer = tracer_of(
        [('deflector', 0.3), ('s1', 1.0), ('s2', 2.5)], lenses)
    units = (1 / ratio(0.3, 1.0), 1 / ratio(1.0, 2.5))
    return tracer, units


def test_two_planes_make_the_ring_of_their_sum():
    # Two aligned spheres: a ray of the axis, at theta on it, crosses the
    # second plane at theta - f12 b1 and reaches the source at
    # theta - f13 b1 - f23 b2: the ring of that radius
    tracer, units = two_planes()
    b1, b2 = 0.9 * units[0], 0.5 * units[1]
    f12 = distance(0.3, 1.0) / distance(1.0)
    f13 = distance(0.3, 2.5) / distance(2.5)
    f23 = distance(1.0, 2.5) / distance(2.5)
    radius = f13 * b1 + f23 * b2
    assert radius > f12 * b1
    rays = tracer.trace(
        circle(radius), [sis_values(0.9), sis_values(0.5)], HOST)
    np.testing.assert_allclose(
        np.hypot(rays[1].x, rays[1].y), radius - f12 * b1, rtol=1e-12)
    np.testing.assert_allclose(np.hypot(rays[2].x, rays[2].y), 0, atol=1e-12)
    for p, q, f in ((0, 1, f12), (0, 2, f13), (1, 2, f23)):
        assert tracer.factor(p, q) == pytest.approx(f, rel=1e-12)


def test_the_lens_of_a_farther_plane_bends_rays_where_they_cross_it():
    # (recursive, not a sum: the second sphere, off the axis, deflects
    # at the points of the rays on its plane)
    tracer, units = two_planes()
    centre = np.array([0.3, 0.1])
    points = SkyPoints(*np.mgrid[-2:2:6j, -2:2:5j] + 0.01, ORIGIN)
    rays = tracer.trace(
        points, [sis_values(0.9), sis_values(0.5, centre)], HOST)

    def alpha(x, y, b, c):
        dx, dy = x - c[0], y - c[1]
        r = np.hypot(dx, dy)
        return b * dx / r, b * dy / r
    a1 = alpha(points.x, points.y, 0.9 * units[0], (0, 0))
    f12 = distance(0.3, 1.0) / distance(1.0)
    x2, y2 = points.x - f12 * a1[0], points.y - f12 * a1[1]
    a2 = alpha(x2, y2, 0.5 * units[1], centre)
    f13 = distance(0.3, 2.5) / distance(2.5)
    f23 = distance(1.0, 2.5) / distance(2.5)
    np.testing.assert_allclose(rays[1].x, x2, atol=1e-12)
    np.testing.assert_allclose(
        rays[2].x, points.x - f13 * a1[0] - f23 * a2[0], atol=1e-12)
    np.testing.assert_allclose(
        rays[2].y, points.y - f13 * a1[1] - f23 * a2[1], atol=1e-12)
    # (to a plane, and from deflections given)
    assert len(tracer.trace(points, [sis_values(0.9), {}], HOST, upto=1)) == 2
    deflections = {
        0: tracer.deflection(0, rays[0], [sis_values(0.9), {}], HOST)}
    np.testing.assert_allclose(
        tracer.ray(1, points, deflections).y, rays[1].y, atol=1e-15)
    with pytest.raises(KeyError):
        tracer.ray(2, points, deflections)


def test_the_jacobian_of_an_isothermal_sphere():
    # A = I - theta_e / r (I - n n^T): along the radius 1, across it
    # 1 - theta_e / r
    tracer = tracer_of([('arc', 2.1)], [sis('lens', 0.35, 2.1)])
    points = SkyPoints(
        np.array([2.4, 0.0, -0.6]), np.array([0.0, 1.0, 0.8]), ORIGIN)
    jacobian = tracer.jacobian(1, points, [sis_values(1.2)], 1e-4, HOST)
    r = np.hypot(points.x, points.y)
    n = np.stack([points.x / r, points.y / r], -1)
    expected = np.eye(2) - (1.2 / r)[:, None, None] * (
        np.eye(2) - n[:, :, None] * n[:, None, :])
    np.testing.assert_allclose(jacobian, expected, atol=1e-7)
    np.testing.assert_allclose(
        stretch(jacobian), 1 / np.minimum(1, np.abs(1 - 1.2 / r)),
        rtol=1e-6)


def test_the_stretch_is_that_of_the_smallest_singular_value():
    # (a Jacobian of several planes need not be symmetric)
    angle = np.radians(30)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    jacobian = np.stack([rotation @ np.diag([1, 0.25]), np.zeros((2, 2))])
    np.testing.assert_allclose(stretch(jacobian), [4, np.inf])


def test_the_planes_of_a_tracer():
    lens = sis('lens', 0.35, 2.1)
    with pytest.raises(ValueError, match="near to far"):
        Tracer([Plane(2.1, (0,), ()), Plane(0.35, (), (0,))], [lens], DEFAULT)
    with pytest.raises(ValueError, match="once"):
        Tracer([Plane(0.35, (), ()), Plane(2.1, (0,), ())], [lens], DEFAULT)
    # (a lens quoting a source in front of its plane)
    with pytest.raises(ValueError, match="not behind"):
        Tracer([Plane(2.2, (), (0,))], [lens], DEFAULT)
