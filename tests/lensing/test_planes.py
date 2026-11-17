"""
Tests of the planes of a lensed sky (gbkfit.lensing.plane_layout):
galaxies and lenses close in redshift share a plane, at one distance,
and the depth order orders only the galaxies of a plane.
"""

import pytest

from gbkfit.lensing import (
    PLANE_LINKING_VELOCITY, Plane, plane_layout, velocity_separation)


C = 299792.458


def redshift_behind(z, velocity):
    """The redshift the given velocity (km/s) behind z."""
    return z + velocity / C * (1 + z)


def test_the_velocity_separation_of_two_redshifts():
    assert velocity_separation(0.5, 0.51) == pytest.approx(
        C * 0.01 / 1.5)
    assert velocity_separation(0.51, 0.5) < 0
    assert velocity_separation(0.3, redshift_behind(0.3, 2500)) \
        == pytest.approx(2500)


def test_a_galaxy_galaxy_lens():
    # A star in front, the lens galaxy and its lens, and a source
    planes = plane_layout(
        [('star', 0.0), ('deflector', 0.351), ('arc', 2.1)],
        [('lens', 0.35)])
    assert planes == (
        Plane(0.0, (0,), ()), Plane(0.35, (1,), (0,)),
        Plane(2.1, (2,), ()))


def test_a_lens_alone_in_its_plane():
    planes = plane_layout([('arc', 2.1)], [('dark', 0.5)])
    assert planes == (Plane(0.5, (), (0,)), Plane(2.1, (0,), ()))


def test_neighbours_link_as_friends_of_friends():
    # 2000 km/s apart each: linked in a chain wider than the velocity
    z = [0.4]
    for _ in range(3):
        z.append(redshift_behind(z[-1], 2000))
    galaxies = [(f'g{i}', zi) for i, zi in enumerate(z)]
    planes = plane_layout(galaxies, [('lens', 0.2)])
    assert [plane.galaxies for plane in planes] == [(), (0, 1, 2, 3)]
    # (just beyond the velocity, apart)
    far = redshift_behind(0.4, PLANE_LINKING_VELOCITY * 1.001)
    planes = plane_layout([('a', 0.4), ('b', far)], [('lens', 0.2)])
    assert [plane.galaxies for plane in planes] == [(), (0,), (1,)]
    near = redshift_behind(0.4, PLANE_LINKING_VELOCITY * 0.999)
    planes = plane_layout([('a', 0.4), ('b', near)], [('lens', 0.2)])
    assert [plane.galaxies for plane in planes] == [(), (0, 1)]


def test_the_distance_of_a_plane_is_that_of_its_lenses():
    # A cluster map and a member's lens at its redshift, with members:
    # the mean of the lenses; without lenses, that of the galaxies
    planes = plane_layout(
        [('bcg', 0.398), ('member', 0.401), ('arc', 1.0), ('pair', 1.002)],
        [('cluster', 0.396), ('perturber', 0.4)])
    assert planes[0].redshift == pytest.approx(0.398)
    assert planes[0].galaxies == (0, 1)
    assert planes[0].lenses == (0, 1)
    assert planes[1].redshift == pytest.approx(1.001)


def test_a_group_of_the_depth_order_is_of_one_plane():
    # A merger: within the velocity, one plane; beyond, an error that
    # names the galaxies and their redshifts
    lenses = [('lens', 0.3)]
    near = redshift_behind(0.5, 2500)
    planes = plane_layout([('a', 0.5), ('b', near)], lenses, [('a', 'b')])
    assert planes[1].galaxies == (0, 1)
    far = redshift_behind(0.5, 5000)
    with pytest.raises(ValueError, match="'a': 0.5") as error:
        plane_layout([('a', 0.5), ('b', far)], lenses, [('a', 'b')])
    assert "'b'" in str(error.value) and "split the group" in str(error.value)


def test_the_depth_order_orders_the_galaxies_of_a_plane_only():
    galaxies = [('a', 0.5), ('b', 0.5001), ('arc', 2.0)]
    lenses = [('lens', 0.5)]
    # In a plane, any order; across planes, that of the redshifts
    plane_layout(galaxies, lenses, [('b',), ('a',), ('arc',)])
    plane_layout(galaxies, lenses, [('a',), ('arc',)])
    with pytest.raises(ValueError, match="'arc' in front of 'a'"):
        plane_layout(galaxies, lenses, [('arc',), ('a',)])
    with pytest.raises(ValueError, match="'arc' in front of 'b'"):
        plane_layout(galaxies, lenses, [('a',), ('arc',), ('b',)])


def test_the_errors_of_a_layout():
    with pytest.raises(ValueError, match="there are none"):
        plane_layout([('a', 0.5)], [])
    with pytest.raises(ValueError, match="have none: \\['b'\\]"):
        plane_layout([('a', 0.5), ('b', None)], [('lens', 0.3)])
    with pytest.raises(ValueError, match="have none: \\[0\\]"):
        plane_layout([(None, None)], [('lens', 0.3)])
    with pytest.raises(ValueError, match="unknown galaxies \\['c'\\]"):
        plane_layout([('a', 0.5)], [('lens', 0.3)], [('c',)])
    with pytest.raises(ValueError, match="repeats"):
        plane_layout([('a', 0.5)], [('lens', 0.3)], [('a',), ('a',)])
