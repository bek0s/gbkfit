"""
Tests of the transforms of model cubes to visibilities, against direct
sums over their pixels at their directions on the sky.
"""

import numpy as np
import pytest

from gbkfit.observation.observables._uv_transforms import uv_transforms
from gbkfit.utils import gridutils


C = 299792458.0
EPS, ERROR = 1e-12, 1e-10

# A grid of 12 x 10 pixels of 60 arcsec, rotated by 30 deg, and a phase
# centre 0.28 deg from its reference pixel; two channels of HI, and
# baselines of up to 2e4 wavelengths, whose w-term reaches 0.33 rad
GRID = gridutils.make_grid((12, 10), 60, rval=(150, -30), rota=30)
PHASE_CENTRE = (150.2, -29.8)
FREQUENCIES = np.array([1.40e9, 1.42e9])
UVW = np.random.default_rng(1).uniform(-4000, 4000, (50, 3))
CUBE = np.random.default_rng(2).uniform(0, 1, (2, 10, 12)).astype(np.float32)


def unit_vectors(ra, dec):
    """Return the directions of RAs and Decs (radians), (..., 3)."""
    return np.stack((
        np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)), -1)


def baseline_vectors(uvw, frequency, centre):
    """
    Return baselines (u, v, w) in wavelengths as vectors, from the axes of
    the phase centre: east (the pole times it) and north.
    """
    direction = unit_vectors(*np.radians(centre))
    east = np.cross([0, 0, 1], direction)
    east /= np.linalg.norm(east)
    north = np.cross(direction, east)
    return uvw @ np.array([east, north, direction]) * (frequency / C)


def tan_directions():
    ra, dec = gridutils._to_sky(*gridutils.sky_positions(GRID), GRID.coords.rval)
    return unit_vectors(ra, dec)


def sin_directions():
    # The inverse of the orthographic projection of the offsets about the
    # reference pixel
    x, y = gridutils.sky_positions(GRID)
    east, north = -np.radians(x / 3600), np.radians(y / 3600)
    n = np.sqrt(1 - east ** 2 - north ** 2)
    ra0, dec0 = np.radians(GRID.coords.rval[:2])
    dec = np.arcsin(north * np.cos(dec0) + n * np.sin(dec0))
    ra = ra0 + np.arctan2(east, n * np.cos(dec0) - north * np.sin(dec0))
    return unit_vectors(ra, dec)


def direct_sums(directions, w_term=True):
    """
    Return the visibilities of the pixels at their directions, V = sum
    a_j exp(-2 pi i b . (s_j - p)); without the w-term about the reference
    pixel c, times exp(2 pi i (b . c) (s_j . c - 1)).
    """
    phase_centre = unit_vectors(*np.radians(PHASE_CENTRE))
    reference = unit_vectors(*np.radians(GRID.coords.rval[:2]))
    directions = directions.reshape(-1, 3)
    visibilities = []
    for frequency, image in zip(FREQUENCIES, CUBE):
        b = baseline_vectors(UVW, frequency, PHASE_CENTRE)
        phases = b @ (directions - phase_centre).T
        if not w_term:
            phases -= np.outer(b @ reference, directions @ reference - 1)
        visibilities.append(
            np.exp(-2j * np.pi * phases) @ image.ravel().astype(float))
    return np.array(visibilities)


def transform(driver, name):
    uv_transform = uv_transforms[name](
        driver, GRID, UVW, FREQUENCIES, PHASE_CENTRE, EPS)
    out = driver.mem_alloc_d((len(FREQUENCIES), len(UVW)), np.complex128)
    uv_transform(driver.mem_copy_h2d(CUBE), out)
    return driver.mem_copy_d2h(out)


def relative_error(result, expected):
    return np.linalg.norm(result - expected) / np.linalg.norm(expected)


@pytest.mark.parametrize('name, directions', [
    ('exact', tan_directions), ('sin', sin_directions)])
def test_transforms_with_the_w_term_are_the_direct_sums(
        driver, name, directions):
    expected = direct_sums(directions())
    assert relative_error(transform(driver, name), expected) < ERROR


def test_the_flat_transform_is_the_direct_sum_without_the_w_term(driver):
    expected = direct_sums(sin_directions(), w_term=False)
    assert relative_error(transform(driver, 'flat'), expected) < ERROR
    # (which matters here)
    assert relative_error(expected, direct_sums(sin_directions())) > 1e-2


@pytest.mark.parametrize('name', ['exact', 'sin', 'flat'])
def test_the_adjoint_is_that_of_the_transform(driver, name):
    # <A a, V> = <a, A^H V>, for a cube a and visibilities V
    uv_transform = uv_transforms[name](
        driver, GRID, UVW, FREQUENCIES, PHASE_CENTRE, EPS)
    rng = np.random.default_rng(3)
    v = rng.standard_normal((2, 50)) + 1j * rng.standard_normal((2, 50))
    forward = transform(driver, name)
    images = driver.mem_alloc_d(CUBE.shape, np.complex128)
    uv_transform.adjoint(driver.mem_copy_h2d(v), images)
    assert np.sum(CUBE * driver.mem_copy_d2h(images)) == pytest.approx(
        np.vdot(forward, v), rel=1e-10)


def test_the_projections_differ_by_the_cube_of_the_angle():
    # SIN pixels are t^3 / 2 nearer to the reference pixel than TAN ones,
    # at an angle t from it: up to 1.6e-9 rad for the corner pixels here
    # (t = 2.3e-3 rad)
    def angle(a, b):
        return 2 * np.arcsin(np.linalg.norm(a - b, axis=-1) / 2)
    reference = unit_vectors(*np.radians(GRID.coords.rval[:2]))
    t = angle(tan_directions(), reference)
    np.testing.assert_allclose(
        angle(tan_directions(), sin_directions()), t ** 3 / 2, rtol=1e-2)


def test_the_tolerance_is_checked(driver):
    with pytest.raises(ValueError, match="eps"):
        uv_transforms['sin'](driver, GRID, UVW, FREQUENCIES, PHASE_CENTRE, 0)
