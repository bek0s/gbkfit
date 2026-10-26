"""
Tests of the orbits of the stars of spheroids (spheroids/_orbits.py, on
the native orbit kernels): the table of the potential against the
gravity it is made of, the energy of the orbits, and their records on
the sky in the geometry of the spheroids.
"""

import numpy as np
import pytest
import scipy.sparse

from gbkfit.model.components.spheroids._orbits import (
    OrbitIntegrator, SkyGrid, potential_at, potential_table)
from gbkfit.model.gravity import GravityNewton
from gbkfit.model.mass import (
    G, MassContext, MassModel, MassPoint, MassSpheroid, ProfileZhao,
    ShapeFlattened)


# A Hernquist spheroid (the Zhao profile of (alpha, beta, gamma) = (1, 4,
# 1)) of the scale radius A (kpc) and the mass M (Msun), round or
# flattened, and a black hole of the mass BH (Msun), at 1 kpc per arcsec
# (lengths in kpc and in arcsec the same)
A = 1.5
M = 1e11
BH = 5e8
# (the context is not used by these mass models)
CONTEXT = MassContext(critical_density=1.0, omega_m=0.3)
# The geometry of the spheroid on the sky (see test_spheroids)
GEOMETRY = dict(xpos=1.5, ypos=-1, posa=30, incl=60)


def gravity(q=None):
    """The gravity of the spheroid (flattened to q, if given)."""
    model = MassModel([
        MassSpheroid(
            'stars', ProfileZhao(parameters='density_radius'),
            None if q is None else ShapeFlattened()),
        MassPoint('bh')])
    params = dict(
        mass_stars_log_rho=np.log10(M / (2 * np.pi * A ** 3)),
        mass_stars_rs=A, mass_stars_alpha=1.0, mass_stars_beta=4.0,
        mass_stars_gamma=1.0, mass_bh_log_m=np.log10(BH))
    if q is not None:
        params['mass_stars_q'] = q
    return GravityNewton().field(model, params, CONTEXT)


def potential_of(q=None):
    """The table of the potential of the gravity (see gravity)."""
    return potential_table(
        gravity(q), 1.0, (1e-6, 1e4), 300, 2 if q is None else 17)


def random_points(rng, n, radii):
    """n points of random directions at radii geometric between two."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.geomspace(*radii, n)[:, None] * directions


def test_the_potential_of_a_sphere_is_exact(driver):
    # That of the Hernquist sphere and the black hole, -G M / (r + A) - G
    # BH / r, from 1e-5 to 1e3 kpc, and its forces
    points = random_points(np.random.default_rng(1), 200, (1e-5, 1e3))
    r = np.linalg.norm(points, axis=1)
    phi, forces = potential_at(driver, np.float64, potential_of(), points)
    exact = -G * M / (r + A) - G * BH / r
    exact_forces = -(G * M / (r + A) ** 2 + G * BH / r ** 2)[:, None] \
        * points / r[:, None]
    np.testing.assert_allclose(phi, exact, rtol=2e-5)
    np.testing.assert_allclose(
        np.linalg.norm(forces - exact_forces, axis=1)
        / np.linalg.norm(exact_forces, axis=1), 0, atol=2e-5)


def test_the_forces_of_a_flattened_spheroid_are_those_of_its_gravity(
        driver):
    # Between the nodes of the table, from 1e-3 to 100 kpc
    q = 0.6
    points = random_points(np.random.default_rng(2), 200, (1e-3, 1e2))
    _, forces = potential_at(driver, np.float64, potential_of(q), points)
    R = np.hypot(points[:, 0], points[:, 1])
    force_R, force_z = gravity(q).forces(R, points[:, 2])
    exact = np.column_stack((
        force_R * points[:, 0] / R, force_R * points[:, 1] / R, force_z))
    np.testing.assert_allclose(
        np.linalg.norm(forces - exact, axis=1)
        / np.linalg.norm(exact, axis=1), 0, atol=1e-4)


@pytest.mark.parametrize('q', [None, 0.6])
def test_orbits_conserve_their_energy(driver, q):
    # 300 orbits from 0.01 to 10 kpc, of speeds of 0.2 to 1.2 times the
    # circular speed in random directions, for 20 periods of the circular
    # orbit of their energy: the time-symmetric steps keep the energy; and
    # all their time is in the cells, which span all space
    rng = np.random.default_rng(3)
    x = random_points(rng, 300, (1e-2, 10))
    table = potential_of(q)
    _, forces = potential_at(driver, np.float64, table, x)
    r = np.linalg.norm(x, axis=1)
    speed = np.sqrt(r * np.linalg.norm(forces, axis=1))
    v = random_points(rng, 300, (1, 1)) \
        * (speed * rng.uniform(0.2, 1.2, 300))[:, None]
    integrator = OrbitIntegrator(
        driver, np.float64, 300, accuracy=0.02, max_steps=10 ** 6,
        nsamples=100, nphi=1, twins=False,
        cell_edges=(np.array([0, 1, 1e5]), np.array([0, 0.5, np.pi / 2])))
    records = integrator.integrate(
        table, np.hstack((x, v)), 20, (GEOMETRY['posa'], GEOMETRY['incl']))
    errors = np.abs(records.energies[:, 1] / records.energies[:, 0] - 1)
    assert np.median(errors) < 1e-4
    assert errors.max() < 1e-2
    np.testing.assert_allclose(records.cells.sum(axis=(1, 2)), 1, rtol=1e-9)


def test_an_orbit_is_seen_in_the_geometry_of_its_spheroid(driver):
    # A circular orbit of the radius R in the plane of the spheroid, seen
    # in the cells of a grid of the sky (each a target) at the geometry:
    # its light is on the ellipse of the ring, at the velocity of each
    # point of it, vc cos(theta) sin(incl) (theta its azimuth, as the
    # spheroids rotate, see test_spheroids): there, at w = 0 of a
    # Gauss-Hermite series, gamma h_0 is sqrt(2) times the light. Its twin
    # rotates the other way.
    R = 5.0
    table = potential_of()
    _, force = potential_at(
        driver, np.float64, table, np.array([[R, 0, 0]]))
    vc = np.sqrt(R * np.abs(force[0, 0]))
    size, step = 400, 0.05
    zero = (GEOMETRY['xpos'] - size * step / 2,
            GEOMETRY['ypos'] - size * step / 2)
    grid = SkyGrid(
        cells=np.array([[1, 0, -zero[0]], [0, 1, -zero[1]]]) / step,
        shape=(size, size),
        shares=scipy.sparse.identity(size * size, format='csr'))
    # (the azimuth of the point of the plane seen at each cell: see
    # test_spheroids.spheroid_frame)
    x, y = np.meshgrid(*(
        (np.arange(size) + 0.5) * step + zero[k] for k in (0, 1)))
    p, i = np.radians(GEOMETRY['posa']), np.radians(GEOMETRY['incl'])
    x, y = x - GEOMETRY['xpos'], y - GEOMETRY['ypos']
    major = -x * np.sin(p) + y * np.cos(p)
    far = (-x * np.cos(p) - y * np.sin(p)) / np.cos(i)
    theta = np.arctan2(far, major).ravel()
    velocity = vc * np.cos(theta) * np.sin(i)
    integrator = OrbitIntegrator(
        driver, np.float64, 1, accuracy=0.01, max_steps=10 ** 6,
        nsamples=4000, nphi=4, twins=True, sky=[grid], order=1)
    records = integrator.integrate(
        table, np.array([[R, 0, 0, 0, vc, 0]]), 1,
        (GEOMETRY['posa'], GEOMETRY['incl']),
        np.array([[GEOMETRY['xpos'], GEOMETRY['ypos']]]),
        (velocity, np.full(size * size, 30.0)))
    light, h0, _ = records.moments[0].T
    seen = light > 0
    assert light.sum() == pytest.approx(1)
    np.testing.assert_allclose(
        np.hypot(major, far).ravel()[seen], R, atol=3 * step / np.cos(i))
    assert np.sum(h0[seen]) / np.sum(light[seen]) \
        == pytest.approx(np.sqrt(2), rel=1e-2)
    twin_light, twin_h0, _ = records.moments[1].T
    np.testing.assert_array_equal(twin_light, light)
    fast = np.abs(velocity) > 0.5 * vc
    assert np.sum(twin_h0[fast]) < 1e-3 * np.sum(h0[fast])
