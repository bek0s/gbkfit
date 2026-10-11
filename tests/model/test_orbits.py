"""
Tests for the Schwarzschild proof of concept (gbkfit.model.orbits): the
Gauss-Hermite series, the exact mock galaxy against independent
integrals, the orbit integration, the weight solver, and the recovery
of the mass of a mock (slow).
"""

import numpy as np
import pytest
import scipy.integrate
import scipy.optimize

from gbkfit.model.mass import G, MassContext
from gbkfit.model.orbits.gauss_hermite import (
    fit_gauss_hermite, gauss_hermite_losvd, hermite_weights, mass_factor)
from gbkfit.model.orbits.kinematics import BinnedField, polar_bins
from gbkfit.model.orbits.library import (
    LibraryOptions, SphericalPotential, build_library, initial_conditions,
    integrate)
from gbkfit.model.orbits.mocks import MockGalaxy, add_noise, observe
from gbkfit.model.orbits.profiles import Hernquist
from gbkfit.model.orbits.schwarzschild import fit_weights, regularised_nnls


LIGHT = Hernquist(total=2.5e10, a=1.5)
UPSILON = 4.0
BH_MASS = 5e8
# (the context is not used by these mass models)
CONTEXT = MassContext(critical_density=1.0, omega_m=0.3)


def jeans_pressure(galaxy: MockGalaxy, r: float) -> float:
    """
    Return nu sigma^2 of the isotropic Jeans model of a mock: int_r^inf
    nu G M / r'^2 dr'.
    """
    def integrand(log_r: float) -> float:
        x = np.exp(log_r)
        mass = galaxy.upsilon * LIGHT.enclosed(x) + galaxy.bh_mass
        return float(LIGHT.density(x) * G * mass / x)
    return scipy.integrate.quad(
        integrand, np.log(r), np.log(1e6), limit=500, epsrel=1e-11)[0]


def test_gauss_hermite_series_are_recovered():
    v = np.linspace(-2000, 2000, 4001)
    losvd = gauss_hermite_losvd(v, 3.0, 40.0, 200.0, (0.05, -0.04))
    fit = fit_gauss_hermite(v, losvd)
    assert fit.gamma == pytest.approx(3.0, rel=1e-6)
    assert fit.mean == pytest.approx(40.0, abs=1e-4)
    assert fit.sigma == pytest.approx(200.0, rel=1e-6)
    assert fit.h == pytest.approx((0.05, -0.04), abs=1e-6)
    # Its mass, and gamma h_m of its histogram in its own basis
    dv = v[1] - v[0]
    assert losvd.sum() * dv == pytest.approx(3.0 * mass_factor((0.05, -0.04)))
    edges = np.linspace(-1500, 1500, 301)
    cumulative = scipy.integrate.cumulative_trapezoid(losvd, v, initial=0)
    histogram = np.diff(np.interp(edges, v, cumulative))
    sums = histogram @ hermite_weights(edges, 40.0, 200.0, 4) / 3.0
    # (to ~ (bin / sigma)^2 / 12, as the contents of a bin are spread
    # uniformly over it)
    assert sums == pytest.approx([1, 0, 0, 0.05, -0.04], abs=3e-4)


@pytest.mark.parametrize('bh_mass', [BH_MASS, 0.0])
def test_mock_distribution_function_is_exact(bh_mass):
    # Its density and its velocity dispersion, from Eddington's formula,
    # against the Hernquist density and the isotropic Jeans equation
    galaxy = MockGalaxy(LIGHT, UPSILON, bh_mass)
    r = np.array([1e-3, 1e-2, 0.1, 1.0, 10.0])
    assert galaxy.density(r) == pytest.approx(LIGHT.density(r), rel=1e-7)
    jeans = np.sqrt([jeans_pressure(galaxy, x) / LIGHT.density(x) for x in r])
    assert galaxy.dispersion(r) == pytest.approx(jeans, rel=1e-7)


@pytest.mark.parametrize('bh_mass', [BH_MASS, 0.0])
def test_mock_losvds_project_the_light(bh_mass):
    # The zeroth and second moments of the LOSVDs: the surface density
    # (Hernquist 1990) and the projected Jeans dispersion, Sigma sigma_p^2
    # = 2 int_R^inf nu sigma^2 r dr / sqrt(r^2 - R^2)
    galaxy = MockGalaxy(LIGHT, UPSILON, bh_mass)
    R = np.array([1e-2, 0.2, 1.0, 5.0])
    v = np.arange(0, 3000.1, 2.0)
    # (trapezoids over v >= 0, doubled)
    weights = np.full(v.size, 4.0)
    weights[[0, -1]] = 2.0
    losvds = galaxy.losvd(R, v)
    surface = losvds @ weights
    assert surface == pytest.approx(LIGHT.surface_density(R), rel=1e-5)
    projected = [2 * scipy.integrate.quad(
        lambda t: jeans_pressure(galaxy, x * np.cosh(t)) * x * np.cosh(t),
        0, np.arccosh(1e6 / x), limit=500, epsrel=1e-9)[0] for x in R]
    assert (losvds * v ** 2) @ weights == pytest.approx(
        projected, rel=1e-5)


def test_integration_conserves_energy_and_angular_momentum():
    model, params = LIGHT.mass_model(UPSILON, BH_MASS)
    potential = SphericalPotential(model, params, CONTEXT)
    x, v, _ = initial_conditions(potential, LIGHT, 300, (1e-3, 15.0), 0)
    energy = potential.potential(np.linalg.norm(x, axis=1)) \
        + np.sum(v ** 2, axis=1) / 2
    periods = 2 * np.pi / potential.circular_frequency(
        potential.circular_radius(energy))
    final_x, final_v = integrate(potential, x, v, 20 * periods, 0.02)
    final_energy = potential.potential(np.linalg.norm(final_x, axis=1)) \
        + np.sum(final_v ** 2, axis=1) / 2
    errors = np.abs(final_energy / energy - 1)
    assert np.median(errors) < 1e-4
    assert errors.max() < 1e-2
    np.testing.assert_allclose(
        np.cross(final_x, final_v), np.cross(x, v), rtol=0, atol=1e-10)


def test_regularised_nnls_is_that_of_the_stacked_system():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(30, 120)) * rng.random((30, 120))
    target = matrix @ rng.random(120) + rng.normal(size=30)
    penalties = rng.uniform(0.05, 0.5, 120)
    w = regularised_nnls(matrix, target, penalties)
    stacked = np.vstack((matrix, np.diag(np.sqrt(penalties))))
    expected, _ = scipy.optimize.nnls(
        stacked, np.concatenate((target, np.zeros(120))))
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-10)


@pytest.mark.slow
def test_schwarzschild_models_recover_the_mock():
    # A small library of the true potential and one without the black
    # hole, fitted to the mock at mass-to-light ratios scaled by 0.8, 1
    # and 1.25: the best is the truth (on the noise-free mock all fit:
    # the regularisation breaks the mass-anisotropy degeneracy)
    edges = [0, 0.2, 0.4, 0.6, 0.85, 1.15, 1.5, 2.0, 2.6, 3.4, 4.4, 5.7]
    field = BinnedField(polar_bins((120, 120), 0.1, edges, 4), 0.1, 0.5)
    data = add_noise(
        observe(MockGalaxy(LIGHT, UPSILON, BH_MASS), field, 16.0), seed=1)
    scale = data.kpc_per_arcsec()
    options = LibraryOptions(
        orbits=1500, radii=(0.02 * scale, 15.0), periods=30,
        shell_edges=tuple(np.concatenate((
            [0], np.geomspace(0.1 * scale, 15.0, 20)))))
    chi2 = {}
    for bh_mass in (BH_MASS, 0.0):
        model, params = LIGHT.mass_model(UPSILON, bh_mass)
        library = build_library(
            model, params, CONTEXT, LIGHT, field, scale, options=options)
        for factor in (0.8, 1.0, 1.25):
            chi2[bh_mass, factor] = fit_weights(
                library, data, LIGHT, scale=factor,
                regularisation=1000).chi2
    assert min(chi2, key=chi2.get) == (BH_MASS, 1.0)
