"""
Tests of the Schwarzschild kinematics of spheroids (its configuration,
the gravity and the observations it takes) and of the objective that
models the bins of their stars from the orbits (objective._orbits): the
sky of the bins, the light of the spheroid in them, and the solved
weights.
"""

import copy

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetRegionMoments
from gbkfit.instrument import Instrument, PSFGauss
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.objective._orbits import _bins_light, sky_grid
from gbkfit.observation import (
    LikelihoodGaussian, ObservationGroup, ObservationSky, PixelSpectra,
    RegionMoments, line_of_sight_parser)
from gbkfit.params import ParamSpace
from gbkfit.region import RegionsBins
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError


# A spheroid of two Gaussians of light of Schwarzschild kinematics in the
# gravity of a Hernquist sphere (a Zhao profile) of 4e10 Msun and a scale
# radius of 1 kpc, and a black hole, at 10 Mpc (A: 1 kpc in arcsec)
A = 1 / (10 * 1e3 * np.pi / 648000)
SPHEROID = dict(
    type='spheroid', name='stars', rmax=30 * A,
    emission=dict(type='stars'), profile=dict(type='mge', ngauss=2),
    kinematics=dict(type='schwarzschild', norbits=200, turns=20))
GALAXY = dict(
    components=[SPHEROID],
    mass=[dict(type='spheroid', name='light',
               profile=dict(type='zhao', parameters='density_radius')),
          dict(type='point', name='bh')],
    distance=dict(type='direct', cosmology='Planck18'))
PROPERTIES = dict(
    xpos=0.5, ypos=-0.3, posa=30, incl=60, vsys=0, distance=10,
    stars_i0=[3.0, 1.0], stars_sigma=[0.3 * A, 1.5 * A], stars_q=[1.0, 1.0],
    mass_light_log_rho=np.log10(4e10 / (2 * np.pi)), mass_light_rs=1.0,
    mass_light_alpha=1.0, mass_light_beta=4.0, mass_light_gamma=1.0,
    mass_bh_log_m=np.log10(5e8))
SOLVE = dict(solve=dict(
    type='gaussian', mean=1, sigma=1, minimum=0, likelihood='profiled'))


def bins(rota=0.0):
    """12 bins of 2 arcsec pixels: the quadrants of 3 annuli."""
    pixels = (np.arange(32) - 15.5) * 2
    x, y = np.meshgrid(pixels, pixels)
    r = np.hypot(x, y)
    index = np.where(
        r < 30, (x > 0) + 2 * (y > 0) + 4 * (r > 8) + 4 * (r > 16), -1)
    return RegionsBins(index, step=(2, 2), rpix=(15.5, 15.5), rota=rota)


def observation(regions, psf=1.0, oversampling=None):
    """
    pPXF kinematics of the stars in the bins (V 0 and sigma 150 km/s, h3
    and h4 0), through a Gaussian PSF of the given dispersion (arcsec).
    """
    n = regions.nregions()
    dataset = DatasetRegionMoments({
        1: Data(np.zeros(n), error=np.full(n, 5.0)),
        2: Data(np.full(n, 150.0), error=np.full(n, 5.0)),
        3: Data(np.zeros(n), error=np.full(n, 0.03)),
        4: Data(np.zeros(n), error=np.full(n, 0.03))}, regions)
    return ObservationSky(
        RegionMoments.from_data(
            dataset, method='gauss_hermite', spec_convention='log',
            spec_rest='5000 Angstrom'),
        line_of_sight_parser.load(dict(type='grid', depth=dict(step=2))),
        instrument=Instrument(psf=None if psf is None else PSFGauss(psf)),
        lines=['stars'], data=dataset, dtype='float64',
        oversampling=oversampling,
        likelihood=LikelihoodGaussian(missing_residual=3))


def galaxy(**changes):
    """The galaxy of the spheroid, with changes to its configuration."""
    return galaxy_parser.load(copy.deepcopy(GALAXY) | changes)


def test_a_schwarzschild_spheroid_round_trips():
    loaded = galaxy()
    assert galaxy_parser.dump(loaded)['components'][0]['kinematics'] \
        == dict(type='schwarzschild', norbits=200, turns=20.0, seed=0)
    # (an element for each orbit and its twin)
    assert loaded.pdescs()['stars_weights'].size() == 400
    assert loaded.components()[0].orbit_weights() == 'weights'


@pytest.mark.parametrize('options, message', [
    (dict(norbits=0), "norbits must be a positive integer"),
    (dict(turns=-1), "turns must be a positive number"),
    (dict(seed=-1), "seed must be a non-negative integer"),
])
def test_invalid_schwarzschild_kinematics_are_refused(options, message):
    spheroid = SPHEROID | dict(kinematics=dict(
        type='schwarzschild') | options)
    with pytest.raises(ConfigError, match=message):
        galaxy(components=[spheroid])


def test_orbits_take_the_forces_of_a_potential():
    # (MOND here is the Newtonian forces times its function: it has curl)
    with pytest.raises(ConfigError, match="gradient of a potential"):
        galaxy(gravity=dict(type='mond', interpolation='simple'))


def test_spectral_cubes_do_not_see_stars_on_orbits(driver):
    spectra = ObservationSky(
        PixelSpectra(size=(8, 8, 16), step=(2, 2, 20)),
        line_of_sight_parser.load(dict(type='grid', depth=dict(step=2))),
        lines=['stars'])
    with pytest.raises(ConfigError, match="spectral cubes do not see them"):
        ObservationGroup([galaxy()], [spectra]).plan(driver)


def test_the_weights_of_the_orbits_are_solved(driver):
    group = ObservationGroup([galaxy()], [observation(bins())])
    params = ParamSpace(group.pdescs(), PROPERTIES | dict(
        stars_weights=[1.0] * 400))
    with pytest.raises(ConfigError, match="give them a solve"):
        Objective(group.plan(driver), params.solves())


@pytest.mark.parametrize('rota, oversampling', [(0, None), (30, (2, 2))])
def test_the_sky_grid_of_bins_holds_their_pixels(rota, oversampling):
    # Without a PSF, the share of its own bin in the light of the centre of
    # each pixel is 1, on a grid rotated and oversampled
    regions = bins(rota)
    grid = sky_grid(observation(regions, None, oversampling))
    x, y = gridutils.sky_positions(regions.grid())
    cx, cy = grid.cells @ np.stack((x.ravel(), y.ravel(), np.ones(x.size)))
    cells = np.floor(cy).astype(int) * grid.shape[1] \
        + np.floor(cx).astype(int)
    index = regions.index().ravel()
    inside = index >= 0
    shares = grid.shares[cells[inside]].toarray()
    np.testing.assert_array_equal(
        shares[np.arange(inside.sum()), index[inside]], 1)
    np.testing.assert_array_equal(shares.sum(axis=1), 1)


def test_the_light_of_bins_is_that_of_the_spheroid():
    # Without a PSF, the surface density at the centres of their pixels
    # times their area; with one, the light that it moves between the
    # bins (with the light of the grid as a whole kept within 1%)
    regions = bins()
    spheroid = galaxy().components()[0]
    values = {name[len('stars_'):] if name.startswith('stars_') else name:
              value for name, value in PROPERTIES.items()}
    centre = np.array([PROPERTIES['xpos'], PROPERTIES['ypos']])
    x, y = gridutils.sky_positions(regions.grid())
    index = regions.index()
    inside = index >= 0
    direct = np.bincount(index[inside], spheroid.surface_density(
        values, x[inside] - centre[0], y[inside] - centre[1]) * 4)
    exact = _bins_light(
        spheroid, values, sky_grid(observation(regions, None)), centre)
    np.testing.assert_allclose(exact, direct, rtol=1e-12)
    blurred = _bins_light(
        spheroid, values, sky_grid(observation(regions, 1.0)), centre)
    assert blurred.sum() == pytest.approx(direct.sum(), rel=0.01)
    assert not np.allclose(blurred, direct, rtol=1e-3)


def test_an_objective_models_the_bins_of_stars_on_orbits(driver):
    # The models of the bins of the solved weights, which are non-negative,
    # and the chi-square: the squared residuals of the bins, and those of
    # the constraints of the light and of the prior of the weights; the
    # data and the mirror symmetry of the orbits (each with its twin) give
    # V = h3 = 0; and the effective number of the weights is at most the
    # number of the residuals of the bins
    group = ObservationGroup([galaxy()], [observation(bins())])
    params = ParamSpace(group.pdescs(), PROPERTIES | dict(
        stars_weights=SOLVE))
    objective = Objective(group.plan(driver), params.solves())
    chi2 = objective.chi2(params.evaluate())
    residuals = objective.residuals(params.evaluate())
    assert chi2 == pytest.approx(np.sum(residuals ** 2))
    bins_residuals = objective.residual_vector_h(
        params.evaluate(), True)[0].reshape(4, -1)
    np.testing.assert_allclose(bins_residuals[[0, 2]], 0, atol=1e-6)
    assert np.all(np.isfinite(bins_residuals))
    weights = np.array(list(objective.solved().values()))
    assert len(weights) == 400 and np.all(weights >= 0)
    assert 0 < objective.nnuisance() <= objective.nresiduals()
