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
from gbkfit.objective._orbits import _bins_light
from gbkfit.observation import (
    LikelihoodGaussian, ObservationGroup, ObservationSky, PixelSpectra,
    RegionMoments, line_of_sight_parser)
from gbkfit.params import ParamSpace
from gbkfit.region import RegionsBins
from gbkfit.utils import gridutils
from gbkfit.utils.parseutils import ConfigError
from test_spheroids import hernquist_mge


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


def observation(regions, psf=1.0, oversampling=None, data=None):
    """
    pPXF kinematics of the stars in the bins (by default V 0 and sigma 150
    km/s, h3 and h4 0; or V, sigma, h3 and h4 by order), of the errors 5
    km/s and 0.03, through a Gaussian PSF of the given dispersion
    (arcsec).
    """
    n = regions.nregions()
    data = data or {1: np.zeros(n), 2: np.full(n, 150.0), 3: np.zeros(n),
                    4: np.zeros(n)}
    errors = {1: 5.0, 2: 5.0, 3: 0.03, 4: 0.03}
    dataset = DatasetRegionMoments({
        m: Data(np.asarray(data[m], float), error=np.full(n, errors[m]))
        for m in range(1, 5)}, regions)
    return ObservationSky(
        RegionMoments.from_data(
            dataset, method='gauss_hermite', spec_convention='log',
            spec_rest='5000 Angstrom'),
        line_of_sight_parser.load(dict(type='grid', depth=dict(step=2))),
        instrument=Instrument(psf=None if psf is None else PSFGauss(psf)),
        lines=['stars'], data=dataset, dtype='float64',
        oversampling=oversampling,
        likelihood=LikelihoodGaussian(missing_residual=3))


def sky_grid(obs):
    """The sky grid of the bins of an observation (see SkyGrid)."""
    return obs.observable().sky_grid(obs.instrument(), obs.oversampling())


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


def test_orbits_are_seen_without_what_their_recording_ignores(driver):
    # (an LSF here; and a varying PSF, a primary beam, a lens, dust:
    # refused when the observation is planned, not ignored)
    from gbkfit.instrument import LSFGauss
    seen = observation(bins())
    with_lsf = ObservationSky(
        seen.observable(), seen.line_of_sight(),
        instrument=Instrument(psf=PSFGauss(1.0), lsf=LSFGauss(20.0)),
        lines=['stars'], data=seen.data(), dtype='float64',
        likelihood=seen.likelihood())
    with pytest.raises(ConfigError, match="without an LSF"):
        ObservationGroup([galaxy()], [with_lsf]).plan(driver)


def test_an_image_sees_the_light_of_stars_on_orbits(driver):
    # (as that of any component: it is no observation of their orbits)
    from gbkfit.dataset import DatasetPixelBrightness
    from gbkfit.observation import PixelBrightness
    dataset = DatasetPixelBrightness(
        Data(np.ones((16, 16)), error=np.ones((16, 16))), step=(2, 2))
    image = ObservationSky(
        PixelBrightness.from_data(dataset),
        line_of_sight_parser.load(dict(type='flat')), data=dataset)
    group = ObservationGroup([galaxy()], [observation(bins()), image])
    params = ParamSpace(group.pdescs(), PROPERTIES | dict(
        stars_weights=dict(solve=SOLVE['solve'])))
    objective = Objective(group.plan(driver), params.solves())
    assert np.isfinite(objective.chi2(params.evaluate()))


def test_bins_of_stars_on_orbits_have_a_positive_sigma(driver):
    # (each series is about the sigma of its bin: 0 gave NaN models)
    n = bins().nregions()
    data = {1: np.zeros(n), 2: np.full(n, 150.0), 3: np.zeros(n),
            4: np.zeros(n)}
    data[2][3] = 0
    group = ObservationGroup([galaxy()], [observation(bins(), data=data)])
    params = ParamSpace(group.pdescs(), PROPERTIES | dict(
        stars_weights=dict(solve=SOLVE['solve'])))
    with pytest.raises(ConfigError, match="have a positive sigma"):
        Objective(group.plan(driver), params.solves())


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
    # Without a PSF, the integral of its surface density over their pixels
    # (here the mean over 64 x 64 points of each pixel, times its area, as
    # the orbits record it; not its value at the centres, 15% off in the
    # central pixels); with one, the light that it moves between the bins
    # (with the light of the grid as a whole kept within 1%)
    regions = bins()
    spheroid = galaxy().components()[0]
    values = {name[len('stars_'):] if name.startswith('stars_') else name:
              value for name, value in PROPERTIES.items()}
    centre = np.array([PROPERTIES['xpos'], PROPERTIES['ypos']])
    x, y = gridutils.sky_positions(regions.grid())
    index = regions.index()
    inside = index >= 0
    offsets = 2 * ((np.arange(64) + 0.5) / 64 - 0.5)
    dx, dy = (d.ravel() for d in np.meshgrid(offsets, offsets))
    density = spheroid.surface_density(
        values, x[inside][:, None] + dx - centre[0],
        y[inside][:, None] + dy - centre[1])
    direct = np.bincount(index[inside], density.mean(axis=1) * 4)
    exact = _bins_light(
        spheroid, values, sky_grid(observation(regions, None)), centre)
    np.testing.assert_allclose(exact, direct, rtol=1e-4)
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


def test_the_outputs_show_the_orbits_in_pixels_and_cells(driver):
    # The extra outputs of the orbits at the weights of their prior (1):
    # the cube of the pixels summed over the pixels of each bin is the
    # spectrum of the bin (of a symmetric PSF), so its moments are the
    # model of the bins; the cells hold most of the light of the orbits;
    # the tables have a row per cell, shell and orbit (and its twin, of
    # the opposite angular momentum)
    from gbkfit.model import Extras
    from gbkfit.observation.observables import _moments
    obs = observation(bins())
    moments = obs.observable()
    extras = Extras()
    model = ObservationGroup([galaxy()], [obs]).plan(driver).model_h(
        PROPERTIES | dict(stars_weights=np.ones(400)), extras)[0]
    cube = extras.get('cube').data
    index = moments.regions().index()
    spectra = np.stack(
        [cube[:, index == b].sum(axis=1) for b in range(index.max() + 1)],
        axis=1)
    plan = _moments.MomentsPlan(
        driver, (spectra.shape[1], 1), moments.orders(),
        moments.flux_floor(), moments.method(), np.float64)
    step = moments.spec_step()
    zero = moments.spec_rval() - (moments.spec_size() / 2 - 0.5) * step
    measured = plan.evaluate(
        (1.0, 1.0, step), (0.0, 0.0, zero),
        driver.mem_copy_h2d(np.ascontiguousarray(spectra[:, None, :])), None)
    for key, value in measured.items():
        np.testing.assert_allclose(
            driver.mem_copy_d2h(value['d'])[0], model[key]['d'],
            rtol=1e-6, atol=1e-6)
    assert extras.get('unbinned_moment2').data.shape == index.shape
    cells = extras.get('orbit_cells', component='stars')
    profile = extras.get('orbit_profile', component='stars')
    assert len(cells) == len(profile) * 5
    total = cells['light_spheroid'].sum()
    assert 0.9 * total < cells['light'].sum() <= total * (1 + 1e-9)
    assert np.all(np.isfinite(profile['beta'][profile['light'] > 0]))
    orbits = extras.get('orbits', component='stars')
    assert len(orbits) == 400
    np.testing.assert_allclose(
        orbits['L_z'][1::2], -orbits['L_z'][::2], rtol=1e-12)
    assert np.all(np.abs(orbits['circularity']) < 1.05)


def test_mock_data_of_orbits_are_the_model_plus_noise(driver):
    # Mock moments of the bins of the noise of the cube of their pixels, as
    # for other stars (an rms of the peak: errors of 8 km/s in V, 0.04 in
    # h3): their errors (of 30 draws) hold the model, about one error from
    # it
    from gbkfit.model import Extras
    from gbkfit.observation import noise_parser
    plan = ObservationGroup([galaxy()], [observation(bins())]).plan(driver)
    values = PROPERTIES | dict(stars_weights=np.ones(400))
    extras = Extras()
    model = plan.model_h(values, extras)[0]
    rms = extras.get('cube').data.max()
    dataset, = plan.mock(
        values, [noise_parser.load(dict(type='gauss', rms=rms))],
        nsamples=30, seed=1)[0]
    z = np.concatenate([
        (dataset[key].data() - model[key]['d']) / dataset[key].error()
        for key in model])
    assert np.all(np.isfinite(z))
    assert 0.5 < np.mean(z ** 2) < 2


def test_the_model_of_the_objective_is_the_series_of_the_data(driver):
    # Data made by the orbits, of the weights of their prior (1): the
    # moments of the spectra of the weighted orbits, as the model of the
    # bins without the objective (e.g. of outputs) gives them. With the
    # weights there (a prior of the factors 1 of a tiny sigma), the model
    # of the objective, of the series of the data and linear in the
    # weights, is the data, but for the h1 and h2 that the data have in
    # their own series (Rix et al. 1997 take them as 0; a fit of V, sigma,
    # h3 and h4 together leaves terms of the order of h4 times the higher
    # orders): within 2% in sigma (where h4 is -0.09) and 0.01 in h4
    group = ObservationGroup([galaxy()], [observation(bins())])
    model = group.plan(driver).model_h(
        PROPERTIES | dict(stars_weights=np.ones(400)))[0]
    data = {m: model[f'moment{m}']['d'] for m in range(1, 5)}
    obs = observation(bins(), data=data)
    group = ObservationGroup([galaxy()], [obs])
    params = ParamSpace(group.pdescs(), PROPERTIES | dict(
        stars_weights=dict(solve=dict(
            type='gaussian', mean=1, sigma=1e-6, minimum=0,
            likelihood='profiled'))))
    objective = Objective(group.plan(driver), params.solves())
    objective.chi2(params.evaluate())
    np.testing.assert_allclose(
        list(objective.solved().values()), 1, rtol=1e-6)
    residuals = objective.residual_nddata_h(params.evaluate(), False)[0]
    errors = {key: obs.data()[key].error() for key in residuals}
    assert np.abs(data[4]).max() > 0.08
    np.testing.assert_allclose(
        residuals['moment1'] * errors['moment1'], 0, atol=1e-6)
    assert np.all(
        np.abs(residuals['moment2'] * errors['moment2']) < 0.02 * data[2])
    np.testing.assert_allclose(
        residuals['moment3'] * errors['moment3'], 0, atol=1e-6)
    np.testing.assert_allclose(
        residuals['moment4'] * errors['moment4'], 0, atol=0.01)


@pytest.mark.slow
def test_schwarzschild_models_recover_the_exact_mock(driver):
    # The exact mock of the proof of concept (model.orbits: a Hernquist
    # sphere of a mass-to-light ratio of 4 and a black hole of 5e8 Msun,
    # isotropic, of Eddington's distribution function) at 16 Mpc, seen in
    # polar bins through a PSF of 0.5 arcsec, with noise; fitted with its
    # MGE in its Hernquist mass at mass-to-light ratios scaled by 0.8, 1
    # and 1.25, with and without the black hole: the chi-square of the
    # data is the least at the truth (as in the proof of concept)
    from gbkfit.model.orbits.kinematics import BinnedField, polar_bins
    from gbkfit.model.orbits.mocks import MockGalaxy, add_noise, observe
    from gbkfit.model.orbits.profiles import Hernquist
    light, upsilon, bh, distance = Hernquist(2.5e10, 1.5), 4.0, 5e8, 16.0
    edges = [0, 0.2, 0.4, 0.6, 0.85, 1.15, 1.5, 2.0, 2.6, 3.4, 4.4, 5.7]
    field = BinnedField(polar_bins((120, 120), 0.1, edges, 4), 0.1, 0.5)
    mock = add_noise(
        observe(MockGalaxy(light, upsilon, bh), field, distance), seed=1)
    regions = RegionsBins(field.bin_map, step=(0.1, 0.1), rpix=(59.5, 59.5))
    dataset = DatasetRegionMoments({
        1: Data(mock.mean, error=mock.mean_error),
        2: Data(mock.sigma, error=mock.sigma_error),
        3: Data(mock.h[:, 0], error=mock.h_error[:, 0]),
        4: Data(mock.h[:, 1], error=mock.h_error[:, 1])}, regions)
    obs = ObservationSky(
        RegionMoments.from_data(
            dataset, method='gauss_hermite', spec_convention='log',
            spec_rest='5000 Angstrom'),
        line_of_sight_parser.load(dict(type='grid', depth=dict(step=2))),
        instrument=Instrument(psf=PSFGauss(0.5 / np.sqrt(8 * np.log(2)))),
        lines=['stars'], data=dataset,
        likelihood=LikelihoodGaussian(missing_residual=3))
    a = light.a / mock.kpc_per_arcsec()
    sigma, i0 = hernquist_mge(a)
    spheroid = SPHEROID | dict(
        rmax=30 * a, profile=dict(type='mge', ngauss=len(i0)),
        kinematics=dict(type='schwarzschild', norbits=1500, turns=30))
    group = ObservationGroup([galaxy(components=[spheroid])], [obs])
    values = dict(
        xpos=0, ypos=0, posa=0, incl=90, vsys=0, distance=distance,
        stars_i0=list(i0), stars_sigma=list(sigma),
        stars_q=[1.0] * len(i0), mass_light_rs=light.a,
        mass_light_alpha=1.0, mass_light_beta=4.0, mass_light_gamma=1.0,
        stars_weights=SOLVE)
    log_rho = np.log10(upsilon * light.total / (2 * np.pi * light.a ** 3))
    objective = None
    chi2 = {}
    for bh_mass in (bh, 1.0):
        for factor in (0.8, 1.0, 1.25):
            params = ParamSpace(group.pdescs(), values | dict(
                mass_light_log_rho=log_rho + np.log10(factor),
                mass_bh_log_m=np.log10(bh_mass)))
            if objective is None:
                objective = Objective(group.plan(driver), params.solves())
            residuals = objective.residual_vector_h(params.evaluate(), True)
            chi2[bh_mass, factor] = float(np.sum(residuals[0] ** 2))
    assert min(chi2, key=chi2.get) == (bh, 1.0)
    # (about one per residual)
    assert chi2[bh, 1.0] < 1.2 * objective.nresiduals()


@pytest.mark.slow
def test_schwarzschild_models_of_jeans_data_find_their_mass(driver):
    # The Gauss-Hermite series of the bins of a flattened (q = 0.7)
    # spheroid, inclined (60 degrees) and rotating (kappa 0.6), of Jeans
    # kinematics in the mass of its light and a black hole, through a
    # PSF: fitted by Schwarzschild models in that mass scaled by 0.8, 1 and
    # 1.25 (a weak prior of the weights), the chi-square of the data is
    # the least at the truth, and their rotation is that of the data.
    # (Their sigma and h4 are not quite: a Gaussian at each point, as the
    # Jeans kinematics take, is not the distribution of a steady state of
    # orbits; 6 to 20 km/s in sigma, of 60 to 80)
    pixels = (np.arange(40) - 19.5) * 0.5
    x, y = np.meshgrid(pixels, pixels)
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    edges = [0, 1, 2, 3, 4.5, 6.5, 9.5]
    ring = np.digitize(r, edges) - 1
    sector = ((theta + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    index = np.where(
        r < edges[-1], np.where(ring == 0, 0, 1 + (ring - 1) * 8 + sector),
        -1)
    regions = RegionsBins(index, step=(0.5, 0.5), rpix=(19.5, 19.5))

    def stars(kinematics):
        return galaxy(
            components=[SPHEROID | dict(rmax=40, kinematics=kinematics)],
            mass=[dict(type='light_spheroid', name='stars',
                       component='stars'),
                  dict(type='point', name='bh')])
    truth = dict(
        xpos=0, ypos=0, posa=0, incl=60, vsys=0, distance=16,
        stars_i0=[50.0, 10.0], stars_sigma=[1.0, 4.0], stars_q=[0.7, 0.7],
        mass_stars_upsilon=1e8, mass_bh_log_m=8.5)
    los = line_of_sight_parser.load(dict(type='grid', depth=dict(step=0.5)))
    instrument = Instrument(psf=PSFGauss(0.4))
    moments = RegionMoments(
        regions, orders=[1, 2, 3, 4], method='gauss_hermite', spec_size=241,
        spec_step=10, spec_rval=0, spec_rest='5000 Angstrom',
        spec_convention='log')
    model = ObservationGroup(
        [stars(dict(type='jeans'))],
        [ObservationSky(moments, los, instrument=instrument, lines=['stars'],
                        dtype='float64')]).plan(driver).model_h(
        truth | dict(stars_beta_z=0.0, stars_kappa=0.6))[0]
    errors = {1: 3.0, 2: 3.0, 3: 0.02, 4: 0.02}
    dataset = DatasetRegionMoments({
        m: Data(model[f'moment{m}']['d'],
                error=np.full(regions.nregions(), errors[m]))
        for m in range(1, 5)}, regions)
    obs = ObservationSky(
        RegionMoments.from_data(
            dataset, method='gauss_hermite', spec_convention='log',
            spec_rest='5000 Angstrom'),
        los, instrument=instrument, lines=['stars'], data=dataset,
        likelihood=LikelihoodGaussian(missing_residual=3))
    group = ObservationGroup([stars(dict(
        type='schwarzschild', norbits=1500, turns=30))], [obs])
    solve = dict(solve=dict(
        type='gaussian', mean=1, sigma=10, minimum=0, likelihood='profiled'))
    objective = None
    chi2 = {}
    for factor in (0.8, 1.0, 1.25):
        params = ParamSpace(group.pdescs(), truth | dict(
            mass_stars_upsilon=1e8 * factor, stars_weights=solve))
        if objective is None:
            objective = Objective(group.plan(driver), params.solves())
        residuals = objective.residual_nddata_h(params.evaluate(), True)[0]
        chi2[factor] = sum(float(np.sum(r ** 2)) for r in residuals.values())
        if factor == 1.0:
            assert np.sum(residuals['moment1'] ** 2) < regions.nregions()
    assert min(chi2, key=chi2.get) == 1.0
