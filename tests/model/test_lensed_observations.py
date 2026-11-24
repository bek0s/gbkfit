"""
Observations of galaxies behind gravitational lenses, end to end: an
Einstein ring against its analytic image, the frames of joint
observations, the light from behind a lens galaxy dimmed by its dust,
every kind of observation of the sky, and the copies of objectives of
workers.
"""

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelBrightness, UVCoverage
from gbkfit.fitting import FitProblem
from gbkfit.lensing import Lens, LensSIE, SourceGrid
from gbkfit.model import galaxy_parser
from gbkfit.objective import Likelihoods, Objective
from gbkfit.observation import (
    ObservationGroup, ObservationPixelBrightness, observation_parser)
from gbkfit.params import ParamSpace
from gbkfit.parallel import WorkersThreads

# The Einstein radius of the lens, and the dispersion of the source
# (arcsec): a Gaussian source on the axis of a singular isothermal sphere,
# whose image is a ring
THETA_E = 0.5
SIGMA = 0.15

# The image: 101 pixels of 0.02 arcsec, one at the centre
SIZE, STEP = 101, 0.02

# A lens at z = 0.5, of the Einstein radius of a source at z = 2
LENS_REDSHIFT, SOURCE_REDSHIFT = 0.5, 2.0


def gauss_disk(name, **options):
    """A face-on thin disk of a Gaussian brightness, in a band."""
    return dict(
        type='thin_disk', name=name, loose=False, tilted=False,
        rnodes=[0, 1, 2], rptraits=dict(type='gauss'),
        emission=dict(type='band')) | options


def sis(source_redshift=SOURCE_REDSHIFT):
    """A singular isothermal sphere (see THETA_E)."""
    return Lens('lens', LENS_REDSHIFT, LensSIE(), source_redshift)


def disk_params(name, **values):
    """The parameters of a gauss_disk at the centre."""
    return {f'{name}_{key}': value for key, value in (dict(
        xpos=0, ypos=0, posa=0, incl=0, rpt_a=1, rpt_s=SIGMA)
        | values).items()}


LENS_PARAMS = dict(
    lens_sie_theta_e=THETA_E, lens_sie_q=1, lens_sie_posa=0, lens_sie_xpos=0,
    lens_sie_ypos=0)


def evaluate(group, properties, driver):
    """
    Return the model images of the observations of a group, planned at
    the values of the parameters (all fixed).
    """
    params = ParamSpace.from_properties(group.pdescs(), properties)
    values = params.evaluate()
    plan = group.plan(driver, reference=values)
    return [model['brightness'].data
            for model in plan.model_h(values)]


def radius() -> np.ndarray:
    """Return the distance of each pixel of the image from its centre."""
    x = (np.arange(SIZE) - SIZE // 2) * STEP
    return np.hypot(*np.meshgrid(x, x))


def test_an_einstein_ring_is_the_analytic_image(driver):
    # Behind a singular isothermal sphere, the light at an angle theta
    # from its centre is that of the source at | |theta| - theta_e |
    source = galaxy_parser.load(dict(
        name='arc', redshift=SOURCE_REDSHIFT,
        components=[gauss_disk('disk')]))
    group = ObservationGroup(
        source, ObservationPixelBrightness(
            size=(SIZE, SIZE), step=STEP, dtype='float64'), [sis()])
    [image] = evaluate(group, disk_params('arc_disk') | LENS_PARAMS, driver)
    r = radius()
    ring = np.exp(-(r - THETA_E) ** 2 / (2 * SIGMA ** 2))
    # (not at the singular centre of the lens)
    beyond = r > STEP
    np.testing.assert_allclose(image[beyond], ring[beyond], atol=2e-4)


def test_the_strengths_of_a_lens_are_scaled_for_each_source(driver):
    # Quoted for a source further than the galaxy, the Einstein radius of
    # the light of the galaxy is theta_e D_ls / D_s, over that of the
    # source of the quote
    source = galaxy_parser.load(dict(
        name='arc', redshift=SOURCE_REDSHIFT,
        components=[gauss_disk('disk')]))
    observation = ObservationPixelBrightness(
        size=(SIZE, SIZE), step=STEP, dtype='float64')
    lens = sis(source_redshift=4.0)
    group = ObservationGroup(source, observation, [lens])
    [image] = evaluate(group, disk_params('arc_disk') | LENS_PARAMS, driver)
    cosmology = group.cosmology()

    def factor(z_s):
        return cosmology.angular_diameter_distance_between(
            LENS_REDSHIFT, z_s) / cosmology.angular_diameter_distance(z_s)
    theta_e = THETA_E * factor(SOURCE_REDSHIFT) / factor(4.0)
    r = radius()
    ring = np.exp(-(r - theta_e) ** 2 / (2 * SIGMA ** 2))
    beyond = r > STEP
    np.testing.assert_allclose(image[beyond], ring[beyond], atol=2e-4)


def test_the_offsets_of_an_observation_move_its_lenses_too(driver):
    # A second image of the arc whose frame is off by its offsets (10
    # pixels) sees the lens and the arc moved by them, together: the first
    # image moved (the grids of the source the same about it)
    source = galaxy_parser.load(dict(
        name='arc', redshift=SOURCE_REDSHIFT,
        components=[gauss_disk('disk')]))
    grid = SourceGrid(step=0.01, extent=(-1, 1, -1, 1))
    images = [
        ObservationPixelBrightness(
            size=(SIZE, SIZE), step=STEP, dtype='float64', name='first',
            source_grid=grid),
        ObservationPixelBrightness(
            size=(SIZE, SIZE), step=STEP, dtype='float64', name='second',
            source_grid=grid, offsets=dict(position=True))]
    group = ObservationGroup(source, images, [sis()])
    first, second = evaluate(
        group, disk_params('arc_disk') | LENS_PARAMS
        | dict(second_dx=10 * STEP, second_dy=0), driver)
    assert np.max(np.abs(first)) > 0.5
    # (but at the singular centre of the lens)
    beyond = radius()[:, :-10] > STEP
    np.testing.assert_allclose(
        second[:, 10:][beyond], first[:, :-10][beyond], atol=1e-12)


def test_the_dust_of_a_lens_galaxy_dims_the_light_behind_it(driver):
    # The light of the arc behind the lens is dimmed by the whole column
    # of the dust of the galaxy of the lens, at its redshift, as that of a
    # galaxy behind it in its plane is (without the lens): the same share
    # of it at each pixel, whatever the light
    dust = dict(
        type='thick_disk', name='dust', loose=False, tilted=False,
        rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
        rhtraits=dict(type='sech2', zmax=4),
        opacity=dict(type='dust', law=dict(type='ccm')))
    deflector = galaxy_parser.load(dict(
        name='deflector', redshift=LENS_REDSHIFT, components=[dust]))
    params = {
        f'deflector_dust_{key}': value for key, value in dict(
            xpos=0.1, ypos=0, posa=0, incl=0, rpt_a=0.5, rpt_s=0.4,
            rht_s=0.05).items()}
    shares = []
    for lenses, arc_redshift, order, sigma in (
            ([sis()], SOURCE_REDSHIFT, None, SIGMA),
            ([], LENS_REDSHIFT, ['deflector', 'arc'], 1.0)):
        arc = galaxy_parser.load(dict(
            name='arc', redshift=arc_redshift,
            components=[gauss_disk('disk')]))
        images = [
            ObservationPixelBrightness(
                size=(SIZE, SIZE), step=STEP, dtype='float64', band='sdss_r',
                galaxies=galaxies, name=name)
            for name, galaxies in (
                ('dimmed', ['deflector', 'arc']), ('clear', ['arc']))]
        group = ObservationGroup(
            [deflector, arc], images, lenses, depth_order=order)
        dimmed, clear = evaluate(
            group, params | disk_params('arc_disk', rpt_s=sigma)
            | (LENS_PARAMS if lenses else {}), driver)
        lit = clear > 1e-3
        shares.append(np.where(lit, dimmed / np.where(lit, clear, 1), np.nan))
    lensed, plain = shares
    assert np.nanmin(lensed) < 0.8
    both = np.isfinite(lensed) & np.isfinite(plain)
    np.testing.assert_allclose(lensed[both], plain[both], rtol=1e-6)


@pytest.mark.parametrize('extent', [(-1, 1, -1, 1), 'auto'])
def test_the_grid_of_the_source_is_laid_out_once(driver, extent):
    # A lens of free parameters bends the rays at each evaluation, onto the
    # grid laid out at the reference values; a larger Einstein radius is a
    # larger ring
    source = galaxy_parser.load(dict(
        name='arc', redshift=SOURCE_REDSHIFT,
        components=[gauss_disk('disk')]))
    group = ObservationGroup(
        source, ObservationPixelBrightness(
            size=(SIZE, SIZE), step=STEP, dtype='float64',
            source_grid=SourceGrid(extent=extent)), [sis()])
    params = ParamSpace.from_properties(
        group.pdescs(), disk_params('arc_disk') | LENS_PARAMS
        | dict(lens_sie_theta_e=dict(value=THETA_E, min=0.2, max=0.8)))
    plan = group.plan(driver, reference=params.evaluate())
    r = radius()
    for theta_e in (0.45, 0.55):
        image = plan.model_h(params.evaluate([theta_e]))[0]['brightness']
        ring = np.exp(-(r - theta_e) ** 2 / (2 * SIGMA ** 2))
        beyond = r > STEP
        np.testing.assert_allclose(
            image.data[beyond], ring[beyond], atol=1e-3)


def test_a_point_behind_a_lens_is_refused(driver):
    # (its flux would fall in one pixel of the source grid, which the rays
    # of the image sample or miss at random)
    source = galaxy_parser.load(dict(
        name='arc', redshift=SOURCE_REDSHIFT, components=[
            gauss_disk('disk'),
            dict(type='point', name='nucleus', emission=dict(type='band'))]))
    group = ObservationGroup(
        source, ObservationPixelBrightness(
            size=(SIZE, SIZE), step=STEP, dtype='float64'), [sis()])
    properties = disk_params('arc_disk') | LENS_PARAMS | dict(
        arc_nucleus_xpos=0, arc_nucleus_ypos=0, arc_nucleus_flux=1)
    with pytest.raises(ValueError, match="a point behind a gravitational"):
        evaluate(group, properties, driver)


# The speed of light (km/s)
C_KMS = 299792.458

# A disk at z = 2 that emits a line, at its centre on the sky, and how
# it is seen by kinds of observations of cubes of the sky, of their
# spectral axes at its redshift (see lensed_kind)
LINE_DISK = dict(
    type='thin_disk', loose=False, tilted=False, rnodes=list(range(13)),
    rptraits=dict(type='exponential'),
    vptraits=dict(type='arctan', direction='tan'),
    dptraits=dict(type='uniform'), emission=dict(type='lines'))
LINE_DISK_PARAMS = dict(
    xpos=0.3, ypos=-0.4, posa=40, incl=55, vsys=10, rpt_a=1, rpt_s=3,
    vpt_rt=2, vpt_vt=150, dpt_a=25)
LINE_LENS_PARAMS = dict(
    lens_sie_theta_e=2, lens_sie_q=0.8, lens_sie_posa=20, lens_sie_xpos=0,
    lens_sie_ypos=0)


def lensed_kind(kind):
    """
    Return an observation of a kind of the disk of LINE_DISK at z = 2:
    visibilities of HI (60 random baselines, 16 channels of 20 km/s), or
    slitless spectra of a line at 4 um at rest.
    """
    z = SOURCE_REDSHIFT
    if kind == 'visibilities':
        hi = 1420.405752e6 / (1 + z)
        velocities = 140 - 20 * np.arange(16)
        uvw = np.random.default_rng(1).uniform(-1, 1, (60, 3)) \
            * [15e3, 15e3, 5e3]
        coverage = UVCoverage(
            uvw, hi * (1 - velocities / C_KMS), (150.001, -30.0005), 'LSRK',
            f'{hi} Hz')
        return observation_parser.load(dict(
            type='visibilities', fov=[32, 28], step=1, rval=[150, -30],
            instrument=dict(lsf=dict(type='hanning')), coverage=coverage))
    wave = f'{4 * (1 + z)} um'
    return observation_parser.load(dict(
        type='slitless_spectra', size=[32, 24], step=[1, 1],
        spectral=dict(size=21, step=30, rest=wave),
        trace=dict(type='polynomial', wave=wave,
                   dx=[0, C_KMS / (4 * (1 + z) * 30)]),
        instrument=dict(psf=dict(type='gauss', sigma=0.6))))


@pytest.mark.parametrize('kind', ['visibilities', 'slitless_spectra'])
def test_visibilities_and_slitless_spectra_see_lensed_light(driver, kind):
    # (their footprints are their whole fields; the visibilities check
    # their sampling with a plan at half their step, of its own rays, at
    # their first evaluation)
    source = galaxy_parser.load(dict(
        redshift=SOURCE_REDSHIFT, components=[LINE_DISK]))
    lens = Lens('lens', LENS_REDSHIFT, LensSIE(), SOURCE_REDSHIFT)
    totals = []
    for lenses, params in (
            ([lens], LINE_DISK_PARAMS | LINE_LENS_PARAMS),
            ([], LINE_DISK_PARAMS)):
        observation = lensed_kind(kind)
        group = ObservationGroup(source, observation, lenses)
        values = ParamSpace.from_properties(group.pdescs(), params).evaluate()
        models = group.plan(driver, reference=values).model_h(values)[0]
        assert set(models) == set(observation.keys())
        for item in models.values():
            assert np.isfinite(item.data).all()
        totals.append(sum(
            np.sum(np.abs(item.data)) for item in models.values()))
    lensed, plain = totals
    # (magnified)
    assert lensed > 1.2 * plain


def test_workers_evaluate_lensed_models_as_the_objective_does(driver):
    # Copies of the objective lay out the grids of the planes behind
    # lenses at the reference values of the parameters, as it does
    source = galaxy_parser.load(dict(
        name='arc', redshift=SOURCE_REDSHIFT,
        components=[gauss_disk('disk')]))
    truth = disk_params('arc_disk') | LENS_PARAMS
    [image] = evaluate(ObservationGroup(source, ObservationPixelBrightness(
        size=(SIZE, SIZE), step=STEP, dtype='float64'), [sis()]),
        truth, driver)
    observation = ObservationPixelBrightness.paired(
        DatasetPixelBrightness(
            Data(image, error=np.full(image.shape, 0.01)), step=STEP),
        dtype='float64')
    group = ObservationGroup(source, observation, [sis()])
    params = ParamSpace.from_properties(
        group.pdescs(), truth | dict(
            lens_sie_theta_e=dict(value=THETA_E, min=0.3, max=0.7),
            arc_disk_rpt_s=dict(value=SIGMA, min=0.1, max=0.2)))
    plan = group.plan(driver, reference=params.evaluate())
    objective = Objective(plan, Likelihoods(group))
    # (the free elements in their order: the dispersion, then the
    # Einstein radius; the truth in the middle)
    assert list(params.free_properties()) == [
        'arc_disk_rpt_s', 'lens_sie_theta_e']
    xs = [[0.14, 0.45], [SIGMA, THETA_E], [0.16, 0.55]]
    alone = FitProblem(objective, params).chi2_batch(xs)
    with WorkersThreads([driver, driver]).start(objective) as pool:
        pooled = FitProblem(objective, params, pool=pool).chi2_batch(xs)
    np.testing.assert_allclose(pooled, alone, rtol=1e-12)
    assert alone[1] < min(alone[0], alone[2])


def test_dust_in_front_of_lensed_light_beyond_its_law_is_refused(driver):
    # A cube at mm wavelengths that sees the dusty lens galaxy (e.g. that
    # of ALMA in a joint fit with MUSE) is told to leave it out: its law is
    # not extrapolated, and its dust and light are negligible there
    dust = dict(
        type='thick_disk', name='dust', loose=False, tilted=False,
        rnodes=[0, 1, 2], rptraits=dict(type='exponential'),
        rhtraits=dict(type='sech2', zmax=4),
        opacity=dict(type='dust', law=dict(type='ccm')))
    deflector = galaxy_parser.load(dict(
        name='deflector', redshift=LENS_REDSHIFT, components=[dust]))
    arc = galaxy_parser.load(dict(
        name='arc', redshift=SOURCE_REDSHIFT,
        components=[LINE_DISK | dict(name='gas')]))
    params = {f'arc_gas_{key}': value
              for key, value in LINE_DISK_PARAMS.items()} \
        | LENS_PARAMS | {
            f'deflector_dust_{key}': value for key, value in dict(
                xpos=0, ypos=0, posa=0, incl=30, rpt_a=1, rpt_s=1,
                rht_s=0.2).items()}
    for galaxies in (['deflector', 'arc'], ['arc']):
        # (CO 3-2 at z = 2)
        cube = observation_parser.load(dict(
            type='pixel_spectra', size=[24, 24], step=[0.5, 0.5],
            spectral=dict(size=20, step=30,
                          rest=f'{867 * (1 + SOURCE_REDSHIFT)} um'),
            galaxies=galaxies))
        group = ObservationGroup([deflector, arc], cube, [sis()])
        values = ParamSpace.from_properties(
            group.pdescs(), params).evaluate()
        if 'deflector' in galaxies:
            with pytest.raises(ValueError, match="leave the galaxy out"):
                group.plan(driver, reference=values)
        else:
            model = group.plan(driver, reference=values).model_h(values)[0]
            assert np.isfinite(model['spectra'].data).all()
