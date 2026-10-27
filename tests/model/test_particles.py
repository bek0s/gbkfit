"""
Tests of particles components (see Particles): particles drawn from a
disk are seen as the disk is.
"""

import copy

import gbkfit.params
import numpy as np
import pytest
from modelutils import group_plan, observation_group


# A thin exponential disk of scale length H and central brightness 1, to
# RMAX, rotating at VT (2 / pi) atan(R / RT), of dispersion DISP
H, RMAX, VT, RT, DISP = 3.0, 30.0, 200.0, 2.0, 20.0
GEOMETRY = dict(xpos=0.4, ypos=-0.3, posa=40, incl=60, vsys=0)
DISK = dict(components=[dict(
    type='disk', loose=False, tilted=False,
    rnodes=list(range(0, int(RMAX) + 1)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        type='traits', vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))])
DISK_PROPERTIES = GEOMETRY | dict(
    pt_a=1, pt_s=H, vpt_rt=RT, vpt_vt=VT, dpt_a=DISP)
CUBE = dict(type='pixel_spectra', size=[40, 40, 40], step=[1, 1, 15])


def disk_particles(path, n=200_000, seed=1):
    """
    Write to a numpy file the particles of the disk: radii of its
    exponential surface density (R exp(-R / H)), azimuths at random, and
    its rotation; each of an equal share of its flux.
    """
    rng = np.random.default_rng(seed)
    r = rng.gamma(2, H, n)
    r = r[r < RMAX]
    theta = rng.uniform(0, 2 * np.pi, len(r))
    v = VT * 2 / np.pi * np.arctan(r / RT)
    total = 2 * np.pi * H ** 2 * (1 - (1 + RMAX / H) * np.exp(-RMAX / H))
    zeros = np.zeros(len(r))
    np.savez(
        path, x=r * np.cos(theta), y=r * np.sin(theta), z=zeros,
        vx=-v * np.sin(theta), vy=v * np.cos(theta), vz=zeros,
        flux=np.full(len(r), total / len(r)), sigma=np.full(len(r), DISP))
    return str(path)


def evaluate(driver, galaxy, properties, line_of_sight='flat'):
    """
    The cube of a galaxy along a line of sight (its type, or its
    options).
    """
    if isinstance(line_of_sight, str):
        line_of_sight = dict(type=line_of_sight)
    plan = group_plan([dict(
        driver=dict(type=driver.type()), galaxy=copy.deepcopy(galaxy),
        line_of_sight=line_of_sight, observable=CUBE)])
    params = gbkfit.params.ParamSpace(plan.group().pdescs(), properties)
    return plan.model_h(params.evaluate())[0]['spectra']['d'].astype(float)


def moments(cube):
    """The image (flux per pixel) and the velocity map of a cube."""
    velocities = (np.arange(len(cube)) - len(cube) / 2) * 15
    image = cube.sum(axis=0)
    with np.errstate(invalid='ignore'):
        return image, np.tensordot(velocities, cube, 1) / image


@pytest.mark.parametrize('line_of_sight', ['flat', 'transfer'])
def test_particles_of_a_disk_are_seen_as_the_disk(
        driver, tmp_path, line_of_sight):
    # Their image and velocity map are those of the disk, to the noise of
    # their number (up to ~10% of the light of a pixel of a fifth of the
    # peak); in the march, the particles of a voxel are one line of their
    # moments
    particles = dict(components=[dict(
        type='particles', file=disk_particles(tmp_path / 'disk.npz'),
        emission=dict(type='lines'))])
    disk_image, disk_velocity = moments(evaluate(
        driver, DISK, DISK_PROPERTIES, line_of_sight))
    image, velocity = moments(evaluate(
        driver, particles, GEOMETRY, line_of_sight))
    assert image.sum() == pytest.approx(disk_image.sum(), rel=0.01)
    bright = disk_image > 0.2 * disk_image.max()
    assert np.max(np.abs(image - disk_image)[bright]
                  / disk_image[bright]) < 0.2
    difference = np.abs(velocity - disk_velocity)[bright]
    assert np.max(difference) < 10 and np.mean(difference) < 2


def test_particles_go_to_their_planes_in_slabs(driver, tmp_path):
    # A march in slabs of two planes of the depth axis (the option slab)
    # places each particle in its plane of the slab that has it: the cube
    # of the whole axis at once
    particles = dict(components=[dict(
        type='particles',
        file=disk_particles(tmp_path / 'disk.npz', n=20_000),
        emission=dict(type='lines'))])
    whole = evaluate(driver, particles, GEOMETRY, 'transfer')
    slabs = evaluate(
        driver, particles, GEOMETRY, dict(type='transfer', slab=2))
    np.testing.assert_allclose(
        slabs, whole, rtol=1e-5, atol=1e-6 * whole.max())


def test_a_fit_to_mock_data_of_particles_recovers_their_disk(tmp_path):
    # Mock data of the particles of the disk (noise of 1% of the peak, and
    # the noise of their number), fitted by the disk from elsewhere. The
    # particles give each pixel the light in it; the disk, sampled at the
    # centres of its pixels (3 per scale length here), would be fitted
    # with its central brightness 5% too low and its inclination 2
    # degrees, so it is oversampled
    from gbkfit.fitting import fitter_parser
    from gbkfit.objective import Objective
    from gbkfit.observation import NoiseGauss, ObservationGroup
    particles = dict(components=[dict(
        type='particles', file=disk_particles(tmp_path / 'disk.npz'),
        emission=dict(type='lines'))])
    plan = group_plan([dict(
        driver=dict(type='host'), galaxy=particles,
        line_of_sight=dict(type='flat'), observable=CUBE)])
    values = gbkfit.params.ParamSpace(
        plan.group().pdescs(), GEOMETRY).evaluate()
    peak = plan.model_h(values)[0]['spectra']['d'].max()
    mock = plan.mock(values, NoiseGauss(rms=0.01 * peak), seed=2)[0][0]
    disk = observation_group([dict(
        driver=dict(type='host'), galaxy=DISK,
        line_of_sight=dict(type='flat'),
        observable=CUBE | dict(oversampling=[4, 4, 1]))])
    objective = Objective(ObservationGroup(
        disk.galaxies(), disk.observations()[0].with_data(mock)).plan())
    properties = DISK_PROPERTIES | dict(
        posa=dict(value=30, min=0, max=90),
        incl=dict(value=50, min=20, max=80),
        vpt_vt=dict(value=150, min=50, max=300),
        pt_a=dict(value=0.5, min=0.1, max=5))
    params = gbkfit.params.ParamSpace(objective.pdescs(), properties)
    best = fitter_parser.load(dict(type='scipy.least_squares')).fit(
        objective, params).best().free
    assert best['posa'] == pytest.approx(40, abs=0.3)
    assert best['incl'] == pytest.approx(60, abs=0.5)
    assert best['vpt_vt'] == pytest.approx(VT, rel=0.01)
    assert best['pt_a'] == pytest.approx(1, rel=0.01)


def test_particles_in_an_image(driver, tmp_path):
    # Each in the pixel that holds it, as surface brightness (the flux of
    # a pixel is its value times its area); off the grid, none
    zeros = np.zeros(3)
    np.savez(
        tmp_path / 'three.npz', x=np.array([0.0, 0.6, 50.0]),
        y=np.array([0.0, 0.0, 0.0]), z=zeros, vx=zeros, vy=zeros,
        vz=zeros, flux=np.array([2.0, 3.0, 4.0]))
    galaxy = dict(components=[dict(
        type='particles', file=str(tmp_path / 'three.npz'),
        emission=dict(type='band'))])
    plan = group_plan([dict(
        driver=dict(type=driver.type()), galaxy=galaxy,
        line_of_sight=dict(type='flat'), observable=dict(
            type='pixel_brightness', size=[10, 8], step=[0.5, 0.5]))])
    params = gbkfit.params.ParamSpace(plan.group().pdescs(), dict(
        xpos=0, ypos=0, posa=0, incl=0))
    image = plan.model_h(params.evaluate())[0]['brightness']['d']
    assert image.sum() * 0.25 == pytest.approx(5)
    assert np.count_nonzero(image) == 2


@pytest.mark.parametrize('columns, message', [
    (dict(x=np.zeros(2), y=np.zeros(2)), "have no columns \\['z', 'vx'"),
    ({key: np.zeros(0) for key in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'flux')},
     "has no particles")])
def test_particles_need_their_columns(tmp_path, columns, message):
    from gbkfit.model import galaxy_parser
    from gbkfit.utils.parseutils import ConfigError
    np.savez(tmp_path / 'some.npz', **columns)
    with pytest.raises(ConfigError, match=message):
        galaxy_parser.load(dict(components=[dict(
            type='particles', file=str(tmp_path / 'some.npz'),
            emission=dict(type='band'))]))
