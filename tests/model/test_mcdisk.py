"""
Tests for the Monte Carlo disk (mcdisk) against the smooth disk (smdisk).
"""

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import ruamel.yaml
from modelutils import config_group


CONFIG_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'mcdisk_vs_smdisk'

# Two different random realisations of the mcdisk differ by about 5e-3.
# The same realisation differs only by float rounding.
SAME_REALISATION = 1e-5


def evaluate_disk(
        disk, driver_type, component=None, properties=None, out_extra=None):
    """
    Evaluate the 'mcdisk' or 'smdisk' configuration with the given
    driver, and return its model cube. The options of the component and
    the parameter properties of the configuration can be overridden. The
    extra outputs go to out_extra, if given.
    """
    import gbkfit.params
    config = ruamel.yaml.YAML(typ='safe').load(CONFIG_DIR / f'{disk}.yaml')
    config['observations'][0]['driver']['type'] = driver_type
    config['gmodels'][0]['components'][0].update(component or {})
    model_group = config_group(config)
    params = gbkfit.params.EvaluationParams(
        model_group.pdescs(),
        config['params']['properties'] | (properties or {}))
    data = model_group.model_h(params.evaluate(), out_extra)
    return data[0]['scube']['d'].copy()


def velocity_field(scube):
    """
    Intensity-weighted mean velocity of every spaxel of a spectral cube,
    in units of channels from the central channel. Also return the total
    intensity of every spaxel.
    """
    channels = np.arange(scube.shape[0]) - (scube.shape[0] - 1) / 2
    intensity = scube.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity = np.tensordot(channels, scube, axes=1) / intensity
    return velocity, intensity


def relative_difference(actual, desired):
    return np.linalg.norm(actual - desired) / np.linalg.norm(desired)


@pytest.mark.parametrize('posa', [30, 120, 250])
def test_mcdisk_velocity_field_matches_smdisk(driver, posa):
    # An off-centre thick disk, with more clouds to reduce the noise
    properties = dict(posa=posa, xpos=5, ypos=-3, vsys=50)
    smdisk = evaluate_disk('smdisk', driver.type(), properties=properties)
    mcdisk = evaluate_disk(
        'mcdisk', driver.type(), dict(cflux=2e-5), properties)
    smdisk_velocity, smdisk_intensity = velocity_field(smdisk)
    mcdisk_velocity, mcdisk_intensity = velocity_field(mcdisk)
    # Compare where the disk is bright enough for little Monte Carlo noise
    bright = smdisk_intensity > 0.05 * smdisk_intensity.max()
    difference = mcdisk_velocity[bright] - smdisk_velocity[bright]
    # The velocities span about +-18 channels; the noise is ~0.015
    assert np.sqrt(np.mean(difference ** 2)) < 0.1
    assert relative_difference(mcdisk_intensity, smdisk_intensity) < 0.02


@pytest.mark.parametrize('trunc', [0, 1], ids=['full', 'truncated'])
@pytest.mark.parametrize('height, shape', [
    ('exponential', {}), ('gauss', {}), ('sech2', {}),
    ('ggauss', dict(bht_b=1.5)), ('ggauss', dict(bht_b=0.6)),
    ('moffat', dict(bht_b=2.5)), ('moffat', dict(bht_b=0.8))],
    ids=['exponential', 'gauss', 'sech2', 'ggauss', 'ggauss_heavy_tails',
         'moffat', 'moffat_heavy_tails'])
def test_mcdisk_vertical_profile_matches_smdisk(driver, height, shape, trunc):
    # An almost edge-on thick disk, with more clouds to reduce the noise.
    # Its image does not depend on the velocity field, so it is not
    # affected by the position angle bug. A truncated profile is cut at
    # trunc times its scale height. The heavy tails of a ggauss of shape
    # 0.6 and of a Moffat of shape 0.8 reach far beyond 5 scale heights.
    component = dict(bhtraits=dict(type=height, trunc=trunc))
    properties = dict(incl=85, bht_s=2) | shape
    mcdisk = evaluate_disk(
        'mcdisk', driver.type(), component | dict(cflux=2e-5), properties)
    smdisk = evaluate_disk('smdisk', driver.type(), component, properties)
    # The Monte Carlo noise is about 0.5%; a wrong profile gives 10-15%
    assert relative_difference(mcdisk.sum(0), smdisk.sum(0)) < 0.02


# A vertical distortion of the midplane at every node of the configs
DISTORTION = dict(zptraits=dict(type='nw_uniform'))
DISTORTED = dict(zpt_a=[2.0] * 11)


def test_mcdisk_vertical_distortion_matches_smdisk(driver):
    # A distortion lifts the midplane of both disks the same way
    properties = DISTORTED | dict(incl=80)
    smdisk = evaluate_disk('smdisk', driver.type(), DISTORTION, properties)
    mcdisk = evaluate_disk(
        'mcdisk', driver.type(), DISTORTION | dict(cflux=2e-5), properties)
    flat = evaluate_disk(
        'smdisk', driver.type(), DISTORTION, properties | dict(
            zpt_a=[0.0] * 11))
    assert relative_difference(smdisk.sum(0), flat.sum(0)) > 0.1
    assert relative_difference(mcdisk.sum(0), smdisk.sum(0)) < 0.02


def test_mcdisk_height_traits_of_a_distorted_disk_match_smdisk(driver):
    # The velocity height traits depend on the height above the
    # (distorted) midplane, in both disks
    component = DISTORTION | dict(vhtraits=dict(type='exponential'))
    properties = DISTORTED | dict(vht_s=1)
    smdisk = evaluate_disk('smdisk', driver.type(), component, properties)
    mcdisk = evaluate_disk(
        'mcdisk', driver.type(), component | dict(cflux=2e-5), properties)
    smdisk_velocity, smdisk_intensity = velocity_field(smdisk)
    mcdisk_velocity, _ = velocity_field(mcdisk)
    bright = smdisk_intensity > 0.05 * smdisk_intensity.max()
    difference = mcdisk_velocity[bright] - smdisk_velocity[bright]
    assert np.sqrt(np.mean(difference ** 2)) < 0.1

@pytest.mark.parametrize('height', ['exponential', 'gauss', 'uniform'])
def test_mcdisk_flux_does_not_depend_on_height_profile(driver, height):
    # All clouds carry the same flux, and with these thin profiles they
    # all land inside the cube, so the total flux must be the same.
    reference = evaluate_disk(
        'mcdisk', driver.type(), dict(bhtraits=dict(type='sech2')))
    total = evaluate_disk(
        'mcdisk', driver.type(), dict(bhtraits=dict(type=height)))
    assert total.sum() == pytest.approx(reference.sum(), rel=1e-5)


def test_mcdisk_same_on_all_drivers(driver):
    # Every driver draws the same random numbers for every cloud
    cube = evaluate_disk('mcdisk', driver.type())
    reference = evaluate_disk('mcdisk', 'host')
    assert relative_difference(cube, reference) < SAME_REALISATION


# Evaluates the mcdisk configuration with the given driver, and saves
# the model cube. Arguments: driver type, output file.
_EVALUATE_SCRIPT = """
import sys
import numpy as np
from test_mcdisk import evaluate_disk
np.save(sys.argv[2], evaluate_disk('mcdisk', sys.argv[1]))
"""


def test_mcdisk_does_not_depend_on_thread_count(tmp_path):
    cubes = []
    for threads in (1, 3):
        output = tmp_path / f'threads_{threads}.npy'
        subprocess.run(
            [sys.executable, '-c', _EVALUATE_SCRIPT, 'host', str(output)],
            env=os.environ | dict(OMP_NUM_THREADS=str(threads)),
            cwd=pathlib.Path(__file__).parent, check=True)
        cubes.append(np.load(output))
    assert relative_difference(cubes[1], cubes[0]) < SAME_REALISATION


def test_mcdisk_cflux_must_be_positive():
    with pytest.raises(Exception, match="cflux must be greater than 0"):
        evaluate_disk('mcdisk', 'host', dict(cflux=-0.01))


def test_mcdisk_of_too_many_clouds_is_an_error(driver):
    # The kernels count the clouds in int32: a disk that needs more (here
    # about 1.6e11) is an error (it was an empty model)
    with pytest.raises(RuntimeError, match="larger cflux"):
        evaluate_disk('mcdisk', driver.type(), dict(cflux=1e-9))

def test_mcdisk_without_clouds(driver):
    # A disk with zero flux has no clouds; its model is empty
    cube = evaluate_disk('mcdisk', driver.type(), properties=dict(bpt_a=0))
    assert not cube.any()


def test_mcdisk_seed(driver):
    # The seed (0 by default) selects the random realisation of the clouds
    default = evaluate_disk('mcdisk', driver.type())
    seed0 = evaluate_disk('mcdisk', driver.type(), dict(seed=0))
    seed1 = evaluate_disk('mcdisk', driver.type(), dict(seed=1))
    assert relative_difference(seed0, default) < SAME_REALISATION
    assert relative_difference(seed1, default) > 1e-3
    # Different realisations of the same disk
    assert seed1.sum() == pytest.approx(default.sum(), rel=1e-5)


@pytest.mark.parametrize('amplitude', [0.3, -0.3])
@pytest.mark.parametrize('order', [0, 2])
def test_mcdisk_harmonic_brightness_matches_smdisk(driver, order, amplitude):
    # An exponential disk plus a harmonic of the given order (a ring for
    # order 0). The clouds of a harmonic carry the sign of cos(k(t - p)).
    component = dict(
        cflux=2e-5,
        bptraits=[
            dict(type='exponential'), dict(type='nw_harmonic', order=order)],
        bhtraits=[dict(type='sech2'), dict(type='sech2')])
    properties = dict(bpt1_a=[amplitude] * 11, bht1_s=1) | (
        dict(bpt1_p=[40] * 11) if order else {})
    mcdisk = evaluate_disk('mcdisk', driver.type(), component, properties)
    del component['cflux']
    smdisk = evaluate_disk('smdisk', driver.type(), component, properties)
    assert relative_difference(mcdisk.sum(0), smdisk.sum(0)) < 0.02


@pytest.mark.parametrize('p', [0, 90, -90, 180, 270])
def test_mcdisk_azimuthal_selection_matches_smdisk(driver, p):
    # Only the azimuths within s / 2 of p are kept, for any p. A third of
    # the disk is kept, so more clouds keep the noise down.
    component = dict(sptraits=dict(type='azrange'))
    properties = dict(spt_p=p, spt_s=120)
    mcdisk = evaluate_disk(
        'mcdisk', driver.type(), component | dict(cflux=1e-5), properties)
    smdisk = evaluate_disk('smdisk', driver.type(), component, properties)
    # The smooth disk selects whole pixels at the edges of the range, which
    # differs by about 2.5%; a wrong range gives 50-100%
    assert relative_difference(mcdisk.sum(0), smdisk.sum(0)) < 0.05


@pytest.mark.parametrize('outer', [0, -0.3], ids=['negative', 'mixed'])
def test_mcdisk_negative_brightness_matches_smdisk(driver, outer):
    # A negative exponential disk, or a positive one whose outer rings
    # are made negative by a second trait: the clouds of a negative ring
    # carry negative flux
    component = dict(
        cflux=2e-5,
        bptraits=[dict(type='exponential'), dict(type='nw_uniform')],
        bhtraits=[dict(type='sech2'), dict(type='sech2')])
    properties = dict(
        bpt_a=-1 if not outer else 1,
        bpt1_a=[0] * 5 + [outer] * 6, bht1_s=1)
    mcdisk = evaluate_disk('mcdisk', driver.type(), component, properties)
    del component['cflux']
    smdisk = evaluate_disk('smdisk', driver.type(), component, properties)
    assert relative_difference(mcdisk.sum(0), smdisk.sum(0)) < 0.02


def test_mcdisk_flux_does_not_depend_on_cloud_flux(driver):
    # The clouds of each ring share its flux exactly, also when a cloud
    # (cflux) holds more than the flux of a ring
    totals = [
        evaluate_disk('mcdisk', driver.type(), dict(cflux=cflux)).sum()
        for cflux in (1e-4, 0.3, 5)]
    np.testing.assert_allclose(totals[1:], totals[0], rtol=1e-4)


def test_mcdisk_harmonic_clouds_take_the_sign_of_the_harmonic(driver):
    # A harmonic alone (the exponential has no flux), with a few clouds
    # per pixel. The clouds are drawn where the harmonic is bright, each
    # with its sign, so where it is clearly positive or negative, the
    # brightness before the PSF has its sign in every pixel.
    component = dict(
        bptraits=[dict(type='exponential'), dict(type='nw_harmonic', order=2)],
        bhtraits=[dict(type='sech2'), dict(type='sech2')])
    properties = dict(bpt_a=0, bpt1_a=[1] * 11, bpt1_p=[40] * 11, bht1_s=1)
    brightness = {}
    for disk, options in [('mcdisk', dict(cflux=0.5)), ('smdisk', {})]:
        extra = {}
        evaluate_disk(
            disk, driver.type(), component | options, properties, extra)
        brightness[disk] = extra['observation0_gmodel_component0_bdata'].data.sum(0)
    smdisk = brightness['smdisk']
    clear = np.abs(smdisk) > 0.5 * np.abs(smdisk).max()
    mcdisk = brightness['mcdisk'][clear]
    expected = np.sign(smdisk[clear])
    has_clouds = mcdisk != 0
    assert has_clouds.sum() > 100
    np.testing.assert_array_equal(
        np.sign(mcdisk[has_clouds]), expected[has_clouds])


def test_mcdisk_velocity_is_the_mean_of_the_clouds(driver):
    # The velocity of each voxel is the mean of those of its clouds,
    # weighted by their flux. Without a PSF and an LSF, the spectrum of a
    # spaxel is the sum of the lines of its clouds, so its first moment
    # is the brightness-weighted mean of the velocity along z (the lines
    # are within the spectral axis).
    import gbkfit.params
    config = ruamel.yaml.YAML(typ='safe').load(CONFIG_DIR / 'mcdisk.yaml')
    observation = config['observations'][0]
    observation['driver']['type'] = driver.type()
    observation['instrument'] = {}
    observation['observable'] = dict(
        type='scube', size=[48, 48, 120], step=[1, 1, 5])
    model_group = config_group(config)
    params = gbkfit.params.EvaluationParams(
        model_group.pdescs(), config['params']['properties'])
    extra = {}
    scube = model_group.model_h(params.evaluate(), extra)[0]['scube']['d']
    velocities = (np.arange(120) - 59.5) * 5
    intensity = scube.sum(axis=0)
    bright = intensity > 0.01 * intensity.max()
    moment1 = np.tensordot(velocities, scube, axes=1)[bright] / \
        intensity[bright]
    brightness = extra['observation0_gmodel_component0_bdata'].data
    velocity = np.nan_to_num(extra['observation0_gmodel_component0_vdata'].data)
    mean = (brightness * velocity).sum(0)[bright] / \
        brightness.sum(0)[bright]
    np.testing.assert_allclose(mean, moment1, atol=0.05)


# Two blobs, the second negative, inside the disk (radius 20)
BLOBS = dict(
    bpt_r=[6.0, 11.0], bpt_t=[30.0, 200.0], bpt_a=[1.0, -0.4],
    bpt_s=[2.0, 3.0], bpt_q=[0.6, 0.8], bpt_p=[20.0, 70.0])


@pytest.mark.parametrize('mixture, shape', [
    ('mixture_exponential', None), ('mixture_gauss', None),
    ('mixture_ggauss', 0.8), ('mixture_ggauss', 1.5),
    ('mixture_ggauss', 3.0), ('mixture_moffat', 2.5)])
def test_mcdisk_mixture_brightness_matches_smdisk(driver, mixture, shape):
    # Each cloud is drawn from a blob chosen by its flux, and carries the
    # sign of its amplitude (the shapes of ggauss cover both ways of
    # drawing its gamma variates: shape 2 / b above and below 1)
    component = dict(
        cflux=1e-5, bptraits=dict(type=mixture, nblobs=2),
        bhtraits=dict(type='sech2'))
    properties = BLOBS | (dict(bpt_b=[shape, shape]) if shape else {})
    mcdisk = evaluate_disk('mcdisk', driver.type(), component, properties)
    del component['cflux']
    smdisk = evaluate_disk('smdisk', driver.type(), component, properties)
    assert relative_difference(mcdisk.sum(0), smdisk.sum(0)) < 0.03


def test_mcdisk_distortion_brightness_matches_smdisk(driver):
    # A ring brighter around an azimuth: the clouds of each ring are drawn
    # around it, within the arc length s
    component = dict(
        cflux=1e-5,
        bptraits=[dict(type='exponential'), dict(type='nw_distortion')],
        bhtraits=[dict(type='sech2'), dict(type='sech2')])
    properties = dict(
        bpt1_a=[0.5] * 11, bpt1_p=[60] * 11, bpt1_s=[3] * 11, bht1_s=1)
    mcdisk = evaluate_disk('mcdisk', driver.type(), component, properties)
    del component['cflux']
    smdisk = evaluate_disk('smdisk', driver.type(), component, properties)
    assert relative_difference(mcdisk.sum(0), smdisk.sum(0)) < 0.03


def test_mcdisk_dispersion_mixture_matches_smdisk(driver):
    # Dispersion traits are evaluated at the clouds, like at the voxels
    component = dict(
        cflux=1e-5,
        dptraits=[dict(type='uniform'), dict(type='mixture_gauss', nblobs=2)])
    properties = {
        'dpt_a': 10, **{k.replace('bpt_', 'dpt1_'): v
                        for k, v in BLOBS.items()}}
    properties['dpt1_a'] = [60.0, 30.0]
    mcdisk = evaluate_disk('mcdisk', driver.type(), component, properties)
    del component['cflux']
    smdisk = evaluate_disk('smdisk', driver.type(), component, properties)
    assert relative_difference(mcdisk, smdisk) < 0.05


def test_mcdisk_moffat_blobs_need_a_finite_flux(driver):
    component = dict(
        bptraits=dict(type='mixture_moffat', nblobs=2),
        bhtraits=dict(type='sech2'))
    with pytest.raises(RuntimeError, match="need b > 1"):
        evaluate_disk('mcdisk', driver.type(), component,
                      BLOBS | dict(bpt_b=[2.0, 0.9]))
