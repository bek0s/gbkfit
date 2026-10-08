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


CONFIG_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'mcdisk_vs_smdisk'


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


@pytest.mark.xfail(
    reason="known bug: the mcdisk position angle is off by 180 degrees")
def test_mcdisk_velocity_field_matches_smdisk(evaluate_model):
    # Both configurations describe the same thick disk
    smdisk = evaluate_model(CONFIG_DIR / 'smdisk.yaml')
    mcdisk = evaluate_model(CONFIG_DIR / 'mcdisk.yaml')
    smdisk_velocity, smdisk_intensity = velocity_field(
        smdisk['model_0_scube_d'])
    mcdisk_velocity, _ = velocity_field(mcdisk['model_0_scube_d'])
    # Compare where the disk is bright enough for little Monte Carlo noise
    bright = smdisk_intensity > 0.05 * smdisk_intensity.max()
    correlation = np.corrcoef(
        smdisk_velocity[bright], mcdisk_velocity[bright])[0, 1]
    assert correlation > 0.95


# Evaluates the mcdisk configuration with the given driver, and saves
# the model cube. Arguments: driver type, output file.
_EVALUATE_SCRIPT = """
import sys
import numpy as np
from test_mcdisk import evaluate_mcdisk
np.save(sys.argv[2], evaluate_mcdisk(sys.argv[1]))
"""


def evaluate_mcdisk(driver_type):
    """Evaluate the mcdisk configuration with the given driver."""
    import gbkfit.model
    import gbkfit.params
    config = ruamel.yaml.YAML(typ='safe').load(CONFIG_DIR / 'mcdisk.yaml')
    config['models'][0]['driver']['type'] = driver_type
    model_group = gbkfit.model.ModelGroup(
        gbkfit.model.model_parser.load(config['models']))
    params = gbkfit.params.EvaluationParams(
        model_group.pdescs(), config['params']['properties'])
    return model_group.model_h(params.evaluate())[0]['scube']['d'].copy()


def relative_difference(actual, desired):
    return np.linalg.norm(actual - desired) / np.linalg.norm(desired)


# Two different random realisations of the mcdisk differ by about 5e-3.
# The same realisation differs only by float rounding.
SAME_REALISATION = 1e-5


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


def test_mcdisk_same_on_all_drivers(driver):
    # Every driver draws the same random numbers for every cloud
    cube = evaluate_mcdisk(driver.type())
    assert relative_difference(cube, evaluate_mcdisk('host')) \
        < SAME_REALISATION
