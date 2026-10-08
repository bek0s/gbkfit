"""
Tests for the Monte Carlo disk (mcdisk) against the smooth disk (smdisk).
"""

import pathlib

import numpy as np
import pytest


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
