"""
Tests for the moment maps data model (mmaps).
"""

import pathlib

import numpy as np


REFERENCE_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'reference_models'


def moments_from_scube(scube, spec_step):
    """
    Moment 1 and 2 maps of a spectral cube whose central channel is at
    velocity zero. Also return the total intensity of every spaxel.
    """
    channels = np.arange(scube.shape[0]) - (scube.shape[0] - 1) / 2
    velocities = channels * spec_step
    intensity = scube.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        moment1 = np.tensordot(velocities, scube, axes=1) / intensity
        moment2 = np.sqrt(
            np.tensordot(velocities ** 2, scube, axes=1) / intensity
            - moment1 ** 2)
    return moment1, moment2, intensity


def test_mmaps_matches_moments_of_scube(evaluate_model):
    # thin_disk_mmaps and thin_disk_scube describe the same galaxy with
    # the same PSF and LSF, so the moment maps must match the moments of
    # the spectral cube
    mmaps = evaluate_model(REFERENCE_DIR / 'thin_disk_mmaps.yaml')
    scube = np.load(REFERENCE_DIR / 'thin_disk_scube.npz')['model_0_scube_d']
    moment1, moment2, intensity = moments_from_scube(
        scube.astype(np.float64), spec_step=10)
    bright = intensity > 0.05 * intensity.max()
    for order, expected in ((1, moment1), (2, moment2)):
        np.testing.assert_allclose(
            mmaps[f'model_0_mmap{order}_d'][bright], expected[bright],
            atol=0.01, err_msg=f"moment {order}")
