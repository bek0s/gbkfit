"""
Tests for the moment maps data model (mmaps).
"""

import pathlib

import numpy as np
import pytest


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


def evaluate_mmaps(driver, dmodel, **properties):
    """
    The moment maps of a thin disk with a constant velocity dispersion
    of 20, evaluated with the given dmodel and parameter properties.
    """
    from gbkfit.model import Model, ModelGroup, gmodel_parser
    from gbkfit.params import EvaluationParams
    gmodel = gmodel_parser.load(dict(
        type='kinematics_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'),
            vptraits=dict(type='tan_arctan'),
            dptraits=dict(type='uniform'))]))
    model_group = ModelGroup([Model(driver, dmodel, gmodel)])
    params = EvaluationParams(model_group.pdescs(), dict(
        vsys=0, xpos=0, ypos=0, posa=30, incl=60,
        bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=40, dpt_a=20) | properties)
    return model_group.model_h(params.evaluate())[0]


def test_higher_moments_of_a_gaussian_line(driver):
    # Without a PSF and an LSF, every spaxel of a thin disk has a single
    # Gaussian line, so its third central moment is zero and its fourth
    # is 3 sigma^4, where sigma is the moment 2 map. A spectral step
    # other than 1 checks that the moments are scaled correctly.
    from gbkfit.model.dmodels import DModelMMaps
    dmodel = DModelMMaps(
        size=(32, 32), spec_size=81, spec_step=5, orders=(1, 2, 3, 4))
    mmaps = evaluate_mmaps(driver, dmodel)
    # One mask for all moments: where the moments are defined.
    # Without weight traits, all weights are 1.
    for key in ('mmap1', 'mmap2', 'mmap3', 'mmap4'):
        defined = np.isfinite(mmaps[key]['d'])
        np.testing.assert_array_equal(mmaps[key]['m'], defined)
        np.testing.assert_array_equal(mmaps[key]['w'], 1)
    sigma = mmaps['mmap2']['d']
    disk = np.isfinite(sigma)
    assert disk.sum() > 100
    np.testing.assert_allclose(sigma[disk], 20, rtol=1e-3)
    np.testing.assert_allclose(
        mmaps['mmap3']['d'][disk], 0, atol=1e-3 * 20 ** 3)
    np.testing.assert_allclose(
        mmaps['mmap4']['d'][disk], 3 * sigma[disk] ** 4, rtol=1e-2)


def test_moment_maps_of_a_galaxy_at_a_high_velocity(driver):
    # B24: the spectral axis was fixed at +-500 around 0. With one around
    # the systemic velocity, the velocity at the centre is vsys.
    from gbkfit.model.dmodels import DModelMMaps
    dmodel = DModelMMaps(
        size=(32, 32), spec_size=201, spec_step=2, spec_rval=1500)
    mmaps = evaluate_mmaps(driver, dmodel, vsys=1500)
    centre = mmaps['mmap1']['d'][15:17, 15:17]
    np.testing.assert_allclose(centre, 1500, atol=15)


@pytest.mark.parametrize('dispersion, spec_size', [
    (None, 401), (20.0, 401), (150.0, 551)])
def test_spectral_axis_from_the_data(dispersion, spec_size):
    # The spectral axis covers the range of mmap1 (1400 to 1600), and
    # three times the largest dispersion on each side: that of mmap2,
    # but at least 100 km/s (also without mmap2), so that the lines of
    # the model are not cut when its dispersion differs from the data's
    from gbkfit.dataset import Data
    from gbkfit.dataset.datasets import DatasetMMaps
    from gbkfit.model import dmodel_parser
    velocity = np.linspace(1400, 1600, 32 * 32).reshape(32, 32)
    maps = dict(mmap0=Data(np.ones((32, 32))), mmap1=Data(velocity))
    if dispersion is not None:
        maps['mmap2'] = Data(np.full((32, 32), dispersion))
    dmodel = dmodel_parser.load(
        dict(type='mmaps', spec_step=2), dataset=DatasetMMaps(**maps))
    assert dmodel.spec_rval() == 1500
    assert dmodel.spec_step() == 2
    assert dmodel.spec_size() == spec_size


def test_at_least_one_moment_order():
    # The moments kernel reads past an empty list of orders
    from gbkfit.model.dmodels import DModelMMaps
    with pytest.raises(RuntimeError, match="at least one moment order"):
        DModelMMaps(size=(8, 8), orders=[])
