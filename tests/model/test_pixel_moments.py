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


def test_pixel_moments_matches_moments_of_pixel_spectra(evaluate_model):
    # thin_disk_pixel_moments and thin_disk_pixel_spectra describe the same galaxy with
    # the same PSF and LSF, so the moment maps must match the moments of
    # the spectral cube. The channels hold the mean of the lines over each
    # channel, which adds the variance of a channel (step^2 / 12) to the
    # lines: the spectral cube has channels of 10, and the cube of the
    # moment maps of 1 (the default).
    mmaps = evaluate_model(REFERENCE_DIR / 'thin_disk_pixel_moments.yaml')
    scube = np.load(REFERENCE_DIR / 'thin_disk_pixel_spectra.npz')['model_0_spectra_d']
    moment1, moment2, intensity = moments_from_scube(
        scube.astype(np.float64), spec_step=10)
    moment2 = np.sqrt(moment2 ** 2 - 10 ** 2 / 12 + 1 ** 2 / 12)
    bright = intensity > 0.05 * intensity.max()
    for order, expected in ((1, moment1), (2, moment2)):
        np.testing.assert_allclose(
            mmaps[f'model_0_moment{order}_d'][bright], expected[bright],
            atol=0.01, err_msg=f"moment {order}")


def evaluate_mmaps(driver, observable, **properties):
    """
    The moment maps of a thin disk with a constant velocity dispersion
    of 20, observed as the given observable with the given parameter
    properties.
    """
    from gbkfit.model import gmodel_parser
    from gbkfit.observation import Observation, ObservationGroup
    from gbkfit.params import EvaluationParams
    gmodel = gmodel_parser.load(dict(
        type='kinematics_2d', components=[dict(
            type='smdisk', loose=False, tilted=False,
            rnodes=list(range(0, 12)),
            bptraits=dict(type='exponential'),
            vptraits=dict(type='tan_arctan'),
            dptraits=dict(type='uniform'))]))
    model_group = ObservationGroup([gmodel], [Observation(observable, driver)])
    params = EvaluationParams(model_group.pdescs(), dict(
        vsys=0, xpos=0, ypos=0, posa=30, incl=60,
        bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=40, dpt_a=20) | properties)
    return model_group.model_h(params.evaluate())[0]


def test_higher_moments_of_a_gaussian_line(driver):
    # Without a PSF and an LSF, every spaxel of a thin disk has a single
    # Gaussian line, so its third central moment is zero and its fourth
    # is 3 sigma^4, where sigma is the moment 2 map. A spectral step
    # other than 1 checks that the moments are scaled correctly. The
    # channels hold the mean of the line over each channel, which adds
    # the variance of a channel (step^2 / 12) to that of the line.
    from gbkfit.observation import PixelMoments
    observable = PixelMoments(
        size=(32, 32), spec_size=81, spec_step=5, orders=(1, 2, 3, 4))
    mmaps = evaluate_mmaps(driver, observable)
    # One mask for all moments: where the moments are defined.
    # Without weight traits, all weights are 1.
    for key in ('moment1', 'moment2', 'moment3', 'moment4'):
        defined = np.isfinite(mmaps[key]['d'])
        np.testing.assert_array_equal(mmaps[key]['m'], defined)
        np.testing.assert_array_equal(mmaps[key]['w'], 1)
    sigma = mmaps['moment2']['d']
    disk = np.isfinite(sigma)
    assert disk.sum() > 100
    np.testing.assert_allclose(
        sigma[disk], np.sqrt(20 ** 2 + 5 ** 2 / 12), rtol=1e-3)
    np.testing.assert_allclose(
        mmaps['moment3']['d'][disk], 0, atol=1e-3 * 20 ** 3)
    np.testing.assert_allclose(
        mmaps['moment4']['d'][disk], 3 * sigma[disk] ** 4, rtol=1e-2)


def test_moment_maps_have_the_weights_of_the_gmodel(driver):
    # The weight of the moments of a spectrum is the flux-weighted mean of
    # its weights: those of a gmodel with the spatial weights 0 on the
    # first row of pixels, and 1 elsewhere
    from gbkfit.model import GModelKinematics2D
    from gbkfit.model.components.disks import SpectralSMDisk2D, traits
    from gbkfit.observation import PixelMoments, Observation, ObservationGroup
    from gbkfit.params import EvaluationParams
    from modelutils import WeightComponent
    disk = SpectralSMDisk2D(
        loose=False, tilted=False, rnodes=list(range(0, 25)),
        bptraits=traits.BPTraitExponential(),
        vptraits=traits.VPTraitTanArctan(), dptraits=traits.DPTraitUniform())
    gmodel = GModelKinematics2D([disk, WeightComponent()])
    observable = PixelMoments(size=(32, 32), spec_size=81, spec_step=5)
    model_group = ObservationGroup([gmodel], [Observation(observable, driver)])
    params = EvaluationParams(model_group.pdescs(), dict(
        vsys=0, xpos=0, ypos=0, posa=30, incl=0,
        bpt_a=1, bpt_s=4, vpt_rt=2, vpt_vt=40, dpt_a=20))
    mmaps = model_group.model_h(params.evaluate())[0]
    for key in ('moment0', 'moment1', 'moment2'):
        defined = np.isfinite(mmaps[key]['d'])
        assert defined.all()
        weights = mmaps[key]['w']
        np.testing.assert_array_equal(weights[0], 0)
        np.testing.assert_array_equal(weights[1:], 1)


def test_moment_maps_of_a_galaxy_at_a_high_velocity(driver):
    # B24: the spectral axis was fixed at +-500 around 0. With one around
    # the systemic velocity, the velocity at the centre is vsys.
    from gbkfit.observation import PixelMoments
    observable = PixelMoments(
        size=(32, 32), spec_size=201, spec_step=2, spec_rval=1500)
    mmaps = evaluate_mmaps(driver, observable, vsys=1500)
    centre = mmaps['moment1']['d'][15:17, 15:17]
    np.testing.assert_allclose(centre, 1500, atol=15)


@pytest.mark.parametrize('dispersion, spec_size', [
    (None, 401), (20.0, 401), (150.0, 551)])
def test_spectral_axis_from_the_data(dispersion, spec_size):
    # The spectral axis covers the range of moment1 (1400 to 1600), and
    # three times the largest dispersion on each side: that of moment2,
    # but at least 100 km/s (also without moment2), so that the lines of
    # the model are not cut when its dispersion differs from the data's
    from gbkfit.dataset import Data
    from gbkfit.dataset import DatasetPixelMoments
    from gbkfit.observation import observable_parser
    velocity = np.linspace(1400, 1600, 32 * 32).reshape(32, 32)
    maps = {0: Data(np.ones((32, 32))), 1: Data(velocity)}
    if dispersion is not None:
        maps[2] = Data(np.full((32, 32), dispersion))
    info = dict(type='pixel_moments', spec_step=2)
    observable = observable_parser.load(
        info, dataset=DatasetPixelMoments(maps))
    assert observable.spec_rval() == 1500
    assert observable.spec_step() == 2
    assert observable.spec_size() == spec_size
    # The configuration is left as it was
    assert info == dict(type='pixel_moments', spec_step=2)


@pytest.mark.parametrize('given, spec_size, spec_rval', [
    # The centre of the data's range (1100 to 1900) with a given size
    (dict(spec_size=401), 401, 1500),
    # A size that covers the range around a given centre
    (dict(spec_rval=1450), 451, 1450)])
def test_spectral_axis_partly_from_the_data(given, spec_size, spec_rval):
    from gbkfit.dataset import Data
    from gbkfit.dataset import DatasetPixelMoments
    from gbkfit.observation import observable_parser
    velocity = np.linspace(1400, 1600, 32 * 32).reshape(32, 32)
    dataset = DatasetPixelMoments(
        {0: Data(np.ones((32, 32))), 1: Data(velocity)})
    observable = observable_parser.load(
        dict(type='pixel_moments', spec_step=2) | given, dataset=dataset)
    assert observable.spec_size() == spec_size
    assert observable.spec_rval() == spec_rval

def test_at_least_one_moment_order():
    # The moments kernel reads past an empty list of orders
    from gbkfit.observation import PixelMoments
    with pytest.raises(RuntimeError, match="at least one moment order"):
        PixelMoments(size=(8, 8), orders=[])
