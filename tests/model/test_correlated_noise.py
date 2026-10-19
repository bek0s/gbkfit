"""
Tests of correlated noise in cubes (see LikelihoodGaussian): the noise
correlated on the sky and along the channels at once.
"""

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelSpectra
from gbkfit.instrument import LSFGauss, PSFGauss
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    LikelihoodGaussian, Observation, ObservationGroup, PixelSpectra,
    line_of_sight_parser)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


SIZE = (12, 12, 31)
STEP = (1, 1, 20)
PARAMS = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, pt_a=1, pt_s=3, vpt_rt=2,
    vpt_vt=150, dpt_a=30)
GALAXY = dict(components=dict(
    type='smdisk', rnodes=list(range(0, 11)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='arctan', direction='tan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines')))


def objective(driver, data, likelihood):
    """The objective of a disk seen in a cube, under a likelihood."""
    dataset = DatasetPixelSpectra(data, step=STEP)
    group = ObservationGroup(
        [galaxy_parser.load(GALAXY)], [Observation(
            PixelSpectra.from_data(dataset),
            line_of_sight_parser.load(dict(type='flat')), driver,
            data=dataset, likelihood=likelihood)])
    return Objective(group), ParamSpace(group.pdescs(), PARAMS).evaluate()


def noisy_cube(rng, mask):
    """
    A cube of noise with errors that differ, a mask, and noise correlated
    on the sky (an elongated beam) and along the channels.
    """
    shape = SIZE[::-1]
    return Data(
        rng.normal(0, 0.01, shape), mask=mask,
        error=rng.uniform(0.005, 0.02, shape),
        spatial_correlation=PSFGauss(1.5, 0.6, 30),
        spectral_correlation=LSFGauss(25))


def test_cubes_have_the_chi2_of_the_product_of_their_correlations(driver):
    # chi2 = r^T (C_z' (x) C_xy')^-1 r over the kept spectra, each
    # correlation with the white floor f: C' = (1 - f) C + f I
    rng = np.random.default_rng(7)
    mask = np.ones(SIZE[::-1])
    mask[:, rng.integers(0, 12, 6), rng.integers(0, 12, 6)] = 0
    data = noisy_cube(rng, mask)
    obj, params = objective(
        driver, data, LikelihoodGaussian(white_floor=0.05))
    model = obj.group().model_h(params)[0]['spectra']['d'].astype(float)
    kept = mask[0] != 0
    r = ((model - data.data()) / data.error())[:, kept]
    y, x = np.nonzero(kept)
    correlation = obj.datasets()[0].noise_correlation('spectra')
    k_xy = correlation.spatial[11 + y[:, None] - y, 11 + x[:, None] - x]
    z = np.arange(SIZE[2])
    k_z = correlation.spectral[30 + z[:, None] - z]
    c = np.kron(
        0.95 * k_z + 0.05 * np.eye(len(k_z)),
        0.95 * k_xy + 0.05 * np.eye(len(k_xy)))
    r = r.ravel()
    assert obj.chi2(params) == pytest.approx(
        r @ np.linalg.solve(c, r), rel=1e-4)
    assert obj.nindependent() == pytest.approx(
        np.trace(np.linalg.solve(c, np.kron(k_z, k_xy))))


def test_cubes_of_correlated_noise_keep_whole_spectra(driver):
    rng = np.random.default_rng(8)
    mask = np.ones(SIZE[::-1])
    mask[3, 5, 5] = 0
    data = noisy_cube(rng, mask)
    with pytest.raises(ConfigError, match="mask_whole_spectra"):
        objective(driver, data, LikelihoodGaussian())
    # The option masks the spectrum of the masked value
    whole = DatasetPixelSpectra(data, step=STEP).mask_whole_spectra()
    np.testing.assert_array_equal(
        whole['spectra'].mask()[:, 5, 5], 0)
    assert np.count_nonzero(whole['spectra'].mask() == 0) == SIZE[2]
    objective(driver, whole['spectra'], LikelihoodGaussian())


def test_cubes_of_correlated_noise_go_without_amplitudes(driver):
    data = noisy_cube(np.random.default_rng(9), None)
    with pytest.raises(ConfigError, match="amplitudes of spectra"):
        objective(driver, data, LikelihoodGaussian(
            amplitudes='per_spectrum', amplitude_sigma=0.5))
