"""
Tests of correlated noise in cubes (see LikelihoodGauss): the noise
correlated on the sky and along the channels at once.
"""

import astropy.units as u
import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelSpectra, NoiseCorrelation
from gbkfit.grid import SpectralAxis
from gbkfit.instrument import LSFGauss, PSFGauss
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    MAX_CORRELATED_PIXELS, LikelihoodGauss, ObservationGroup,
    ObservationSky, PixelSpectra, line_of_sight_parser, whitening)
from gbkfit.params import ParamSpace


SIZE = (12, 12, 31)
STEP = (1, 1, 20)
SPECTRAL = SpectralAxis(step=STEP[2], convention='relativistic')
PARAMS = dict(
    xpos=0, ypos=0, posa=30, incl=60, vsys=0, rpt_a=1, rpt_s=3, vpt_rt=2,
    vpt_vt=150, dpt_a=30)
GALAXY = dict(components=dict(
    type='thin_disk', rnodes=list(range(0, 11)),
    rptraits=dict(type='exponential'),
    vptraits=dict(type='arctan', direction='tan'),
    dptraits=dict(type='uniform'),
    emission=dict(type='lines')))


def objective(driver, data, likelihood):
    """The objective of a disk seen in a cube, under a likelihood."""
    dataset = DatasetPixelSpectra(data, step=STEP[:2], spectral=SPECTRAL)
    group = ObservationGroup(
        [galaxy_parser.load(GALAXY)], [ObservationSky(
            PixelSpectra.from_data(dataset),
            line_of_sight_parser.load(dict(type='flat')), data=dataset,
            likelihood=likelihood)])
    return (Objective(group.plan(driver)),
            ParamSpace.from_properties(group.pdescs(), PARAMS).evaluate())


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
        spectral_correlation=LSFGauss(25 * u.km / u.s))


@pytest.mark.parametrize('blank', [[], [0, 17]])
def test_cubes_have_the_chi2_of_the_product_of_their_correlations(
        driver, blank):
    # chi2 = r^T (C_z' (x) C_xy')^-1 r over the kept spectra and channels
    # (channels blank in every spectrum are left out), each correlation
    # with the white floor f: C' = (1 - f) C + f I
    rng = np.random.default_rng(7)
    mask = np.ones(SIZE[::-1])
    mask[:, rng.integers(0, 12, 6), rng.integers(0, 12, 6)] = 0
    mask[blank] = 0
    data = noisy_cube(rng, mask)
    obj, params = objective(
        driver, data, LikelihoodGauss(white_floor=0.05))
    model = obj.group_plan().model_h(
        params)[0]['spectra'].data.astype(float)
    kept = np.any(mask != 0, axis=0)
    z = np.flatnonzero(np.any(mask != 0, axis=(1, 2)))
    r = ((model - data.data()) / data.error())[z][:, kept]
    y, x = np.nonzero(kept)
    correlation = obj.datasets()[0].noise_correlation('spectra')
    k_xy = correlation.spatial[11 + y[:, None] - y, 11 + x[:, None] - x]
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
    with pytest.raises(ValueError, match="mask_whole_spectra"):
        objective(driver, data, LikelihoodGauss())
    # The option masks the spectrum of the masked value
    whole = DatasetPixelSpectra(data, step=STEP[:2], spectral=SPECTRAL).mask_whole_spectra()
    np.testing.assert_array_equal(
        whole['spectra'].mask()[:, 5, 5], 0)
    assert np.count_nonzero(whole['spectra'].mask() == 0) == SIZE[2]
    objective(driver, whole['spectra'], LikelihoodGauss())


def test_whole_spectra_leave_out_the_channels_masked_everywhere(driver):
    # (a plane blank in every spectrum, e.g. flagged for RFI, masked every
    # spectrum, and so the whole cube)
    rng = np.random.default_rng(10)
    mask = np.ones(SIZE[::-1])
    mask[5] = 0
    mask[3, 5, 5] = 0
    dataset = DatasetPixelSpectra(
        noisy_cube(rng, mask), step=STEP[:2], spectral=SPECTRAL)
    whole = dataset.mask_whole_spectra()['spectra']
    masked = whole.mask() == 0
    assert np.all(masked[5]) and np.all(masked[:, 5, 5])
    assert np.count_nonzero(masked) == SIZE[0] * SIZE[1] + SIZE[2] - 1
    objective(driver, whole, LikelihoodGauss())


def test_noise_correlated_along_the_channels_alone_has_no_limit_of_pixels():
    # (the limit is of the dense matrix of a correlation on the sky)
    mask = np.ones((5, 150, 200))
    shape = np.array([0, 0, 0.1, 0.5, 1, 0.5, 0.1, 0, 0])
    spectral = whitening('spectra', mask, NoiseCorrelation(None, shape), 0.01)
    assert spectral.pixels.size == 30000 > MAX_CORRELATED_PIXELS
    spatial = NoiseCorrelation(np.ones((299, 399)), shape)
    with pytest.raises(ValueError, match="correlated on the sky over 30000"):
        whitening('spectra', mask, spatial, 0.01)


def test_cubes_of_correlated_noise_go_without_solved_scales(driver):
    from gbkfit.params import SolveGauss
    dataset = DatasetPixelSpectra(
        noisy_cube(np.random.default_rng(9), None), step=STEP[:2],
        spectral=SPECTRAL)
    group = ObservationGroup(
        [galaxy_parser.load(GALAXY)], [ObservationSky(
            PixelSpectra.from_data(dataset),
            line_of_sight_parser.load(dict(type='flat')), data=dataset,
            flux_scale='one', name='obs')])
    with pytest.raises(ValueError, match="has correlated noise"):
        Objective(group.plan(driver),
                  dict(obs_flux_scale=SolveGauss(1, 0.5, 0)))
