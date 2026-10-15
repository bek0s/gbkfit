"""
Tests for the making of observations: their observables from data, and
the checks of their options.
"""

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelMoments
from gbkfit.instrument import Instrument, LSFGauss
from gbkfit.observation import (
    Flat, LikelihoodGaussian, Observation, PixelBrightness, PixelMoments,
    observation_parser)
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


def test_observable_from_data():
    # The data give the observable its grid and the orders of its
    # moments, and its spectral axis covers their velocities; the driver
    # is the host by default
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))},
        step=2, rota=30)
    observable = PixelMoments.from_data(data, spec_step=5)
    assert observable.grid() == data.grid()
    assert observable.orders() == (0, 1)
    assert observable.spec_rval() == 1500
    observation = Observation(
        observable, Flat(), data=data,
        likelihood=LikelihoodGaussian(missing_residual=3))
    assert observation.driver().type() == 'host'


def test_a_model_that_can_lack_values_needs_their_residual():
    # A velocity map: the model has no velocity where its flux is below
    # the mask cutoff, and the residual there is the user's choice
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))})
    observable = PixelMoments.from_data(data, spec_step=5)
    with pytest.raises(ConfigError, match="option missing_residual"):
        Observation(observable, Flat(), data=data)


def test_observable_options_are_checked():
    with pytest.raises(ConfigError, match="mask_apply needs a mask_cutoff"):
        PixelBrightness(size=(8, 8), mask_apply=True)
    with pytest.raises(ConfigError, match="no spectral axis for an LSF"):
        Observation(
            PixelBrightness(size=(8, 8)), Flat(),
            instrument=Instrument(lsf=LSFGauss(10)))


def test_instrument_warnings_have_the_path_of_the_configuration(caplog):
    # The instrument is checked when the observation is made, so its
    # warnings have the path of the observation
    with parseutils.config_path('observations'):
        observation_parser.load([dict(
            line_of_sight=dict(type='flat'),
            observable=dict(
                type='pixel_spectra', size=[4, 4, 4], smooth_weights=True))])
    assert "observations[0]: smooth_weights is true" in caplog.text


@pytest.mark.parametrize('dtype', ['float16', 'int32', 'complex'])
def test_dtypes_without_kernels_are_errors(dtype):
    # (they failed later, in the native code)
    dataset = Data(np.ones((8, 8)))
    from gbkfit.dataset import DatasetPixelBrightness
    observable = PixelBrightness.from_data(DatasetPixelBrightness(dataset))
    with pytest.raises(ConfigError, match="dtype must be one of"):
        Observation(observable, Flat(), dtype=dtype)


def test_ffts_are_float64_by_default_and_not_less_precise_than_models():
    observable = PixelBrightness(size=(8, 8))
    assert Observation(observable, Flat()).fft_dtype() == np.float64
    with pytest.raises(ConfigError, match=r"fft_dtype \(float32\) is less"):
        Observation(observable, Flat(), dtype='float64', fft_dtype='float32')
