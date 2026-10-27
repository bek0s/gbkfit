"""
Tests for the making of observations: their observables from data, and
the checks of their options.
"""

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelMoments
from gbkfit.instrument import Instrument, LSFGauss
from gbkfit.observation import (
    Flat, LikelihoodGaussian, ObservationSky, PixelBrightness, PixelMoments,
    observation_parser)
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError


def test_observable_from_data():
    # The data give the observable its grid and the orders of its
    # moments, and its spectral axis covers their velocities
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))},
        step=2, rota=30)
    observable = PixelMoments.from_data(data, spec_step=5,
                                        spec_convention='relativistic')
    assert observable.grid() == data.grid()
    assert observable.orders() == (0, 1)
    assert observable.spec_rval() == 1500
    ObservationSky(
        observable, Flat(), data=data,
        likelihood=LikelihoodGaussian(missing_residual=3))


@pytest.mark.parametrize('dataset_class, observable_class, shape', [
    ('DatasetPixelSpectra', 'PixelSpectra', (10, 4, 4)),
    ('DatasetSlitSpectra', 'SlitSpectra', (10, 4))])
def test_spectra_keep_the_convention_of_their_data(
        dataset_class, observable_class, shape):
    # Spectra on an axis of log velocities (WAVE-LOG) make an observable of
    # their grid, its convention included
    import gbkfit.dataset
    import gbkfit.observation
    data = getattr(gbkfit.dataset, dataset_class)(
        Data(np.zeros(shape)), rest='6564.61 Angstrom', convention='log')
    observable = getattr(gbkfit.observation, observable_class).from_data(data)
    assert observable.grid() == data.grid()
    assert observable.grid().coords.convention == 'log'


def test_a_model_that_can_lack_values_needs_their_residual():
    # A velocity map: the model has no velocity where its flux is below
    # the mask cutoff, and the residual there is the user's choice
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))})
    observable = PixelMoments.from_data(data, spec_step=5,
                                        spec_convention='relativistic')
    with pytest.raises(ConfigError, match="option missing_residual"):
        ObservationSky(observable, Flat(), data=data)


def test_observable_options_are_checked():
    with pytest.raises(ConfigError, match="flux_floor is a fraction"):
        PixelMoments(size=(8, 8), flux_floor=1, spec_convention='relativistic')
    with pytest.raises(ConfigError, match="no spectral axis for an LSF"):
        ObservationSky(
            PixelBrightness(size=(8, 8)), Flat(),
            instrument=Instrument(lsf=LSFGauss(10)))


def test_observations_need_their_type():
    info = dict(
        line_of_sight=dict(type='flat'),
        observable=dict(type='pixel_brightness', size=[4, 4]))
    with pytest.raises(ConfigError, match="'type' is required.*'sky'"):
        observation_parser.load(info)
    observation = observation_parser.load(info | dict(type='sky'))
    assert observation_parser.dump(observation)['type'] == 'sky'


def test_instrument_warnings_have_the_path_of_the_configuration(caplog):
    # The instrument is checked when the observation is made, so its
    # warnings have the path of the observation
    with parseutils.config_path('observations'):
        observation_parser.load([dict(
            type='sky',
            line_of_sight=dict(type='flat'),
            observable=dict(
                type='pixel_spectra', size=[4, 4, 4], smooth_weights=True))])
    assert "observations[0] [sky]: smooth_weights is true" in caplog.text


@pytest.mark.parametrize('dtype', ['float16', 'int32', 'complex'])
def test_dtypes_without_kernels_are_errors(dtype):
    # (they failed later, in the native code)
    dataset = Data(np.ones((8, 8)))
    from gbkfit.dataset import DatasetPixelBrightness
    observable = PixelBrightness.from_data(DatasetPixelBrightness(dataset))
    with pytest.raises(ConfigError, match="dtype must be one of"):
        ObservationSky(observable, Flat(), dtype=dtype)


def test_ffts_are_float64_by_default_and_not_less_precise_than_models():
    observable = PixelBrightness(size=(8, 8))
    assert ObservationSky(observable, Flat()).fft_dtype() == np.float64
    with pytest.raises(ConfigError, match=r"fft_dtype \(float32\) is less"):
        ObservationSky(
            observable, Flat(), dtype='float64', fft_dtype='float32')


@pytest.mark.parametrize('options, message', [
    (dict(weights=dict(moment2=2)), r"does not have: \['moment2'\]"),
    (dict(upper_limits=dict(momnet0=dict(limit=1, error=0.1))),
     r"does not have: \['momnet0'\]")])
def test_likelihoods_name_items_of_their_observation(options, message):
    # A typo in the name of an item of the options of a likelihood would
    # leave the option out without a word
    data = DatasetPixelMoments(
        {0: Data(np.ones((8, 20))), 1: Data(np.full((8, 20), 1500.0))})
    with pytest.raises(ConfigError, match=message):
        ObservationSky(
            PixelMoments.from_data(
                data, spec_step=5, spec_convention='relativistic'),
            Flat(), data=data,
            likelihood=LikelihoodGaussian(missing_residual=3, **options))


def test_residuals_of_upper_limits_are_not_linear_in_the_model():
    # (so that no elements of the model are solved from them)
    from gbkfit.dataset import DatasetPixelSpectra
    from gbkfit.driver.drivers.host import DriverHost
    from gbkfit.objective.comparisons import comparison
    from gbkfit.observation import PixelSpectra
    data = DatasetPixelSpectra(Data(np.ones((10, 4, 4))))
    observation = ObservationSky(
        PixelSpectra.from_data(data), Flat(), data=data,
        likelihood=LikelihoodGaussian(
            upper_limits=dict(spectra=dict(limit=0.1, error=0.05))))
    comparison_ = comparison(observation, DriverHost())
    for form in (comparison_.linear_form, comparison_.pointwise_form):
        assert "has an upper limit" in form(['spectra'])
