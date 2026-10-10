"""
Tests for the making of observations: their observables from data, and
the checks of their options.
"""

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelMoments
from gbkfit.instrument import Instrument, LSFGauss
from gbkfit.observation import (
    Observation, PixelBrightness, PixelMoments, observation_parser)
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
    observation = Observation(observable, data=data)
    assert observation.driver().type() == 'host'


def test_observable_options_are_checked():
    with pytest.raises(ConfigError, match="mask_apply needs a mask_cutoff"):
        PixelBrightness(size=(8, 8), mask_apply=True)
    with pytest.raises(ConfigError, match="no spectral axis for an LSF"):
        Observation(
            PixelBrightness(size=(8, 8)),
            instrument=Instrument(lsf=LSFGauss(10)))


def test_instrument_warnings_have_the_path_of_the_configuration(caplog):
    # The instrument is checked when the observation is made, so its
    # warnings have the path of the observation
    with parseutils.config_path('observations'):
        observation_parser.load([dict(observable=dict(
            type='pixel_spectra', size=[4, 4, 4], smooth_weights=True))])
    assert "observations[0]: smooth_weights is true" in caplog.text
