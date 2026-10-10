"""
Tests for the objective: the totals that the fitters use agree with the
residuals, and the residuals are new arrays.
"""

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelBrightness
from gbkfit.model import model_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    LikelihoodGaussian, Observation, ObservationGroup, PixelBrightness)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


MODEL = dict(type='intensity_2d', components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    bptraits=dict(type='exponential'))])

PROPERTIES = dict(xpos=0.0, ypos=0.0, posa=50, incl=60, bpt_a=1, bpt_s=2)


def make_objective(driver, data, likelihood=None):
    dataset = DatasetPixelBrightness(data)
    observation = Observation(
        PixelBrightness.from_data(dataset), driver, data=dataset,
        likelihood=likelihood)
    group = ObservationGroup(model_parser.load(dict(MODEL)), observation)
    params = ParamSpace(group.pdescs(), PROPERTIES).evaluate()
    return Objective(group), params


def test_totals_agree_with_the_residuals(driver):
    rng = np.random.default_rng(1)
    data = Data(rng.normal(0.1, 1, (24, 32)), error=np.full((24, 32), 0.5))
    objective, params = make_objective(
        driver, data, LikelihoodGaussian(weights=2.0))
    residuals = objective.residuals(params)
    assert residuals.dtype == np.float64
    assert objective.chi2(params) == pytest.approx(
        np.sum(residuals ** 2), rel=1e-5)
    assert objective.log_likelihood(params) == pytest.approx(
        -0.5 * objective.chi2(params))


def test_residuals_are_a_new_array_on_every_call(driver):
    # (scipy's least_squares keeps the arrays of earlier calls)
    data = Data(np.ones((24, 32)))
    objective, params = make_objective(driver, data)
    first = objective.residuals(params)
    second = objective.residuals(params)
    assert first is not second
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, second)


def test_masked_values_do_not_count(driver):
    mask = np.ones((24, 32))
    mask[:5] = 0
    data = Data(np.ones((24, 32)), mask=mask)
    objective, params = make_objective(driver, data)
    assert objective.nresiduals(params) == 24 * 32 - 5 * 32
    residuals = objective.residuals(params).reshape(24, 32)
    np.testing.assert_array_equal(residuals[:5], 0)


def test_an_observation_without_data_is_an_error(driver):
    dataset = DatasetPixelBrightness(Data(np.ones((24, 32))))
    observation = Observation(PixelBrightness.from_data(dataset), driver)
    group = ObservationGroup(model_parser.load(dict(MODEL)), observation)
    with pytest.raises(ConfigError, match="needs data"):
        Objective(group)
