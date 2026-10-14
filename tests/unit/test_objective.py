"""
Tests for the objective: the totals that the fitters use agree with the
residuals, and the residuals are new arrays.
"""

import numpy as np
import pytest

from gbkfit.dataset import Data, DatasetPixelBrightness, DatasetPixelMoments
from gbkfit.model import galaxy_parser
from gbkfit.objective import Objective
from gbkfit.observation import (
    Flat, LikelihoodGaussian, Observation, ObservationGroup, PixelBrightness,
    PixelMoments, observable_parser)
from gbkfit.params import ParamSpace
from gbkfit.utils.parseutils import ConfigError


GALAXY = dict(components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 14)),
    density=dict(ptraits=dict(type='exponential')),
    emission=dict(type='band'))])

PROPERTIES = dict(xpos=0.0, ypos=0.0, posa=50, incl=60, pt_a=1, pt_s=2)


def make_objective(driver, data, likelihood=None):
    dataset = DatasetPixelBrightness(data)
    observation = Observation(
        PixelBrightness.from_data(dataset), Flat(), driver, data=dataset,
        likelihood=likelihood)
    group = ObservationGroup(galaxy_parser.load(GALAXY), observation)
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
    assert objective.nresiduals() == 24 * 32 - 5 * 32
    residuals = objective.residuals(params).reshape(24, 32)
    np.testing.assert_array_equal(residuals[:5], 0)


def test_an_observation_without_data_is_an_error(driver):
    dataset = DatasetPixelBrightness(Data(np.ones((24, 32))))
    observation = Observation(
        PixelBrightness.from_data(dataset), Flat(), driver)
    group = ObservationGroup(galaxy_parser.load(GALAXY), observation)
    with pytest.raises(ConfigError, match="needs data"):
        Objective(group)


KINEMATICS = dict(components=[dict(
    type='smdisk', loose=False, tilted=False,
    rnodes=list(range(0, 30)),
    density=dict(ptraits=dict(type='exponential')),
    kinematics=dict(
        vptraits=dict(type='tan_arctan'),
        dptraits=dict(type='uniform')),
    emission=dict(type='lines'))])

DISK = dict(
    vsys=0, xpos=0, ypos=0, posa=40, incl=60, pt_a=1, pt_s=4, vpt_rt=2,
    vpt_vt=150, dpt_a=20)


def test_a_model_without_values_where_the_data_have_some_pays(driver):
    # A model can no longer lower the chi-square by having no flux (and so
    # no velocity) where the data have some: its moment 0 there is 0, and
    # its moments 1 and 2 count as misses of 3 errors (they counted 0)
    def group(observable, dataset=None):
        likelihood = None if dataset is None \
            else LikelihoodGaussian(missing_residual=3)
        observation = Observation(
            observable, Flat(), driver, data=dataset, likelihood=likelihood)
        return ObservationGroup(galaxy_parser.load(KINEMATICS), observation)
    image = group(observable_parser.load(dict(
        type='pixel_moments', size=[24, 24], orders=[0, 1, 2])))
    model = image.model_h(ParamSpace(image.pdescs(), DISK).evaluate())[0]
    errors = dict(moment0=0.01, moment1=5.0, moment2=5.0)
    dataset = DatasetPixelMoments({
        order: Data(model[f'moment{order}']['d'], error=np.full(
            (24, 24), errors[f'moment{order}']))
        for order in (0, 1, 2)})
    objective = Objective(group(PixelMoments.from_data(dataset), dataset))
    truth = ParamSpace(objective.pdescs(), DISK).evaluate()
    compact = ParamSpace(
        objective.pdescs(), DISK | dict(pt_s=0.1)).evaluate()
    compact_moment1 = objective.group().model_h(compact)[0]['moment1']['d']
    without = np.isnan(compact_moment1)
    has_data = dataset['moment1'].mask() != 0
    residuals = objective.residual_nddata_h(compact, False)[0]
    assert np.any(without & has_data)
    np.testing.assert_allclose(
        residuals['moment1'][without & has_data], 3)
    np.testing.assert_allclose(
        residuals['moment0'][without & has_data],
        (0 - dataset['moment0'].data()[without & has_data]) / 0.01,
        rtol=1e-4, atol=1e-3)
    assert objective.chi2(compact) > objective.chi2(truth)
