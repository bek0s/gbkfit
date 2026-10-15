"""
Tests for the priors on several parameters: their residuals, and how the
parameter space keeps them.
"""

import numpy as np
import pytest

from gbkfit.fitting import joint_prior_residuals
from gbkfit.params import ParamScalarDesc, ParamSpace, ParamVectorDesc
from gbkfit.utils.parseutils import ConfigError


PDESCS = dict(a=ParamScalarDesc(), b=ParamScalarDesc(), v=ParamVectorDesc(5))
PROPERTIES = dict(a=1, b=2, v=0)
# (v = k^2: first differences 1, 3, 5, 7, second differences 2)
VALUES = dict(a=1.0, b=2.0, v=np.array([0.0, 1.0, 4.0, 9.0, 16.0]))


def residuals(info, constants=None):
    """The residuals of one prior at VALUES."""
    params = ParamSpace(
        PDESCS, PROPERTIES, constants=constants, priors=[info])
    (function,) = joint_prior_residuals(params)
    return function(VALUES)


def test_gauss_and_mvn_give_the_chi_square_of_their_gaussian():
    np.testing.assert_allclose(residuals(dict(
        type='gauss', expr='v[1:3] * a', mean=[1, 2], std=2)), [0, 1])
    cov = np.array([[4.0, 1.0], [1.0, 2.0]])
    r = residuals(dict(
        type='mvn', exprs=['a + b', 'b'], mean=[2, 1], cov=cov.tolist()))
    d = np.array([1.0, 1.0])
    assert r @ r == pytest.approx(d @ np.linalg.solve(cov, d))


def test_smooth_penalises_jumps_or_bends_however_the_points_are_spaced():
    smooth = dict(type='smooth', expr='v', std=2)
    np.testing.assert_allclose(
        residuals(smooth | dict(order=1)), [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(residuals(smooth | dict(order=2)), [1, 1, 1])
    # (evenly spaced positions change nothing)
    np.testing.assert_allclose(residuals(
        smooth | dict(order=2, positions='r'), dict(r=[0, 10, 20, 30, 40])),
        [1, 1, 1])
    # v is a straight line over the positions v (spaced 1, 3, 5, 7): no
    # bends, and the penalty of its slope of four even steps of 4
    line = smooth | dict(positions='v')
    np.testing.assert_allclose(residuals(line | dict(order=2)), 0, atol=1e-12)
    r = residuals(line | dict(order=1))
    assert r @ r == pytest.approx(4 * (4 / 2) ** 2)


def test_bound_penalises_only_beyond_its_bounds():
    # (differences 1, 3, 5, 7: one below 2, one above 6, each by 1)
    np.testing.assert_allclose(residuals(dict(
        type='bound', expr='np.diff(v)', lower=2, upper=6, std=0.5)),
        [2, 0, 0, 0, 0, 0, 0, 2])


def test_the_parameter_space_keeps_its_priors():
    info = dict(type='smooth', expr='v', order=1, std=2)
    params = ParamSpace(PDESCS, PROPERTIES, priors=[info])
    assert params.priors() == [info]
    assert ParamSpace.load(params.dump(), PDESCS).priors() == [info]
    assert params.with_values({}).priors() == [info]
    with pytest.raises(ConfigError, match="must be dicts"):
        ParamSpace(PDESCS, PROPERTIES, priors=['smooth'])


@pytest.mark.parametrize('info, message', [
    (dict(type='gauss', expr='c', mean=0, std=1), "invalid expression"),
    (dict(type='gauss', expr='a', mean=0, std=0), "std"),
    (dict(type='smooth', expr='v', order=3, std=1), "order"),
    (dict(type='bound', expr='a', std=1), "lower, upper"),
    (dict(type='mvn', exprs=['a', 'b'], mean=[0, 0], cov=[[1, 2], [2, 1]]),
     "positive definite"),
])
def test_invalid_priors_are_config_errors(info, message):
    with pytest.raises(ConfigError, match=message):
        residuals(info)
