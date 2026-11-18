"""Tests of the search for fixed points (gbkfit.model._fixed_point)."""

import numpy as np

from gbkfit.model._fixed_point import fixed_point


def largest(x, mapped):
    return float(np.max(np.abs(mapped - x)))


def test_a_contracting_map_settles_on_its_fixed_point():
    # x = cos(x), each component: Picard takes about 60 passes to 1e-12
    result = fixed_point(np.cos, np.zeros(3), largest, 1e-12, 100)
    assert result.settled and not result.growing
    assert result.passes < 15
    np.testing.assert_allclose(result.x, 0.7390851332151607, rtol=1e-11)


def test_a_map_that_does_not_contract_settles_where_picard_diverges():
    # (a slope of -1.5: Picard oscillates and grows; the secant does not)
    def g(x):
        return 1 - 1.5 * x
    result = fixed_point(g, np.zeros(2), largest, 1e-12, 100)
    assert result.settled
    np.testing.assert_allclose(result.x, 0.4, rtol=1e-11)


def test_the_fixed_point_is_the_same_every_time():
    def g(x):
        return np.cos(x) + 0.1 * x[::-1]
    first = fixed_point(g, np.zeros(4), largest, 1e-12, 100)
    again = fixed_point(g, np.zeros(4), largest, 1e-12, 100)
    np.testing.assert_array_equal(first.x, again.x)
    assert first.passes == again.passes


def test_a_map_without_a_fixed_point_does_not_settle():
    # (x = x + 1 + x^2 has no real solution)
    def g(x):
        return x + 1 + x ** 2
    result = fixed_point(g, np.zeros(1), largest, 1e-12, 50)
    assert not result.settled
    assert result.passes <= 50
