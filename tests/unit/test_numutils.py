"""
Tests for the helpers for numbers.
"""

import numpy as np
import pytest

from gbkfit.utils import numutils


@pytest.mark.parametrize('origin, expected', [
    (0, [1, 3, 6, 10]), (1, [3, 2, 5, 9]), (3, [10, 9, 7, 4]),
    (-1, [10, 9, 7, 4])])
def test_cumsum_from(origin, expected):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(numutils.cumsum_from(x, origin), expected)
    # In place
    numutils.cumsum_from(x, origin, out=x)
    np.testing.assert_array_equal(x, expected)


def test_cumsum_from_widens_small_integers_and_bools():
    # As in np.cumsum, the sums do not fit the type of the values
    np.testing.assert_array_equal(
        numutils.cumsum_from(np.array([100, 100, 100], np.int8), 0),
        [100, 200, 300])
    np.testing.assert_array_equal(
        numutils.cumsum_from(np.array([True, True, True]), 2), [3, 2, 1])


def test_cumsum_from_errors():
    with pytest.raises(IndexError, match="outside"):
        numutils.cumsum_from(np.ones(4), 4)
    with pytest.raises(ValueError, match="non-empty"):
        numutils.cumsum_from(np.ones(0), 0)


def test_nativify_makes_numpy_values_writable_as_json():
    import json
    x = dict(
        array=np.arange(3), flag=np.bool_(True), number=np.float32(1.5),
        nested=[(np.int64(2), np.str_('a'))])
    native = numutils.nativify(x)
    assert native == dict(array=[0, 1, 2], flag=True, number=1.5,
                          nested=[[2, 'a']])
    json.dumps(native)


def test_is_even_and_is_odd():
    for num in [0, 0.0, 2, 2.0, 2.5]:
        assert numutils.is_even(+num)
        assert numutils.is_even(-num)
        assert not numutils.is_odd(+num)
        assert not numutils.is_odd(-num)
    for num in [1, 1.0, 1.5]:
        assert not numutils.is_even(+num)
        assert not numutils.is_even(-num)
        assert numutils.is_odd(+num)
        assert numutils.is_odd(-num)


def test_round_functions():
    # round even
    assert numutils.roundu_even(0.0) == 0
    assert numutils.roundu_even(+3.0) == +4
    assert numutils.roundu_even(-3.0) == -2
    assert numutils.roundu_even(+3.5) == +4
    assert numutils.roundu_even(-3.5) == -2
    # round odd
    assert numutils.roundu_odd(0.0) == 1
    assert numutils.roundu_odd(+2.0) == +3
    assert numutils.roundu_odd(-2.0) == -1
    assert numutils.roundu_odd(+2.5) == +3
    assert numutils.roundu_odd(-2.5) == -1
    # round multiple
    assert numutils.roundu_multiple(+1.0, 2.0) == +2
    assert numutils.roundu_multiple(+5.0, 2.0) == +6
    assert numutils.roundu_multiple(-1.0, 2.0) == 0
    assert numutils.roundu_multiple(-5.0, 2.0) == -4
    # round power of two
    assert numutils.roundu_po2(3.0) == 4
    assert numutils.roundu_po2(3.5) == 4
