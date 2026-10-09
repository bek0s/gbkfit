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
    # (in place)
    numutils.cumsum_from(x, origin, out=x)
    np.testing.assert_array_equal(x, expected)


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
