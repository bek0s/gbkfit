"""
Tests for gbkfit.utils.iterutils.
"""

import numpy as np
import pytest

from gbkfit.utils import iterutils


def test_merge_with_prefixes():
    merged, mappings = iterutils.merge_with_prefixes(
        [dict(a=1, b=2), dict(a=3)], ['', 'cmp1_'])
    assert merged == dict(a=1, b=2, cmp1_a=3)
    assert mappings == (dict(a='a', b='b'), dict(a='cmp1_a'))


def test_merge_with_prefixes_rejects_repeated_names():
    # e.g. a component named 'disk' with the parameter 'bpt_a', and one
    # named 'disk_bpt' with the parameter 'a'
    with pytest.raises(RuntimeError, match=r"repeated: \['disk_bpt_a'\]"):
        iterutils.merge_with_prefixes(
            [dict(bpt_a=1), dict(a=2)], ['disk_', 'disk_bpt_'])


def test_listify_takes_none_as_empty_and_wraps_other_objects():
    assert iterutils.listify(None) == []
    assert iterutils.tuplify('disk') == ('disk',)
    assert iterutils.listify(np.array([1, 2])) == [1, 2]
    assert iterutils.setify([1, 1, 2]) == {1, 2}


def test_helpers_of_order_and_indices():
    assert iterutils.duplicates([1, 2, 2, 3, 3, 3]) == {2, 3}
    assert iterutils.sorted_by_order(
        ['c', 'x', 'a'], ['a', 'b', 'c'], on_missing='end') == ['a', 'c', 'x']
    with pytest.raises(ValueError, match="not in the order"):
        iterutils.sorted_by_order(['x'], ['a'])
    assert iterutils.normalize_indices([0, -1], 4) == [0, 3]
    with pytest.raises(IndexError):
        iterutils.normalize_index(4, 4)
