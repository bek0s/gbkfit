"""
Tests for gbkfit.utils.miscutils.
"""

import pytest

from gbkfit.utils import miscutils


def test_merge_with_prefixes():
    merged, mappings = miscutils.merge_with_prefixes(
        [dict(a=1, b=2), dict(a=3)], ['', 'cmp1_'])
    assert merged == dict(a=1, b=2, cmp1_a=3)
    assert mappings == (dict(a='a', b='b'), dict(a='cmp1_a'))


def test_merge_with_prefixes_rejects_repeated_names():
    # e.g. a component named 'disk' with the parameter 'bpt_a', and one
    # named 'disk_bpt' with the parameter 'a'
    with pytest.raises(RuntimeError, match=r"repeated: \['disk_bpt_a'\]"):
        miscutils.merge_with_prefixes(
            [dict(bpt_a=1), dict(a=2)], ['disk_', 'disk_bpt_'])
