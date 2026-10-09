"""
Tests for the helpers for functions.
"""

from gbkfit.utils import funcutils


def test_parameter_names_are_those_passed_by_name():
    # self, *args, **kwargs and positional-only parameters are left out

    class Thing:
        def __init__(self, a, /, b, c=1, *args, d, e=2, **kwargs):
            pass

    names = funcutils.parameter_names(Thing.__init__)
    assert names.required == ('b', 'd')
    assert names.optional == ('c', 'e')
    assert names.all == ('b', 'd', 'c', 'e')
