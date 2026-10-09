"""
Tests for the helpers for functions.
"""

import pytest

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


def test_load_function_from_file(tmp_path):
    path = tmp_path / 'functions.py'
    path.write_text('def double(x):\n    return 2 * x\n\nvalue = 3\n')
    assert funcutils.load_function_from_file(str(path), 'double')(4) == 8
    # Not a function, and a file that fails to run (with its error)
    with pytest.raises(RuntimeError, match="does not define a function"):
        funcutils.load_function_from_file(str(path), 'value')
    path.write_text('1 / 0\n')
    with pytest.raises(RuntimeError, match="error running") as info:
        funcutils.load_function_from_file(str(path), 'double')
    assert isinstance(info.value.__cause__, ZeroDivisionError)
