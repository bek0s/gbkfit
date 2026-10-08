"""
Tests for the parsing of configurations.
"""

import pytest

from gbkfit.utils import parseutils


DIMENSIONAL = dict(size=int, step=int | float)


@pytest.mark.parametrize('info, sanitized', [
    (dict(size=4, step=0.5), dict(size=[4, 4, 4], step=[0.5, 0.5, 0.5])),
    (dict(size=[1, 2, 3]), dict(size=[1, 2, 3])),
    (dict(size=[1, 2, 3, 4]), dict(size=[1, 2, 3])),
    (dict(size=None), dict(size=None)),
    (dict(), dict())])
def test_sanitize_dimensional_options(info, sanitized):
    parseutils.sanitize_dimensional_options(info, DIMENSIONAL, 3)
    assert info == sanitized


@pytest.mark.parametrize('info', [
    dict(size=[1, 2]),
    dict(size=1.5),
    dict(step='a')])
def test_sanitize_dimensional_options_errors(info):
    with pytest.raises(RuntimeError, match=f"option '{next(iter(info))}'"):
        parseutils.sanitize_dimensional_options(info, DIMENSIONAL, 3)
