"""
Tests for the parameter descriptions, and the parsing of the keys and the
expressions of parameter properties. Their behaviour as a whole is tested
in test_paramspace.py.
"""

import numpy as np
import pytest

from gbkfit.params.expressions import Expression, InvalidExpressionError
from gbkfit.params.keys import InvalidKeyError, parse_key
from gbkfit.params.pdescs import *


def describe(pdesc):
    """The properties of a parameter description."""
    return (
        type(pdesc), pdesc.name(), pdesc.size(), pdesc.desc(),
        pdesc.default(), pdesc.minimum(), pdesc.maximum())


def test_param_desc_scalar():
    pdesc = ParamScalarDesc('name', 'description', 10, -10)
    assert describe(pdesc) == (
        ParamScalarDesc, 'name', 1, 'description', 10, -10, np.inf)
    # The dump leaves out the infinite bounds, and loads to the same desc
    info = dict(
        type='scalar', name='name', desc='description',
        default=10, minimum=-10)
    assert pdesc.dump() == info
    assert describe(pdesc_parser.load(info)) == describe(pdesc)


def test_param_desc_vector():
    pdesc = ParamVectorDesc('name', 5, 'description', 10, -10)
    assert describe(pdesc) == (
        ParamVectorDesc, 'name', 5, 'description', 10, -10, np.inf)
    info = dict(
        type='vector', name='name', size=5, desc='description',
        default=10, minimum=-10)
    assert pdesc.dump() == info
    assert describe(pdesc_parser.load(info)) == describe(pdesc)


PDESCS = dict(a=ParamScalarDesc('a'), v=ParamVectorDesc('v', 12))


@pytest.mark.parametrize('key, name, indices, element', [
    ('a', 'a', None, False),
    (' a ', 'a', None, False),
    ('v', 'v', range(12), False),
    ('v[0]', 'v', [0], True),
    ('v[-1]', 'v', [11], True),
    ('v [ 2 ]', 'v', [2], True),
    ('v[2:5]', 'v', [2, 3, 4], False),
    ('v[::4]', 'v', [0, 4, 8], False),
    ('v[1::10]', 'v', [1, 11], False),
    ('v[-2:]', 'v', [10, 11], False),
    ('v[::-5]', 'v', [11, 6, 1], False),
    ('v[[0, 5, -1]]', 'v', [0, 5, 11], False),
    ('v[[3]]', 'v', [3], False)])
def test_keys(key, name, indices, element):
    parsed = parse_key(key, PDESCS)
    assert parsed.name == name
    assert parsed.element == element
    if indices is None:
        assert parsed.indices is None
    else:
        np.testing.assert_array_equal(parsed.indices, list(indices))


def test_keys_of_unknown_parameters():
    assert parse_key('b', PDESCS) is None
    assert parse_key('b[0]', PDESCS) is None


@pytest.mark.parametrize('key, reason', [
    ('1a', "invalid syntax"),
    ('a b', "invalid syntax"),
    ('v.x', "invalid syntax"),
    ('v[0][1]', "invalid syntax"),
    ('a[0]', "scalar"),
    ('v[12]', "out of range"),
    ('v[[0, 12]]', "out of range"),
    ('v[1.5]', "integer constants"),
    ('v[a]', "integer constants"),
    ('v[[]]', "integer constants"),
    ('v[03]', "invalid syntax")])
def test_invalid_keys(key, reason):
    with pytest.raises(InvalidKeyError, match=reason):
        parse_key(key, PDESCS)


@pytest.mark.parametrize('source, value', [
    ('1 + 2 * 3 - 4 / 2 + 7 // 2 + 7 % 2 + 2 ** 3 + -1', 16),
    ('a * 2', 6),
    ('v[0] + v[-1]', 11),
    ('np.sum(v[1:3])', 3),
    ('np.pi', np.pi),
    ('np.linalg.norm(v[:2])', 1),
    ('abs(-a) + min(a, 1) + max(a, 1) + round(1.6) + len(v)', 21),
    ('sum([1, 2]) + sum((3, 4))', 10),
    ('1 if a > 2 else 0', 1),
    ('1 if a > 2 and a <= 3 or not a else 0', 1),
    ('np.mean(v, axis=0)', 5.5)])
def test_expressions(source, value):
    namespace = dict(a=3.0, v=np.arange(12.0))
    expression = Expression(source, PDESCS)
    assert expression.evaluate(namespace) == pytest.approx(value)


@pytest.mark.parametrize('source, reason', [
    ("__import__('os')", "__import__"),
    ("open('file')", "open"),
    ("a.__class__", "not allowed"),
    # numpy holds modules (e.g. os) and functions with files and state
    ("np.f2py.os.getcwd()", "cannot be called"),
    ("np.ndarray.__subclasses__()", "cannot be called"),
    ("np.sin.__class__", "not allowed"),
    ("np.add.reduce(v)", "cannot be called"),
    ("np.loadtxt", "not allowed"),
    ("np.vectorize(np.sin)", "cannot be called"),
    ("v.sum()", "cannot be called"),
    ("'text'", "numbers"),
    ("True", "numbers"),
    ("lambda: 1", "not allowed"),
    ("[x for x in v]", "not allowed"),
    ("b + 1", "unknown name 'b'"),
    ("a[0]", "scalar"),
    ("v[12]", "out of range"),
    ("a +", "syntax")])
def test_invalid_expressions(source, reason):
    with pytest.raises(InvalidExpressionError, match=reason):
        Expression(source, PDESCS)


@pytest.mark.parametrize('source, reads', [
    ('1', {}),
    ('a', {'a': None}),
    ('v[0] + v[[2, 3]]', {'v': {0, 2, 3}}),
    ('v[1:3] + a', {'v': {1, 2}, 'a': None}),
    ('np.sum(v)', {'v': set(range(12))}),
    # An index that is not a constant can be any element
    ('v[np.int64(a)]', {'v': set(range(12)), 'a': None})])
def test_expression_reads(source, reads):
    assert Expression(source, PDESCS).reads == reads
