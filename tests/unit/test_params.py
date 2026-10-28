"""
Tests for the parameter descriptions (and the units of those of the
models), and the parsing of the keys and the expressions of parameter
properties. Their behaviour as a whole is tested in test_paramspace.py.
"""

import pathlib

import numpy as np
import pytest

import gbkfit
from gbkfit.params._expressions import Expression, InvalidExpressionError
from gbkfit.params._keys import InvalidKeyError, parse_key
from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc, pdesc_parser
from gbkfit.utils.parseutils import ConfigError


def describe(pdesc):
    """The properties of a parameter description."""
    return (
        type(pdesc), pdesc.is_scalar(), pdesc.size(), pdesc.desc(),
        pdesc.minimum(), pdesc.maximum())


def test_param_desc_scalar():
    pdesc = ParamScalarDesc(desc='description', min=-10)
    assert describe(pdesc) == (
        ParamScalarDesc, True, 1, 'description', -10, np.inf)
    # The dump leaves out the infinite bounds, and loads to the same desc
    info = dict(type='scalar', desc='description', min=-10)
    assert pdesc.dump() == info
    assert describe(pdesc_parser.load(info)) == describe(pdesc)


def test_param_desc_vector():
    pdesc = ParamVectorDesc(5, desc='description', max=10)
    assert describe(pdesc) == (
        ParamVectorDesc, False, 5, 'description', -np.inf, 10)
    info = dict(type='vector', size=5, desc='description', max=10)
    assert pdesc.dump() == info
    assert describe(pdesc_parser.load(info)) == describe(pdesc)


@pytest.mark.parametrize('make, message', [
    (lambda: ParamVectorDesc(0), "'size' must be at least 1"),
    (lambda: ParamVectorDesc(-1), "'size' must be at least 1"),
    (lambda: ParamVectorDesc(1.5), "'size' must be of type int"),
    (lambda: ParamVectorDesc(True), "'size' must be of type int"),
    (lambda: ParamScalarDesc(min=1, max=0), "above its max")])
def test_invalid_param_descs(make, message):
    with pytest.raises(ConfigError, match=message):
        make()


GOLDEN_DIR = pathlib.Path(__file__).parents[1] / 'data' / 'golden'

# Golden configurations of most kinds of parameters, each with those of its
# parameters that have no unit: the light of components whose emission has
# none (in the units of the data: the amplitudes of their density traits
# and light profiles, the flux of points), the mass per unit of that
# light, and the jitter of a noise of data of no unit
UNITLESS_PARAMS = dict(
    blobs_maps_selection={'pt_a', 'pt1_a', 'pt2_a'},
    disk_cone_lines={'disk_pt_a', 'cone_pt_a'},
    dust_extinction_band=set(),
    equilibrium_heights_mond={'gas_pt_a', 'mass_gas_upsilon'},
    lens_foreground={'pt_a'},
    points_particles_band={'nucleus_flux'},
    profile_table_mass=set(),
    spiral_bar_flows={'gas_pt_a'},
    stars_orbits={'stars_i0', 'mass_lum_upsilon'},
    upper_limits_fitted_noise={'pt_a', 'halpha_noise_jitter_moment1'})


@pytest.mark.parametrize('case, unitless', UNITLESS_PARAMS.items())
def test_params_of_models_have_units(case, unitless):
    group = gbkfit.describe(str(GOLDEN_DIR / case / 'config.yaml')).group
    assert {name for name, pdesc in group.pdescs().items()
            if pdesc.unit() is None} == unitless


PDESCS = dict(a=ParamScalarDesc(), v=ParamVectorDesc(12))


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
    ('v[03]', "invalid syntax"),
    (3, "must be strings")])
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
    # out would write into a parameter, and where leaves values unset
    ("np.add(v, 1, out=v)", "keyword 'out'"),
    ("np.sin(v, where=v > 0)", "keyword 'where'"),
    ("'text'", "numbers"),
    ("True", "numbers"),
    ("lambda: 1", "not allowed"),
    ("[x for x in v]", "not allowed"),
    ("b + 1", "unknown name 'b'"),
    ("a[0]", "'a' is a scalar"),
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
