"""
The behaviour of the parameters of the configurations: which elements
are free, tied and fixed, the values they evaluate to, and the errors.

Each case goes through the same path as a configuration file
(param_space_parser), or makes a ParamSpace in Python.
"""

import re

import numpy as np
import pytest

from gbkfit.params import (
    ParamEvaluationError, ParamFunctions, ParamKind, ParamModeIncrements,
    ParamModeOffsets, ParamScalarDesc, ParamSpace, ParamVectorDesc,
    param_space_parser)
from gbkfit.utils.parseutils import ConfigError


def make_pdescs(**sizes):
    """Parameter descriptions: a scalar for None, else a vector."""
    return {
        name: ParamScalarDesc() if size is None else ParamVectorDesc(size)
        for name, size in sizes.items()}


def load_params(pdescs, properties, transforms=None):
    info = dict(properties=properties)
    if transforms:
        info['transforms'] = transforms
    return param_space_parser.load(info, pdescs=pdescs)


def evaluate(pdescs, properties, transforms=None):
    return load_params(pdescs, properties, transforms).evaluate()


def write_transforms(path, source):
    """Write a transforms function to a file, and return its config."""
    path.write_text(source)
    return dict(file=str(path), func='transforms')


S = None

# Evaluation: the parameters, their properties, and their values
EVALUATION_CASES = dict(
    scalar=(
        make_pdescs(a=S), {'a': 1.5},
        {'a': 1.5}),
    vector_from_number=(
        make_pdescs(v=3), {'v': 2},
        {'v': [2, 2, 2]}),
    vector_from_list=(
        make_pdescs(v=3), {'v': [1, 2, 3]},
        {'v': [1, 2, 3]}),
    elements=(
        make_pdescs(v=3), {'v[0]': 1, 'v[1]': 2, 'v[2]': 3},
        {'v': [1, 2, 3]}),
    negative_index_and_slice=(
        make_pdescs(v=3), {'v[-1]': 7, 'v[:-1]': [1, 2]},
        {'v': [1, 2, 7]}),
    slice_with_step=(
        make_pdescs(v=6), {'v[::2]': 1, 'v[1::2]': 2},
        {'v': [1, 2, 1, 2, 1, 2]}),
    slice_with_step_of_ten=(
        make_pdescs(v=12), {'v[::10]': 5, 'v[1:10]': 2, 'v[11]': 3},
        {'v': [5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 3]}),
    index_list=(
        make_pdescs(v=4), {'v[[0, 2]]': 1, 'v[[1, 3]]': 'v[0] + 5'},
        {'v': [1, 6, 1, 6]}),
    white_space_in_keys=(
        make_pdescs(a=S, v=2), {'a ': 1, 'v[ 0 ]': 2, 'v [1]': 3},
        {'a': 1, 'v': [2, 3]}),
    expression=(
        make_pdescs(a=S, b=S), {'a': 1, 'b': 'a + 1'},
        {'a': 1, 'b': 2}),
    expressions_in_any_order=(
        make_pdescs(a=S, b=S, c=S), {'c': 'a + b', 'a': 1, 'b': '1 + 1'},
        {'a': 1, 'b': 2, 'c': 3}),
    expressions_between_elements=(
        make_pdescs(v=3),
        {'v[0]': 1, 'v[1]': 'v[0] + 1', 'v[2]': 'v[1] * 10'},
        {'v': [1, 2, 20]}),
    scalar_expression_for_vector=(
        make_pdescs(a=S, v=3), {'a': 2, 'v': 'a * 2'},
        {'a': 2, 'v': [4, 4, 4]}),
    vector_expression=(
        make_pdescs(a=S, v=3), {'a': 2, 'v': 'a * np.arange(3)'},
        {'a': 2, 'v': [0, 2, 4]}),
    list_of_numbers_and_expressions=(
        make_pdescs(a=S, v=3), {'a': 2, 'v': [1, 'a', 'a * 3']},
        {'a': 2, 'v': [1, 2, 6]}),
    expression_of_vector=(
        make_pdescs(a=S, v=3), {'v': [1, 2, 3], 'a': 'np.sum(v)'},
        {'a': 6, 'v': [1, 2, 3]}),
    expression_of_slice=(
        make_pdescs(a=S, v=4), {'v': [1, 2, 3, 4], 'a': 'np.sum(v[1:3])'},
        {'a': 5, 'v': [1, 2, 3, 4]}),
    expression_with_numpy=(
        make_pdescs(a=S, b=S),
        {'a': 1, 'b': 'np.degrees(np.arctan(a)) + abs(-1)'},
        {'a': 1, 'b': 46}),
    value_of_fitting_properties=(
        make_pdescs(a=S, v=2),
        {'a': {'value': 1, 'min': 0, 'max': 2}, 'v': [{'value': 4}, 3]},
        {'a': 1, 'v': [4, 3]}),
    spread_fitting_properties=(
        make_pdescs(v=3), {'v': {'*value': [1, 2, 3], 'min': 0}},
        {'v': [1, 2, 3]}),
)


@pytest.mark.parametrize(
    'pdescs, properties, desired',
    EVALUATION_CASES.values(), ids=EVALUATION_CASES)
def test_evaluation(pdescs, properties, desired):
    values = evaluate(pdescs, properties)
    assert values.keys() == desired.keys()
    for name, value in desired.items():
        np.testing.assert_array_equal(values[name], value, err_msg=name)


def test_transforms(tmp_path):
    # Parameters set to None are tied by a user function
    transforms = write_transforms(
        tmp_path / 'transforms.py',
        "def transforms(params):\n"
        "    params['b'] = 2 * params['a']\n"
        "    params['v'][1:] = params['v'][0] * np.array([2, 3])\n"
        "\n"
        "import numpy as np\n")
    pdescs = make_pdescs(a=S, b=S, v=3)
    properties = {'a': 3, 'b': None, 'v[0]': 1, 'v[1:]': None}
    values = evaluate(pdescs, properties, transforms)
    assert values['a'] == 3
    assert values['b'] == 6
    np.testing.assert_array_equal(values['v'], [1, 2, 3])


def test_dump_keeps_the_transforms(tmp_path):
    transforms = write_transforms(
        tmp_path / 'transforms.py',
        "def transforms(params):\n"
        "    params['b'] = 2 * params['a']\n")
    pdescs = make_pdescs(a=S, b=S)
    info = load_params(pdescs, {'a': 3, 'b': None}, transforms).dump()
    assert info == dict(properties={'a': 3, 'b': None}, transforms=transforms)
    assert param_space_parser.load(info, pdescs=pdescs).evaluate()['b'] == 6
    # A function that is not at the top level of a file cannot be dumped
    params = ParamSpace(
        pdescs, {'a': 3, 'b': None}, transforms=lambda p: p.update(b=1))
    with pytest.raises(ConfigError, match="cannot be dumped"):
        params.dump()


def test_dump_gives_prior_objects_as_configurations():
    # (Python callers can give a Prior object; result.json needs its
    # configuration, which loads back)
    import json
    from gbkfit.params import PriorGauss
    pdescs = make_pdescs(a=S, v=2)
    params = ParamSpace(pdescs, {
        'a': dict(value=1, prior=PriorGauss(1, 2)),
        'v': [dict(value=0, prior=PriorGauss(0, 1)), 2]})
    info = json.loads(json.dumps(params.dump()))
    assert info['properties']['a']['prior'] == dict(type='gauss', mean=1, std=2)
    loaded = param_space_parser.load(info, pdescs=pdescs)
    assert loaded.dump() == info


def test_element_values():
    pdescs = make_pdescs(a=S, v=2)
    params = load_params(pdescs, {'a': 1, 'v': 'a + np.arange(2)'})
    values = params.evaluate()
    assert params.element_values(values) == {'a': 1, 'v[0]': 1, 'v[1]': 2}


# Evaluation errors: the parameters, their properties, and a pattern the
# error message must contain (which names the problem)
ERROR_CASES = dict(
    missing_parameter=(
        make_pdescs(a=S, b=S), {'a': 1},
        r"'b'"),
    bad_key=(
        make_pdescs(a=S), {'a': 1, '1a': 2},
        r"'1a'"),
    scalar_with_subscript=(
        make_pdescs(a=S, b=S), {'a': 1, 'b[0]': 2},
        r"'b\[0\]'"),
    index_out_of_range=(
        make_pdescs(v=3), {'v[:2]': 1, 'v[3]': 2},
        r"'v\[3\]'"),
    repeated_element=(
        make_pdescs(v=3), {'v': 1, 'v[0]': 2},
        r"'v\[0\]'"),
    list_of_wrong_length=(
        make_pdescs(v=3), {'v': [1, 2]},
        r"'v'"),
    expression_syntax=(
        make_pdescs(a=S, b=S), {'a': 1, 'b': 'a +'},
        r"'b'"),
    expression_with_subscript_of_scalar=(
        make_pdescs(a=S, b=S, c=S), {'a': 1, 'b': 'a + 1', 'c': 'b[0]'},
        r"b\[0\]"),
    expression_with_unknown_name=(
        make_pdescs(a=S), {'a': 'zzz + 1'},
        r"zzz"),
    expression_cycle=(
        make_pdescs(a=S, b=S), {'a': 'b', 'b': 'a'},
        r"(?i)circular|cycle"),
    expression_not_finite=(
        make_pdescs(a=S, b=S), {'a': 0, 'b': 'np.log(a)'},
        r"finite"),
    expression_of_wrong_size=(
        make_pdescs(v=3), {'v': 'np.arange(2)'},
        r"size 2.*size 3"),
    vector_expression_for_scalar=(
        make_pdescs(a=S), {'a': 'np.arange(2)'},
        r"scalar"),
    boolean_value=(
        make_pdescs(a=S), {'a': True},
        r"'a'"),
    none_without_transforms=(
        make_pdescs(a=S), {'a': None},
        r"None"),
    fitting_properties_without_value=(
        make_pdescs(a=S), {'a': {'min': 0}},
        r"'a'"),
    python_beyond_numpy=(
        make_pdescs(a=S), {'a': "__import__('os').getpid() * 0"},
        r"__import__"),
    expression_writing_into_a_parameter=(
        make_pdescs(a=S, v=3),
        {'v': [1, 2, 3], 'a': 'np.add(v, 10, out=v)[0]'},
        r"keyword 'out'"),
    subscript_of_a_prefixed_scalar=(
        make_pdescs(hi_a=S, b=S), {'hi_a': 1, 'b': 'hi_a[0]'},
        r"'hi_a' is a scalar"),
    name_of_a_builtin=(
        make_pdescs(max=S), {'max': 1},
        r"'max' has an invalid name: expressions use it"),
    name_of_a_keyword=(
        make_pdescs(**{'lambda': S}), {'lambda': 1},
        r"'lambda' has an invalid name: it is a Python keyword"),
    key_that_is_not_a_string=(
        make_pdescs(a=S), {'a': 1, 3: 2},
        r"keys must be strings"),
)


@pytest.mark.parametrize(
    'pdescs, properties, pattern',
    ERROR_CASES.values(), ids=ERROR_CASES)
def test_evaluation_error(pdescs, properties, pattern):
    with pytest.raises(Exception) as error:
        evaluate(pdescs, properties)
    assert re.search(pattern, str(error.value))


def test_expressions_and_transforms_are_exclusive(tmp_path):
    transforms = write_transforms(
        tmp_path / 'transforms.py',
        "def transforms(params):\n"
        "    params['c'] = 1\n")
    pdescs = make_pdescs(a=S, b=S, c=S)
    with pytest.raises(Exception, match="mutually exclusive"):
        evaluate(pdescs, {'a': 1, 'b': 'a', 'c': None}, transforms)


# The parameters and properties of a fit, with free parameters (their
# properties are dicts), tied parameters and fixed parameters
FIT_PDESCS = make_pdescs(a=S, b=S, c=S, v=3, w=2)

FIT_PROPERTIES = {
    'a': 1,
    'b': 'a * 2',
    'c': {'value': 1, 'min': 0, 'max': 2},
    'v': [{'value': 1}, 'v[0] * 2', 3],
    'w': {'*value': [1, 2], 'max': 5}}


def test_fit_classification():
    params = ParamSpace(FIT_PDESCS, FIT_PROPERTIES)
    assert params.kinds() == {
        'a': ParamKind.FIXED, 'b': ParamKind.TIED, 'c': ParamKind.FREE,
        'v[0]': ParamKind.FREE, 'v[1]': ParamKind.TIED,
        'v[2]': ParamKind.FIXED, 'w[0]': ParamKind.FREE,
        'w[1]': ParamKind.FREE}
    assert params.free_properties() == {
        'c': {'value': 1, 'min': 0, 'max': 2},
        'v[0]': {'value': 1},
        'w[0]': {'value': 1, 'max': 5},
        'w[1]': {'value': 2, 'max': 5}}


def test_fit_evaluation():
    params = ParamSpace(FIT_PDESCS, FIT_PROPERTIES)
    values = params.evaluate({'c': 1.5, 'v[0]': 2, 'w[0]': 3, 'w[1]': 4})
    assert values['a'] == 1
    assert values['b'] == 2
    assert values['c'] == 1.5
    np.testing.assert_array_equal(values['v'], [2, 4, 3])
    np.testing.assert_array_equal(values['w'], [3, 4])


@pytest.mark.parametrize('free, pattern', [
    ({'v[0]': 2, 'w[0]': 3, 'w[1]': 4}, r"'c'"),
    ({'a': 1, 'c': 1.5, 'v[0]': 2, 'w[0]': 3, 'w[1]': 4}, r"'a'")])
def test_fit_evaluation_needs_the_free_parameters(free, pattern):
    # Every free parameter must be given, and only those
    params = ParamSpace(FIT_PDESCS, FIT_PROPERTIES)
    with pytest.raises(ValueError) as error:
        params.evaluate(free)
    assert re.search(pattern, str(error.value))


def test_evaluation_does_not_keep_state():
    # The values returned are new arrays, and changing them does not
    # change the next evaluation
    params = ParamSpace(FIT_PDESCS, FIT_PROPERTIES)
    free = {'c': 1.5, 'v[0]': 2, 'w[0]': 3, 'w[1]': 4}
    first = params.evaluate(free)
    first['v'][:] = -1
    second = params.evaluate(free)
    np.testing.assert_array_equal(second['v'], [2, 4, 3])


def test_constants():
    # Expressions can use constants (e.g. the radial nodes of a disk)
    space = ParamSpace(
        make_pdescs(v=3), {'v': 'rnodes * 2'},
        constants=dict(rnodes=[0, 1, 2]))
    np.testing.assert_array_equal(space.evaluate()['v'], [0, 2, 4])


def test_constants_cannot_have_the_names_of_parameters():
    with pytest.raises(ConfigError, match="constants cannot have the names"):
        ParamSpace(
            make_pdescs(a=S), {'a': 1}, constants=dict(a=1))


def test_unknown_parameters_are_errors_or_ignored_in_lenient_mode():
    from gbkfit.utils import parseutils
    properties = {'a': 1, 'old': 2, 'old[0]': 3}
    with pytest.raises(Exception, match=r"unknown parameters: 'old'"):
        ParamSpace(make_pdescs(a=S), properties)
    with parseutils.strict_mode(False):
        assert evaluate(make_pdescs(a=S), properties) == {'a': 1}


def load_moded_params(properties, modes):
    """The params of a vector v of 4 elements and scalars s and t."""
    return param_space_parser.load(
        dict(properties=properties, modes=modes),
        pdescs=make_pdescs(v=4, s=None, t=None))


@pytest.mark.parametrize('mode, expected', [
    # Each element an offset from the value at the origin
    (dict(type='offsets', origin=1), [3, 2, 5, 6]),
    # Each element an increment over its neighbour towards the origin
    (dict(type='increments', origin=1), [3, 2, 5, 9])])
def test_modes_decode_the_values(mode, expected):
    properties = dict(v=[1, 2, 3, 4], s=0, t=0)
    params = load_moded_params(properties, dict(v=mode))
    np.testing.assert_array_equal(params.evaluate()['v'], expected)
    # The dump has the modes, and loads to the same values
    info = params.dump()
    assert info['modes'] == dict(v=mode)
    reloaded = param_space_parser.load(
        info, pdescs=make_pdescs(v=4, s=None, t=None))
    np.testing.assert_array_equal(reloaded.evaluate()['v'], expected)


def test_expressions_read_the_decoded_values():
    # The elements of v are coded, one of them tied to s, and the
    # expression of t reads the decoded values of v
    params = load_moded_params(
        dict(v=[10, 1, 2, 's * 3'], s=1, t='v[3]'),
        dict(v=dict(type='offsets')))
    values = params.evaluate()
    np.testing.assert_array_equal(values['v'], [10, 11, 12, 13])
    assert values['t'] == 13


@pytest.mark.parametrize('properties, modes, pattern', [
    (dict(), dict(s=dict(type='offsets')), "only vector parameters"),
    (dict(), dict(w=dict(type='offsets')), "unknown parameter: 'w'"),
    (dict(), dict(v=dict(type='offsets', origin=4)),
     "an index of the 4 elements"),
    (dict(), dict(v=dict(type='offsets', origin=1.5)),
     "'origin' must be of type int"),
    # An element of v cannot read v, whose decoded values need it
    (dict(v=[1, 2, 3, 'v[0]']), dict(v=dict(type='offsets')),
     "in a cycle: the mode of 'v'")])
def test_mode_errors(properties, modes, pattern):
    with pytest.raises(Exception, match=pattern):
        load_moded_params(dict(v=[1, 2, 3, 4], s=0, t=0) | properties, modes)


SOLVE = dict(type='gaussian', mean=1, sigma=0.5, minimum=0)


def test_solved_elements_take_their_values_without_data_or_given():
    pdescs = dict(
        s=ParamScalarDesc(minimum=0), v=ParamVectorDesc(3, minimum=0))
    params = load_params(pdescs, {
        's': dict(solve=SOLVE),
        'v[:2]': dict(solve=SOLVE | dict(mean=-1)),
        'v[2]': 2})
    assert params.kinds() == {
        's': ParamKind.SOLVED, 'v[0]': ParamKind.SOLVED,
        'v[1]': ParamKind.SOLVED, 'v[2]': ParamKind.FIXED}
    solves = params.solves()
    assert list(solves) == ['s', 'v[0]', 'v[1]']
    assert solves['v[0]'] is solves['v[1]']
    assert params.free_properties() == {}
    # (without data, the mean, or the minimum below it)
    values = params.evaluate()
    assert values['s'] == 1
    np.testing.assert_array_equal(values['v'], [0, 0, 2])
    values = params.evaluate(solved={'s': 1.5, 'v[1]': 0.5})
    assert values['s'] == 1.5
    np.testing.assert_array_equal(values['v'], [0, 0.5, 2])
    with pytest.raises(ValueError, match=r"not solved: \['v\[2\]'\]"):
        params.evaluate(solved={'v[2]': 1})
    assert load_params(pdescs, params.dump()['properties']).kinds() \
        == params.kinds()


@pytest.mark.parametrize('properties, modes, pattern', [
    (dict(s=dict(solve=SOLVE, value=1)), {}, r"its 'solve' alone"),
    (dict(s=dict(solve=dict(type='gaussian', mean=1, sigma=0))), {},
     "sigma must be finite and positive"),
    (dict(s=dict(solve=SOLVE | dict(minimum=-1))), {},
     r"gives values in \[-1.0, inf\], outside its valid range"),
    (dict(s=dict(solve=dict(type='gaussian', mean=1, sigma=1))), {},
     "outside its valid range"),
    (dict(v=dict(solve=SOLVE)), dict(v=dict(type='offsets')),
     "with a mode cannot have solved elements"),
    (dict(t='s * 2'), {}, r"'t' reads \['s'\], which have solved elements")])
def test_solved_elements_refused(properties, modes, pattern):
    pdescs = dict(
        s=ParamScalarDesc(minimum=0), t=ParamScalarDesc(),
        v=ParamVectorDesc(3, minimum=0))
    info = dict(
        properties=dict(s=dict(solve=SOLVE), t=1, v=1) | properties,
        modes=modes)
    with pytest.raises(ConfigError, match=pattern):
        param_space_parser.load(info, pdescs=pdescs)


def test_expressions_cannot_read_solved_elements():
    params = load_params(
        dict(s=ParamScalarDesc(minimum=0)), dict(s=dict(solve=SOLVE)))
    with pytest.raises(ConfigError, match="it has solved elements"):
        params.expression('s * 2')


def test_error_types():
    # Invalid properties are configuration errors; failed evaluations are
    # evaluation errors
    with pytest.raises(ConfigError, match="have no value"):
        ParamSpace(make_pdescs(a=S), {})
    space = ParamSpace(make_pdescs(a=S, b=S), {'a': 1, 'b': 'np.inf * a'})
    with pytest.raises(ParamEvaluationError, match="finite"):
        space.evaluate()


def test_valid_ranges():
    pdescs = dict(
        s=ParamScalarDesc(minimum=0),
        v=ParamVectorDesc(3, minimum=0, maximum=1))
    # A fixed value outside its range is a configuration error
    with pytest.raises(ConfigError, match=r"'v\[1\]' is 2.0, outside"):
        ParamSpace(pdescs, {'s': 1, 'v': [0, 2, 1]})
    # An evaluated one fails its evaluation
    space = ParamSpace(pdescs, {'s': {'value': 1}, 'v': 's * 0.5'})
    space.evaluate()
    with pytest.raises(ParamEvaluationError, match=r"'s' is -1.0"):
        space.evaluate({'s': -1})
    # The values of a parameter with a mode are checked once decoded
    space = ParamSpace(
        pdescs, {'s': 1, 'v': [0.5, 0.2, 0.7]},
        modes=dict(v=ParamModeOffsets()))
    with pytest.raises(ParamEvaluationError, match=r"'v\[2\]' is 1.2"):
        space.evaluate()


def test_with_values_starts_the_free_elements_there():
    # (e.g. at the solution of a fit, to start another one): through
    # every form of key, leaving the fixed and tied elements alone
    pdescs = make_pdescs(a=None, b=None, v=4, w=3)
    params = ParamSpace(pdescs, {
        'a': dict(value=1, min=0, max=5),
        'b': 'a * 2',
        'v[0]': dict(value=2),
        'v[1:]': {'value': 0, '*min': [-1, -2, -3]},
        'w': [dict(value=1), 7.0, 'w[0] + 1']})
    start = params.with_values(
        {'a': 3, 'b': 99, 'v[0]': 5, 'v[1]': 6, 'v[2]': 7, 'v[3]': 8,
         'w[0]': 4, 'w[1]': 99})
    values = start.evaluate()
    assert (values['a'], values['b']) == (3, 6)
    np.testing.assert_array_equal(values['v'], [5, 6, 7, 8])
    np.testing.assert_array_equal(values['w'], [4, 7, 5])
    # The other properties stay (the bounds, and those of the elements)
    assert start.properties()['a'] == dict(value=3, min=0, max=5)
    assert start.properties()['v[1:]'] == {
        '*min': [-1, -2, -3], '*value': [6, 7, 8]}
    assert start.free_properties()['v[3]'] == dict(value=8, min=-3)


@pytest.mark.parametrize('mode', [
    ParamModeOffsets(origin=1), ParamModeIncrements(origin=2)])
def test_with_values_codes_the_values_of_parameters_with_a_mode(mode):
    pdescs = make_pdescs(v=4)
    params = ParamSpace(pdescs, {'v': {'*value': [1, 2, 3, 4]}},
                        modes=dict(v=mode))
    decoded = [10, 12, 15, 11]
    start = params.with_values(
        {f'v[{i}]': value for i, value in enumerate(decoded)})
    np.testing.assert_allclose(start.evaluate()['v'], decoded)
    with pytest.raises(ValueError, match="give all its values"):
        params.with_values({'v[0]': 10})


class Functions(ParamFunctions):
    """
    Functions of a model: vcirc(R) = a R, of the parameter a, and the
    number G = 2; they count their bindings.
    """

    def __init__(self):
        self.nbinds = 0

    def names(self):
        return ('vcirc', 'G')

    def bind(self, values):
        self.nbinds += 1
        return dict(vcirc=lambda R: values['a'] * np.asarray(R), G=2.0)


def derived_params(derived, properties=None, functions=None):
    """Parameters a and b, with derived values (see Functions)."""
    return param_space_parser.load(
        dict(properties=properties or dict(a=3, b={'value': 2}),
             derived=derived),
        pdescs=make_pdescs(a=S, b=S),
        functions=functions or Functions())


def test_derived_values():
    # Each derived value reads the functions, and those before it; only
    # those wanted are evaluated, with those they read, and the functions
    # are bound only for those that use them
    functions = Functions()
    params = derived_params(
        dict(v='vcirc(2 * b)', m='v ** 2 / G', c='a + 1',
             vs='vcirc(np.array([1, 2]))'),
        functions=functions)
    values = params.evaluate()
    derived = params.derived_values(values)
    np.testing.assert_array_equal(derived.pop('vs'), [3, 6])
    assert derived == dict(v=12, m=72, c=4)
    assert params.derived_values(values, ['m']) == dict(v=12, m=72)
    assert functions.nbinds == 2
    assert params.derived_values(values, ['c']) == dict(c=4)
    assert functions.nbinds == 2
    dumped = params.dump()
    assert dumped['derived'] == params.derived()
    assert params.with_values(dict(b=1)).derived_values(
        params.evaluate([1]), ['v']) == dict(v=6)


@pytest.mark.parametrize('derived, properties, message', [
    (dict(v='vcirc(b)'), dict(a=3, b='v'),
     "'v' cannot be read here: it is a derived value"),
    (dict(v='vcirc(b)'), dict(a=3, b='vcirc(1)'),
     "'vcirc' cannot be read here: it is a function of the models"),
    (dict(m='v * 2', v='vcirc(b)'), None,
     "'v' cannot be read here: derived values can read only those before"),
    (dict(a='vcirc(b)'), None, "'a' has the name of a parameter"),
    (dict(v=1), None, "'v' must be an expression"),
    (dict(v='unknown(b)'), None, "'unknown' cannot be called"),
])
def test_derived_values_refused(derived, properties, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        derived_params(derived, properties)


def test_expressions_read_derived_values():
    # (as the priors on several parameters: not the functions, and the
    # derived values they read must be finite)
    params = derived_params(dict(v='vcirc(b)', w='v * np.nan'))
    values = params.evaluate()
    np.testing.assert_array_equal(params.expression('v / 2')(values), [3])
    with pytest.raises(ConfigError, match='derive a value with it'):
        params.expression('vcirc(1)')
    with pytest.raises(ParamEvaluationError, match="'w' is nan"):
        params.expression('w')(values)


def test_failed_derived_value():
    params = derived_params(dict(v='vcirc(b)[3]'))
    with pytest.raises(ParamEvaluationError, match="derived value 'v'"):
        params.derived_values(params.evaluate())
