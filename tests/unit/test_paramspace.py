"""
The behaviour of the parameters of the configurations: which elements
are free, tied and fixed, the values they evaluate to, and the errors.

Each case goes through the same path as a configuration file: the
params of the eval task (EvaluationParams), or those of a fit
(ParamSpace).
"""

import re

import numpy as np
import pytest

import gbkfit.params
from gbkfit.params import ParamScalarDesc, ParamVectorDesc


def make_pdescs(**sizes):
    """Parameter descriptions: a scalar for None, else a vector."""
    return {
        name: ParamScalarDesc(name) if size is None
        else ParamVectorDesc(name, size)
        for name, size in sizes.items()}


def load_eval_params(pdescs, properties, transforms=None):
    info = dict(properties=properties)
    if transforms:
        info['transforms'] = transforms
    return gbkfit.params.evaluation_params_parser.load(info, pdescs=pdescs)


def evaluate(pdescs, properties, transforms=None):
    return load_eval_params(pdescs, properties, transforms).evaluate()


def load_fit_params(pdescs, properties):
    return gbkfit.params.ParamSpace(pdescs, properties)


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
    unknown_keys_are_ignored=(
        make_pdescs(a=S), {'a': 1, 'old': 2, 'old[0]': 3},
        {'a': 1}),
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


def test_exploded_values():
    pdescs = make_pdescs(a=S, v=2)
    params = load_eval_params(pdescs, {'a': 1, 'v': 'a + np.arange(2)'})
    exploded = {}
    params.evaluate(out_exploded_params=exploded)
    assert exploded == {'a': 1, 'v[0]': 1, 'v[1]': 2}


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
    params = load_fit_params(FIT_PDESCS, FIT_PROPERTIES)
    assert params.names(free=True, tied=False, fixed=False) == [
        'c', 'v[0]', 'w[0]', 'w[1]']
    assert params.names(free=False, tied=True, fixed=False) == [
        'b', 'v[1]']
    assert params.names(free=False, tied=False, fixed=True) == [
        'a', 'v[2]']
    assert params.free_properties() == {
        'c': {'value': 1, 'min': 0, 'max': 2},
        'v[0]': {'value': 1},
        'w[0]': {'value': 1, 'max': 5},
        'w[1]': {'value': 2, 'max': 5}}


def test_fit_evaluation():
    params = load_fit_params(FIT_PDESCS, FIT_PROPERTIES)
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
    params = load_fit_params(FIT_PDESCS, FIT_PROPERTIES)
    with pytest.raises(Exception) as error:
        params.evaluate(free)
    assert re.search(pattern, str(error.value))


def test_evaluation_does_not_keep_state():
    # The values returned are new arrays, and changing them does not
    # change the next evaluation
    params = load_fit_params(FIT_PDESCS, FIT_PROPERTIES)
    free = {'c': 1.5, 'v[0]': 2, 'w[0]': 3, 'w[1]': 4}
    first = params.evaluate(free)
    first['v'][:] = -1
    second = params.evaluate(free)
    np.testing.assert_array_equal(second['v'], [2, 4, 3])


def test_constants():
    # Expressions can use constants (e.g. the radial nodes of a disk)
    space = gbkfit.params.ParamSpace(
        make_pdescs(v=3), {'v': 'rnodes * 2'},
        constants=dict(rnodes=[0, 1, 2]))
    np.testing.assert_array_equal(space.evaluate()['v'], [0, 2, 4])


def test_constants_cannot_have_the_names_of_parameters():
    with pytest.raises(Exception, match="constants cannot have the names"):
        gbkfit.params.ParamSpace(
            make_pdescs(a=S), {'a': 1}, constants=dict(a=1))


def test_unknown_parameters_are_errors_in_strict_mode():
    from gbkfit.utils import parseutils
    with parseutils.strict_mode():
        with pytest.raises(Exception, match=r"unknown parameters: 'old'"):
            gbkfit.params.ParamSpace(make_pdescs(a=S), {'a': 1, 'old': 2})


def load_moded_params(properties, modes):
    """The params of a vector v of 4 elements and scalars s and t."""
    return gbkfit.params.evaluation_params_parser.load(
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
    reloaded = gbkfit.params.evaluation_params_parser.load(
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
