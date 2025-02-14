
import numbers

import numpy as np

from gbkfit.params.interpreter import *
from gbkfit.params.params import *
from gbkfit.params.parsers import *
from gbkfit.params.pdescs import *
from gbkfit.params.symbols import *


def test_param_desc_scalar():
    pdesc_name = 'name'
    pdesc_desc = 'description'
    pdesc_default = 10
    pdesc_minimum = -10
    pdesc_maximum = +np.inf
    # Test __init__
    pdesc = ParamScalarDesc(
        pdesc_name, pdesc_desc, pdesc_default,
        pdesc_minimum, pdesc_maximum)
    assert pdesc.name() == pdesc_name
    assert pdesc.size() == 1
    assert pdesc.desc() == pdesc_desc
    assert pdesc.default() == pdesc_default
    assert pdesc.minimum() == pdesc_minimum
    assert pdesc.maximum() == pdesc_maximum
    # Test load
    info = dict(
        type='scalar',
        name=pdesc_name,
        desc=pdesc_desc,
        default=pdesc_default,
        minimum=pdesc_minimum)
    desc_loaded = pdesc_parser.load(info)
    assert desc_loaded.__class__ == pdesc.__class__ and \
           desc_loaded.__dict__ == pdesc.__dict__
    # Test dump
    info_dumped = pdesc.dump()
    assert info_dumped == info


def test_param_desc_vector():
    pdesc_name = 'name'
    pdesc_size = 5
    pdesc_desc = 'description'
    pdesc_default = 10
    pdesc_minimum = -10
    pdesc_maximum = +np.inf
    # Test __init__
    pdesc = ParamVectorDesc(
        pdesc_name, pdesc_size, pdesc_desc, pdesc_default,
        pdesc_minimum, pdesc_maximum)
    assert pdesc.name() == pdesc_name
    assert pdesc.size() == pdesc_size
    assert pdesc.desc() == pdesc_desc
    assert pdesc.default() == pdesc_default
    assert pdesc.minimum() == pdesc_minimum
    assert pdesc.maximum() == pdesc_maximum
    # Test load
    info = dict(
        type='vector',
        name=pdesc_name,
        size=pdesc_size,
        desc=pdesc_desc,
        default=pdesc_default,
        minimum=pdesc_minimum)
    desc_loaded = pdesc_parser.load(info)
    assert desc_loaded.__class__ == pdesc.__class__ and \
           desc_loaded.__dict__ == pdesc.__dict__
    # Test dump
    info_dumped = pdesc.dump()
    assert info_dumped == info


def test_param_symbols():
    # is_param_symbol()
    assert is_param_symbol('a ')
    assert is_param_symbol(' a')
    assert is_param_symbol(' a ')
    assert is_param_symbol('a_1')
    assert is_param_symbol('a[:]')
    assert is_param_symbol('a[1]')
    assert is_param_symbol('a [1]')
    assert is_param_symbol('a[+1]')
    assert is_param_symbol('a[-1]')
    assert is_param_symbol('a[[1,2]]')
    assert is_param_symbol('a[[1,2,2]]')
    assert is_param_symbol('a[1:2]')
    assert is_param_symbol('a[1:2:1]')
    assert is_param_symbol('a[1:2:-1]')
    assert not is_param_symbol('1')
    assert not is_param_symbol('*')
    assert not is_param_symbol('1a')
    assert not is_param_symbol('*a')
    assert not is_param_symbol('a*')
    assert not is_param_symbol('a a')
    assert not is_param_symbol('a[b]')
    # is_param_attrib_symbol()
    assert is_param_attrib_symbol('a')
    assert not is_param_attrib_symbol('*a')
    assert not is_param_attrib_symbol('a*')
    assert not is_param_attrib_symbol('a[1]')
    # make_param_symbol()
    assert make_param_symbol('a', None) == 'a'
    assert make_param_symbol('a', 0) == 'a[0]'
    assert make_param_symbol('a', [0]) == 'a[0]'
    assert make_param_symbol('a', [0, 1]) == 'a[[0, 1]]'
    assert make_param_symbol('a', [1, 0]) == 'a[[1, 0]]'
    # make_param_symbol_subscript_[bindx|slice|aindx]()
    assert make_param_symbol_subscript_bindx(1) == '[1]'
    assert make_param_symbol_subscript_bindx(-1) == '[-1]'
    assert make_param_symbol_subscript_slice(1, 3) == '[1:3:]'
    assert make_param_symbol_subscript_slice(1, 3, -1) == '[1:3:-1]'
    assert make_param_symbol_subscript_aindx([1, 2]) == '[[1, 2]]'
    assert make_param_symbol_subscript_aindx([2, 2, 1]) == '[[2, 2, 1]]'
    # make_param_symbols_from_name[s]_and_indices()
    assert make_param_symbols_from_name_and_indices(
        'a', [1, 2]) == ['a[1]', 'a[2]']
    assert make_param_symbols_from_names_and_indices(
        ['a', 'b'], [[0, 1], [2, 3]]) == ['a[0]', 'a[1]', 'b[2]', 'b[3]']
    # make_param_symbols_from_pdesc[s]()
    assert make_param_symbols_from_pdesc(
        ParamScalarDesc('a')) == ['a']
    assert make_param_symbols_from_pdesc(
        ParamScalarDesc('a'), 'b') == ['b']
    assert make_param_symbols_from_pdesc(
        ParamVectorDesc('a', 2)) == ['a[0]', 'a[1]']
    assert make_param_symbols_from_pdesc(
        ParamVectorDesc('a', 2), 'b') == ['b[0]', 'b[1]']
    assert make_param_symbols_from_pdescs([
        ParamScalarDesc('a'), ParamVectorDesc('b', 2)]
    ) == ['a', 'b[0]', 'b[1]']
    assert make_param_symbols_from_pdescs([
        ParamScalarDesc('a'), ParamVectorDesc('b', 2)], ['c', 'd']
    ) == ['c', 'd[0]', 'd[1]']
    # parse_param_symbol()
    assert parse_param_symbol('a', None) == ('a', None, None)
    assert parse_param_symbol('a', 3) == ('a', [0, 1, 2], [])
    assert parse_param_symbol('a[:]', 3) == ('a', [0, 1, 2], [])
    assert parse_param_symbol('a[0]', 3) == ('a', [0], [])
    assert parse_param_symbol('a[3]', 3) == ('a', [], [3])
    assert parse_param_symbol('a[-1]', 3) == ('a', [2], [])
    assert parse_param_symbol('a[-4]', 3) == ('a', [], [-4])
    assert parse_param_symbol('a[[1, 2]]', 3) == ('a', [1, 2], [])
    assert parse_param_symbol('a[[2, 2, 1]]', 3) == ('a', [2, 2, 1], [])


def test_param_parsers():

    #
    # parse_param_keys()
    #

    pdescs = dict(
        a=ParamScalarDesc('a'),
        b=ParamVectorDesc('b', 5),
        c=ParamVectorDesc('c', 5),
        d=ParamVectorDesc('d', 5),
        e=ParamVectorDesc('b', 5),
    )

    params = {
        # valid keys
        'a ': 1,
        'b': 1,
        'c[0:2]': 1,
        'd': 1,
        # invalid keys
        '1invalid': 1,
        'unknown': 1,
        'a[0]': 1,
        'd[0]': 1,
        'e[5]': 1
    }

    parse_param_keys_result = parse_param_keys(
        params, pdescs,
        silent_errors=True,
        silent_warnings=True,
        throw_on_errors=False,
        throw_on_warnings=False)

    assert parse_param_keys_result.keys == ['a', 'b', 'c[0:2]']
    assert parse_param_keys_result.values == [1, 1, 1]
    assert parse_param_keys_result.param_names == ['a', 'b', 'c']
    assert parse_param_keys_result.param_indices == [
        None, [0, 1, 2, 3, 4], [0, 1]]

    assert parse_param_keys_result.invalid_keys_syntax == ['1invalid']
    assert parse_param_keys_result.invalid_keys_unknown == ['unknown']
    assert parse_param_keys_result.invalid_keys_repeated == {
        'd': ['d[0]'], 'd[0]': ['d[0]']}
    assert parse_param_keys_result.invalid_keys_bad_scalar == ['a[0]']
    assert parse_param_keys_result.invalid_keys_bad_vector == {'e[5]': [5]}

    #
    # parse_param_values()
    #

    class InvalidValueType:
        pass

    pdescs = dict(
        a=ParamScalarDesc('a'),
        b=ParamScalarDesc('b'),
        c=ParamScalarDesc('c'),
        d=ParamVectorDesc('d', 3),
        e=ParamVectorDesc('e', 3),
        f=ParamVectorDesc('f', 3),
        g=ParamVectorDesc('g', 3),
        h=ParamVectorDesc('h', 3),
        i=ParamVectorDesc('i', 3),
        j=ParamVectorDesc('j', 3),
        k=ParamVectorDesc('k', 3),
        l=ParamVectorDesc('l', 3),
        m=ParamVectorDesc('m', 3),
        n=ParamVectorDesc('n', 3),
        o=ParamVectorDesc('o', 3)
    )

    params = {
        'a': None,
        'b': 1,
        'c': '1',
        'd': None,
        'e': 1,
        'f': '1',
        'g': [0, 1, 2],
        'h': [None, 1, '1'],
        'i[:]': 1,
        'j[0]': 0,
        'j[1:3]': [1, 2],
        'k[[0, 1, 2]]': [0, 1, 2],
        # Invalid values
        'l': [0, 1],
        'm': InvalidValueType(),
        'n': [InvalidValueType(), 1, 2],
        'o[0]': [1]
    }

    parse_param_values_result = parse_param_values(
        params, pdescs,
        is_value_fun=lambda x: isinstance(x, (numbers.Real,)),
        silent_errors=True,
        silent_warnings=True,
        throw_on_errors=False,
        throw_on_warnings=False)

    assert parse_param_values_result.parse_param_keys_result.param_names == [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'j', 'k', 'l', 'm',
        'n', 'o']
    assert parse_param_values_result.parse_param_keys_result.param_indices == [
        None, None, None, [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
        [0, 1, 2], [0, 1, 2], 0, [1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
        [0, 1, 2], 0]
    assert parse_param_values_result.exploded_params == {
        'b': 1, 'e[0]': 1, 'e[1]': 1, 'e[2]': 1, 'g[0]': 0, 'g[1]': 1,
        'g[2]': 2, 'h[1]': 1, 'i[0]': 1, 'i[1]': 1, 'i[2]': 1, 'j[0]': 0,
        'j[1]': 1, 'j[2]': 2, 'k[0]': 0, 'k[1]': 1, 'k[2]': 2, 'n[1]': 1,
        'n[2]': 2}
    assert parse_param_values_result.expressions == {
        'a': None, 'c': '1', 'd': None, 'f': '1', 'h[0]': None, 'h[2]': '1'}

    assert parse_param_values_result.invalid_values_bad_value == [
        'm']
    assert parse_param_values_result.invalid_values_bad_evalue == {
        'n': ['n[0]']}
    assert parse_param_values_result.invalid_values_bad_length == [
        'l', 'o[0]']

    #
    # parse_param_expressions()
    #

    pdescs = dict(
        a=ParamScalarDesc('a'),
        b=ParamScalarDesc('b'),
        c=ParamScalarDesc('c'),
        d=ParamVectorDesc('d', 3),
        e=ParamVectorDesc('e', 3),
        f=ParamVectorDesc('f', 3),
        g=ParamVectorDesc('g', 3)
    )

    params = {
        'a': 'b',
        'b': 'c',
        'c': '1 + 1',
        'd[0]': 'd[1]',
        'd[1]': 'd[2]',
        'd[2]': 'np.cos(a)',
        # Invalid expressions
        'e': InvalidValueType(),
        'f': [1, 2, InvalidValueType()],
        'g[0]': 'syntax error',
        'g[1]': 'a[100] + a[101]',
        'g[2]': 'd[[0, 100]]'
    }

    parse_param_expressions_result = parse_param_expressions(
        params, pdescs,
        silent_errors=True,
        silent_warnings=True,
        throw_on_errors=False,
        throw_on_warnings=False)

    assert parse_param_expressions_result.expression_keys == [
        'c', 'b', 'a', 'd[2]', 'd[1]', 'd[0]']

    assert parse_param_expressions_result.invalid_expressions_bad_value == [
        'e', 'f']
    assert parse_param_expressions_result.invalid_expressions_bad_syntax == [
        'g[0]']
    assert parse_param_expressions_result.invalid_expressions_bad_scalar == {
        'g[1]': {'a[100]', 'a[101]'}}
    assert parse_param_expressions_result.invalid_expressions_bad_vector == {
        'g[2]': {'d[[0, 100]]': [100]}}

    #
    # prepare_param_info()
    #

    pdescs = dict(
        a=ParamScalarDesc('a'),
        b=ParamScalarDesc('b'),
        c=ParamVectorDesc('c', 2),
        d=ParamVectorDesc('d', 2),
        e=ParamVectorDesc('e', 8),
        f=ParamVectorDesc('f', 5),
        g=ParamVectorDesc('g', 5),
        h=ParamVectorDesc('h', 5)
    )

    params = {
        'a': 1,
        'b': {'attr1': 1},
        'c': {'*attr1': [1, 2], 'attr2': 3},
        'd': {'*attr1': [1, 2], '*attr2': [3, 4]},
        'e': [
            {'attr1': 1},
            {'*attr1': [1, 2], 'attr2': 3},
            {'*attr1': [1, 2], '*attr2': [3, 4]},
            {'1invalid': 1},
            {'*attr1': [1, 2], '*attr2': 3},
            {'*attr1': [1, 2], '*attr2': [3, 4, 5]},
        ],
        'f': {'1invalid': 1},
        'g': {'*attr1': [1, 2], '*attr2': 3},
        'h': {'*attr1': [1, 2], '*attr2': [3, 4, 5]}
    }

    parse_param_info_result = parse_param_info(
        params, pdescs,
        silent_errors=True,
        silent_warnings=True,
        throw_on_errors=False,
        throw_on_warnings=False)

    assert parse_param_info_result.info == {
        'a': 1,
        'b': {'attr1': 1},
        'c': [
            {'attr1': 1, 'attr2': 3},
            {'attr1': 2, 'attr2': 3}
        ],
        'd': [
            {'attr1': 1, 'attr2': 3},
            {'attr1': 2, 'attr2': 4}
        ],
        'e': [
            {'attr1': 1},
            {'attr1': 1, 'attr2': 3},
            {'attr1': 2, 'attr2': 3},
            {'attr1': 1, 'attr2': 3},
            {'attr1': 2, 'attr2': 4}
        ]
    }
    assert parse_param_info_result.invalid_infos_bad_attr_name == {
        'f': ['1invalid'], 'e': [(3, ['1invalid'])]}
    assert parse_param_info_result.invalid_infos_bad_attr_value == {
        'g': ['attr2'], 'e': [(4, ['attr2'])]}
    assert parse_param_info_result.invalid_infos_bad_attr_length == {
        'h': ['attr1', 'attr2'], 'e': [(5, ['attr1', 'attr2'])]}


def test_param_interpreter():

    pdescs = dict(
        a=ParamScalarDesc('a'),
        b=ParamScalarDesc('b'),
        c=ParamScalarDesc('c'),
        d=ParamVectorDesc('d', 3),
        e=ParamVectorDesc('e', 3),
        f=ParamVectorDesc('f', 3))

    expressions_dict = {
        'a': 1,
        'b': '1 + 1',
        'c': 'a + b',
        'd': 4,
        'e[0]': 5,
        'e[1:]': [6, 7],
        'f[1]': 'f[0]',
        'f[2]': 'f[1] + 8'
    }

    expressions_dict = {
        'a': 1,
        'b': None,
        'c': None,
        'd': 4,
        'e[0]': 5,
        'e[1:]': [6, 7],
        'f[1]': None,
        'f[2]': None
    }

    def expressions_func(params):
        params['a'] = 1
        params['b'] = 1 + 1
        params['c'] = params['a'] + params['b']
        params['d'] = 4
        params['e'][0] = 5
        params['e'][1:] = [6, 7]
        params['f'][1] = params['f'][0]
        params['f'][2] = params['f'][1] + 8
        return params

    interpreter01 = Interpreter(pdescs, expressions_dict, expressions_func)
    enames_free = interpreter01.enames(fixed=False, tied=False, free=True)
    enames_tied = interpreter01.enames(fixed=False, tied=True, free=False)
    enames_fixed = interpreter01.enames(fixed=True, tied=False, free=False)
    assert enames_free == ['f[0]']
    assert enames_tied == ['b', 'c', 'f[1]', 'f[2]']
    assert enames_fixed == ['a', 'd[0]', 'd[1]', 'd[2]', 'e[0]', 'e[1]', 'e[2]']

    eparams = {}
    params = interpreter01.evaluate({'f[0]': 1}, eparams, True)
    assert params['a'] == 1
    assert params['b'] == 2
    assert params['c'] == 3
    assert np.array_equal(params['d'], (4, 4, 4))
    assert np.array_equal(params['e'], (5, 6, 7))
    assert np.array_equal(params['f'], (1, 1, 9))
    assert eparams == {
        'a': 1, 'b': 2, 'c': 3,
        'd[0]': 4, 'd[1]': 4, 'd[2]': 4,
        'e[0]': 5, 'e[1]': 6, 'e[2]': 7,
        'f[0]': 1, 'f[1]': 1, 'f[2]': 9}


# def test_evaluation_params():
#
#     pdescs = dict(
#         a=ParamScalarDesc('a'),
#         b=ParamScalarDesc('b'),
#         c=ParamVectorDesc('c', 3))
#
#     parameters = {
#         'a': 1,
#         'b': '2',
#         'c': 'a + b'
#     }
#
#     params = EvaluationParams(pdescs, parameters)
#     enames_tied = params.enames(fixed=False, tied=True)
#     enames_fixed = params.enames(fixed=True, tied=False)
#     assert enames_tied == ['b', 'c[0]', 'c[1]', 'c[2]']
#     assert enames_fixed == ['a']


if __name__ == '__main__':
    test_param_desc_scalar()
    test_param_desc_vector()
    test_param_symbols()
    test_param_parsers()
    test_param_interpreter()
    # test_evaluation_params()
