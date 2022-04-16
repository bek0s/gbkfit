
import re

from gbkfit.utils import iterutils, stringutils


_REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON = r'(?!.*\D0+[1-9])'

_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX = (
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*([-+]?\s*\d+)\s*\]')

_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE = (
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*([+-]?\s*\d+)?\s*:\s*([+-]?\s*\d+)?\s*(:\s*([+-]?\s*[1-9]+)?\s*)?\]')

_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX = (
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*\[\s*([-+]?\s*\d+\s*,\s*)*\s*([-+]?\s*\d+\s*)?\]\s*,?\s*\]')

_REGEX_PARAM_SYMBOL_SUBSCRIPT = (
    r'('
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}|'
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}|'
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}'
    r')')

_REGEX_PARAM_SYMBOL_NAME = r'[_a-zA-Z]\w*'

_REGEX_PARAM_SYMBOL_SCALAR = _REGEX_PARAM_SYMBOL_NAME

_REGEX_PARAM_SYMBOL_VECTOR_BINDX = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}\s*')

_REGEX_PARAM_SYMBOL_VECTOR_SLICE = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}\s*')

_REGEX_PARAM_SYMBOL_VECTOR_AINDX = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}\s*')

_REGEX_PARAM_SYMBOL_VECTOR = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT}\s*')

_REGEX_PARAM_SYMBOL = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT}?\s*')

_REGEX_PARAM_ATTRIB_NAME = r'[_a-zA-Z]\w*'


def is_param_symbol(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL}$', x)


def is_param_symbol_name(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_NAME}$', x)


def is_param_symbol_scalar(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SCALAR}$', x)


def is_param_symbol_vector(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR}$', x)


def is_param_symbol_vector_bindx(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_BINDX}$', x)


def is_param_symbol_vector_slice(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_SLICE}$', x)


def is_param_symbol_vector_aindx(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_AINDX}$', x)


def is_param_symbol_subscript(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT}$', x)


def is_param_symbol_subscript_bindx(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}$', x)


def is_param_symbol_subscript_slice(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}$', x)


def is_param_symbol_subscript_aindx(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}$', x)


def is_param_attrib_symbol(x):
    return re.match(fr'^{_REGEX_PARAM_ATTRIB_NAME}$', x)


def make_param_symbol_subscript_bindx(index):
    return f'[{index}]'


def make_param_symbol_subscript_slice(start='', stop='', step=''):
    return f'[{start}:{stop}:{step}]'


def make_param_symbol_subscript_aindx(indices):
    return f'[[{", ".join(str(i) for i in indices)}]]'


def make_param_symbol(name, indices):
    indices = iterutils.tuplify(indices, False)
    if not indices:
        result = name
    elif len(indices) == 1:
        result = f'{name}{make_param_symbol_subscript_bindx(indices[0])}'
    else:
        result = f'{name}{make_param_symbol_subscript_aindx(indices)}'
    return result


def _parse_param_symbol_subscript_bindx(x):
    x = stringutils.remove_white_space(x).strip('[]')
    return int(x),


def _parse_param_symbol_subscript_slice(x, size):
    x = stringutils.remove_white_space(x).strip('[]')
    x += ':' * (2 - x.count(':'))
    strt_str, stop_str, step_str = x.split(':')
    strt = int(strt_str) if strt_str else None
    stop = int(stop_str) if stop_str else None
    step = int(step_str) if step_str else None
    return tuple(range(*slice(strt, stop, step).indices(size)))


def _parse_param_symbol_subscript_aindx(x):
    x = stringutils.remove_white_space(x).strip('[],')
    return tuple([int(i) for i in x.split(',')])


def parse_param_symbol_subscript(x, size):
    if is_param_symbol_subscript_bindx(x):
        indices = _parse_param_symbol_subscript_bindx(x)
    elif is_param_symbol_subscript_slice(x):
        indices = _parse_param_symbol_subscript_slice(x, size)
    elif is_param_symbol_subscript_aindx(x):
        indices = _parse_param_symbol_subscript_aindx(x)
    else:
        raise RuntimeError(f"invalid subscript syntax: {x}")
    return indices


def parse_param_symbol_into_name_and_subscript_str(x):
    x = stringutils.remove_white_space(x)
    name = x[:x.find('[')].strip() if '[' in x else x
    sbsc = x[x.find('['):].strip() if '[' in x else None
    return name, sbsc


def parse_param_symbol(x, vector_size=None):
    x = stringutils.remove_white_space(x)
    name, subscript = parse_param_symbol_into_name_and_subscript_str(x)
    valid_indices = None
    invalid_indices = None
    if vector_size is not None and not subscript:
        subscript = '[:]'
    if subscript:
        indices = parse_param_symbol_subscript(subscript, vector_size)
        valid_indices, invalid_indices = iterutils.validate_sequence_indices(
            indices, vector_size)
        valid_indices = iterutils.unwrap_sequence_indices(
            valid_indices, vector_size)
    return name, valid_indices, invalid_indices
