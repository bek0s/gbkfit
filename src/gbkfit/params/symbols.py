
import re

from gbkfit.params.pdescs import ParamVectorDesc
from gbkfit.utils import iterutils, stringutils


__all__ = [
    'is_param_symbol',
    'is_param_symbol_name',
    'is_param_symbol_scalar',
    'is_param_symbol_vector',
    'is_param_symbol_vector_bindx',
    'is_param_symbol_vector_slice',
    'is_param_symbol_vector_aindx',
    'is_param_symbol_subscript',
    'is_param_symbol_subscript_bindx',
    'is_param_symbol_subscript_slice',
    'is_param_symbol_subscript_aindx',
    'is_param_attrib_symbol',
    'make_param_symbol_subscript_bindx',
    'make_param_symbol_subscript_slice',
    'make_param_symbol_subscript_aindx',
    'make_param_symbol',
    'parse_param_symbol_subscript',
    'parse_param_symbol_into_name_and_subscript_str',
    'parse_param_symbol',
    'make_param_symbols_from_name_and_indices',
    'make_param_symbols_from_names_and_indices',
    'make_param_symbols_from_pdesc',
    'make_param_symbols_from_pdescs'
]

# Prevents leading zeros in nonzero indices (numpy syntax).
# Example: disallows `[03]`, allows `[0]`.
_REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON = r'(?!.*\D0+[1-9])'

# Matches basic (single-element) indexing (numpy syntax).
# Example: `[0]`.
_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX = (
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*([-+]?\s*\d+)\s*\]')

# Matches slice indexing (numpy syntax).
# Example: `[0:1:2]`.
_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE = (
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*([+-]?\s*\d+)?\s*:\s*([+-]?\s*\d+)?\s*(:\s*([+-]?\s*[1-9]+)?\s*)?\]')

# Matches advanced (list-based) indexing (numpy syntax).
# Example: `[[0, 1, 2]]`.
_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX = (
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*\[\s*([-+]?\s*\d+\s*,\s*)*\s*([-+]?\s*\d+\s*)?\]\s*,?\s*\]')

# Matches any supported indexing mode (numpy syntax).
_REGEX_PARAM_SYMBOL_SUBSCRIPT = (
    r'('
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}|'
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}|'
    fr'{_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}'
    r')')

# Matches symbol names.
_REGEX_PARAM_SYMBOL_NAME = r'[_a-zA-Z]\w*'

# Matches a scalar symbol (no indexing).
# Example: `A`.
_REGEX_PARAM_SYMBOL_SCALAR = _REGEX_PARAM_SYMBOL_NAME

# Matches basic (single-element) indexing (numpy syntax).
# Example: `A[3]`.
_REGEX_PARAM_SYMBOL_VECTOR_BINDX = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}\s*')

# Matches slice indexing (numpy syntax).
# Example: `A[1:5]`.
_REGEX_PARAM_SYMBOL_VECTOR_SLICE = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}\s*')

# Matches advanced (list-based) indexing (numpy syntax).
# Example: `A[[1, 3, 7]]`.
_REGEX_PARAM_SYMBOL_VECTOR_AINDX = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}\s*')

# Matches any supported indexing mode (numpy syntax).
_REGEX_PARAM_SYMBOL_VECTOR = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT}\s*')

# Matches any supported indexing mode (numpy syntax).
_REGEX_PARAM_SYMBOL = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}'
    fr'\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT}?\s*')

# Matches attribute names.
_REGEX_PARAM_ATTRIB_SYMBOL_NAME = r'[_a-zA-Z]\w*'


def is_param_symbol(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL}$', x)


def is_param_symbol_name(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_NAME}$', x)


def is_param_symbol_scalar(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SCALAR}$', x)


def is_param_symbol_vector(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR}$', x)


def is_param_symbol_vector_bindx(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_BINDX}$', x)


def is_param_symbol_vector_slice(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_SLICE}$', x)


def is_param_symbol_vector_aindx(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_AINDX}$', x)


def is_param_symbol_subscript(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT}$', x)


def is_param_symbol_subscript_bindx(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}$', x)


def is_param_symbol_subscript_slice(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}$', x)


def is_param_symbol_subscript_aindx(x: str):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}$', x)


def is_param_attrib_symbol(x: str):
    return re.match(fr'^{_REGEX_PARAM_ATTRIB_SYMBOL_NAME}$', x)


def make_param_symbol_subscript_bindx(index: int) -> str:
    """Create a basic bracket indexing expression."""
    return f'[{index}]'


def make_param_symbol_subscript_slice(
        start: int = '', stop: int = '', step: int = ''
) -> str:
    """Create a slice indexing expression."""
    return f'[{start}:{stop}:{step}]'


def make_param_symbol_subscript_aindx(
        indices: tuple[int, ...] | list[int]
) -> str:
    """Create an advanced indexing expression with multiple indices."""
    if not indices:
        raise RuntimeError("advanced index list cannot be empty")
    return f'[[{", ".join(str(i) for i in indices)}]]'


def make_param_symbol(
        name: str,
        indices: None | int | tuple[int, ...] | list[int]
) -> str:
    """Create a parameter symbol with optional indexing."""
    indices = iterutils.tuplify(indices, False)
    if not indices:
        result = name
    elif len(indices) == 1:
        result = f'{name}{make_param_symbol_subscript_bindx(indices[0])}'
    else:
        result = f'{name}{make_param_symbol_subscript_aindx(indices)}'
    return result


def _parse_param_symbol_subscript_bindx(x: str) -> tuple[int]:
    """
    Parse a basic bracket indexing expression and extract the index.
    """
    x = stringutils.remove_white_space(x).strip('[]')
    return int(x),


def _parse_param_symbol_subscript_slice(x, size):
    x = stringutils.remove_white_space(x).strip('[]')
    x += ':' * (2 - x.count(':'))
    start_str, stop_str, step_str = x.split(':')
    start = int(start_str) if start_str else None
    stop = int(stop_str) if stop_str else None
    step = int(step_str) if step_str else None
    return tuple(range(*slice(start, stop, step).indices(size)))


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
        raise RuntimeError(f"invalid subscript syntax: '{x}'")
    return indices


def parse_param_symbol_into_name_and_subscript_str(x):
    x = stringutils.remove_white_space(x)
    name = x[:x.find('[')].strip() if '[' in x else x
    subscript = x[x.find('['):].strip() if '[' in x else None
    return name, subscript


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


def make_param_symbols_from_name_and_indices(name, indices):
    symbols = []
    for index in iterutils.listify(indices):
        symbols.append(make_param_symbol(name, index))
    return symbols


def make_param_symbols_from_names_and_indices(name_list, indices_list):
    symbols = []
    for name, indices in zip(name_list, indices_list, strict=True):
        symbols.extend(make_param_symbols_from_name_and_indices(name, indices))
    return symbols


def make_param_symbols_from_pdesc(pdesc, override_name=None):
    name = override_name if override_name else pdesc.name()
    indices = None
    if isinstance(pdesc, ParamVectorDesc):
        indices = list(range(pdesc.size()))
    return make_param_symbols_from_name_and_indices(name, indices)


def make_param_symbols_from_pdescs(pdescs, override_names=None):
    symbols = []
    names = override_names if override_names else [d.name() for d in pdescs]
    for name, pdesc in zip(names, pdescs, strict=True):
        symbols.extend(make_param_symbols_from_pdesc(pdesc, name))
    return symbols
