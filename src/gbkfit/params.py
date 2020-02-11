
import abc
import ast
import collections
import itertools
import logging
import re

import astor
import networkx
import networkx.algorithms.dag

from gbkfit.utils import iterutils


log = logging.getLogger(__name__)


def _log_msg(level, throw, msg):
    if throw:
        raise RuntimeError(msg)
    else:
        log.log(level, msg)


REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON = r'(?!.*\D0+[1-9])'

REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX = (
    fr'{REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*([-+]?\s*\d+)\s*\]')

REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE = (
    fr'{REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*([+-]?\s*\d+)?\s*:\s*([+-]?\s*\d+)?\s*(:\s*([+-]?\s*[1-9]+)?\s*)?\]')

REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX = (
    fr'{REGEX_PARAM_SYMBOL_SUBSCRIPT_COMMON}'
    r'\[\s*\[\s*([-+]?\s*\d+\s*,\s*)*\s*([-+]?\s*\d+\s*)?\]\s*,?\s*\]')

REGEX_PARAM_SYMBOL_SUBSCRIPT = (
    r'('
    fr'{REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}|'
    fr'{REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}|'
    fr'{REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}'
    r')')

REGEX_PARAM_SYMBOL_NAME = r'[_a-zA-Z]\w*'

REGEX_PARAM_SYMBOL_SCALAR = REGEX_PARAM_SYMBOL_NAME

REGEX_PARAM_SYMBOL_VECTOR_BINDX = (
    fr'\s*{REGEX_PARAM_SYMBOL_NAME}\s*{REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}\s*')

REGEX_PARAM_SYMBOL_VECTOR_SLICE = (
    fr'\s*{REGEX_PARAM_SYMBOL_NAME}\s*{REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}\s*')

REGEX_PARAM_SYMBOL_VECTOR_AINDX = (
    fr'\s*{REGEX_PARAM_SYMBOL_NAME}\s*{REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}\s*')

REGEX_PARAM_SYMBOL_VECTOR = (
    fr'\s*{REGEX_PARAM_SYMBOL_NAME}\s*{REGEX_PARAM_SYMBOL_SUBSCRIPT}\s*')

REGEX_PARAM_SYMBOL = (
    fr'\s*{REGEX_PARAM_SYMBOL_NAME}\s*{REGEX_PARAM_SYMBOL_SUBSCRIPT}?\s*')


def _is_param_symbol(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL}$', x)


def _is_param_symbol_name(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_NAME}$', x)


def _is_param_symbol_scalar(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_SCALAR}$', x)


def _is_param_symbol_vector(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_VECTOR}$', x)


def _is_param_symbol_vector_bindx(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_VECTOR_BINDX}$', x)


def _is_param_symbol_vector_slice(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_VECTOR_SLICE}$', x)


def _is_param_symbol_vector_aindx(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_VECTOR_AINDX}$', x)


def _is_param_symbol_subscript(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_SUBSCRIPT}$', x)


def _is_param_symbol_subscript_bindx(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}$', x)


def _is_param_symbol_subscript_slice(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}$', x)


def _is_param_symbol_subscript_aindx(x):
    return re.match(fr'^{REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}$', x)


def _split_param_symbol(x):
    x = x.replace(' ', '')
    name = x[:x.find('[')].strip() if '[' in x else x
    sbsc = x[x.find('['):].strip() if '[' in x else None
    return name, sbsc


def _parse_param_symbol_subscript_bindx(x):
    x = x.replace(' ', '').strip('[]')
    return [int(x)]


def _parse_param_symbol_subscript_slice(x, size):
    x = x.replace(' ', '').strip('[]')
    x += ':' * (2 - x.count(':'))
    strt_str, stop_str, step_str = x.split(':')
    strt = int(strt_str) if strt_str else None
    stop = int(stop_str) if stop_str else None
    step = int(step_str) if step_str else None
    return list(range(*slice(strt, stop, step).indices(size)))


def _parse_param_symbol_subscript_aindx(x):
    x = x.replace(' ', '').strip('[],')
    return [int(i) for i in x.split(',')]


def _parse_param_symbol_subscript(x, size):
    if _is_param_symbol_subscript_bindx(x):
        indices = _parse_param_symbol_subscript_bindx(x)
    elif _is_param_symbol_subscript_slice(x):
        indices = _parse_param_symbol_subscript_slice(x, size)
    else:  # _is_param_symbol_subscript_aindx(x)
        indices = _parse_param_symbol_subscript_aindx(x)
    return indices


def make_param_symbol(name, index):
    return name if index is None else f'{name}[{index}]'


def explode_param_symbol(name, indices):
    if isinstance(indices, int):
        indices = [indices]
    return [make_param_symbol(name, i) for i in indices] \
        if indices else [name]


def explode_param_symbols(names, indices_list):
    eparams = []
    for name, indices in zip(names, indices_list):
        eparams += explode_param_symbol(name, indices)
    return eparams


def _validate_param_indices(indices, size):
    valid_indices = set()
    invalid_indices = set()
    for i in indices:
        if -size <= i < size:
            valid_indices.add(i)
        else:
            invalid_indices.add(i)
    return valid_indices, sorted(invalid_indices)


def _unwrap_param_indices(indices, size):
    return [i + size if i < 0 else i for i in indices]


def _is_param_expr(x):
    return (isinstance(x, (int, float, str))
            or (isinstance(x, (tuple, list))
                and iterutils.is_sequence_of_type(x, (int, float))))


def _is_param_value(x):
    return _is_param_value_scalar(x) or _is_param_value_vector(x)


def _is_param_value_scalar(x):
    return isinstance(x, (int, float))


def _is_param_value_vector(x):
    return (isinstance(x, (tuple, list))
            and iterutils.is_sequence_of_type(x, (int, float)))


def _explode_param_vector(value, size):
    values = []

    for i in range(size):
        values.append({})
        for akey, avalue in value.items():
            akey = akey.strip()
            if akey.startswith('*'):
                akey = akey[1:]
                avalue = iterutils.listify(avalue)
                values[i][akey] = avalue[i] if i < len(avalue) else avalue[-1]
            else:
                values[i][akey] = avalue

    return values


def extract_params_of_type(params, types):
    types = iterutils.tuplify(types)
    return {key: value for key, value in params.items()
            if isinstance(value, types)}


def convert_params_free_to_fixed(params):
    return {key: value['init'] if isinstance(value, dict) else value
            for key, value in params.items()}


class _ParamExprVisitor(ast.NodeVisitor):

    def __init__(self, descs):
        self._descs = dict(descs)
        self._symbols = collections.defaultdict(set)
        self._invalid_scalars = set()
        self._invalid_vectors = dict()

    def symbols(self):
        return dict(self._symbols)

    def invalid_scalars(self):
        return set(self._invalid_scalars)

    def invalid_vectors(self):
        return dict(self._invalid_vectors)

    def visit_Name(self, node):
        code = astor.to_source(node).strip('\n')
        name = code
        desc = self._descs.get(name)
        # If symbol name is not recognised, ignore this node.
        if not desc:
            return
        # If symbol is supposed to be a vector,
        # generate all possible indices.
        indices = _parse_param_symbol_subscript('[:]', desc.size()) \
            if desc.is_vector() else None
        # Store symbol name along with its indices (if it is a vector).
        self._symbols[name] = indices

    def visit_Subscript(self, node):
        code = astor.to_source(node).strip('\n')
        name, subscript = _split_param_symbol(code)
        desc = self._descs.get(name)
        # If symbol name is not recognised,
        # ignore this node and keep traversing this branch.
        if not desc:
            self.generic_visit(node)
            return
        # If symbol is supposed to be a scalar,
        # mark symbol as invalid and ignore this node.
        if desc.is_scalar():
            self._invalid_scalars.add(code)
            return
        # If symbol subscript syntax is not supported,
        # mark symbol as invalid and ignore this node.
        if not _is_param_symbol_subscript(subscript):
            self._invalid_vectors[code] = None
            return
        # Extract indices while unwrapping the negative ones.
        size = desc.size()
        indices = _parse_param_symbol_subscript(subscript, size)
        indices, invalid_indices = _validate_param_indices(indices, size)
        indices = _unwrap_param_indices(indices, size)
        # If out-of-range indices are found,
        # mark symbol as invalid and ignore this node.
        if invalid_indices:
            self._invalid_vectors[code] = invalid_indices
            return
        # Symbol is recognised and valid, add it to the list.
        self._symbols[name].update(indices)


class ParamDesc(abc.ABC):

    def __init__(self, name, size):
        assert _is_param_symbol_scalar(name)
        self._name = name
        self._size = size

    def name(self):
        return self._name

    def size(self):
        return self._size

    def is_scalar(self):
        return not self.is_vector()

    @abc.abstractmethod
    def is_vector(self):
        pass


class ParamScalarDesc(ParamDesc):

    def __init__(self, name):
        super().__init__(name, 1)

    def is_vector(self):
        return False

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name()!r})'


class ParamVectorDesc(ParamDesc):

    def __init__(self, name, size):
        super().__init__(name, size)

    def is_vector(self):
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name()!r}, {self.size()!r})'


def parse_param_keys(
        params, descs,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_keys_syntax=None,
        invalid_keys_unknown=None,
        invalid_keys_repeated=None,
        invalid_keys_bad_scalar=None,
        invalid_keys_bad_vector=None):

    if invalid_keys_syntax is None:
        invalid_keys_syntax = []
    if invalid_keys_unknown is None:
        invalid_keys_unknown = []
    if invalid_keys_repeated is None:
        invalid_keys_repeated = {}
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    keys = []
    values = []
    param_names = []
    param_indices_list = []

    exploded_params = collections.defaultdict(list)

    for key, value in params.items():

        # Skip keys with invalid syntax
        if not _is_param_symbol(key):
            invalid_keys_syntax.append(key)
            continue

        # Extract parameter name and subscript
        name, subscript = _split_param_symbol(key)
        # Extract parameter description
        desc = descs.get(name)

        # Skip keys with unknown parameters
        if not desc:
            invalid_keys_unknown.append(key)
            continue

        if desc.is_scalar():

            # Skip keys which are supposed to be scalars,
            # but they have a subscript
            if subscript is not None:
                invalid_keys_bad_scalar.append(key)
                continue

            # Scalar are exploded already
            exploded_params[key.strip()].append(key)
            # Scalars do not have indices
            indices = None

        else:

            size = desc.size()

            # Explicitly define vector syntax if subscript is missing
            if subscript is None:
                #key += '[:]'
                subscript = '[:]'

            # Extract indices
            indices = _parse_param_symbol_subscript(subscript, size)
            # Find invalid indices
            indices, invalid_indices = _validate_param_indices(indices, size)
            # Skip keys with invalid indices
            if invalid_indices:
                invalid_keys_bad_vector[key] = invalid_indices
                continue
            # Ensure positive indices
            indices = _unwrap_param_indices(indices, size)

            # Explode vector and store its items in a key=>value dict.
            # Each key corresponds to an exploded vector item,
            # and each value corresponds to the parent key of that item.
            for i in indices:
                exploded_params[make_param_symbol(name, i)].append(key)

            # We differentiate basic indexing from slicing or advanced
            # indexing by using an integer instead if a list of integers.
            if _is_param_symbol_subscript_bindx(subscript):
                indices = indices[0]

        keys.append(key)
        values.append(value)
        param_names.append(name)
        param_indices_list.append(indices)

    # Collect all keys that refer to repeated exploded parameters
    for eparam, parents in exploded_params.items():
        if len(parents) > 1:
            for parent in parents:
                if parent not in invalid_keys_repeated:
                    invalid_keys_repeated[parent] = []
                invalid_keys_repeated[parent].append(eparam)

    # Remove all information related to repeated keys
    for key in invalid_keys_repeated:
        i = keys.index(key)
        del keys[i]
        del values[i]
        del param_names[i]
        del param_indices_list[i]

    if invalid_keys_syntax:
        _log_msg(
            logging.WARNING,
            throw_on_warnings,
            f"keys with invalid syntax found: "
            f"{str(invalid_keys_syntax)[1:-1]}")
    if invalid_keys_unknown:
        _log_msg(
            logging.WARNING,
            throw_on_warnings,
            f"keys with unknown parameters found: "
            f"{str(invalid_keys_unknown)[1:-1]}")
    if invalid_keys_repeated:
        _log_msg(
            logging.ERROR,
            throw_on_errors,
            f"keys with repeated parameters found: "
            f"{str(invalid_keys_repeated)[1:-1]}")
    if invalid_keys_bad_scalar:
        _log_msg(
            logging.ERROR,
            throw_on_errors,
            f"scalar parameter keys with vector syntax found: "
            f"{str(invalid_keys_bad_scalar)[1:-1]}")
    if invalid_keys_bad_vector:
        _log_msg(
            logging.ERROR,
            throw_on_errors,
            f"vector parameter keys with out-of-range indices found: "
            f"{str(invalid_keys_bad_vector)[1:-1]}")

    return keys, values, param_names, param_indices_list


def parse_param_exprs(
        params, descs,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_keys_syntax=None,
        invalid_keys_unknown=None,
        invalid_keys_repeated=None,
        invalid_keys_bad_scalar=None,
        invalid_keys_bad_vector=None,
        invalid_exprs_bad_value=None,
        invalid_exprs_bad_syntax=None,
        invalid_exprs_bad_scalar=None,
        invalid_exprs_bad_vector=None):

    if invalid_keys_syntax is None:
        invalid_keys_syntax = []
    if invalid_keys_unknown is None:
        invalid_keys_unknown = []
    if invalid_keys_repeated is None:
        invalid_keys_repeated = {}
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    if invalid_exprs_bad_value is None:
        invalid_exprs_bad_value = {}
    if invalid_exprs_bad_syntax is None:
        invalid_exprs_bad_syntax = {}
    if invalid_exprs_bad_scalar is None:
        invalid_exprs_bad_scalar = {}
    if invalid_exprs_bad_vector is None:
        invalid_exprs_bad_vector = {}

    # First, parse and validate the param keys
    keys, values, key_names, key_indices_list = parse_param_keys(
        params, descs,
        throw_on_errors,
        throw_on_warnings,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)

    keys_2 = []
    values_2 = []
    key_names_2 = []
    key_indices_list_2 = []
    expr_symbols = []

    # Parse and validate the param expressions
    for key, value, key_name, key_indices in zip(
            keys, values, key_names, key_indices_list):

        # Skip invalid expressions
        if not _is_param_expr(value):
            invalid_exprs_bad_value[key] = value
            continue

        # All expressions can be converted to string
        value = str(value)

        # Parse expression and retrieve AST root node
        try:
            ast_root = ast.parse(value)
        except Exception as e:
            ast_root = None

        # Skip expressions with invalid syntax
        if ast_root is None:
            invalid_exprs_bad_syntax[key] = value
            continue

        # Discover all symbols in expression
        visitor = _ParamExprVisitor(descs)
        visitor.visit(ast_root)

        # Skip expressions with invalid symbols
        if visitor.invalid_scalars():
            invalid_exprs_bad_scalar[key] = visitor.invalid_scalars()
        if visitor.invalid_vectors():
            invalid_exprs_bad_vector[key] = visitor.invalid_vectors()
        if visitor.invalid_scalars() or visitor.invalid_vectors():
            continue

        keys_2.append(key)
        values_2.append(value)
        key_names_2.append(key_name)
        key_indices_list_2.append(key_indices)
        expr_symbols.append(visitor.symbols())

    if invalid_exprs_bad_value:

        _log_msg(
            logging.ERROR,
            throw_on_errors,
            f"the rhs of the following expressions is an invalid value"
            f"{invalid_exprs_bad_value}")
        raise RuntimeError(invalid_exprs_bad_value)

    if invalid_exprs_bad_syntax:
        _log_msg(
            logging.ERROR,
            throw_on_errors,
            f"the rhs of the following expressions is an invalid value"
            f"{invalid_exprs_bad_syntax}")

        raise RuntimeError(invalid_exprs_bad_syntax)

    if invalid_exprs_bad_scalar:

        _log_msg(
            logging.ERROR,
            throw_on_errors,
            f"the rhs of the following expressions contain scalar parameters"
            f"{invalid_exprs_bad_scalar}")

        raise RuntimeError(invalid_exprs_bad_scalar)


    if invalid_exprs_bad_vector:
        raise RuntimeError(invalid_exprs_bad_vector)

    # Build a dict which maps exploded params to their parent keys
    eparam_to_key_dict = {}
    for key, name, indices in zip(keys_2, key_names_2, key_indices_list_2):
        eparams = explode_param_symbol(name, indices)
        eparam_to_key_dict.update({eparam: key for eparam in eparams})

    # Build dependency graph
    graph = networkx.DiGraph()
    for key, expr_symbols in zip(keys_2, expr_symbols):
        lhs = [key]
        eparams = explode_param_symbols(expr_symbols.keys(), expr_symbols.values())
        rhs = set([eparam_to_key_dict[eparam] for eparam in eparams if eparam in eparam_to_key_dict])
        for pair in itertools.product(lhs, rhs):
            graph.add_edge(pair[0], pair[1])

    # Perform topological sorting to the graph in order to calculate
    # the order in which the expressions should be evaluated with.
    try:
        keys_ordered = list(networkx.algorithms.dag.topological_sort(graph))
        keys_ordered = keys_ordered[::-1]
        keys_ordered += list(set(keys_2).difference(set(keys_ordered)))
    except networkx.NetworkXUnfeasible:
        raise RuntimeError()

    # Apply the correct order to the output arrays
    keys_ordered_indices = [keys_2.index(key) for key in keys_ordered]
    keys_2 = [keys_2[i] for i in keys_ordered_indices]
    values_2 = [values_2[i] for i in keys_ordered_indices]
    key_names_2 = [key_names_2[i] for i in keys_ordered_indices]
    key_indices_list_2 = [key_indices_list_2[i] for i in keys_ordered_indices]

    return keys_2, values_2, key_names_2, key_indices_list_2


def parse_param_values(
        params, descs,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_keys_syntax=None,
        invalid_keys_unknown=None,
        invalid_keys_repeated=None,
        invalid_keys_bad_scalar=None,
        invalid_keys_bad_vector=None,
        invalid_values_bad_value=None,
        invalid_values_bad_size=None,
        invalid_values_bad_scalar=None):

    if invalid_keys_syntax is None:
        invalid_keys_syntax = []
    if invalid_keys_unknown is None:
        invalid_keys_unknown = []
    if invalid_keys_repeated is None:
        invalid_keys_repeated = {}
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    if invalid_values_bad_value is None:
        invalid_values_bad_value = {}
    if invalid_values_bad_size is None:
        invalid_values_bad_size = {}
    if invalid_values_bad_scalar is None:
        invalid_values_bad_scalar = {}

    # First, parse and validate the param keys
    keys, values, key_names, key_indices_list = parse_param_keys(
        params, descs,
        throw_on_errors,
        throw_on_warnings,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)

    keys_2 = []
    values_2 = []
    key_names_2 = []
    key_indices_list_2 = []

    # Parse and validate the param values
    for key, value, key_name, key_indices in zip(
            keys, values, key_names, key_indices_list):

        # Skip invalid values
        if not _is_param_value(value):
            invalid_values_bad_value[key] = value
            continue

        # lhs is a scalar or a vector item
        if key_indices is None or isinstance(key_indices, int):
            # Only accept scalar rhs values
            if _is_param_value_vector(value):
                invalid_values_bad_scalar[key] = value
                continue
        # lhs is a vector
        else:
            # rhs is a vector
            if _is_param_value_vector(value):
                # lhs and rhs vectors must have the same length
                if len(key_indices) != len(value):
                    invalid_values_bad_size[key] = value
                    continue
            # rhs is a scalar
            else:
                # repeat value
                value = [value] * len(key_indices)

        keys_2.append(key)
        values_2.append(value)
        key_names_2.append(key_name)
        key_indices_list_2.append(key_indices)

    if invalid_values_bad_value:

        _log_msg(
            logging.ERROR,
            True,
            f"pairs with invalid values found: "
            f"{str(invalid_values_bad_value)}")

        raise RuntimeError(invalid_values_bad_value)
    if invalid_values_bad_size:

        _log_msg(logging.ERROR, True, "parameters with invalid values found")

        raise RuntimeError(invalid_values_bad_size)
    if invalid_values_bad_scalar:
        raise RuntimeError(invalid_values_bad_scalar)

    return keys_2, values_2, key_names_2, key_indices_list_2


def parse_param_fit_info(
        params, descs,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_keys_syntax=None,
        invalid_keys_unknown=None,
        invalid_keys_repeated=None,
        invalid_keys_bad_scalar=None,
        invalid_keys_bad_vector=None,
        invalid_values_bad_value=None,
        invalid_values_attrib_bad_value=None,
        invalid_values_attrib_bad_size=None):

    if invalid_keys_syntax is None:
        invalid_keys_syntax = []
    if invalid_keys_unknown is None:
        invalid_keys_unknown = []
    if invalid_keys_repeated is None:
        invalid_keys_repeated = {}
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    if invalid_values_bad_value is None:
        invalid_values_bad_value = {}
    if invalid_values_attrib_bad_value is None:
        invalid_values_attrib_bad_value = {}
    if invalid_values_attrib_bad_size is None:
        invalid_values_attrib_bad_size = {}

    # First, parse and validate the param keys
    keys, values, key_names, key_indices_list = parse_param_keys(
        params, descs,
        throw_on_errors,
        throw_on_warnings,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)

    ekeys = explode_param_symbols(key_names, key_indices_list)
    evalues = []



    for key, value, key_indices in zip(keys, values, key_indices_list):

        # Fit info values must be dicts
        if not isinstance(value, dict):
            invalid_values_bad_value[key] = value
            continue

        # When the key is just a single value
        if not isinstance(key_indices, list):
            evalues.append(value)
        # ..
        else:
            curr_evalues = iterutils.make_list((len(key_indices),), {}, True)
            for akey, avalue in value.items():
                for i in range(len(key_indices)):
                    akey = akey.strip()
                    if akey.startswith('*'):
                        aakey = akey[1:]
                        if not isinstance(avalue, list):
                            invalid_values_attrib_bad_value[key] = akey
                            break
                        if len(avalue) != len(key_indices):
                            invalid_values_attrib_bad_size[key] = akey
                            break
                        curr_evalues[i][aakey] = avalue[i]
                    else:
                        curr_evalues[i][akey] = avalue

            evalues.extend(curr_evalues)

    if invalid_values_bad_value:
        raise RuntimeError()

    if invalid_values_attrib_bad_value:
        raise RuntimeError()

    if invalid_values_attrib_bad_size:
        raise RuntimeError()

    parset = dict(zip(ekeys, evalues))

    return parset
