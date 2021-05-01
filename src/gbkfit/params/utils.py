
import ast
import collections
import copy
import itertools
import logging
import numbers
import re

import astor
import networkx
import networkx.algorithms.dag
import numpy as np

from gbkfit.params.descs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import iterutils


log = logging.getLogger(__name__)


def _log_msg(level, silent, throw, msg):
    if throw:
        raise RuntimeError(msg)
    elif not silent:
        log.log(level, msg)


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
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}\s*')

_REGEX_PARAM_SYMBOL_VECTOR_SLICE = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}\s*')

_REGEX_PARAM_SYMBOL_VECTOR_AINDX = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}\s*')

_REGEX_PARAM_SYMBOL_VECTOR = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT}\s*')

_REGEX_PARAM_SYMBOL = (
    fr'\s*{_REGEX_PARAM_SYMBOL_NAME}\s*{_REGEX_PARAM_SYMBOL_SUBSCRIPT}?\s*')

_REGEX_PARAM_ATTRIB_NAME = r'\*?[_a-zA-Z]'


def _is_param_symbol(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL}$', x)


def _is_param_symbol_name(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_NAME}$', x)


def _is_param_symbol_scalar(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SCALAR}$', x)


def _is_param_symbol_vector(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR}$', x)


def _is_param_symbol_vector_bindx(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_BINDX}$', x)


def _is_param_symbol_vector_slice(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_SLICE}$', x)


def _is_param_symbol_vector_aindx(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_VECTOR_AINDX}$', x)


def _is_param_symbol_subscript(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT}$', x)


def _is_param_symbol_subscript_bindx(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_BINDX}$', x)


def _is_param_symbol_subscript_slice(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_SLICE}$', x)


def _is_param_symbol_subscript_aindx(x):
    return re.match(fr'^{_REGEX_PARAM_SYMBOL_SUBSCRIPT_AINDX}$', x)


def _is_param_attrib_name(x):
    return re.match(fr'^{_REGEX_PARAM_ATTRIB_NAME}$', x)


def _remove_white_space(x):
    return ''.join(x.split())


def _split_param_symbol(x):
    x = _remove_white_space(x)
    name = x[:x.find('[')].strip() if '[' in x else x
    sbsc = x[x.find('['):].strip() if '[' in x else None
    return name, sbsc


def _parse_param_symbol_subscript_bindx(x):
    x = _remove_white_space(x).strip('[]')
    return [int(x)]


def _parse_param_symbol_subscript_slice(x, size):
    x = _remove_white_space(x).strip('[]')
    x += ':' * (2 - x.count(':'))
    strt_str, stop_str, step_str = x.split(':')
    strt = int(strt_str) if strt_str else None
    stop = int(stop_str) if stop_str else None
    step = int(step_str) if step_str else None
    return list(range(*slice(strt, stop, step).indices(size)))


def _parse_param_symbol_subscript_aindx(x):
    x = _remove_white_space(x).strip('[],')
    return [int(i) for i in x.split(',')]


def _parse_param_symbol_subscript(x, size):
    if _is_param_symbol_subscript_bindx(x):
        indices = _parse_param_symbol_subscript_bindx(x)
    elif _is_param_symbol_subscript_slice(x):
        indices = _parse_param_symbol_subscript_slice(x, size)
    elif _is_param_symbol_subscript_aindx(x):
        indices = _parse_param_symbol_subscript_aindx(x)
    else:
        raise RuntimeError()
    return indices


def _validate_param_indices(indices, size):
    def is_valid(i): return -size <= i < size
    def is_invalid(i): return not is_valid(i)
    valid_indices = set(filter(is_valid, indices))
    invalid_indices = set(filter(is_invalid, indices))
    return sorted(valid_indices), sorted(invalid_indices)


def _unwrap_param_indices(indices, size):
    return [i + size if i < 0 else i for i in indices]


def make_param_symbol_subscript_bindx(index):
    return f'[{index}]'


def make_param_symbol_subscript_slice(start='', stop='', step=''):
    return f'[{start}:{stop}:{step}]'


def make_param_symbol_subscript_aindx(indices):
    return f'[{", ".join(indices)}]'


def make_param_symbol(name, index):
    return name if index is None \
        else f'{name}{make_param_symbol_subscript_bindx(index)}'


def explode_pname(name, indices):
    eparams = []
    for index in iterutils.listify(indices):
        eparams.append(make_param_symbol(name, index))
    return eparams


def explode_pnames(name_list, indices_list):
    eparams = []
    for name, indices in zip(name_list, indices_list):
        eparams.extend(explode_pname(name, indices))
    return eparams


def explode_pdesc(desc, name=None):
    enames = []
    name = name if name else desc.name()
    if isinstance(desc, ParamScalarDesc):
        enames.append(explode_pname(name, None)[0])
    elif isinstance(desc, ParamVectorDesc):
        enames.extend(explode_pname(name, list(range(desc.size()))))
    return enames


def explode_pdescs(descs, names=None):
    enames = []
    names = names if names else [desc.name() for desc in descs]
    for name, desc in zip(names, descs):
        enames.extend(explode_pdesc(desc, name))
    return enames


def is_param_value_expr(x, accept_num=True, accept_vec=True):
    ntypes = (numbers.Number,)
    vtypes = (tuple, list, np.ndarray)
    is_str = isinstance(x, str)
    is_num = isinstance(x, ntypes)
    is_vec = isinstance(x, vtypes) and all(isinstance(n, ntypes) for n in x)
    is_none = x is None
    return is_str or (accept_num and is_num) or (accept_vec and is_vec) or is_none


class _ParamExprVisitor(ast.NodeVisitor):

    def __init__(self, descs):
        self._descs = copy.deepcopy(descs)
        self._symbols = collections.defaultdict(set)
        self._invalid_scalars = set()
        self._invalid_vectors = dict()

    def symbols(self):
        return self._symbols

    def invalid_scalars(self):
        return self._invalid_scalars

    def invalid_vectors(self):
        return self._invalid_vectors

    def visit_Name(self, node):
        code = astor.to_source(node).strip('\n')
        name = code
        desc = self._descs.get(name)
        # If symbol is not recognised, ignore this node.
        if not desc:
            return
        # If symbol is a vector, generate all possible indices.
        indices = list(range(desc.size())) \
            if isinstance(desc, ParamVectorDesc) else None
        # Store symbol name along with its indices (if a vector).
        self._symbols[name] = indices

    def visit_Subscript(self, node):
        code = astor.to_source(node).strip('\n')
        name, subscript = _split_param_symbol(code)
        desc = self._descs.get(name)
        # If symbol is not recognised,
        # ignore this node but keep traversing this branch.
        if not desc:
            self.generic_visit(node)
            return
        # If symbol is a scalar,
        # mark symbol as invalid and ignore this node.
        if isinstance(desc, ParamScalarDesc):
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


def parse_param_keys(
        params, descs,
        silent_errors=False,
        silent_warnings=False,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_keys_syntax=None,
        invalid_keys_unknown=None,
        invalid_keys_repeated=None,
        invalid_keys_bad_scalar=None,
        invalid_keys_bad_vector=None):

    if params is None:
        params = {}
    if invalid_keys_syntax is None:
        invalid_keys_syntax = []
    if invalid_keys_unknown is None:
        invalid_keys_unknown = []
    if invalid_keys_repeated is None:
        invalid_keys_repeated = collections.defaultdict(list)
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    rkeys = []
    skeys = []
    values = []
    param_names = []
    param_indices = []
    eparams_to_keys = collections.defaultdict(list)

    for rkey, value in params.items():
        skey = _remove_white_space(rkey)
        # Skip keys with invalid syntax
        if not _is_param_symbol(rkey):
            invalid_keys_syntax.append(rkey)
            continue
        # Extract parameter name and subscript as strings
        # The latter will be None if key has scalar syntax
        name, subscript = _split_param_symbol(rkey)
        # Skip keys with unknown parameters
        desc = descs.get(name)
        if not desc:
            invalid_keys_unknown.append(rkey)
            continue
        if isinstance(desc, ParamScalarDesc):
            # Skip keys which are supposed to be scalars,
            # but have a subscript
            if subscript is not None:
                invalid_keys_bad_scalar.append(rkey)
                continue
            # Scalars are exploded already
            eparams_to_keys[skey].append(rkey)
            # Scalars do not have indices
            indices = None
        else:
            size = desc.size()
            # Explicitly define vector syntax if subscript is missing
            if subscript is None:
                subscript = '[:]'
            # Extract indices
            indices = _parse_param_symbol_subscript(subscript, size)
            # Validate indices
            indices, invalid_indices = _validate_param_indices(indices, size)
            # Skip keys with invalid indices
            if invalid_indices:
                invalid_keys_bad_vector[rkey] = invalid_indices
                continue
            # Ensure positive indices
            indices = _unwrap_param_indices(indices, size)
            # Explode vector
            for i in indices:
                eparams_to_keys[make_param_symbol(name, i)].append(rkey)
            # We differentiate basic indexing from slicing or advanced
            # indexing by using an integer instead of a list of integers
            if _is_param_symbol_subscript_bindx(subscript):
                indices = indices[0]
        # This is a valid key-value pair
        rkeys.append(rkey)
        skeys.append(skey)
        values.append(value)
        param_names.append(name)
        param_indices.append(indices)

    # Collect all keys that refer to repeated exploded parameters
    for eparam, parents in eparams_to_keys.items():
        if len(parents) > 1:
            for parent in parents:
                invalid_keys_repeated[parent].append(eparam)

    # Remove all information related to repeated keys
    for rkey in invalid_keys_repeated:
        i = rkeys.index(rkey)
        del rkeys[i]
        del skeys[i]
        del values[i]
        del param_names[i]
        del param_indices[i]

    if invalid_keys_unknown:
        _log_msg(
            logging.WARNING,
            silent_warnings,
            throw_on_warnings,
            f"keys with unknown parameters found: "
            f"{str(invalid_keys_unknown)}")
    if invalid_keys_syntax:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with invalid syntax found: "
            f"{str(invalid_keys_syntax)}")
    if invalid_keys_repeated:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with repeated parameters found: "
            f"{str(invalid_keys_repeated)}")
    if invalid_keys_bad_scalar:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"scalar parameter keys with vector syntax found: "
            f"{str(invalid_keys_bad_scalar)}")
    if invalid_keys_bad_vector:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"vector parameter keys with out-of-range indices found: "
            f"{str(invalid_keys_bad_vector)}")

    return skeys, values, param_names, param_indices


def parse_param_exprs(
        params, descs,
        silent_errors=False,
        silent_warnings=False,
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
        invalid_keys_repeated = collections.defaultdict(list)
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    if invalid_exprs_bad_value is None:
        invalid_exprs_bad_value = []
    if invalid_exprs_bad_syntax is None:
        invalid_exprs_bad_syntax = []
    if invalid_exprs_bad_scalar is None:
        invalid_exprs_bad_scalar = {}
    if invalid_exprs_bad_vector is None:
        invalid_exprs_bad_vector = {}

    keys, values, param_names, param_indices = parse_param_keys(
        params, descs,
        silent_errors,
        silent_warnings,
        throw_on_errors,
        throw_on_warnings,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)

    keys_2 = []
    values_2 = []
    param_names_2 = []
    param_indices_2 = []
    ast_root_nodes = []
    eparams_to_keys = {}
    expr_param_symbols = []

    for key, value, name, indices in zip(
            keys, values, param_names, param_indices):
        # Skip invalid expressions
        if not is_param_value_expr(value, True, True):
            invalid_exprs_bad_value.append(key)
            continue
        # All expressions can be converted to str
        value = str(value)
        # Parse expression and retrieve AST root node
        try:
            ast_root = ast.parse(value)
        except SyntaxError:
            invalid_exprs_bad_syntax.append(key)
            continue
        # Discover all symbols in expression
        visitor = _ParamExprVisitor(descs)
        visitor.visit(ast_root)
        # Keep track of invalid symbols
        if visitor.invalid_scalars():
            invalid_exprs_bad_scalar[key] = visitor.invalid_scalars()
        if visitor.invalid_vectors():
            invalid_exprs_bad_vector[key] = visitor.invalid_vectors()
        # Skip expressions with invalid symbols
        if visitor.invalid_scalars() or visitor.invalid_vectors():
            continue
        # Keep track of the parent key for each exploded param
        eparams_to_keys.update(
            {eparam: key for eparam in explode_pname(name, indices)})
        # This is a valid key-value pair
        keys_2.append(key)
        values_2.append(value)
        param_names_2.append(name)
        param_indices_2.append(indices)
        ast_root_nodes.append(ast_root)
        expr_param_symbols.append(visitor.symbols())

    if invalid_exprs_bad_value:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"the value of the following parameter keys "
            f"cannot be converted to an expression: "
            f"{str(invalid_exprs_bad_value)}")
    if invalid_exprs_bad_syntax:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"the value of the following parameter keys "
            f"contains an expression with syntax errors: "
            f"{str(invalid_exprs_bad_syntax)}")
    if invalid_exprs_bad_scalar:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"the value of the following parameter keys "
            f"contains an expression with invalid scalar symbols: "
            f"{str(invalid_exprs_bad_scalar)}")
    if invalid_exprs_bad_vector:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"the value of the following parameter keys "
            f"contains an expression with invalid vector symbols: "
            f"{str(invalid_exprs_bad_vector)}")

    # Build dependency graph of all exploded parameters
    graph = networkx.DiGraph()
    for name, indices, symbols in zip(
            param_names_2, param_indices_2, expr_param_symbols):
        eparams_lhs = explode_pname(name, indices)
        eparams_rhs = explode_pnames(symbols.keys(), symbols.values())
        for pair in itertools.product(eparams_lhs, eparams_rhs):
            graph.add_edge(pair[0], pair[1])

    # Perform topological sorting on the graph
    # This is a fatal error
    try:
        sorted_eparams = list(reversed(list(
            networkx.algorithms.dag.topological_sort(graph))))
    except networkx.NetworkXUnfeasible:
        raise RuntimeError(
            "circular dependencies found between param expressions")

    # Using the sorted exploded parameters derive a list of keys
    # The order of keys reflects the expression evaluation order
    sorted_keys = []
    for eparam in sorted_eparams:
        key = eparams_to_keys.get(eparam)
        if key and key not in sorted_keys:
            sorted_keys.append(key)
    for key in keys_2:
        if key not in sorted_keys:
            sorted_keys.append(key)

    # Apply the correct order to the output arrays
    order = [keys_2.index(key) for key in sorted_keys]
    keys_2 = [keys_2[i] for i in order]
    values_2 = [values_2[i] for i in order]
    param_names_2 = [param_names_2[i] for i in order]
    param_indices_2 = [param_indices_2[i] for i in order]
    ast_root_nodes = [ast_root_nodes[i] for i in order]

    return keys_2, values_2, param_names_2, param_indices_2, ast_root_nodes


def parse_param_values(
        params, descs,
        is_value_fun=None,
        silent_errors=False,
        silent_warnings=False,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_keys_syntax=None,
        invalid_keys_unknown=None,
        invalid_keys_repeated=None,
        invalid_keys_bad_scalar=None,
        invalid_keys_bad_vector=None,
        invalid_values_bad_value=None,
        invalid_values_bad_evalue=None,
        invalid_values_bad_length=None):

    if invalid_keys_syntax is None:
        invalid_keys_syntax = []
    if invalid_keys_unknown is None:
        invalid_keys_unknown = []
    if invalid_keys_repeated is None:
        invalid_keys_repeated = collections.defaultdict(list)
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    if invalid_values_bad_value is None:
        invalid_values_bad_value = []
    if invalid_values_bad_evalue is None:
        invalid_values_bad_evalue = collections.defaultdict(list)
    if invalid_values_bad_length is None:
        invalid_values_bad_length = []

    keys, values, param_names, param_indices = parse_param_keys(
        params, descs,
        silent_errors,
        silent_warnings,
        throw_on_errors,
        throw_on_warnings,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)

    exprs = {}
    enames = []
    evalues = []

    for key, value, name, indices in zip(
            keys, values, param_names, param_indices):
        if is_value_fun(value):
            if indices is None:
                enames.append(name)
                evalues.append(value)
            else:
                indices = iterutils.listify(indices)
                enames.extend(explode_pname(name, indices))
                evalues.extend([copy.deepcopy(value) for _ in indices])
        elif isinstance(value, (tuple, list, np.ndarray)):
            ienames = explode_pname(name, indices)
            if len(ienames) == len(value):
                for iename, ievalue in zip(ienames, value):
                    if is_value_fun(ievalue):
                        enames.append(iename)
                        evalues.append(ievalue)
                    elif isinstance(ievalue, (str, numbers.Number)):
                        exprs[iename] = ievalue
                    else:
                        invalid_values_bad_evalue[key].append(iename)
            else:
                invalid_values_bad_length.append(key)
        elif is_param_value_expr(value, True, False):
            exprs[key] = value
        else:
            invalid_values_bad_value.append(key)

    if invalid_values_bad_value:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with invalid value found: "
            f"{str(invalid_values_bad_value)}")
    if invalid_values_bad_evalue:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with invalid exploded value(s) found: "
            f"{str(invalid_values_bad_evalue)}")
    if invalid_values_bad_length:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with value of incompatible length found: "
            f"{str(invalid_values_bad_length)}")

    return keys, values, param_names, param_indices, dict(zip(enames, evalues)), exprs


def parse_param_info(
        params, descs,
        silent_errors=False,
        silent_warnings=False,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_keys_syntax=None,
        invalid_keys_unknown=None,
        invalid_keys_repeated=None,
        invalid_keys_bad_scalar=None,
        invalid_keys_bad_vector=None,
        invalid_infos_bad_value=None,
        invalid_infos_bad_attr_name=None,
        invalid_infos_bad_attr_value=None,
        invalid_infos_bad_attr_length=None):

    if invalid_keys_syntax is None:
        invalid_keys_syntax = []
    if invalid_keys_unknown is None:
        invalid_keys_unknown = []
    if invalid_keys_repeated is None:
        invalid_keys_repeated = collections.defaultdict(list)
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    if invalid_infos_bad_value is None:
        invalid_infos_bad_value = []
    if invalid_infos_bad_attr_name is None:
        invalid_infos_bad_attr_name = {}
    if invalid_infos_bad_attr_value is None:
        invalid_infos_bad_attr_value = {}
    if invalid_infos_bad_attr_length is None:
        invalid_infos_bad_attr_length = {}

    keys, values, param_names, param_indices = parse_param_keys(
        params, descs,
        silent_errors,
        silent_warnings,
        throw_on_errors,
        throw_on_warnings,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)

    keys_2 = []
    values_2 = []
    param_names_2 = []
    param_indices_2 = []
    exprs = {}
    eparams = []
    evalues = []

    for key, value, name, indices in zip(
            keys, values, param_names, param_indices):
        # value contains parameter info
        if isinstance(value, dict):
            error = False
            ieparams = []
            ievalues = []
            # This is a scalar or a vector with basic indexing
            if not isinstance(indices, list):
                ieparams.append(key)
                ievalues.append(value)
            # This is a vector with slice or advanced indexing
            else:
                nindices = len(indices)
                ieparams = explode_pname(name, indices)
                ievalues = iterutils.make_list(nindices, {}, True)
                for akey, avalue in value.items():
                    if _is_param_attrib_name(akey):
                        invalid_infos_bad_attr_name[key].append(akey)
                        error = True
                        continue
                    for i in range(nindices):
                        if akey.startswith('*'):
                            if not iterutils.is_sequence(avalue):
                                invalid_infos_bad_attr_value[key] = akey
                                error = True
                                break
                            if len(avalue) != nindices:
                                invalid_infos_bad_attr_length[key] = akey
                                error = True
                                break
                            ievalues[i][akey[1:]] = avalue[i]
                        else:
                            ievalues[i][akey] = avalue
            # If an error was occurred during attribute parsing,
            # we need to discard the entire key-value pair
            if error:
                continue
            eparams.extend(ieparams)
            evalues.extend(ievalues)
        # Value contains expression
        elif is_param_value_expr(value, True, True):
            exprs[key] = value
        # Value contains something unexpected
        else:
            invalid_infos_bad_value.append(key)
            continue
        # This is a valid key-value pair
        keys_2.append(key)
        values_2.append(value)
        param_names_2.append(name)
        param_indices_2.append(indices)

    if invalid_infos_bad_value:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with invalid values found: "
            f"{str(invalid_infos_bad_value)}")
    if invalid_infos_bad_attr_name:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with invalid attributes found (bad name): "
            f"{str(invalid_infos_bad_attr_name)}")
    if invalid_infos_bad_attr_value:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with invalid attributes found (bad value): "
            f"{str(invalid_infos_bad_attr_value)}")
    if invalid_infos_bad_attr_length:
        _log_msg(
            logging.ERROR,
            silent_errors,
            throw_on_errors,
            f"keys with invalid attributes found (bad length): "
            f"{str(invalid_infos_bad_attr_length)}")

    return (keys_2, values_2, param_names_2, param_indices_2,
            dict(zip(eparams, evalues)), exprs)


import inspect
import textwrap

from gbkfit.utils import miscutils, parseutils





def load_expressions(info):
    if not info:
        return None
    desc = 'parameter expressions'
    opts = parseutils.parse_options(info, desc, ['file', 'func'])
    return miscutils.get_attr_from_file(opts['file'], opts['func'])


def load_econstraints(info):
    if not info:
        return None
    desc = 'parameter equality constraints'
    opts = parseutils.parse_options(info, desc, ['file', 'func'])
    return miscutils.get_attr_from_file(opts['file'], opts['func'])


def load_iconstraints(info):
    if not info:
        return None
    desc = 'parameter inequality constraints'
    opts = parseutils.parse_options(info, desc, ['file', 'func'])
    return miscutils.get_attr_from_file(opts['file'], opts['func'])


def _dump_function(func, file):
    with open(file, 'a') as f:
        f.write('\n')
        f.write(textwrap.dedent(inspect.getsource(func)))
        f.write('\n')
    return dict(file=file, func=func.__name__)


def dump_expressions(func, file='gbkfit_config_expressions.py'):
    return _dump_function(func, file) if func else None


def dump_econstraints(func, file='gbkfit_config_econstraints.py'):
    return _dump_function(func, file) if func else None


def dump_iconstraints(func, file='gbkfit_config_iconstraints.py'):
    return _dump_function(func, file) if func else None


def order_eparams(descs, enames):
    enames_all = explode_pdescs(descs.values(), descs.keys())
    return [ename for ename in enames_all if ename in enames]
