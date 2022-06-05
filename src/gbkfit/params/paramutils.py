
import ast
import collections
import copy
import graphlib
import inspect
import itertools
import logging
import numbers
import textwrap

import numpy as np

from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
# from gbkfit.params.paramutils import *
from gbkfit.params.symbols import *
from gbkfit.utils import miscutils, parseutils, stringutils


# __all__ = [
#     'explode_param_name_from_indices',
#     'explode_param_name_from_pdesc',
#     'explode_param_names_from_indices',
#     'explode_param_names_from_pdescs',
#     'sort_param_enames'
# ]


_log = logging.getLogger(__name__)


def explode_param_name_from_indices(name, indices):
    eparams = []
    for index in iterutils.listify(indices):
        eparams.append(make_param_symbol(name, index))
    return eparams


def explode_param_names_from_indices(name_list, indices_list):
    # todo: add strict=True to zip() (python 3.10)
    eparams = []
    for name, indices in zip(name_list, indices_list):
        eparams.extend(explode_param_name_from_indices(name, indices))
    return eparams


def explode_param_name_from_pdesc(pdesc, override_name=None):
    enames = []
    name = override_name if override_name else pdesc.name()
    if isinstance(pdesc, ParamScalarDesc):
        enames.extend(explode_param_name_from_indices(name, None))
    elif isinstance(pdesc, ParamVectorDesc):
        enames.extend(explode_param_name_from_indices(
            name, list(range(pdesc.size()))))
    else:
        raise RuntimeError()
    return enames


def explode_param_names_from_pdescs(pdescs, override_names=None):
    # todo: add strict=True to zip() (python 3.10)
    enames = []
    names = override_names if override_names else [d.name() for d in pdescs]
    for name, pdesc in zip(names, pdescs):
        enames.extend(explode_param_name_from_pdesc(pdesc, name))
    return enames


def sort_param_enames(pdescs, enames):
    # todo: check if enames is indeed a subset?
    enames_all = explode_param_names_from_pdescs(pdescs.values(), pdescs.keys())
    return [ename for ename in enames_all if ename in enames]


def _log_msg(level, silent, throw, msg):
    if throw:
        raise RuntimeError(msg)
    elif not silent:
        _log.log(level, msg)


def _is_param_value_expr(x, accept_vec=True):
    def is_nil(i): return isinstance(i, type(None))
    def is_str(i): return isinstance(i, str)
    def is_num(i): return isinstance(i, numbers.Real)
    def is_scalar(i): return is_nil(i) or is_str(i) or is_num(i)
    def is_vector(i): return isinstance(i, (tuple, list, np.ndarray))*accept_vec
    return is_scalar(x) or (is_vector(x) and all(is_scalar(i) for i in x))


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
        code = ast.unparse(node).strip('\n')
        name = code
        desc = self._descs.get(name)
        # If symbol is not recognised, ignore this node.
        if not desc:
            return
        # # If symbol is a vector, generate all possible indices.
        indices = list(range(desc.size())) \
            if isinstance(desc, ParamVectorDesc) else None
        # # Store symbol name along with its indices (if a vector).
        # print("foo:", desc)
        # self._symbols[name] = set(indices)

        if isinstance(desc, ParamScalarDesc):
            self._symbols[name] = None
        elif isinstance(desc, ParamVectorDesc):
            self._symbols[name] = set(indices)
        else:
            assert False

    def visit_Subscript(self, node):
        code = ast.unparse(node).strip('\n')
        name, subscript = parse_param_symbol_into_name_and_subscript_str(code)
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
        if not is_param_symbol_subscript(subscript):
            self._invalid_vectors[code] = None
            return
        # Extract indices while unwrapping the negative ones.
        size = desc.size()
        indices = parse_param_symbol_subscript(subscript, size)
        indices, invalid_indices = iterutils.validate_sequence_indices(
            indices, size)
        indices = iterutils.unwrap_sequence_indices(indices, size)
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
        invalid_keys_repeated = {}
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
        skey = stringutils.remove_white_space(rkey)
        # Skip keys with invalid syntax
        if not is_param_symbol(rkey):
            invalid_keys_syntax.append(rkey)
            continue
        # Extract parameter name and subscript as strings
        # The latter will be None if key has scalar syntax
        name, subscript = parse_param_symbol_into_name_and_subscript_str(rkey)
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
            indices = parse_param_symbol_subscript(subscript, size)
            # Validate indices
            indices, invalid_indices = iterutils.validate_sequence_indices(
                indices, size)
            # Skip keys with invalid indices
            if invalid_indices:
                invalid_keys_bad_vector[rkey] = invalid_indices
                continue
            # Ensure positive indices
            indices = iterutils.unwrap_sequence_indices(indices, size)
            # Explode vector
            for i in indices:
                eparams_to_keys[make_param_symbol(name, i)].append(rkey)
            # We differentiate basic indexing from slicing or advanced
            # indexing by using an integer instead of a list of integers
            if is_param_symbol_subscript_bindx(subscript):
                indices = indices[0]
        # This is a valid key-value pair
        rkeys.append(rkey)
        skeys.append(skey)
        values.append(value)
        param_names.append(name)
        param_indices.append(indices)

    # Collect all keys that refer to repeated exploded parameters
    invalid_keys_repeated_ = collections.defaultdict(list)
    for eparam, parents in eparams_to_keys.items():
        if len(parents) > 1:
            for parent in parents:
                invalid_keys_repeated_[parent].append(eparam)
    invalid_keys_repeated.update(invalid_keys_repeated_)

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
        invalid_keys_repeated = {}
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
        if not _is_param_value_expr(value, True):
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
            {eparam: key for eparam in explode_param_name_from_indices(name, indices)})
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
    graph = graphlib.TopologicalSorter()
    for name, indices, symbols in zip(
            param_names_2, param_indices_2, expr_param_symbols):
        print(f"name: {name}, symbols: {dict(symbols)}, values: {list(symbols.values())}")
        eparams_lhs = explode_param_name_from_indices(name, indices)
        eparams_rhs = explode_param_names_from_indices(symbols.keys(), symbols.values())
        for pair in itertools.product(eparams_lhs, eparams_rhs):
            graph.add(pair[1], pair[0])

    # Perform topological sorting on the graph
    try:
        sorted_eparams = list(reversed(list(graph.static_order())))
    except graphlib.CycleError as e:
        raise RuntimeError(
            f"circular dependencies found between param expressions; "
            f"have a look at the following cycle: {e.args[1]}") from e

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
        params, pdescs,
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
        invalid_keys_repeated = {}
    if invalid_keys_bad_scalar is None:
        invalid_keys_bad_scalar = []
    if invalid_keys_bad_vector is None:
        invalid_keys_bad_vector = {}

    if invalid_values_bad_value is None:
        invalid_values_bad_value = []
    if invalid_values_bad_evalue is None:
        invalid_values_bad_evalue = {}
    if invalid_values_bad_length is None:
        invalid_values_bad_length = []

    keys, values, param_names, param_indices = parse_param_keys(
        params, pdescs,
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

    # Use a default dicts internally.
    # The api should use normal dicts through.
    invalid_values_bad_evalue_ = collections.defaultdict(list)

    for key, value, name, indices in zip(
            keys, values, param_names, param_indices):

        if is_value_fun(value):
            if indices is None:
                enames.append(name)
                evalues.append(value)
            else:
                indices = iterutils.listify(indices)
                enames.extend(explode_param_name_from_indices(name, indices))
                evalues.extend(iterutils.make_list(len(indices), value))

        elif iterutils.is_sequence(value):
            ienames = explode_param_name_from_indices(name, indices)
            if len(ienames) == len(value) and not is_param_symbol_vector_bindx(key):
                for iename, ievalue in zip(ienames, value):
                    if is_value_fun(ievalue):
                        enames.append(iename)
                        evalues.append(ievalue)
                    elif isinstance(ievalue, (type(None), numbers.Real, str)):
                        exprs[iename] = ievalue
                    else:
                        invalid_values_bad_evalue_[key].append(iename)
            else:
                invalid_values_bad_length.append(key)
        elif _is_param_value_expr(value, False):
            exprs[key] = value
        else:
            invalid_values_bad_value.append(key)

    invalid_values_bad_evalue.update(invalid_values_bad_evalue_)

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

    eparams = dict(zip(enames, evalues))
    return keys, values, param_names, param_indices, eparams, exprs


def _parse_param_info_item(info):
    bad_keys = []
    bad_vals = []
    bad_lens = False
    expandable_keys = []
    # Validate attributes
    sizes = []
    for key, val in info.items():
        needs_expansion = False
        if key.startswith('*'):
            key = key[1:]
            needs_expansion = True
            expandable_keys.append(key)
        # Attribute name must be valid
        if not is_param_attrib_symbol(key):
            bad_keys.append(key)
            continue
        # If attribute needs expansion, value must be a sequence
        if needs_expansion and not iterutils.is_sequence(val):
            bad_vals.append(key)
            continue
        # If attribute needs expansion, keep track of the value length
        if needs_expansion:
            sizes.append(len(val))
    # All expandable attributes must have the same value length
    if len(set(sizes)) > 1:
        bad_lens = True
    # Any error causes failure
    if bad_lens or bad_keys or bad_vals:
        return [], bad_keys, bad_vals, bad_lens, expandable_keys
    # Expansion not needed
    if len(set(sizes)) == 0:
        return info, bad_keys, bad_vals, bad_lens, expandable_keys
    # Expansion needed (len(set(sizes)) == 1)
    values = iterutils.make_list(sizes.pop(), {})
    for key, val in info.items():
        for i in range(len(values)):
            needs_expansion = key.startswith('*')
            if needs_expansion:
                values[i][key[1:]] = val[i]
            else:
                values[i][key] = val
    return values, bad_keys, bad_vals, bad_lens, expandable_keys


def prepare_param_info(
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
        invalid_infos_bad_attr_name=None,
        invalid_infos_bad_attr_value=None,
        invalid_infos_bad_attr_length=None):

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

    if invalid_infos_bad_attr_name is None:
        invalid_infos_bad_attr_name = {}
    if invalid_infos_bad_attr_value is None:
        invalid_infos_bad_attr_value = {}
    if invalid_infos_bad_attr_length is None:
        invalid_infos_bad_attr_length = {}

    keys, values = parse_param_keys(
        params, descs,
        silent_errors,
        silent_warnings,
        throw_on_errors,
        throw_on_warnings,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)[:2]

    # Use default dicts internally.
    # The api should use normal dicts through.
    invalid_infos_bad_attr_name_ = collections.defaultdict(list)
    invalid_infos_bad_attr_value_ = collections.defaultdict(list)
    invalid_infos_bad_attr_length_ = collections.defaultdict(list)

    # These will hold all valid key=>value pairs
    keys_2 = []
    values_2 = []

    # Validate key=>value pairs while exploding values if needed
    # We only explode mappings with attributes requesting explosion
    for key, value in zip(keys, values):
        if iterutils.is_mapping(value):
            # Explode value
            (evalues,
             bad_attr_keys,
             bad_attr_vals,
             bad_attr_lens,
             eattr_keys) = _parse_param_info_item(value)
            # Check for errors
            if bad_attr_keys:
                invalid_infos_bad_attr_name_[key].extend(bad_attr_keys)
            if bad_attr_vals:
                invalid_infos_bad_attr_value_[key].extend(bad_attr_vals)
            if bad_attr_lens:
                invalid_infos_bad_attr_length_[key].extend(eattr_keys)
            # On any error, ignore current value
            if bad_attr_keys or bad_attr_vals or bad_attr_lens:
                continue
            # Replace value with exploded values
            value = evalues
        elif iterutils.is_sequence(value):
            evalues = []
            for j, item in enumerate(value):
                if iterutils.is_mapping(item):
                    # Explode value item
                    (ievalues,
                     bad_attr_keys,
                     bad_attr_vals,
                     bad_attr_lens,
                     eattr_keys) = _parse_param_info_item(item)
                    # Check for errors
                    if bad_attr_keys:
                        invalid_infos_bad_attr_name_[key].append(
                            (j, bad_attr_keys))
                    if bad_attr_vals:
                        invalid_infos_bad_attr_value_[key].append(
                            (j, bad_attr_vals))
                    if bad_attr_lens:
                        invalid_infos_bad_attr_length_[key].append(
                            (j, eattr_keys))
                    # On any error, ignore current value item
                    if bad_attr_keys or bad_attr_vals or bad_attr_lens:
                        continue
                    evalues.extend(iterutils.listify(ievalues))
                # Not a mapping, just copy it
                else:
                    evalues.append(item)
            # Replace values with (potentially) exploded values
            value = evalues
        # key-value is now a valid pair
        keys_2.append(key)
        values_2.append(value)

    # Update the contents of the internal dicts to the api dicts
    invalid_infos_bad_attr_name.update(invalid_infos_bad_attr_name_)
    invalid_infos_bad_attr_value.update(invalid_infos_bad_attr_value_)
    invalid_infos_bad_attr_length.update(invalid_infos_bad_attr_length_)

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

    return dict(zip(keys_2, values_2))


def explode_param_info_2(
        params, descs,
        export_singles=True,
        export_sequences=True,
        silent_errors=False,
        silent_warnings=False,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_keys_syntax=None,
        invalid_keys_unknown=None,
        invalid_keys_repeated=None,
        invalid_keys_bad_scalar=None,
        invalid_keys_bad_vector=None,
        invalid_infos_bad_attr_name=None,
        invalid_infos_bad_attr_value=None,
        invalid_infos_bad_attr_length=None):

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

    if invalid_infos_bad_attr_name is None:
        invalid_infos_bad_attr_name = {}
    if invalid_infos_bad_attr_value is None:
        invalid_infos_bad_attr_value = {}
    if invalid_infos_bad_attr_length is None:
        invalid_infos_bad_attr_length = {}

    keys, values = parse_param_keys(
        params, descs,
        silent_errors,
        silent_warnings,
        throw_on_errors,
        throw_on_warnings,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)[:2]

    # Use default dicts internally.
    # The api should use normal dicts through.
    invalid_infos_bad_attr_name_ = collections.defaultdict(list)
    invalid_infos_bad_attr_value_ = collections.defaultdict(list)
    invalid_infos_bad_attr_length_ = collections.defaultdict(list)

    # These will hold all valid key=>value pairs
    keys_2 = []
    values_2 = []

    # Validate key=>value pairs while exploding values if needed
    # We only explode mappings with attributes requesting explosion
    for key, value in zip(keys, values):
        if iterutils.is_mapping(value):
            # Explode value
            (evalues,
             bad_attr_keys,
             bad_attr_vals,
             bad_attr_lens,
             eattr_keys) = _parse_param_info_item(value)
            # Check for errors
            if bad_attr_keys:
                invalid_infos_bad_attr_name_[key].extend(bad_attr_keys)
            if bad_attr_vals:
                invalid_infos_bad_attr_value_[key].extend(bad_attr_vals)
            if bad_attr_lens:
                invalid_infos_bad_attr_length_[key].extend(eattr_keys)
            # On any error, ignore current value
            if bad_attr_keys or bad_attr_vals or bad_attr_lens:
                continue
            # Replace value with exploded values
            value = evalues
        elif iterutils.is_sequence(value):
            evalues = []
            for j, item in enumerate(value):
                if iterutils.is_mapping(item):
                    # Explode value item
                    (ievalues,
                     bad_attr_keys,
                     bad_attr_vals,
                     bad_attr_lens,
                     eattr_keys) = _parse_param_info_item(item)
                    # Check for errors
                    if bad_attr_keys:
                        invalid_infos_bad_attr_name_[key].append(
                            (j, bad_attr_keys))
                    if bad_attr_vals:
                        invalid_infos_bad_attr_value_[key].append(
                            (j, bad_attr_vals))
                    if bad_attr_lens:
                        invalid_infos_bad_attr_length_[key].append(
                            (j, eattr_keys))
                    # On any error, ignore current value item
                    if bad_attr_keys or bad_attr_vals or bad_attr_lens:
                        continue
                    evalues.extend(iterutils.listify(ievalues))
                # Not a mapping, just copy it
                else:
                    evalues.append(item)
            # Replace values with (potentially) exploded values
            value = evalues
        # key-value is now a valid pair
        keys_2.append(key)
        values_2.append(value)

    # Update the contents of the internal dicts to the api dicts
    invalid_infos_bad_attr_name.update(invalid_infos_bad_attr_name_)
    invalid_infos_bad_attr_value.update(invalid_infos_bad_attr_value_)
    invalid_infos_bad_attr_length.update(invalid_infos_bad_attr_length_)

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

    params = dict(zip(keys_2, values_2))
    def is_dict(x): return iterutils.is_mapping(x)
    return parse_param_values(params, descs, is_dict)



def explode_param_info(
        params,
        silent_errors=False,
        silent_warnings=False,
        throw_on_errors=True,
        throw_on_warnings=False,
        invalid_infos_bad_attr_name=None,
        invalid_infos_bad_attr_value=None,
        invalid_infos_bad_attr_length=None):

    if invalid_infos_bad_attr_name is None:
        invalid_infos_bad_attr_name = {}
    if invalid_infos_bad_attr_value is None:
        invalid_infos_bad_attr_value = {}
    if invalid_infos_bad_attr_length is None:
        invalid_infos_bad_attr_length = {}

    keys = list(params.keys())
    values = list(params.values())

    # Use default dicts internally.
    # The api should use normal dicts through.
    invalid_infos_bad_attr_name_ = collections.defaultdict(list)
    invalid_infos_bad_attr_value_ = collections.defaultdict(list)
    invalid_infos_bad_attr_length_ = collections.defaultdict(list)

    # These will hold all valid key=>value pairs
    keys_2 = []
    values_2 = []

    # Validate key=>value pairs while exploding values if needed
    # We only explode mappings with attributes requesting explosion
    for key, value in zip(keys, values):
        if iterutils.is_mapping(value):
            # Explode value
            (evalues,
             bad_attr_keys,
             bad_attr_vals,
             bad_attr_lens,
             eattr_keys) = _parse_param_info_item(value)
            # Check for errors
            if bad_attr_keys:
                invalid_infos_bad_attr_name_[key].extend(bad_attr_keys)
            if bad_attr_vals:
                invalid_infos_bad_attr_value_[key].extend(bad_attr_vals)
            if bad_attr_lens:
                invalid_infos_bad_attr_length_[key].extend(eattr_keys)
            # On any error, ignore current value
            if bad_attr_keys or bad_attr_vals or bad_attr_lens:
                continue
            # Replace value with exploded values
            value = evalues
        elif iterutils.is_sequence(value):
            evalues = []
            for j, item in enumerate(value):
                if iterutils.is_mapping(item):
                    # Explode value item
                    (ievalues,
                     bad_attr_keys,
                     bad_attr_vals,
                     bad_attr_lens,
                     eattr_keys) = _parse_param_info_item(item)
                    # Check for errors
                    if bad_attr_keys:
                        invalid_infos_bad_attr_name_[key].append(
                            (j, bad_attr_keys))
                    if bad_attr_vals:
                        invalid_infos_bad_attr_value_[key].append(
                            (j, bad_attr_vals))
                    if bad_attr_lens:
                        invalid_infos_bad_attr_length_[key].append(
                            (j, eattr_keys))
                    # On any error, ignore current value item
                    if bad_attr_keys or bad_attr_vals or bad_attr_lens:
                        continue
                    evalues.extend(iterutils.listify(ievalues))
                # Not a mapping, just copy it
                else:
                    evalues.append(item)
            # Replace values with (potentially) exploded values
            value = evalues
        # key-value is now a valid pair
        keys_2.append(key)
        values_2.append(value)

    # Update the contents of the internal dicts to the api dicts
    invalid_infos_bad_attr_name.update(invalid_infos_bad_attr_name_)
    invalid_infos_bad_attr_value.update(invalid_infos_bad_attr_value_)
    invalid_infos_bad_attr_length.update(invalid_infos_bad_attr_length_)

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

    return dict(zip(keys_2, values_2))


def parse_param_values_strict(descs, params_dict, value_type):
    names, indices, values, exprs = parse_param_values(
        params_dict, descs, lambda x: isinstance(x, value_type))[2:]
    enames_from_params = explode_param_names_from_indices(names, indices)
    enames_from_pdescs = explode_param_names_from_pdescs(descs.values(), descs.keys())
    if missing := set(enames_from_pdescs) - set(enames_from_params):
        raise RuntimeError(
            f"information for the following parameters is required "
            f"but not provided: {str(missing)}")
    return values, exprs


def load_params_parameters(info, pdescs, types, loader):
    info, expressions = parse_param_values(
        info, pdescs, lambda x: isinstance(x, types))[4:]
    for key, val in info.items():
        try:
            info[key] = loader(val)
        except RuntimeError as e:
            raise RuntimeError(
                f"could not parse information for parameter '{key}'; "
                f"reason: {str(e)}") from e
    return info | expressions


def dump_params_parameters(parameters, param_types, param_dumper):
    info = dict()
    for key, value in parameters.items():
        if isinstance(value, param_types):
            value = param_dumper(value)
        elif iterutils.is_sequence(value):
            value = [param_dumper(p) if isinstance(p, param_types) else p
                     for p in value]
        info[key] = value
    return info


def _load_function(info, desc):
    if not info:
        return info
    opts = parseutils.parse_options(info, desc, ['file', 'func'])
    return miscutils.get_attr_from_file(opts['file'], opts['func'])


def _dump_function(func, file):
    if not func:
        return func
    with open(file, 'a') as f:
        f.write('\n')
        f.write(textwrap.dedent(inspect.getsource(func)))
        f.write('\n')
    return dict(file=file, func=func.__name__)


def load_params_conversions(info):
    return _load_function(info, 'params value conversions')


def dump_params_conversions(func, file):
    return _dump_function(func, file) if func else None


def load_params_parameters_conversions(
        info, pdescs, param_types, param_loader):
    info = copy.deepcopy(info)
    info['parameters'] = load_params_parameters(
        info['parameters'], pdescs, param_types, param_loader)
    if 'conversions' in info:
        info['conversions'] = load_params_conversions(
            info['conversions'])
    return info


def dump_params_parameters_conversions(
        params, param_types, param_dumper, conversions_file):
    info = dict()
    info.update(parameters=dump_params_parameters(
        params.parameters(), param_types, param_dumper))
    if params.conversions():
        info.update(conversions=dump_params_conversions(
            params.conversions(), conversions_file))
    return info
