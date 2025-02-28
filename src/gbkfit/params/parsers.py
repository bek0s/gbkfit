
import ast
import collections
import copy
import graphlib  # noqa: pycharm bug
import inspect
import itertools
import logging
import numbers
import textwrap

from collections.abc import Callable
from dataclasses import dataclass
from types import NoneType
from typing import Any

import numpy as np

from gbkfit.params import utils as param_utils
from gbkfit.params.pdescs import *
from gbkfit.params.symbols import *
from gbkfit.utils import iterutils, miscutils, parseutils, stringutils


__all__ = [
    'parse_param_keys',
    'parse_param_values',
    'parse_param_expressions',
    'parse_param_info',
    'load_params_parameters',
    'dump_params_parameters'
]


_log = logging.getLogger(__name__)


def _log_msg(level: int, silent: bool, throw: bool, msg: str):
    if throw:
        raise RuntimeError(msg)
    elif not silent:
        _log.log(level, msg)


@dataclass(frozen=True)
class ParseParamKeysResult:
    keys: list[str]
    values: list[Any]
    param_names: list[str]
    param_indices: list[None | int | list[int]]
    invalid_keys_syntax: list[str]
    invalid_keys_unknown: list[str]
    invalid_keys_repeated: dict[str, list[str]]
    invalid_keys_bad_scalar: list[str]
    invalid_keys_bad_vector: dict[str, list[int]]


def parse_param_keys(
        params: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        silent_errors: bool = False,
        silent_warnings: bool = False,
        throw_on_errors: bool = True,
        throw_on_warnings: bool = False):
    """
    Parses parameter keys and validates their syntax, existence, and
    indexing.

    This function processes a dictionary of parameter keys and their
    values, ensuring that the keys follow the correct syntax, exist in
    the provided parameter descriptions, and have valid indexing if
    they correspond to vector parameters. It also detects repeated
    parameters mapped to different keys.

    Parameters
    ----------
    params : dict[str, Any] | None
        Dictionary of parameter keys and their associated values.
        If `None`, an empty dictionary is used.
    pdescs : dict[str, ParamDesc]
        Dictionary mapping parameter names to their descriptions.
    silent_errors : bool, optional
        If `True`, suppresses error messages in logs. Default is `False`.
    silent_warnings : bool, optional
        If `True`, suppresses warning messages in logs. Default is `False`.
    throw_on_errors : bool, optional
        If `True`, raises an exception on errors instead of logging them.
        Default is `True`.
    throw_on_warnings : bool, optional
        If `True`, raises an exception on warnings instead of logging them.
        Default is `False`.
    """

    invalid_keys_syntax: list[str] = []
    invalid_keys_unknown: list[str] = []
    invalid_keys_repeated: dict[str, list[str]] = {}
    invalid_keys_bad_scalar: list[str] = []
    invalid_keys_bad_vector: dict[str, list[int]] = {}

    keys: list[str] = []
    values: list[Any] = []
    param_names: list[str] = []
    param_indices: list[None | int | list[int]] = []
    eparams_to_keys = collections.defaultdict(list)

    for key, value in params.items():
        key = stringutils.remove_white_space(key)
        # Skip keys with invalid syntax
        if not is_param_symbol(key):
            invalid_keys_syntax.append(key)
            continue
        # Extract parameter name and subscript as strings
        # The latter will be None if key has scalar syntax
        name, subscript = parse_param_symbol_into_name_and_subscript_str(key)
        # Skip keys with unknown parameters
        pdesc = pdescs.get(name)
        if not pdesc:
            invalid_keys_unknown.append(key)
            continue
        if isinstance(pdesc, ParamScalarDesc):
            # Skip keys which are supposed to be scalars,
            # but have a subscript
            if subscript is not None:
                invalid_keys_bad_scalar.append(key)
                continue
            # Scalars are exploded already
            eparams_to_keys[key].append(key)
            # Scalars do not have indices
            indices = None
        else:
            size = pdesc.size()
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
                invalid_keys_bad_vector[key] = invalid_indices
                continue
            # Ensure positive indices
            indices = iterutils.unwrap_sequence_indices(indices, size)
            # Explode vector
            for i in indices:
                eparams_to_keys[make_param_symbol(name, i)].append(key)
            # We differentiate basic indexing from slicing or advanced
            # indexing by using an integer instead of a list of integers
            if is_param_symbol_subscript_bindx(subscript):
                indices = indices[0]
        # This is a valid key-value pair
        keys.append(key)
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
    for key in invalid_keys_repeated:
        i = keys.index(key)
        del keys[i]
        del values[i]
        del param_names[i]
        del param_indices[i]

    # Handle issues
    for label, items, level, silent, throw in [
        ("keys with unknown parameters found:",
         invalid_keys_unknown,
         logging.WARNING, silent_warnings, throw_on_warnings),
        ("keys with invalid syntax found:",
         invalid_keys_syntax,
         logging.ERROR, silent_errors, throw_on_errors),
        ("keys with repeated parameters found:",
         invalid_keys_repeated,
         logging.ERROR, silent_errors, throw_on_errors),
        ("scalar parameter keys with vector syntax found:",
         invalid_keys_bad_scalar,
         logging.ERROR, silent_errors, throw_on_errors),
        ("vector parameter keys with out-of-range indices found:",
         invalid_keys_bad_vector,
         logging.ERROR, silent_errors, throw_on_errors),
    ]:
        if items:
            _log_msg(level, silent, throw, f"{label} {items}")

    return ParseParamKeysResult(
        keys,
        values,
        param_names,
        param_indices,
        invalid_keys_syntax,
        invalid_keys_unknown,
        invalid_keys_repeated,
        invalid_keys_bad_scalar,
        invalid_keys_bad_vector)


@dataclass(frozen=True)
class ParseParamValuesResult:
    exploded_params: dict[str, Any]
    expressions: dict[str, str]
    invalid_values_bad_value: list[str]
    invalid_values_bad_evalue: dict[str, list[str]]
    invalid_values_bad_length: list[str]
    parse_param_keys_result: ParseParamKeysResult


def _is_param_expression(x: Any, accept_vec: bool = True) -> bool:
    """
    Determine whether a value can be considered a parameter expression.

    A value is considered a parameter expression if it is:
    - `None`, a string, or a real number.
    - A vector (list, tuple, or 1D NumPy array) if `accept_vec` is
      True, where all elements must be `None`, strings, or real
      numbers.

    This function does not evaluate whether the value is a valid
    expression in a specific computational context—it only checks its
    structural validity.
    """
    def is_nil(i): return isinstance(i, NoneType)
    def is_str(i): return isinstance(i, str)
    def is_num(i): return isinstance(i, numbers.Real)
    def is_single(i): return is_nil(i) or is_str(i) or is_num(i)
    def is_numpy(i): return isinstance(i, np.ndarray) and i.ndim == 1
    def is_vector(i): return isinstance(i, (tuple, list)) or is_numpy(i)

    return is_single(x) or (
            accept_vec and is_vector(x) and all(is_single(i) for i in x))


def parse_param_values(
        params: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        is_value_fun: Callable[[Any], bool],
        silent_errors: bool = False,
        silent_warnings: bool = False,
        throw_on_errors: bool = True,
        throw_on_warnings: bool = False
) -> ParseParamValuesResult:
    """
    Parse parameter values, validating their structure and categorizing
    them.

    This function processes a dictionary of parameter values,
    ensuring their validity based on descriptions (`pdescs`). It
    categorizes values as valid exploded parameters, expressions,
    or invalid values due to type mismatches or length issues.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of parameter names and their associated values.
    pdescs : dict[str, ParamDesc]
        Dictionary of parameter descriptions.
    is_value_fun : Callable[[Any], bool]
        Function to determine whether a value is valid.
    silent_errors : bool, optional
        If True, suppresses error messages, by default False.
    silent_warnings : bool, optional
        If True, suppresses warnings, by default False.
    throw_on_errors : bool, optional
        If True, raises exceptions on errors, by default True.
    throw_on_warnings : bool, optional
        If True, raises exceptions on warnings, by default False.
    """

    parse_param_keys_result = parse_param_keys(
        params, pdescs,
        silent_errors,
        silent_warnings,
        throw_on_errors,
        throw_on_warnings)

    keys = parse_param_keys_result.keys
    values = parse_param_keys_result.values
    param_names = parse_param_keys_result.param_names
    param_indices = parse_param_keys_result.param_indices

    invalid_values_bad_value: list[str] = []
    invalid_values_bad_evalue: dict[str, list[str]] = {}
    invalid_values_bad_length: list[str] = []

    expressions = {}
    exploded_names = []
    exploded_values = []

    # Use default dicts internally for convenience.
    # The api should use normal dicts through.
    invalid_values_bad_evalue_ = collections.defaultdict(list)

    # Iterate over lhs key, name, indices and rhs values.
    for key, value, name, indices in zip(
            keys, values, param_names, param_indices, strict=True):
        if is_value_fun(value):
            # If key is a scalar, just append the name/value
            if indices is None:
                exploded_names.append(name)
                exploded_values.append(value)
            # If key is a vector, make all symbol names and create
            # copies of the value. Then, append those to the output
            else:
                indices = iterutils.listify(indices)
                exploded_names.extend(
                    make_param_symbols_from_name_and_indices(name, indices))
                exploded_values.extend(
                    iterutils.make_list(len(indices), value))
        elif iterutils.is_sequence(value):
            # Make symbol names from name and indices
            i_exploded_names = make_param_symbols_from_name_and_indices(
                name, indices)
            # Mark the value as invalid and ignore it, if any is true:
            # - Its length is not equal to the number of symbols
            # - The key is a vector symbol with basic indexing.
            #   We can not set a vector element with a sequence.
            if (len(i_exploded_names) != len(value)
                    or is_param_symbol_vector_bindx(key)):
                invalid_values_bad_length.append(key)
                continue
            # At this point the lhs and rhs have compatible lengths
            for i_exploded_name, i_exploded_value in zip(
                    i_exploded_names, value, strict=True):
                if is_value_fun(i_exploded_value):
                    exploded_names.append(i_exploded_name)
                    exploded_values.append(i_exploded_value)
                elif _is_param_expression(i_exploded_value, accept_vec=False):
                    expressions[i_exploded_name] = i_exploded_value
                # i_exploded_value is not of a recognized type
                else:
                    invalid_values_bad_evalue_[key].append(i_exploded_name)
        elif _is_param_expression(value, accept_vec=False):
            expressions[key] = value
        # value is not of a recognized type
        else:
            invalid_values_bad_value.append(key)

    # Copy the contents of the internal dict to the api dict
    invalid_values_bad_evalue.update(invalid_values_bad_evalue_)

    # Handle issues
    for label, items, level, silent, throw in [
        ("keys with invalid value found:",
         invalid_values_bad_value,
         logging.ERROR, silent_warnings, throw_on_warnings),
        ("keys with invalid exploded value(s) found:",
         invalid_values_bad_evalue,
         logging.ERROR, silent_errors, throw_on_errors),
        ("keys with value of incompatible length found:",
         invalid_values_bad_length,
         logging.ERROR, silent_errors, throw_on_errors)
    ]:
        if items:
            _log_msg(level, silent, throw, f"{label} {items}")

    exploded_params = dict(zip(exploded_names, exploded_values, strict=True))

    return ParseParamValuesResult(
        exploded_params,
        expressions,
        invalid_values_bad_value,
        invalid_values_bad_evalue,
        invalid_values_bad_length,
        parse_param_keys_result)


def parse_param_values_strict(
        params: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        value_types: tuple[Any, ...]
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Strictly parse parameter values, enforcing type constraints.

    This function ensures that all parameter values conform to the
    given `value_types` and that all required parameters are
    present. It raises errors on any missing or unknown parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of parameter names and their associated values.
    pdescs : dict[str, ParamDesc]
        Dictionary of parameter descriptions.
    value_types : tuple[Any, ...]
        Allowed types for parameter values.
    """
    parse_param_values_result = parse_param_values(
        params, pdescs, lambda x: isinstance(x, value_types),
        throw_on_errors=True,
        throw_on_warnings=True)
    values_dict, expressions_dict = (
        parse_param_values_result.exploded_params,
        parse_param_values_result.expressions)
    # Make sure we have exactly the parameters we need.
    # No missing or unknown parameters are allowed.
    exploded_names = make_param_symbols_from_names_and_indices(
        parse_param_values_result.parse_param_keys_result.param_names,
        parse_param_values_result.parse_param_keys_result.param_indices)
    param_utils.ensure_no_missing_params(exploded_names, pdescs)
    param_utils.ensure_no_unknown_params(exploded_names, pdescs)
    return values_dict, expressions_dict


class _ParamExprVisitor(ast.NodeVisitor):
    """
    AST visitor that collects parameter symbols and validates their usage.
    """

    def __init__(self, pdescs: dict[str, ParamDesc]):
        self._pdescs = pdescs
        self._symbols = dict()
        self._invalid_scalars = list()
        self._invalid_vectors = dict()

    def symbols(self) -> dict[str, None | list[int]]:
        """
        Returns a dictionary of recognized parameter symbols.

        - Scalars are mapped to `None`.
        - Vectors are mapped to a list of valid indices.
        """
        return self._symbols

    def invalid_scalars(self) -> list[str]:
        """
        Returns a list of invalid scalar expressions.

        These occur when a scalar parameter is accessed as if it were
        a vector.
        """
        return self._invalid_scalars

    def invalid_vectors(self) -> dict[str, None | list[int]]:
        """
        Returns a dictionary of invalid vector expressions.

        - If a subscript syntax is unsupported, the value is `None`.
        - If some indices are out of range, the value is a list of
          invalid indices.
        """
        return self._invalid_vectors

    def visit_Name(self, node: ast.Name) -> None:
        """Processes a standalone symbol reference."""
        code = ast.unparse(node).strip('\n')
        name = code
        pdesc = self._pdescs.get(name)
        # If symbol is not recognized, ignore it.
        if not pdesc:
            return
        # Store symbol name along with its indices (if a vector).
        self._symbols[name] = None
        if isinstance(pdesc, ParamVectorDesc):
            indices = list(range(pdesc.size()))
            self._symbols[name] = indices

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Processes an indexed (subscript) parameter reference."""
        code = ast.unparse(node).strip('\n')
        name, subscript = parse_param_symbol_into_name_and_subscript_str(code)
        pdesc = self._pdescs.get(name)
        # If symbol is not recognized,
        # traverse deeper but ignore the node itself.
        if not pdesc:
            self.generic_visit(node)
            return
        # If symbol is supposed to be a scalar,
        # mark symbol as invalid and ignore this node.
        if isinstance(pdesc, ParamScalarDesc):
            self._invalid_scalars.append(code)
            return
        # If symbol subscript syntax is not supported,
        # mark symbol as invalid and ignore this node.
        if not is_param_symbol_subscript(subscript):
            self._invalid_vectors[code] = None
            return
        # Parse and validate indices, handling out-of-range cases.
        size = pdesc.size()
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
        # The method visit_Subscript() can be called multiple times for
        # the same expression. This is why we need to use list.extend()
        self._symbols.setdefault(name, list()).extend(indices)


@dataclass(frozen=True)
class ParseParamExpressionsResult:
    expression_keys: list[str]
    expression_values: list[str]
    expression_param_names: list[str]
    expression_param_indices: list[None | int | list[int]]
    expression_ast_root_nodes: list[Any]
    invalid_expressions_bad_value: list[str]
    invalid_expressions_bad_syntax: list[str]
    invalid_expressions_bad_scalar: dict[str, None | set[str]]
    invalid_expressions_bad_vector: dict[str, dict[str, None | list[int]]]
    parse_param_keys_result: ParseParamKeysResult


def parse_param_expressions(
        params: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        silent_errors: bool = False,
        silent_warnings: bool = False,
        throw_on_errors: bool = True,
        throw_on_warnings: bool = False
) -> ParseParamExpressionsResult:
    """
    Parse parameter expressions in the provided parameters and validate
    them.

    This function processes a dictionary of parameters and their
    associated descriptions, identifies parameters with expressions,
    parses them into abstract syntax trees (AST), and validates the
    expressions for syntax errors and invalid symbols. It also handles
    dependencies between parameters and returns the results in a
    structured format.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of parameter names and their associated values.
    pdescs : dict[str, ParamDesc]
        Dictionary of parameter descriptions.
    silent_errors : bool, optional
        If True, suppresses error messages, by default False.
    silent_warnings : bool, optional
        If True, suppresses warnings, by default False.
    throw_on_errors : bool, optional
        If True, raises exceptions on errors, by default True.
    throw_on_warnings : bool, optional
        If True, raises exceptions on warnings, by default False.
    """
    parse_param_keys_result = parse_param_keys(
        params, pdescs,
        silent_errors,
        silent_warnings,
        throw_on_errors,
        throw_on_warnings)

    keys = parse_param_keys_result.keys
    values = parse_param_keys_result.values
    param_names = parse_param_keys_result.param_names
    param_indices = parse_param_keys_result.param_indices

    invalid_expressions_bad_value: list[str] = []
    invalid_expressions_bad_syntax: list[str] = []
    invalid_expressions_bad_scalar: dict[str, None | list[str]] = {}
    invalid_expressions_bad_vector: dict[str, dict[str, None | list[int]]] = {}

    expression_keys = []
    expression_values = []
    expression_param_names = []
    expression_param_indices = []
    expression_ast_root_nodes = []
    eparams_to_keys = {}
    expression_param_symbols = []

    for key, value, name, indices in zip(
            keys, values, param_names, param_indices, strict=True):
        # Skip invalid expressions
        if not _is_param_expression(value, accept_vec=True):
            invalid_expressions_bad_value.append(key)
            continue
        # All expressions can be converted to str
        value = str(value)
        # Parse expression and retrieve AST root node
        try:
            ast_root = ast.parse(value)
        except SyntaxError:
            invalid_expressions_bad_syntax.append(key)
            continue
        # Discover all symbols in expression
        visitor = _ParamExprVisitor(pdescs)
        visitor.visit(ast_root)
        # Keep track of invalid symbols
        if visitor.invalid_scalars():
            invalid_expressions_bad_scalar[key] = visitor.invalid_scalars()
        if visitor.invalid_vectors():
            invalid_expressions_bad_vector[key] = visitor.invalid_vectors()
        # Skip expressions with invalid symbols
        if visitor.invalid_scalars() or visitor.invalid_vectors():
            continue
        # Keep track of the parent key for each exploded param
        eparams_to_keys.update(
            {eparam: key for eparam in
             make_param_symbols_from_name_and_indices(name, indices)})
        # This is a valid key-value pair
        expression_keys.append(key)
        expression_values.append(value)
        expression_param_names.append(name)
        expression_param_indices.append(indices)
        expression_ast_root_nodes.append(ast_root)
        expression_param_symbols.append(visitor.symbols())

        # Handle issues
        for label, items, level, silent, throw in [
            ("the value of the following parameter keys "
             "cannot be converted to an expression:",
             invalid_expressions_bad_value,
             logging.WARNING, silent_warnings, throw_on_warnings),
            ("the value of the following parameter keys "
             "contains an expression with syntax errors:",
             invalid_expressions_bad_syntax,
             logging.ERROR, silent_errors, throw_on_errors),
            ("the value of the following parameter keys "
             f"contains an expression with invalid scalar symbols:",
             invalid_expressions_bad_scalar,
             logging.ERROR, silent_errors, throw_on_errors),
            ("the value of the following parameter keys "
             "contains an expression with invalid vector symbols:",
             invalid_expressions_bad_vector,
             logging.ERROR, silent_errors, throw_on_errors),
        ]:
            if items:
                _log_msg(level, silent, throw, f"{label} {items}")

    # Build dependency graph of all exploded parameters
    graph = graphlib.TopologicalSorter()
    for name, indices, symbols in zip(
            expression_param_names,
            expression_param_indices,
            expression_param_symbols,
            strict=True):
        # lhs depends on rhs (obviously!)
        eparams_lhs = make_param_symbols_from_name_and_indices(name, indices)
        eparams_rhs = make_param_symbols_from_names_and_indices(
            symbols.keys(), symbols.values())
        for pair in itertools.product(eparams_lhs, eparams_rhs):
            # pair[0] depends on pair[1]
            graph.add(pair[1], pair[0])

    # Perform topological sorting on the graph
    try:
        sorted_eparams = list(reversed(list(graph.static_order())))
    except graphlib.CycleError as e:
        cycle_details = " -> ".join(e.args[1]) if len(
            e.args) > 1 else "Unknown cycle"
        raise RuntimeError(
            f"Circular dependencies detected in parameter expressions. "
            f"The following cycle was found: {cycle_details}") from e

    # Using the sorted exploded parameters derive a list of keys
    # The order of keys reflects the expression evaluation order
    sorted_keys = []
    for eparam in sorted_eparams:
        key = eparams_to_keys.get(eparam)
        if key and key not in sorted_keys:
            sorted_keys.append(key)
    for key in expression_keys:
        if key not in sorted_keys:
            sorted_keys.append(key)

    # Apply the correct order to the output arrays
    order = [expression_keys.index(key) for key in sorted_keys]
    expression_keys = [expression_keys[i] for i in order]
    expression_values = [expression_values[i] for i in order]
    expression_param_names = [expression_param_names[i] for i in order]
    expression_param_indices = [expression_param_indices[i] for i in order]
    expression_ast_root_nodes = [expression_ast_root_nodes[i] for i in order]

    return ParseParamExpressionsResult(
        expression_keys,
        expression_values,
        expression_param_names,
        expression_param_indices,
        expression_ast_root_nodes,
        invalid_expressions_bad_value,
        invalid_expressions_bad_syntax,
        invalid_expressions_bad_scalar,
        invalid_expressions_bad_vector,
        parse_param_keys_result)


def _explode_param_info_item(
        info: dict[str, Any]
) -> tuple[
        list[dict[str, Any]] | dict[str, Any],
        list[str], list[str], bool, list[str]]:
    """
    Validates and expands parameter attributes if required.

    This function processes a dictionary of parameter attributes,
    ensuring that attribute names are valid and expandable attributes
    have consistent sequence lengths. If expansion is required, it
    generates a list of expanded attribute dictionaries.

    Parameters
    ----------
    info : dict[str, Any]
        Dictionary mapping attribute names to their values. Attribute
        names prefixed with '*' indicate that their values must be
        expanded as sequences.

    Returns
    -------
    values : list[dict[str, Any]] | dict[str, Any]
        List of dictionaries with expanded attributes if expansion is
        needed. Otherwise, returns the original dictionary.
    bad_keys : list[str]
        List of attribute names with invalid syntax.
    bad_vals : list[str]
        List of attribute names whose values are not sequences but
        required expansion.
    bad_lens : bool
        `True` if attribute expansion fails due to mismatched sequence
        lengths, `False` otherwise.
    expandable_keys : list[str]
        List of attributes that require expansion.
    """
    bad_keys = []
    bad_vals = []
    expandable_keys = []
    # Validate attributes
    sizes = []
    for key, val in info.items():
        needs_expansion = key.startswith('*')
        if needs_expansion:
            key = key[1:]
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
    bad_lens = len(set(sizes)) > 1
    # Any error causes failure
    if bad_lens or bad_keys or bad_vals:
        return [], bad_keys, bad_vals, bad_lens, expandable_keys
    # Expansion not needed
    if len(set(sizes)) == 0:
        return info, bad_keys, bad_vals, bad_lens, expandable_keys
    # Expansion needed (len(set(sizes)) == 1)
    values = iterutils.make_list(sizes[0], {})
    for key, val in info.items():
        needs_expansion = key.startswith('*')
        key = key[1:] if needs_expansion else key  # Strip '*' once
        for i in range(len(values)):
            values[i][key] = val[i] if needs_expansion else val
    return values, bad_keys, bad_vals, bad_lens, expandable_keys


@dataclass(frozen=True)
class ParseParamInfoResult:
    info: dict[str, Any]
    invalid_infos_bad_attr_name: dict[str, list[str] | list[tuple[int, str]]]
    invalid_infos_bad_attr_value: dict[str, list[str] | list[tuple[int, str]]]
    invalid_infos_bad_attr_length: dict[str, list[str] | list[tuple[int, str]]]
    parse_param_keys_result: ParseParamKeysResult


def parse_param_info(
        params: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        silent_errors: bool = False,
        silent_warnings: bool = False,
        throw_on_errors: bool = True,
        throw_on_warnings: bool = False
) -> ParseParamInfoResult:
    """
    Validates and processes parameter attributes.

    This function verifies the syntax and validity of parameter keys,
    expands attribute mappings when necessary, and collects errors or
    warnings related to attribute validity.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary mapping parameter names to their values.
    pdescs : dict[str, ParamDesc]
        Dictionary of parameter descriptions, used for validation.
    silent_errors : bool, optional
        Whether to suppress error messages, by default False.
    silent_warnings : bool, optional
        Whether to suppress warning messages, by default False.
    throw_on_errors : bool, optional
        Whether to raise exceptions on errors, by default True.
    throw_on_warnings : bool, optional
        Whether to raise exceptions on warnings, by default False.
    """

    parse_param_keys_result = parse_param_keys(
        params, pdescs,
        silent_errors,
        silent_warnings,
        throw_on_errors,
        throw_on_warnings)

    keys = parse_param_keys_result.keys
    values = parse_param_keys_result.values

    invalid_infos_bad_attr_name = collections.defaultdict(list)
    invalid_infos_bad_attr_value = collections.defaultdict(list)
    invalid_infos_bad_attr_length = collections.defaultdict(list)

    # These will hold all valid key=>value pairs
    valid_keys: list[str] = []
    valid_values: list[Any] = []

    # Validate key=>value pairs while exploding values if needed
    # We only explode mappings with attributes requesting explosion
    for key, value in zip(keys, values, strict=True):
        if iterutils.is_mapping(value):
            # Explode value
            (exploded_values,
             bad_attr_keys,
             bad_attr_vals,
             bad_attr_lens,
             exploded_attr_keys) = _explode_param_info_item(value)
            # Check for errors
            if bad_attr_keys:
                invalid_infos_bad_attr_name[key].extend(bad_attr_keys)
            if bad_attr_vals:
                invalid_infos_bad_attr_value[key].extend(bad_attr_vals)
            if bad_attr_lens:
                invalid_infos_bad_attr_length[key].extend(exploded_attr_keys)
            # On any error, ignore current value
            if bad_attr_keys or bad_attr_vals or bad_attr_lens:
                continue
            # Replace value with exploded values
            value = exploded_values
        elif iterutils.is_sequence(value):
            exploded_values = []
            for j, item in enumerate(value):
                if iterutils.is_mapping(item):
                    # Explode value item
                    (i_exploded_values,
                     bad_attr_keys,
                     bad_attr_vals,
                     bad_attr_lens,
                     exploded_attr_keys) = _explode_param_info_item(item)
                    # Check for errors
                    if bad_attr_keys:
                        invalid_infos_bad_attr_name[key].append(
                            (j, bad_attr_keys))
                    if bad_attr_vals:
                        invalid_infos_bad_attr_value[key].append(
                            (j, bad_attr_vals))
                    if bad_attr_lens:
                        invalid_infos_bad_attr_length[key].append(
                            (j, exploded_attr_keys))
                    # On any error, ignore current value item
                    if bad_attr_keys or bad_attr_vals or bad_attr_lens:
                        continue
                    exploded_values.extend(
                        iterutils.listify(i_exploded_values))
                # Not a mapping, just copy it
                else:
                    exploded_values.append(item)
            # Replace values with (potentially) exploded values
            value = exploded_values
        # key-value is now a valid pair
        valid_keys.append(key)
        valid_values.append(value)

    # Convert defaultdict to dict
    invalid_infos_bad_attr_name = dict(invalid_infos_bad_attr_name)
    invalid_infos_bad_attr_value = dict(invalid_infos_bad_attr_value)
    invalid_infos_bad_attr_length = dict(invalid_infos_bad_attr_length)

    for label, items, level, silent, throw in [
        ("keys with invalid attributes found (bad name):",
         invalid_infos_bad_attr_name,
         logging.ERROR, silent_warnings, throw_on_warnings),
        ("keys with invalid attributes found (bad value):",
         invalid_infos_bad_attr_value,
         logging.ERROR, silent_errors, throw_on_errors),
        ("keys with invalid attributes found (bad length):",
         invalid_infos_bad_attr_length,
         logging.ERROR, silent_errors, throw_on_errors)
    ]:
        if items:
            _log_msg(level, silent, throw, f"{label} {items}")

    valid_info = dict(zip(valid_keys, valid_values, strict=True))
    return ParseParamInfoResult(
        valid_info,
        invalid_infos_bad_attr_name,
        invalid_infos_bad_attr_value,
        invalid_infos_bad_attr_length,
        parse_param_keys_result)


def load_params_parameters(
        info: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        param_types: type | tuple[type, ...],
        param_loader
):
    def is_value(x): return isinstance(x, param_types)
    parse_result = parse_param_values(info, pdescs, is_value)
    values, expressions = (
        parse_result.exploded_params, parse_result.expressions)
    for key, val in values.items():
        try:
            values[key] = param_loader(val)
        except Exception as e:
            raise RuntimeError(
                f"could not parse info for parameter '{key}'; "
                f"reason: {e}") from e
    return values | expressions


def dump_params_parameters(
        parameters,
        param_types,
        param_dumper
):
    info = dict()
    for key, val in parameters.items():
        if isinstance(val, param_types):
            val = param_dumper(val)
        elif iterutils.is_sequence(val):
            val = [param_dumper(p) if isinstance(p, param_types)
                   else p for p in val]
        info[key] = val
    return info


def _load_function(info, desc):
    opts = parseutils.parse_options(info, desc, {'file', 'func'})
    return miscutils.get_attr_from_file(opts['file'], opts['func'])


def _dump_function(func, file):
    with open(file, 'a') as f:
        f.write('\n')
        f.write(textwrap.dedent(inspect.getsource(func)))
        f.write('\n')
    return dict(file=file, func=func.__name__)


def load_params_conversions(info):
    return _load_function(info, 'params value conversions')


def dump_params_conversions(func, file):
    return _dump_function(func, file)


def load_params_parameters_conversions(
        info: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        param_types: type | tuple[type, ...],
        param_loader
):
    info = copy.deepcopy(info)
    # parseutils.load_option_and_update_info(param_loader, info, 'properties', )
    info['properties'] = load_params_parameters(
        info['properties'], pdescs, param_types, param_loader)
    if 'conversions' in info:
        info['conversions'] = load_params_conversions(
            info['conversions'])

    # parseutils.load_option_and_update_info()
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


def _merge_pdescs(pdescs1: dict, pdescs2: dict):
    pdescs1 = pdescs1 or {}
    pdescs2 = pdescs2 or {}
    if intersection := set(pdescs1) & set(pdescs2):
        raise RuntimeError(
            f"the following pdescs conflict with "
            f"the parameters of the objective function: "
            f"{str(intersection)}; "
            f"please choose different names")
    return pdescs1 | pdescs2


def load_params_common(
        info: dict[str, Any],
        pdescs: dict[str, ParamDesc],
        param_types: tuple[type, ...],
        param_loader
):
    info = copy.deepcopy(info)
    if 'descriptions' in info:
        pdescs_new = load_pdescs_dict(info['descriptions'])
        pdescs = _merge_pdescs(pdescs, pdescs_new)
    info['parameters'] = load_params_parameters(
        info['parameters'], pdescs, param_types, param_loader)
    if 'conversions' in info:
        info['conversions'] = load_params_conversions(
            info['conversions'])
    return info, pdescs
