
import ast
import copy
import logging
import types
from collections.abc import Callable
from numbers import Real
from typing import Any

import numpy as np

from gbkfit.params import parsers as param_parsers, utils as param_utils
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from gbkfit.params.symbols import *
from gbkfit.utils import iterutils, miscutils


__all__ = [
    'Interpreter'
]


_log = logging.getLogger(__name__)


class _Transformer(ast.NodeTransformer):
    """
    AST Transformer that modifies variable references to be subscript
    lookups within a 'params' dictionary.

    This class extends ast.NodeTransformer to convert specified
    variable names into dictionary lookups, facilitating parameterized
    expression evaluation.
    """

    def __init__(self, pdescs: dict[str, ParamDesc]):
        self._pdescs = pdescs

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """
        Visits a variable name node and transforms it into a dictionary
        lookup if the variable exists in `pdescs`.

        This transformation applies to both left-hand side and
        right-hand side variable references, preserving the original
        context.

        Examples
        --------
        If `node.id` is `"a"` and `"a"` is in `pdescs` but `b` is not,
        this method changes:

            a = b + 1

        to:

            params['a'] = b + 1
        """
        if node.id in self._pdescs:
            node = ast.Subscript(
                value=ast.Name(id='params', ctx=ast.Load()),
                slice=ast.Constant(value=node.id),
                ctx=node.ctx)
        return node


def _make_expressions_func(
        pdescs: dict[str, ParamDesc], expressions: dict[str, str]
) -> tuple[Callable, str]:
    """
    Generates a function that evaluates parameter expressions.

    This function creates an executable function based on the provided
    parameter descriptions and expressions. The LHS of each expression
    corresponds to one or more parameters being assigned values, while
    the RHS contains an expression that may involve other parameters.

    The function transforms parameter references into dictionary
    lookups using the _Transformer class and returns both the
    compiled function and its source code as a string.

    The generated function expects a 'params' dictionary as its single
    argument, which will be used for parameter lookups and assignments.

    Examples
    --------
    Given:
        pdescs = {'a': ParamScalarDesc(), 'b': ParamScalarDesc()}
        expressions = {'a': 'b + 10'}

    This would generate a function equivalent to:

        def expressions(params):
            params['a'] = params['b'] + 10
    """
    indent = ' ' * 4
    globals_ = dict(np=np)
    source = 'def expressions(params):\n'
    for key, val in expressions.items():
        line_ast = _Transformer(pdescs).visit(ast.parse(f'{key} = {val}'))
        line_src = ast.unparse(line_ast).strip('\n')
        source += f'{indent}{line_src}\n'
    try:
        code_obj = compile(source, filename='<string>', mode='exec')
        func_obj = types.FunctionType(code_obj.co_consts[0], globals_)
        return func_obj, source
    except Exception as e:
        raise RuntimeError(
            f"exception thrown while compiling "
            f"auto-generated expressions function: {str(e)}") from e


class Interpreter:

    def __init__(
            self,
            pdescs: dict[str, ParamDesc],
            expressions_dict: dict[str, Any],
            expressions_func: Callable | None = None
    ):
        self._pdescs = copy.deepcopy(pdescs)
        # Setup parameter storage and mappings
        self._iparams, self._eparams_nmapping, self._eparams_imapping = (
            self._setup_parameter_storage_and_mappings(pdescs))
        # From the expression dict, extract pairs with:
        # - Expressions (tied parameters)
        # - Nones (tied parameters)
        # - Reals (fixed parameters)
        def is_none(x): return isinstance(x, type(None))
        def is_real(x): return isinstance(x, Real)
        parse_param_values_result = param_parsers.parse_param_values(
            expressions_dict, pdescs, lambda x: is_none(x) or is_real(x))
        evalues, expressions = (
            parse_param_values_result.exploded_params,
            parse_param_values_result.expressions)
        nones_dict = dict(filter(lambda x: is_none(x[1]), evalues.items()))
        reals_dict = dict(filter(lambda x: is_real(x[1]), evalues.items()))
        # Apply exploded None and Real values to the parameter storage
        self._apply_exploded_params(evalues)
        # Parse expressions and extract various ordered information
        parse_result = param_parsers.parse_param_expressions(
            expressions, pdescs)
        expr_keys, expr_values, expr_keys_names, expr_keys_indices = (
            parse_result.expression_keys,
            parse_result.expression_values,
            parse_result.expression_param_names,
            parse_result.expression_param_indices)
        # From now on, work with the ordered expressions
        expressions = dict(zip(expr_keys, expr_values, strict=True))
        # Extract the exploded names for all parameters and create
        # various groups for convenience
        enames_all = param_utils.make_param_symbols_from_pdescs(
            pdescs.values(), pdescs.keys())
        enames_none = list(nones_dict.keys())
        enames_fixed = list(reals_dict.keys())
        enames_tied = param_utils.make_param_symbols_from_names_and_indices(
            expr_keys_names, expr_keys_indices)
        # Parameters with value None are considered tied because
        # their value is expected to be set in the expression function
        enames_tied += enames_none
        enames_notfree = enames_tied + enames_fixed
        # Order exploded names based on the supplied descriptions
        enames_free = [n for n in enames_all if n not in enames_notfree]
        enames_tied = [n for n in enames_all if n in enames_tied]
        enames_fixed = [n for n in enames_all if n in enames_fixed]
        enames_notfree = [n for n in enames_all if n in enames_notfree]
        # Validate expressions configuration and
        # setup expressions function
        expressions_func_obj, expressions_func_src, expressions_func_gen = (
            self._setup_expressions_function(
                pdescs, expressions, expressions_func, enames_none))
        # Assign member variables.
        self._fixed_dict = reals_dict
        self._expressions_func_obj = expressions_func_obj
        self._expressions_func_src = expressions_func_src
        self._expressions_func_gen = expressions_func_gen
        self._enames_all = enames_all
        self._enames_free = enames_free
        self._enames_tied = enames_tied
        self._enames_fixed = enames_fixed
        self._enames_notfree = enames_notfree

    @staticmethod
    def _setup_parameter_storage_and_mappings(
            pdescs: dict[str, ParamDesc]
    ) -> tuple[
        dict[str, Real | np.ndarray],
        dict[str, str],
        dict[str, None | list[int]]
    ]:
        """
        Set up storage and mappings for imploded and exploded
        parameters.

        Create mapping (iparams):
        - imploded parameter name => imploded parameter value
          For scalar parameter values we use floats.
          For vector parameter values we use numpy arrays of floats.
        This mapping holds the final imploded parameters after
        expression evaluation.

        Create mappings (eparams_nmapping, eparams_imapping):
        - exploded parameter name => imploded parameter name
        - exploded parameter name => imploded parameter indices
        These mappings allow to quickly map exploded to imploded
        parameters.
        """
        iparams: dict[str, Real | np.ndarray] = dict()
        eparams_nmapping: dict[str, str] = dict()
        eparams_imapping: dict[str, None | list[int]] = dict()
        for name, pdesc in pdescs.items():
            if isinstance(pdesc, ParamScalarDesc):
                iparams[name] = np.nan
                eparams_nmapping[name] = name
                eparams_imapping[name] = None
            elif isinstance(pdesc, ParamVectorDesc):
                size = pdesc.size()
                indices = list(range(size))
                enames = param_utils.make_param_symbols_from_name_and_indices(
                    name, indices)
                iparams[name] = np.full(size, np.nan)
                eparams_nmapping.update(
                    dict(zip(enames, [name] * size, strict=True)))
                eparams_imapping.update(
                    dict(zip(enames, indices, strict=True)))
            else:
                raise RuntimeError("impossible")
        return iparams, eparams_nmapping, eparams_imapping

    @staticmethod
    def _setup_expressions_function(
            pdescs: dict[str, ParamDesc],
            expressions: dict[str, str],
            expressions_func: Callable,
            enames_none: list[str]
    ) -> tuple[Callable, str, bool]:
        """
        Validate the expression configuration and setup expressions
        function.

        If expressions are given via `expressions`, generate the
        expressions function.

        If expressions are given via `expressions_func`, try to
        retrieve, cleanup, and store its source code. The source code
        may not always be available.
        """
        def get_func_details(func: Callable) -> str:
            """Extract readable details from a function object."""
            qualname = func.__qualname__
            filename = func.__code__.co_filename \
                if hasattr(func, "__code__") else "<unknown file>"
            return f"{filename}:{qualname}"
        # We cannot have expressions and an expressions function at the
        # same time, because it is very hard to implement it robustly
        # while avoiding any ambiguities.
        if expressions and expressions_func:
            func_details = get_func_details(expressions_func)
            raise RuntimeError(
                f"the following expressions were provided: "
                f"{expressions}; "
                f"the following expressions function was provided: "
                f"{func_details}; "
                f"expressions and expression function are mutually exclusive")
        # Parameters with value None are considered tied and
        # their value must be set in the expression function
        if enames_none and not expressions_func:
            raise RuntimeError(
                f"the following parameters are set to None: {enames_none}; "
                f"this implies that they are tied parameters and "
                f"their value must be set in an expression function; "
                f"however, an expression function was not provided")
        # Providing an expression function while no parameters are
        # marked as tied is pointless, unless the user wants to do
        # something sneaky. Let them do it, but emit warning.
        if not enames_none and expressions_func:
            func_details = get_func_details(expressions_func)
            _log.warning(
                f"the following expression function was provided: "
                f"{func_details}; "
                f"however, no tied parameters have been defined; "
                f"are you certain you know what you are doing?")
        # Setup expressions function
        expressions_func_obj = None
        expressions_func_src = None
        expressions_func_gen = False
        if expressions:
            expressions_func_obj, expressions_func_src = (
                _make_expressions_func(pdescs, expressions))
            expressions_func_gen = True
        elif expressions_func:
            expressions_func_obj = expressions_func
            expressions_func_src = miscutils.get_source(expressions_func)
        return expressions_func_obj, expressions_func_src, expressions_func_gen

    def exploded_names(
            self, fixed: bool = True, tied: bool = True, free: bool = True
    ) -> list[str]:
        """
        Filter exploded parameter names based on specified criteria.

        This method returns a list of exploded parameter names that
        match the specified flags for fixed, tied, and free parameters.
        The returned names follow the order defined by the parameter
        descriptions passed in the constructor.
        """
        return [p for p in self._enames_all if
                (p in self._enames_fixed * fixed) or
                (p in self._enames_tied * tied) or
                (p in self._enames_free * free)]

    def exploded_values_fixed(self) -> dict[str, Real]:
        """
        Retrieve a dictionary with the values of all the fixed exploded
        parameters.
        """
        return copy.deepcopy(self._fixed_dict)

    def evaluate(
            self,
            exploded_free_params: dict[str, Real],
            validate_free_params: bool,
            out_exploded_all_params: dict[str, Real] | None = None
    ) -> dict[str, Real | np.ndarray]:
        # Verify supplied eparams
        if validate_free_params:
            self._check_eparams(exploded_free_params)
        # Apply supplied exploded params on the imploded params
        self._apply_exploded_params(exploded_free_params)
        # Apply expressions on the imploded params
        self._apply_expressions()
        # Extract all exploded params from the imploded params
        if out_exploded_all_params is not None:
            self._extract_exploded_params(out_exploded_all_params)
        # Return a copy of the params (for safety)
        return copy.deepcopy(self._iparams)

    def _check_eparams(self, eparams: dict[str, Real]):
        if missing := set(self._enames_free) - set(eparams):
            raise RuntimeError(
                f"the following parameters are missing: "
                f"{param_utils.sort_param_enames(missing, self._pdescs)}")
        if notfree := set(self._enames_notfree) & set(eparams):
            raise RuntimeError(
                f"the following parameters are not free: "
                f"{param_utils.sort_param_enames(notfree, self._pdescs)}")
        if unknown := set(eparams) - set(self._enames_all):
            raise RuntimeError(
                f"the following parameters are not recognised: "
                f"{unknown}")

    def _apply_exploded_params(self, eparams):
        for key, val in eparams.items():
            name = self._eparams_nmapping[key]
            index = self._eparams_imapping[key]
            if index is None:
                self._iparams[name] = val
            else:
                self._iparams[name][index] = val

    def _extract_exploded_params(self, out_eparams):
        if not out_eparams:
            foo = make_param_symbols_from_pdescs(self._pdescs.values(), self._pdescs.keys())
            for k in foo:
                out_eparams[k] = None
        for ename in out_eparams:
            assert ename in self.exploded_names()  # todo: provide better error message
            name = self._eparams_nmapping[ename]
            index = self._eparams_imapping[ename]
            out_eparams[ename] = self._iparams[name] \
                if index is None else self._iparams[name][index]

    def _apply_expressions(self):
        """
        Evaluate parameter expressions and validate parameter values.

        This function creates a deep copy of `_iparams` and applies
        `_expressions_func_obj` to it. The resulting parameter values
        are then validated to ensure they are finite real numbers and
        conform to expected types and shapes. Only tied parameters
        are copied back to `_iparams`.
        """
        if self._expressions_func_obj is None:
            return
        # Apply the expressions on a copy of the iparams dict.
        # This is done to avoid issues that may arise in case
        # the user does something silly.
        result = copy.deepcopy(self._iparams)
        try:
            self._expressions_func_obj(result)
        except Exception as e:
            raise RuntimeError(
                f"exception thrown while evaluating parameter expressions; "
                f"see preceding exception for additional information; "
                f"the expression function source code is the following:\n"
                f"{'-' * 50}\n"
                f"{str(self._expressions_func_src)}"
                f"{'-' * 50}") from e
        # Validate result values
        for result_key, result_val in result.items():
            # Ignore unknown parameters
            # This may occur if the expression function
            # adds new parameters to the parameter dict
            if result_key not in self._pdescs:
                _log.warning(
                    f"unknown parameter found "
                    f"after evaluating parameters expressions; "
                    f"name: {result_key}; value: {result_val}; "
                    f"This may indicate an unintended addition "
                    f"of parameters or a typo in parameter names; "
                    f"the parameter will be ignored")
                continue
            pdesc = self._pdescs[result_key]
            pdesc_is_scalar = isinstance(pdesc, ParamScalarDesc)
            pdesc_is_vector = isinstance(pdesc, ParamVectorDesc)
            result_is_scalar = isinstance(result_val, Real)
            result_is_vector = iterutils.is_sequence(result_val, strict=True)
            # Ensure all values are finite real numbers
            def is_num(x): return isinstance(x, Real) and np.isfinite(x)
            if any(not is_num(x) for x in np.atleast_1d(result_val)):
                raise RuntimeError(
                    f"failed to validate parameter value "
                    f"after evaluating parameter expressions; "
                    f"values must be finite real numbers or vectors thereof; "
                    f"however, the following parameter does not respect that; "
                    f"name: {result_key}; value: {result_val}")
            # Ensure the parameter value has the correct type/length
            pdesc_val_len = 1 if pdesc_is_scalar else pdesc.size()
            result_val_len = 1 if result_is_scalar else len(result_val)
            if pdesc_is_vector and result_is_scalar:
                result[result_key] = np.full(pdesc.size(), result_val)
            elif pdesc_is_scalar and result_is_vector:
                raise RuntimeError(
                    f"failed to validate parameter values "
                    f"after evaluating parameter expressions; "
                    f"cannot assign sequence of size {result_val_len} to "
                    f"scalar parameter; "
                    f"{result_key}: {result_val}")
            elif pdesc_is_vector and result_is_vector \
                    and pdesc_val_len != result_val_len:
                raise RuntimeError(
                    f"failed to validate parameter values "
                    f"after evaluating parameter expressions; "
                    f"cannot assign sequence of size {result_val_len} to "
                    f"vector parameter of size {pdesc_val_len}; "
                    f"the offending parameter is {result_key}: {result_val}")
        # Copy *only* the tied parameter values
        # from the result dict to the iparams dict
        for ename in self._enames_tied:
            name = self._eparams_nmapping[ename]
            index = self._eparams_imapping[ename]
            if index is None:
                self._iparams[name] = result[name]
            else:
                self._iparams[name][index] = result[name][index]
