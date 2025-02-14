
import ast
import copy
import logging
import numbers
import types

import numpy as np

from gbkfit.params import parsers as param_parsers, utils as param_utils
from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.params.symbols import *
from gbkfit.utils import miscutils


__all__ = [
    'Interpreter'
]


_log = logging.getLogger(__name__)


class _Transformer(ast.NodeTransformer):

    def __init__(self, pdescs):
        self._pdescs = pdescs

    def visit_Name(self, node):
        if node.id in self._pdescs:
            node = ast.Subscript(
                value=ast.Name(id='params', ctx=ast.Load()),
                slice=ast.Index(value=ast.Constant(value=node.id)),
                ctx=node.ctx)
        return node


def _make_exprs_func(pdescs, exprs):
    indent = ' ' * 4
    globals_ = dict(np=np)
    source = 'def expressions(params):\n'
    for key, val in exprs.items():
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
            f"auto-generated expression function: {str(e)}") from e


class Interpreter:

    def __init__(self, pdescs, exprs_dict=None, exprs_func=None):
        self._pdescs = copy.deepcopy(pdescs)
        # Create mapping:
        # - imploded parameter name => imploded parameter value
        # For scalar parameter values we use floats
        # For vector parameter values we use numpy arrays
        # Create mappings:
        # - exploded parameter name => imploded parameter name
        # - exploded parameter name => imploded parameter indices
        self._iparams = dict()
        self._eparams_nmapping = dict()
        self._eparams_imapping = dict()
        for name, pdesc in pdescs.items():
            if isinstance(pdesc, ParamScalarDesc):
                self._iparams[name] = np.nan
                self._eparams_nmapping[name] = name
                self._eparams_imapping[name] = None
            elif isinstance(pdesc, ParamVectorDesc):
                size = pdesc.size()
                names = [name] * size
                indices = list(range(size))
                enames = param_utils.make_param_symbols_from_name_and_indices(
                    name, indices)
                self._iparams[name] = np.full(size, np.nan)
                self._eparams_nmapping.update(zip(enames, names, strict=True))
                self._eparams_imapping.update(zip(enames, indices, strict=True))
            else:
                raise RuntimeError("impossible")
        # From the expression dict, extract pairs with:
        # - Nones (tied parameters)
        # - Reals (fixed parameters)
        # - Expressions (tied parameters)
        def is_none(x): return isinstance(x, type(None))
        def is_real(x): return isinstance(x, numbers.Real)
        # TODO
        parse_result = param_parsers.parse_param_values(
            exprs_dict, pdescs, lambda x: is_none(x) or is_real(x))
        values, exprs = parse_result.exploded_params, parse_result.expressions
        nones_dict = dict(filter(lambda x: is_none(x[1]), values.items()))
        reals_dict = dict(filter(lambda x: is_real(x[1]), values.items()))
        # Apply None and Real values to the params storage
        self._apply_eparams(values)
        # Parse expressions and extract various ordered information
        parse_result = param_parsers.parse_param_expressions(exprs, pdescs)
        expr_keys, expr_values, expr_keys_names, expr_keys_indices = (
            parse_result.expression_keys,
            parse_result.expression_values,
            parse_result.expression_param_names,
            parse_result.expression_param_indices)
        # From now on, work with the ordered expressions
        exprs = dict(zip(expr_keys, expr_values, strict=True))
        # Extract the exploded names for all parameters and create
        # various groups for convenience
        enames_all = param_utils.make_param_symbols_from_pdescs(
            pdescs.values(), pdescs.keys())
        enames_none = list(nones_dict.keys())
        enames_fixed = list(reals_dict.keys())
        enames_tied = param_utils.make_param_symbols_from_names_and_indices(
            expr_keys_names, expr_keys_indices)
        enames_tied += enames_none
        enames_notfree = enames_tied + enames_fixed
        # Order exploded names based on the supplied descriptions
        enames_free = [n for n in enames_all if n not in enames_notfree]
        enames_tied = [n for n in enames_all if n in enames_tied]
        enames_fixed = [n for n in enames_all if n in enames_fixed]
        enames_notfree = [n for n in enames_all if n in enames_notfree]
        # We cannot have expression strings and an expression function
        # at the same time, because it is very hard to implement it
        # robustly and without any ambiguities.
        if exprs and exprs_func:
            func_file = exprs_func.__code__.co_filename
            func_name = exprs_func.__qualname__
            func_full = f"{func_file}:{func_name}"
            raise RuntimeError(
                f"the following expression strings were provided: {exprs}; "
                f"the following expression function was provided: {func_full}; "
                f"expression strings and expression function "
                f"are mutually exclusive")
        # Parameters with value None are considered tied parameters
        # and their value must be defined in the expression function
        if enames_none and not exprs_func:
            raise RuntimeError(
                f"the following parameters are set to None: {enames_none}; "
                f"this implies that they are tied parameters and "
                f"their value must be defined in an expression function; "
                f"however, an expression function was not provided")
        # Providing an expression function while no parameters are
        # marked as tied is pointless, unless the user wants to do
        # something sneaky. Let them do it, but emit warning.
        if not enames_none and exprs_func:
            func_file = exprs_func.__code__.co_filename
            func_name = exprs_func.__qualname__
            func_full = f"{func_file}:{func_name}"
            _log.warning(
                f"the following expression function was provided: {func_full}; "
                f"however, no tied parameters have been defined. "
                f"are you certain you know what you are doing?")
        exprs_func_obj = None
        exprs_func_src = None
        exprs_func_gen = False
        # If expressions are given via exprs_dict,
        # generate the expression function
        if exprs:
            exprs_func_obj, exprs_func_src = _make_exprs_func(pdescs, exprs)
            exprs_func_gen = True
        # If we are given the expression function directly,
        # try to retrieve, cleanup, and store its source code.
        # The source code may not always be available.
        elif exprs_func:
            exprs_func_obj = exprs_func
            exprs_func_src = miscutils.get_source(exprs_func)
        self._exprs_dict = exprs_dict
        self._fixed_dict = reals_dict
        self._exprs_func_obj = exprs_func_obj
        self._exprs_func_src = exprs_func_src
        self._exprs_func_gen = exprs_func_gen
        self._enames_all = enames_all
        self._enames_free = enames_free
        self._enames_tied = enames_tied
        self._enames_fixed = enames_fixed
        self._enames_notfree = enames_notfree

    def exprs_dict(self):
        return copy.deepcopy(self._exprs_dict)

    def fixed_dict(self):
        return copy.deepcopy(self._fixed_dict)

    def enames(self, fixed=True, tied=True, free=True):
        return [p for p in self._enames_all if
                (p in self._enames_fixed * fixed) or
                (p in self._enames_tied * tied) or
                (p in self._enames_free * free)]

    def evaluate(self, eparams, out_eparams=None, check=True):
        # Verify supplied eparams
        if check:
            self._check_eparams(eparams)
        # Apply supplied eparams on the iparams
        self._apply_eparams(eparams)
        # Apply expressions on the iparams
        self._apply_exprs()
        # Extract all eparams from the iparams
        if out_eparams is not None:
            self._extract_eparams(out_eparams)
        # Return a copy of the params (for safety)
        return copy.deepcopy(self._iparams)

    def _check_eparams(self, eparams):
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

    def _apply_eparams(self, eparams):
        for key, val in eparams.items():
            name = self._eparams_nmapping[key]
            index = self._eparams_imapping[key]
            if index is None:
                self._iparams[name] = val
            else:
                self._iparams[name][index] = val

    def _extract_eparams(self, out_eparams):
        if not out_eparams:
            foo = make_param_symbols_from_pdescs(self._pdescs.values(), self._pdescs.keys())
            for k in foo:
                out_eparams[k] = None
        for ename in out_eparams:
            assert ename in self.enames()  # todo: provide better error message
            name = self._eparams_nmapping[ename]
            index = self._eparams_imapping[ename]
            out_eparams[ename] = self._iparams[name] \
                if index is None else self._iparams[name][index]

    def _apply_exprs(self):
        # If we have no expression function, return immediately
        if self._exprs_func_obj is None:
            return
        # Apply the expressions on a copy of the iparams dict.
        # This is done to avoid issues that may arise in case
        # the user does something silly.
        result = copy.deepcopy(self._iparams)
        try:
            self._exprs_func_obj(result)
        except Exception as e:
            raise RuntimeError(
                f"exception thrown while evaluating parameter expressions; "
                f"see preceding exception for additional information; "
                f"the expression function source code is the following:\n"
                f"{'-' * 50}\n"
                f"{str(self._exprs_func_src)}"
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
                    f"the parameter will be ignored")
                continue
            pdesc = self._pdescs[result_key]
            pdesc_is_scalar = isinstance(pdesc, ParamScalarDesc)
            pdesc_is_vector = isinstance(pdesc, ParamVectorDesc)
            result_is_scalar = isinstance(result_val, numbers.Real)
            result_is_vector = isinstance(result_val, (tuple, list, np.ndarray))
            # Ensure all values are finite real numbers
            def is_num(x): return isinstance(x, numbers.Real) and np.isfinite(x)
            if any([not is_num(x) for x in np.atleast_1d(result_val)]):
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
                    f"{result_key}: {result_val}")
        # Copy *only* the tied parameter values
        # from the result dict to the iparams dict
        for ename in self._enames_tied:
            # todo: investigate whether we can avoid the map lookups
            name = self._eparams_nmapping[ename]
            index = self._eparams_imapping[ename]
            if index is None:
                self._iparams[name] = result[name]
            else:
                self._iparams[name][index] = result[name][index]
