
import ast
import copy
import inspect
import numbers
import textwrap
import types

import numpy as np

from gbkfit.utils import miscutils, parseutils
from . import ParamScalarDesc, ParamVectorDesc, utils


class _Transformer(ast.NodeTransformer):

    def __init__(self, descs):
        self._descs = descs

    def visit_Name(self, node):
        if node.id in self._descs:
            node = ast.Subscript(
                value=ast.Name(id='params', ctx=ast.Load()),
                slice=ast.Index(value=ast.Constant(value=node.id)),
                ctx=node.ctx)
        return node


def _make_exprs_func(descs, exprs):
    indent = ' ' * 4
    globals_ = dict(np=np)
    source = 'def expressions(params):\n'
    for key, val in exprs.items():
        line_ast = _Transformer(descs).visit(ast.parse(f'{key} = {val}'))
        line_src = ast.unparse(line_ast).strip('\n')
        source += f'{indent}{line_src}\n'
    source += f'{indent}return params\n'
    try:
        codeobj = compile(source, filename='<string>', mode='exec')
        funcobj = types.FunctionType(codeobj.co_consts[0], globals_)
        return funcobj, source
    except Exception as e:
        raise RuntimeError(
            f"exception thrown while compiling "
            f"auto-generated expression function: {str(e)}") from e


def load_exprs_file(info):
    desc = 'parameter expressions'
    opts = parseutils.parse_options(info, desc, ['file', 'func'])
    return miscutils.get_attr_from_file(opts['file'], opts['func'])


def dump_exprs_file(file, func_obj, func_src):
    if not func_src:
        raise RuntimeError(
            "failed to dump expression function "
            "because its source code is not available")
    with open(file, 'a') as f:
        f.write(func_src)
    return dict(file=file, func=func_obj.__name__)


class Interpreter:

    def __init__(self, descs, exprs_dict=None, exprs_func=None):
        self._descs = copy.deepcopy(descs)
        # Create mapping of imploded parameters (name=>value)
        # For scalar parameters we use floats
        # For vector parameters we use numpy arrays
        # Also create mappings that map exploded parameter names
        # to imploded parameter names and indices
        self._iparams = dict()
        self._eparams_nmapping = dict()
        self._eparams_imapping = dict()
        for name, desc in descs.items():
            if isinstance(desc, ParamScalarDesc):
                self._iparams[name] = np.nan
                self._eparams_nmapping[name] = name
                self._eparams_imapping[name] = None
            elif isinstance(desc, ParamVectorDesc):
                size = desc.size()
                indices = list(range(size))
                eparams = utils.explode_pname(name, indices)
                self._iparams[name] = np.full(desc.size(), np.nan)
                self._eparams_nmapping.update(zip(eparams, [name] * size))
                self._eparams_imapping.update(zip(eparams, indices))
            else:
                assert False
        # Extract pairs with:
        # - Nones (tied parameters)
        # - Reals (fixed parameters)
        # - Expressions (tied parameters)
        def is_none(x): return isinstance(x, type(None))
        def is_numb(x): return isinstance(x, numbers.Real)
        values, exprs = utils.parse_param_values(
            exprs_dict, descs, lambda x: is_none(x) or is_numb(x))[4:6]
        nones_dict = dict(filter(lambda x: is_none(x[1]), values.items()))
        numbs_dict = dict(filter(lambda x: is_numb(x[1]), values.items()))
        # Apply None (tied) and Real (fixed) values
        # on the iparams storage
        self._apply_eparams(values)
        # Extract the name and indices of each expression
        expr_names, expr_indices = utils.parse_param_exprs(exprs, descs)[2:4]
        # Extract the exploded names for all parameters and create
        # various groups for convenience
        enames_all = utils.explode_pdescs(descs.values(), descs.keys())
        enames_none = list(nones_dict.keys())
        enames_fixed = list(numbs_dict.keys())
        enames_tied = utils.explode_pnames(expr_names, expr_indices)
        enames_tied += enames_none
        enames_notfree = enames_tied + enames_fixed
        # Order exploded names based on the supplied descriptions
        enames_free = [n for n in enames_all if n not in enames_notfree]
        enames_tied = [n for n in enames_all if n in enames_tied]
        enames_fixed = [n for n in enames_all if n in enames_fixed]
        enames_notfree = [n for n in enames_all if n in enames_notfree]
        # We cannot have expression strings and an expression function
        # This is because it is just very hard to implement it robustly
        if exprs and exprs_func:
            func_file = exprs_func.__code__.co_filename
            func_name = exprs_func.__qualname__
            func_full = f"{func_file}:{func_name}"
            raise RuntimeError(
                f"the following expression strings were provided: {exprs}; "
                f"the following expression function was provided: {func_full}; "
                f"expression strings and expression functions "
                f"are mutually exclusive")
        # Parameters with value None are considered tied parameters and
        # their value is expected to be defined in the supplied function
        if enames_none and not exprs_func:
            raise RuntimeError(
                f"the following parameters are set to None: {enames_none}; "
                f"this implies that they are tied parameters and hence "
                f"they are expected to be defined in an expression function; "
                f"however, an expression function was not provided")
        exprs_func_obj = None
        exprs_func_src = None
        exprs_func_gen = False
        # If expressions are given in the form of strings,
        # generate the expression function
        if exprs:
            exprs_func_obj, exprs_func_src = _make_exprs_func(descs, exprs)
            exprs_func_gen = True
        # If we are given the expression function directly,
        # try to retrieve, cleanup, and store its source code.
        # The source code may not always be available.
        elif exprs_func:
            exprs_func_obj = exprs_func
            try:
                exprs_func_src = textwrap.dedent(inspect.getsource(exprs_func))
            except AttributeError:
                exprs_func_src = ''
        # ...
        self._exprs_dict = exprs_dict
        self._fixed_dict = numbs_dict
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

    def exprs_func_obj(self):
        return self._exprs_func_obj

    def exprs_func_src(self):
        return self._exprs_func_src

    def exprs_func_gen(self):
        return self._exprs_func_gen

    def enames(self, free=True, tied=True, fixed=True):
        return [p for p in self._enames_all if
                (p in self._enames_free * free) or
                (p in self._enames_tied * tied) or
                (p in self._enames_fixed * fixed)]

    def evaluate(self, eparams, check=True, out_eparams=None):
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
        if missing := set(self._enames_free).difference(eparams):
            raise RuntimeError(
                f"the following parameters are missing: "
                f"{utils.sort_eparams(self._descs, missing)}")
        if notfree := set(self._enames_notfree).intersection(eparams):
            raise RuntimeError(
                f"the following parameters are not free: "
                f"{utils.sort_eparams(self._descs, notfree)}")
        if unknown := set(eparams).difference(self._enames_all):
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
        for ename in out_eparams:
            assert ename in self.enames()  # todo: provide better error message
            name = self._eparams_nmapping[ename]
            index = self._eparams_imapping[ename]
            out_eparams[ename] = self._iparams[name] \
                if index is None else self._iparams[name][index]

    def _apply_exprs(self):
        if self._exprs_func_obj is None:
            return
        # Apply the expressions on a copy of the param dict
        # This will leave the original dict intact in case of error
        try:
            result = copy.deepcopy(self._iparams)
            self._exprs_func_obj(result)
        except Exception as e:
            raise RuntimeError(
                f"exception thrown while evaluating parameter expressions; "
                f"see preceding exception for additional information; "
                f"the expressions function source code is the following:\n"
                f"{'-' * 50}\n"
                f"{str(self._exprs_func_src)}"
                f"{'-' * 50}")
        # Validate resulting values
        for lhs, rhs in result.items():
            # Ignore unknown parameters
            # This may occur if the expression function
            # adds new parameters in the 'result' dict
            if lhs not in self._descs:
                continue
            desc = self._descs[lhs]
            lhs_is_scalar = isinstance(desc, ParamScalarDesc)
            lhs_is_vector = isinstance(desc, ParamVectorDesc)
            rhs_is_scalar = isinstance(rhs, numbers.Real)
            rhs_is_vector = isinstance(rhs, (tuple, list, np.ndarray))
            def is_num(x): return isinstance(x, numbers.Real) and np.isfinite(x)
            if any([not is_num(x) for x in np.atleast_1d(rhs)]):
                raise RuntimeError(
                    f"failed to validate parameter values "
                    f"after evaluating parameter expressions; "
                    f"invalid value(s) encountered; "
                    f"{lhs}: {rhs}")
            lhs_length = 1 if lhs_is_scalar else desc.size()
            rhs_length = 1 if rhs_is_scalar else len(rhs)
            if lhs_is_vector and rhs_is_scalar:
                rhs = np.full(desc.size(), rhs)
            if lhs_is_scalar and rhs_is_vector:
                raise RuntimeError(
                    f"failed to validate parameter values "
                    f"after evaluating parameter expressions; "
                    f"cannot assign sequence of size {rhs_length} to "
                    f"scalar parameter; "
                    f"{lhs}: {rhs}")
            if lhs_is_vector and rhs_is_vector and lhs_length != rhs_length:
                raise RuntimeError(
                    f"failed to validate parameter values "
                    f"after evaluating parameter expressions; "
                    f"cannot assign sequence of size {rhs_length} to "
                    f"vector parameter of size {lhs_length}; "
                    f"{lhs}: {rhs}")
            result[lhs] = rhs
        # Copy results to the iparams
        for ename in self._enames_tied:
            name = self._eparams_nmapping[ename]
            index = self._eparams_imapping[ename]
            value = result[name] if index is None else result[name][index]
            if index is None:
                self._iparams[name] = value
            else:
                self._iparams[name][index] = value
