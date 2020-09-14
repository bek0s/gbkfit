
import ast
import astor
import copy
import inspect
import math
import numbers
import textwrap
import types

import numpy as np

from . import ParamScalarDesc, ParamVectorDesc, utils

from gbkfit.utils import miscutils, parseutils


class Transformer(ast.NodeTransformer):
    def visit_Name(self, node):
        return ast.Subscript(
            value=ast.Name(id='params', ctx=ast.Load()),
            slice=ast.Index(value=ast.Constant(value=node.id)),
            ctx=node.ctx)


def _create_exprs_func(exprs):
    source = 'def expressions(params):\n'
    for key, val in exprs.items():
        source += f'    {key} = {val}\n'
    source = astor.to_source(Transformer().visit(ast.parse(source)))
    source += '    return params\n'
    try:
        codeobj = compile(source, filename='<string>', mode='exec')
        funcobj = types.FunctionType(codeobj.co_consts[0], globals())
        return funcobj, source
    except SyntaxError as e:
        raise RuntimeError() from e
    except ValueError as e:
        raise RuntimeError() from e


def load_expr_file(info):
    if info is None:
        return info
    desc = 'parameter expressions'
    opts = parseutils.parse_options(info, desc, ['file', 'func'])
    return miscutils.get_attr_from_file(opts['file'], opts['func'])


def dump_expr_file(exprs, force=False, file='gbkfit_config_expressions.py'):
    func_obj = exprs.exprs_func_obj()
    func_gen = exprs.exprs_func_gen()
    func_src = exprs.exprs_func_src()
    if func_gen and not force:
        raise RuntimeError(0)
    if func_obj is None:
        raise RuntimeError(1)
    if func_obj and not func_src:
        raise RuntimeError(2)
    with open(file, 'a') as f:
        f.write(func_src)
    return dict(file=file, func=func_obj.__name__)


class Expressions:




    def __init__(self, descs, exprs_dict, exprs_func=None):
        self._descs = copy.deepcopy(descs)
        # ...
        self._nmapping = dict()
        self._imapping = dict()
        self._values = dict()
        for name, desc in descs.items():
            if isinstance(desc, ParamScalarDesc):
                self._nmapping[name] = name
                self._imapping[name] = None
                self._values[name] = np.nan
            elif isinstance(desc, ParamVectorDesc):
                indices = list(range(desc.size()))
                eparams = utils.explode_pname(name, indices)
                self._nmapping.update(zip(eparams, [name] * desc.size()))
                self._imapping.update(zip(eparams, indices))
                self._values[name] = np.full(desc.size(), np.nan)
            else:
                raise RuntimeError()
        # ...
        def is_num(x): return isinstance(x, numbers.Number)
        def is_nil(x): return isinstance(x, type(None))
        def is_val(x): return is_nil(x) or is_num(x)
        values, exprs = utils.parse_param_values(exprs_dict, descs, is_val)[4:6]
        expr_names, exprs_indices = utils.parse_param_exprs(exprs, descs)[2:4]
        consts_num = dict(filter(lambda x: is_num(x[1]), values.items()))
        consts_nil = [key for key, val in values.items() if is_nil(val)]
        consts_xpr = utils.explode_pnames(expr_names, exprs_indices)
        consts_all = consts_nil + consts_xpr + list(consts_num.keys())
        consts_nilxpr = consts_nil + consts_xpr
        if exprs and exprs_func:
            raise RuntimeError(
                "expression strings and expression function "
                "are mutually exclusive")
        if consts_nil and not exprs_func:
            raise RuntimeError(
                "expression nones require an expression function")
        # ...
        enames = utils.explode_pdescs(descs.values(), descs.keys())
        enames_consts = [n for n in enames if n in consts_all]
        enames_nilstr = [n for n in enames if n in consts_nilxpr]
        enames_params = [n for n in enames if n not in consts_all]
        # ...
        exprs_func_obj = None
        exprs_func_src = None
        exprs_func_gen = False
        if exprs:
            exprs_func_obj, exprs_func_src = _create_exprs_func(exprs)
            exprs_func_gen = True
        elif exprs_func:
            exprs_func_obj = exprs_func
            try:
                exprs_func_src = textwrap.dedent(inspect.getsource(exprs_func))
            except AttributeError:
                exprs_func_src = ''
        # ...
        self._exprs_dict = exprs_dict
        self._exprs_func_obj = exprs_func_obj
        self._exprs_func_src = exprs_func_src
        self._exprs_func_gen = exprs_func_gen
        self._enames = enames
        self._enames_consts = enames_consts
        self._enames_nilstr = enames_nilstr
        self._enames_params = enames_params
        # ...
        self._assign_eparams(consts_num)

    def exprs_dict(self):
        return copy.deepcopy(self._exprs_dict)

    def exprs_func_obj(self):
        return self._exprs_func_obj

    def exprs_func_src(self):
        return self._exprs_func_src

    def exprs_func_gen(self):
        return self._exprs_func_gen

    def names(self, params=True, consts=False):
        return self._enames_params * params + self._enames_consts * consts

    def evaluate(self, eparams):
        # Verify all required eparams are provided
        self._verify_eparams(eparams)
        # Assign eparam values
        self._assign_eparams(eparams)
        # Apply expressions
        self._apply_exprs()
        # Make sure all variables have valid values
        #self._verify_evalues()
        # Return a copy for safety
        return copy.deepcopy(self._values)

    def _assign_eparams(self, eparams):
        for key, val in eparams.items():
            name = self._nmapping[key]
            index = self._imapping[key]
            if index is None:
                self._values[name] = val
            else:
                self._values[name][index] = val

    def _apply_exprs(self):
        if self._exprs_func_obj is None:
            return
        result = self._exprs_func_obj(self._values)
        for lhs, rhs in result.items():
            desc = self._descs[lhs]
            lhs_is_scalar = isinstance(desc, ParamScalarDesc)
            lhs_is_vector = isinstance(desc, ParamVectorDesc)
            rhs_is_scalar = isinstance(rhs, numbers.Number)
            rhs_is_vector = isinstance(rhs, (tuple, list, np.ndarray))
            def is_num(x): return isinstance(x, numbers.Real) and np.isfinite(x)
            if any([not is_num(x) for x in np.atleast_1d(rhs)]):
                raise RuntimeError(0)
            if lhs_is_vector and rhs_is_scalar:
                result[lhs] = np.full(desc.size(), rhs)
            if lhs_is_scalar and rhs_is_vector:
                raise RuntimeError(1)
            lhs_length = 1 if lhs_is_scalar else desc.size()
            rhs_length = 1 if rhs_is_scalar else len(rhs)
            if lhs_is_vector and rhs_is_vector and lhs_length != rhs_length:
                raise RuntimeError(2)
        self._values = result





    def _verify_eparams(self, eparams):
        missing = set(self._enames_params).difference(eparams)
        if missing:
            raise RuntimeError(
                f"the following parameters are missing: "
                f"{missing}")
        fixed = set(self._enames_consts).intersection(eparams)
        if fixed:
            raise RuntimeError(
                f"the following parameters are supposed to be fixed: "
                f"{fixed}")
        unknown = set(eparams).difference(self._enames)
        if unknown:
            raise RuntimeError(
                f"the following parameters are not recognised: "
                f"{unknown}")

    def _verify_evalues(self):
        for lhs, rhs in self._values.items():
            desc = self._descs[lhs]
            lhs_is_scalar = isinstance(desc, ParamScalarDesc)
            lhs_is_vector = isinstance(desc, ParamVectorDesc)
            rhs_is_scalar = isinstance(rhs, numbers.Number)
            rhs_is_vector = isinstance(rhs, (tuple, list, np.ndarray))
            if not (rhs_is_scalar or (rhs_is_vector and (None not in rhs))):
                raise RuntimeError(1)
            lhs_length = 1 if lhs_is_scalar else desc.size()
            rhs_length = 1 if rhs_is_scalar else len(rhs)
            if lhs_is_scalar != rhs_is_scalar:
                raise RuntimeError(2)
            if lhs_is_vector and rhs_is_vector and lhs_length != rhs_length:
                raise RuntimeError(3)

    def _extract_eparams(self, enames):
        eparams = dict()
        for ename in enames:
            name = self._nmapping[ename]
            index = self._imapping[ename]
            eparams[ename] = self._values[name][index] \
                if index else self._values[name]
        return eparams
