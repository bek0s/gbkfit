
import ast
import astor
import copy
import inspect
import numbers
import textwrap
import types

import numpy as np

from . import ParamScalarDesc, ParamVectorDesc, utils

from gbkfit.utils import miscutils, parseutils


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


def _create_exprs_func(descs, exprs):
    globals_ = dict(np=np)
    source = 'def expressions(params):\n'
    for key, val in exprs.items():
        line_ast = _Transformer(descs).visit(ast.parse(f'{key} = {val}'))
        line_src = astor.to_source(line_ast).strip('\n')
        source += f'    {line_src}\n'
    source += '    return params\n'
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


def dump_exprs_file(file, exprs):
    func_obj = exprs.exprs_func_obj()
    func_src = exprs.exprs_func_src()
    if func_obj and not func_src:
        raise RuntimeError(
            "failed to dump expression function "
            "because its source code is not available")
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
        def is_none(x): return isinstance(x, type(None))
        def is_number(x): return isinstance(x, numbers.Number)
        values, exprs = utils.parse_param_values(
            exprs_dict, descs, lambda x: is_none(x) or is_number(x))[4:6]
        self._apply_eparams(values)
        # ...
        enames_all = utils.explode_pdescs(descs.values(), descs.keys())
        enames_none = list(dict(filter(lambda x: is_none(x[1]), values.items())).keys())
        enames_fixed = list(dict(filter(lambda x: is_number(x[1]), values.items())).keys())
        expr_names, expr_indices = utils.parse_param_exprs(exprs, descs)[2:4]
        enames_tied = utils.explode_pnames(expr_names, expr_indices)
        enames_tied += enames_none
        enames_notfree = enames_tied + enames_fixed
        # ...
        enames_free = [n for n in enames_all if n not in enames_notfree]
        enames_tied = [n for n in enames_all if n in enames_tied]
        enames_fixed = [n for n in enames_all if n in enames_fixed]
        enames_notfree = [n for n in enames_all if n in enames_notfree]
        # ...
        if exprs and exprs_func:
            func_file = exprs_func.__code__.co_filename
            func_name = exprs_func.__name__
            func_full = f"{func_file}:{func_name}"
            raise RuntimeError(
                f"the following expression strings were provided: {exprs}; "
                f"the following expression function was provided: {func_full}; "
                f"these two are mutually exclusive")
        if enames_none and not exprs_func:
            raise RuntimeError(
                f"the following parameters are set to None: {enames_none}; "
                f"an expression function must be provided")
        # ...
        exprs_func_obj = None
        exprs_func_src = None
        exprs_func_gen = False
        if exprs:
            exprs_func_obj, exprs_func_src = _create_exprs_func(descs, exprs)
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

        self._enames_all = enames_all
        self._enames_free = enames_free
        self._enames_tied = enames_tied
        self._enames_fixed = enames_fixed
        self._enames_notfree = enames_notfree

    def exprs_dict(self):
        return copy.deepcopy(self._exprs_dict)

    def exprs_func_obj(self):
        return self._exprs_func_obj

    def exprs_func_src(self):
        return self._exprs_func_src

    def exprs_func_gen(self):
        return self._exprs_func_gen

    def enames(self, free=True, tied=True, fixed=True):
        return [p for p in self.enames_all() if
                (p in self.enames_free() * free) or
                (p in self.enames_tied() * tied) or
                (p in self.enames_fixed() * fixed)]

    def enames_all(self):
        return self._enames_all

    def enames_free(self):
        return self._enames_free

    def enames_tied(self):
        return self._enames_tied

    def enames_fixed(self):
        return self._enames_fixed

    def enames_notfree(self):
        return self._enames_notfree

    def evaluate(
            self, eparams, check=True,
            out_eparams_all=None, out_eparams_free=None,
            out_eparams_tied=None, out_eparams_fixed=None,
            out_eparams_notfree=None):
        # Verify all required eparams are provided
        self._check_eparams(eparams)
        # Assign eparam values
        self._apply_eparams(eparams)
        # Apply expressions
        self._apply_exprs()
        # ...
        self._extract_eparams(self.enames_all(), out_eparams_all)
        self._extract_eparams(self.enames_free(), out_eparams_free)
        self._extract_eparams(self.enames_tied(), out_eparams_tied)
        self._extract_eparams(self.enames_fixed(), out_eparams_fixed)
        self._extract_eparams(self.enames_notfree(), out_eparams_notfree)
        # Return a copy for safety
        return copy.deepcopy(self._values)

    def _check_eparams(self, eparams):
        missing = set(self._enames_free).difference(eparams)
        if missing:
            raise RuntimeError(
                f"the following parameters are missing: "
                f"{missing}")
        fixed = set(self._enames_notfree).intersection(eparams)
        if fixed:
            raise RuntimeError(
                f"the following parameters are not free: "
                f"{fixed}")
        unknown = set(eparams).difference(self._enames_all)
        if unknown:
            raise RuntimeError(
                f"the following parameters are not recognised: "
                f"{unknown}")

    def _apply_eparams(self, eparams):
        for key, val in eparams.items():
            name = self._nmapping[key]
            index = self._imapping[key]
            if index is None:
                self._values[name] = val
            else:
                self._values[name][index] = val

    def _extract_eparams(self, enames, out_eparams):
        if out_eparams is None:
            return
        for ename in enames:
            name = self._nmapping[ename]
            index = self._imapping[ename]
            out_eparams[ename] = self._values[name][index] \
                if index is not None else self._values[name]

    def _apply_exprs(self):
        if self._exprs_func_obj is None:
            return
        # Apply the expressions on a copy of the main dict
        # This will leave the main dict intact in case of error
        try:
            temp = copy.deepcopy(self._values)
            result = self._exprs_func_obj(temp)
        except Exception as e:
            raise RuntimeError(
                f"exception thrown while evaluating parameter expressions: "
                f"{str(e)}") from e
        # Validate parameter values and copy them to the main dict
        for lhs, rhs in result.items():
            if lhs not in self._descs:
                continue
            desc = self._descs[lhs]
            lhs_is_scalar = isinstance(desc, ParamScalarDesc)
            lhs_is_vector = isinstance(desc, ParamVectorDesc)
            rhs_is_scalar = isinstance(rhs, numbers.Number)
            rhs_is_vector = isinstance(rhs, (tuple, list, np.ndarray))
            def is_num(x): return isinstance(x, numbers.Real) and np.isfinite(x)
            if any([not is_num(x) for x in np.atleast_1d(rhs)]):
                raise RuntimeError(
                    f"invalid value(s) encountered after parameter evaluation; "
                    f"{lhs}: {rhs}")
            lhs_length = 1 if lhs_is_scalar else desc.size()
            rhs_length = 1 if rhs_is_scalar else len(rhs)
            if lhs_is_vector and rhs_is_scalar:
                result[lhs] = np.full(desc.size(), rhs)
            if lhs_is_scalar and rhs_is_vector:
                raise RuntimeError(
                    f"cannot assign sequence of size {rhs_length} to "
                    f"scalar parameter; "
                    f"{lhs}: {rhs}")
            if lhs_is_vector and rhs_is_vector and lhs_length != rhs_length:
                raise RuntimeError(
                    f"cannot assign sequence of size {rhs_length} to "
                    f"vector parameter of size {lhs_length}; "
                    f"{lhs}: {rhs}")
            self._values[lhs] = rhs
