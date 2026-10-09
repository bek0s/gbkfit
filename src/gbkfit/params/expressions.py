"""
The expressions of parameter properties (e.g. 'vpt_vt': 'a * 2'). They
can use numbers, parameters and their elements, constants, arithmetic,
comparisons, conditional expressions, tuples and lists, a few builtins,
and of numpy (np, np.linalg): its ufuncs (e.g. np.sin), numbers (e.g.
np.pi), scalar types (e.g. np.int64) and the functions on numbers and
arrays of _NUMPY_FUNCTIONS. Nothing else, so a configuration cannot run
arbitrary code (numpy also holds modules such as os, and functions that
read and write files).
"""

import ast
import builtins

import numpy as np

from gbkfit.params.keys import (
    InvalidKeyError, parse_subscript, subscript_value)
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc


__all__ = [
    'Expression',
    'InvalidExpressionError'
]


class InvalidExpressionError(ValueError):
    """An invalid expression; the message explains why."""


# The builtins that expressions can call
BUILTINS = {
    name: getattr(builtins, name)
    for name in ('abs', 'min', 'max', 'round', 'sum', 'len')}

# The nodes that expressions can have (and their operators)
_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow,
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Not,
    ast.BoolOp, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.IfExp, ast.Tuple, ast.List, ast.Subscript, ast.Slice,
    ast.Call, ast.keyword, ast.Attribute)


# The numpy modules whose members expressions can use
_NUMPY_MODULES = {'np': np, 'np.linalg': np.linalg}

# The numpy functions that expressions can call, besides the ufuncs: they
# compute on numbers and arrays only
_NUMPY_FUNCTIONS = {
    'np': frozenset((
        'all', 'allclose', 'amax', 'amin', 'any', 'append', 'arange',
        'argmax', 'argmin', 'argsort', 'around', 'array', 'asarray',
        'atleast_1d', 'average', 'clip', 'concatenate', 'convolve', 'cross',
        'cumprod', 'cumsum', 'diff', 'dot', 'flip', 'full', 'full_like',
        'geomspace', 'gradient', 'hstack', 'inner', 'interp', 'isclose',
        'linspace', 'logspace', 'max', 'mean', 'median', 'min', 'nanmax',
        'nanmean', 'nanmin', 'nansum', 'ones', 'ones_like', 'outer',
        'percentile', 'polyval', 'prod', 'ptp', 'quantile', 'ravel',
        'repeat', 'reshape', 'roll', 'round', 'select', 'sort', 'squeeze',
        'stack', 'std', 'sum', 'take', 'tile', 'trapezoid', 'unique', 'var',
        'vstack', 'where', 'zeros', 'zeros_like')),
    'np.linalg': frozenset(('det', 'inv', 'norm', 'solve'))}


def _dotted_name(node):
    """The dotted name of a node (e.g. 'np.linalg.norm'), or None."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return '.'.join([node.id] + names[::-1])


def _is_numpy(node):
    """
    Whether a node names one of _NUMPY_MODULES, or a member of one that
    expressions can use: a ufunc, a number, a scalar type, or one of
    _NUMPY_FUNCTIONS.
    """
    name = _dotted_name(node)
    if name in _NUMPY_MODULES:
        return True
    module, _, member = (name or '').rpartition('.')
    if module not in _NUMPY_MODULES or member.startswith('_'):
        return False
    value = getattr(_NUMPY_MODULES[module], member, None)
    return value is not None and (
        isinstance(value, (np.ufunc, int, float))
        or (isinstance(value, type) and issubclass(value, np.generic))
        or member in _NUMPY_FUNCTIONS[module])


class Expression:
    """
    An expression, compiled once. It reads the elements of parameters in
    reads: a dict of parameter name to the set of indices it reads (None
    for a scalar).
    """

    def __init__(
            self,
            source: str,
            pdescs: dict[str, ParamDesc],
            constants: dict[str, np.ndarray] | None = None):
        constants = constants or {}
        try:
            tree = ast.parse(source.strip(), mode='eval')
        except SyntaxError as e:
            raise InvalidExpressionError(f"invalid syntax: {e.msg}") from None
        self.source = source
        self.reads = {}
        self._check(tree, pdescs, constants)
        self._code = compile(tree, '<expression>', 'eval')

    def evaluate(self, namespace):
        """Evaluate with the values of the parameters and constants."""
        return eval(self._code, {'__builtins__': BUILTINS, 'np': np},
                    namespace)

    def _check(self, tree, pdescs, constants):
        """Check the nodes and names, and find the elements it reads."""
        subscripted = set()
        for node in ast.walk(tree):
            if not isinstance(node, _NODES):
                raise InvalidExpressionError(
                    f"'{ast.unparse(node)}' is not allowed in expressions")
            if isinstance(node, ast.Constant) and not (
                    isinstance(node.value, (int, float))
                    and not isinstance(node.value, bool)):
                raise InvalidExpressionError(
                    f"{node.value!r} is not allowed in expressions; "
                    f"constants must be numbers")
            if isinstance(node, ast.Attribute) and not _is_numpy(node):
                raise InvalidExpressionError(
                    f"'{ast.unparse(node)}' is not allowed in expressions; "
                    f"of the attributes, only numpy ufuncs, numbers, "
                    f"scalar types and functions on arrays are (e.g. "
                    f"np.sin, np.pi, np.sum)")
            if isinstance(node, ast.Call) and not (
                    _is_numpy(node.func) or (
                        isinstance(node.func, ast.Name)
                        and node.func.id in BUILTINS)):
                raise InvalidExpressionError(
                    f"'{ast.unparse(node.func)}' cannot be called in "
                    f"expressions; only numpy ufuncs, scalar types and "
                    f"functions on arrays, and {', '.join(BUILTINS)} can")
            if isinstance(node, ast.Name) and not (
                    node.id in pdescs or node.id in constants
                    or node.id in BUILTINS or node.id == 'np'):
                raise InvalidExpressionError(f"unknown name '{node.id}'")
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in pdescs):
                self._read_subscript(node, pdescs[node.value.id])
                subscripted.add(node.value)
        # A parameter without a subscript reads all its elements
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and node.id in pdescs
                    and node not in subscripted):
                self._read(node.id, pdescs[node.id], None)

    def _read_subscript(self, node, pdesc):
        code = ast.unparse(node)
        if isinstance(pdesc, ParamScalarDesc):
            raise InvalidExpressionError(
                f"'{code}': '{pdesc.name()}' is a scalar and has no "
                f"elements")
        # A subscript that is not constant (e.g. v[int(a)]) can read any
        # element
        indices = None
        if subscript_value(node.slice) is not None:
            try:
                indices, _ = parse_subscript(node.slice, pdesc.size())
            except InvalidKeyError as e:
                raise InvalidExpressionError(f"'{code}': {e}") from None
        self._read(node.value.id, pdesc, indices)

    def _read(self, name, pdesc, indices):
        if isinstance(pdesc, ParamScalarDesc):
            self.reads[name] = None
            return
        if indices is None:
            indices = range(pdesc.size())
        self.reads.setdefault(name, set()).update(int(i) for i in indices)
