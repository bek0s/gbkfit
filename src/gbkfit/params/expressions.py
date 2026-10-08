"""
The expressions of parameter properties (e.g. 'vpt_vt': 'a * 2'). They
can use numbers, parameters and their elements, constants, arithmetic,
comparisons, conditional expressions, tuples and lists, numpy (np.*),
and a few builtins. Nothing else, so a configuration cannot run
arbitrary code.
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


def _is_numpy(node):
    """Whether a node is np or an attribute of it (e.g. np.linalg.norm)."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id == 'np'


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
                    f"only numpy attributes are (e.g. np.pi)")
            if isinstance(node, ast.Call) and not (
                    _is_numpy(node.func) or (
                        isinstance(node.func, ast.Name)
                        and node.func.id in BUILTINS)):
                raise InvalidExpressionError(
                    f"'{ast.unparse(node.func)}' cannot be called in "
                    f"expressions; only numpy functions and "
                    f"{', '.join(BUILTINS)} can")
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
