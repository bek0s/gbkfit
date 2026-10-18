"""
The expressions of parameter properties (e.g. 'hi_vpt_vt': 'a * 2'). They
can use numbers, parameters and their elements, constants, arithmetic,
comparisons, conditional expressions, tuples and lists, a few builtins,
of numpy (np, np.linalg): its ufuncs (e.g. np.sin), numbers (e.g.
np.pi), scalar types (e.g. np.int64) and the functions on numbers and
arrays of _NUMPY_FUNCTIONS, and the scaling relations of gbkfit.relations
(e.g. relations.dz_c); and the other names they are given (e.g. derived
values, and the functions of the models, see ParamSpace). Nothing else,
so a configuration cannot run arbitrary code (numpy also holds modules
such as os, and functions that read and write files), and an expression
cannot change the values it reads.
"""

import ast
import builtins
import keyword
from collections.abc import Collection, Mapping
from types import CodeType
from typing import Any

import numpy as np

from gbkfit import relations
from gbkfit.params._keys import (
    InvalidKeyError, parse_subscript, subscript_value)
from gbkfit.params.pdescs import ParamDesc


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

# The keywords that calls cannot have: out writes into an array (e.g. a
# parameter), and where leaves the values it does not select unset
_KEYWORDS = frozenset(('out', 'where'))

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


def check_name(name: str) -> str | None:
    """
    Return why a parameter or a constant cannot have a name, or None if
    it can: it must be a Python identifier that is not a keyword, and not
    a name of expressions (np, and the builtins they can call).
    """
    if not isinstance(name, str) or not name.isidentifier():
        return "it is not a valid name"
    if keyword.iskeyword(name):
        return "it is a Python keyword"
    if name in ('np', 'relations') or name in BUILTINS:
        return "expressions use it"
    return None


def _dotted_name(node: ast.expr) -> str | None:
    """Return the dotted name of a node (e.g. 'np.linalg.norm'), or None."""
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return '.'.join([node.id] + names[::-1])


def _is_relation(node: ast.expr) -> bool:
    """Check whether a node names a relation of gbkfit.relations."""
    module, _, member = (_dotted_name(node) or '').rpartition('.')
    return module == 'relations' and member in relations.__all__


def _is_numpy(node: ast.expr) -> bool:
    """
    Check whether a node names one of _NUMPY_MODULES, or a member of one
    that expressions can use: a ufunc, a number, a scalar type, or one of
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
    An expression, checked and compiled once.

    Its attribute reads has the elements of the parameters it reads: the
    set of indices of each vector parameter, and None for each scalar; and
    its attribute uses, the other names it reads or calls (of names).
    Raises InvalidExpressionError if it is invalid.

    Parameters
    ----------
    source : str
        The expression.
    pdescs : Mapping
        The descriptions of the parameters it can read, by name.
    constants : Mapping, optional
        The constants it can read, by name.
    names : Collection of str, optional
        Other names it can read and call (e.g. derived values, and the
        functions of the models; see ParamSpace).
    unavailable : Mapping of str to str, optional
        Names it cannot read, and why (e.g. derived values, in a tie).
    """

    def __init__(
            self,
            source: str,
            pdescs: Mapping[str, ParamDesc],
            constants: Mapping[str, np.ndarray] | None = None,
            names: Collection[str] = (),
            unavailable: Mapping[str, str] | None = None
    ):
        constants = constants or {}
        try:
            tree = ast.parse(source.strip(), mode='eval')
        except SyntaxError as e:
            raise InvalidExpressionError(f"invalid syntax: {e.msg}") from None
        self.source = source
        self.reads: dict[str, set[int] | None] = {}
        self.uses: set[str] = set()
        self._check(tree, pdescs, constants, names, unavailable or {})
        self._code: CodeType = compile(tree, '<expression>', 'eval')

    def evaluate(self, namespace: Mapping[str, Any]) -> Any:
        """Evaluate with the values of the parameters and constants."""
        return eval(self._code, {
            '__builtins__': BUILTINS, 'np': np, 'relations': relations},
            namespace)

    def _check(
            self,
            tree: ast.Expression,
            pdescs: Mapping[str, ParamDesc],
            constants: Mapping[str, np.ndarray],
            names: Collection[str],
            unavailable: Mapping[str, str]
    ) -> None:
        """
        Check the nodes and names, and find the elements and the other
        names it reads.
        """
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
            if isinstance(node, ast.Attribute) \
                    and not (_is_numpy(node) or _is_relation(node)):
                raise InvalidExpressionError(
                    f"'{ast.unparse(node)}' is not allowed in expressions; "
                    f"of the attributes, only numpy ufuncs, numbers, "
                    f"scalar types and functions on arrays are (e.g. "
                    f"np.sin, np.pi, np.sum), and the relations of "
                    f"gbkfit.relations (e.g. relations.dz_c)")
            if isinstance(node, ast.Call) and not (
                    _is_numpy(node.func) or _is_relation(node.func) or (
                        isinstance(node.func, ast.Name)
                        and (node.func.id in BUILTINS
                             or node.func.id in names
                             or node.func.id in unavailable))):
                raise InvalidExpressionError(
                    f"'{ast.unparse(node.func)}' cannot be called in "
                    f"expressions; only numpy ufuncs, scalar types and "
                    f"functions on arrays, the relations of "
                    f"gbkfit.relations, {', '.join(BUILTINS)}, and in "
                    f"derived values, the functions of the models can")
            if isinstance(node, ast.keyword) and node.arg in _KEYWORDS:
                raise InvalidExpressionError(
                    f"the keyword '{node.arg}' is not allowed in expressions")
            if isinstance(node, ast.Name) and node.id in unavailable:
                raise InvalidExpressionError(
                    f"'{node.id}' cannot be read here: "
                    f"{unavailable[node.id]}")
            if isinstance(node, ast.Name) and node.id in names:
                self.uses.add(node.id)
            elif isinstance(node, ast.Name) and not (
                    node.id in pdescs or node.id in constants
                    or node.id in BUILTINS or node.id in ('np', 'relations')):
                raise InvalidExpressionError(f"unknown name '{node.id}'")
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in pdescs):
                self._read_subscript(node, node.value.id, pdescs)
                subscripted.add(node.value)
        # A parameter without a subscript reads all its elements
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and node.id in pdescs
                    and node not in subscripted):
                self._read(node.id, pdescs[node.id], None)

    def _read_subscript(
            self,
            node: ast.Subscript,
            name: str,
            pdescs: Mapping[str, ParamDesc]
    ) -> None:
        """Record the elements that a subscript of a parameter reads."""
        code = ast.unparse(node)
        pdesc = pdescs[name]
        if pdesc.is_scalar():
            raise InvalidExpressionError(
                f"'{code}': '{name}' is a scalar and has no elements")
        # A subscript that is not constant (e.g. v[int(a)]) can read any
        # element
        indices = None
        if subscript_value(node.slice) is not None:
            try:
                indices, _ = parse_subscript(node.slice, pdesc.size())
            except InvalidKeyError as e:
                raise InvalidExpressionError(f"'{code}': {e}") from None
        self._read(name, pdesc, indices)

    def _read(
            self,
            name: str,
            pdesc: ParamDesc,
            indices: np.ndarray | None
    ) -> None:
        """Record the elements of a parameter that it reads (all if None)."""
        if pdesc.is_scalar():
            self.reads[name] = None
            return
        if indices is None:
            indices = np.arange(pdesc.size())
        self.reads.setdefault(name, set()).update(int(i) for i in indices)
