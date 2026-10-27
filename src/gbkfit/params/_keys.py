"""
The keys of parameter properties: a parameter name, optionally with a
numpy subscript that selects elements of a vector parameter, e.g. 'a',
'v', 'v[0]', 'v[-1]', 'v[1:4]', 'v[::2]' or 'v[[0, 2, 5]]'.
"""

import ast
import dataclasses
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from gbkfit.params.pdescs import ParamDesc


class InvalidKeyError(ValueError):
    """An invalid key; the message explains why."""


@dataclasses.dataclass(frozen=True)
class Key:
    """
    A parsed key: the name of its parameter, and the indices of the
    elements it selects (None for a scalar parameter). element is True
    if it selects one element with a single index (e.g. 'v[0]').
    """
    name: str
    indices: np.ndarray | None
    element: bool = False


def element_name(name: str, index: int | None = None) -> str:
    """Return the name of an element of a parameter, e.g. 'a' or 'v[3]'."""
    return name if index is None else f'{name}[{index}]'


def element_names(name: str, indices: Iterable[int] | None) -> list[str]:
    """Return the names of the given elements of a parameter."""
    if indices is None:
        return [name]
    return [element_name(name, int(i)) for i in indices]


def with_elements(
        values: Mapping[str, float | np.ndarray],
        elements: Sequence[tuple[str, int | None]],
        new: float | Sequence[float] | np.ndarray
) -> dict[str, float | np.ndarray]:
    """
    Return the values of parameters with the given elements (each the
    name of its parameter and its index, None for a scalar) at new values
    (one for all, or one for each); copies of the vectors it changes.
    """
    result = dict(values)
    new = np.broadcast_to(np.asarray(new, dtype=float), len(elements))
    copied = set()
    for (name, index), value in zip(elements, new):
        if index is None:
            result[name] = float(value)
            continue
        if name not in copied:
            result[name] = np.array(result[name], dtype=float)
            copied.add(name)
        result[name][index] = value
    return result


def subscript_value(node: ast.expr) -> int | slice | list[int] | None:
    """
    Return the value of a subscript of integer constants: an int, a
    slice, or a list of ints; None if it is anything else.
    """
    def integer(n: ast.expr | None) -> int | None:
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            value = integer(n.operand)
            return None if value is None else -value
        if (isinstance(n, ast.Constant) and isinstance(n.value, int)
                and not isinstance(n.value, bool)):
            return n.value
        return None
    if isinstance(node, ast.Slice):
        parts = [node.lower, node.upper, node.step]
        values = [None if n is None else integer(n) for n in parts]
        if any(v is None and n is not None for v, n in zip(values, parts)):
            return None
        return slice(*values)
    if isinstance(node, ast.List):
        items = [integer(n) for n in node.elts]
        if not items or any(i is None for i in items):
            return None
        return [i for i in items if i is not None]
    return integer(node)


def parse_subscript(node: ast.expr, size: int) -> tuple[np.ndarray, bool]:
    """
    Return the indices of the elements of a vector of the given size that
    a subscript (an ast node) selects, and whether it selects a single
    element with an index.

    Raises InvalidKeyError if the subscript is not one of integer
    constants, or is out of range.
    """
    subscript = subscript_value(node)
    if subscript is None:
        raise InvalidKeyError(
            f"subscripts must be an index, a slice or a list of indices "
            f"of integer constants; found: [{ast.unparse(node)}]")
    try:
        indices = np.arange(size)[subscript]
    except IndexError:
        raise InvalidKeyError(
            f"index out of range for a vector of size {size}") from None
    return np.atleast_1d(indices), isinstance(subscript, int)


def parse_key(key: str, pdescs: Mapping[str, ParamDesc]) -> Key | None:
    """
    Parse a key. Return None if its parameter is not in pdescs.

    Raises InvalidKeyError if it is invalid.
    """
    if not isinstance(key, str):
        raise InvalidKeyError(f"keys must be strings; found: {key!r}")
    try:
        node = ast.parse(key.strip(), mode='eval').body
    except SyntaxError:
        raise InvalidKeyError("invalid syntax") from None
    if isinstance(node, ast.Name):
        name, subscript = node.id, None
    elif (isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)):
        name, subscript = node.value.id, node.slice
    else:
        raise InvalidKeyError("invalid syntax")
    pdesc = pdescs.get(name)
    if pdesc is None:
        return None
    if pdesc.is_scalar():
        if subscript is not None:
            raise InvalidKeyError(f"'{name}' is a scalar and has no elements")
        return Key(name, None)
    if subscript is None:
        return Key(name, np.arange(pdesc.size()))
    indices, element = parse_subscript(subscript, pdesc.size())
    return Key(name, indices, element)
