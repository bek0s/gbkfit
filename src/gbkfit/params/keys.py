"""
The keys of parameter properties: a parameter name, optionally with a
numpy subscript that selects elements of a vector parameter, e.g. 'a',
'v', 'v[0]', 'v[-1]', 'v[1:4]', 'v[::2]' or 'v[[0, 2, 5]]'.
"""

import ast
import dataclasses

import numpy as np

from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc


__all__ = [
    'Key',
    'InvalidKeyError',
    'parse_key',
    'parse_subscript',
    'subscript_value',
    'element_name',
    'element_names'
]


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

    def element_names(self):
        return element_names(self.name, self.indices)


def element_name(name, index=None):
    """The name of an element of a parameter, e.g. 'a' or 'v[3]'."""
    return name if index is None else f'{name}[{index}]'


def element_names(name, indices=None):
    if indices is None:
        return [name]
    return [element_name(name, i) for i in indices]


def subscript_value(node):
    """
    The value of a subscript of integer constants: an int, a slice, or
    a list of ints. None if the subscript is anything else.
    """
    def integer(n):
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
        values = [integer(n) for n in node.elts]
        return None if not values or None in values else values
    return integer(node)


def parse_subscript(node, size):
    """
    The indices of the elements of a vector of the given size selected by
    the subscript (an ast node), and whether it selects a single element
    with an index. Raise InvalidKeyError if the subscript is not one of
    integer constants, or if it is out of range.
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


def parse_key(key: str, pdescs: dict[str, ParamDesc]) -> Key | None:
    """
    Parse a key. Return None if its parameter is not in pdescs, and raise
    InvalidKeyError if it is invalid.
    """
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
    if isinstance(pdesc, ParamScalarDesc):
        if subscript is not None:
            raise InvalidKeyError(f"'{name}' is a scalar and has no elements")
        return Key(name, None)
    if subscript is None:
        return Key(name, np.arange(pdesc.size()))
    indices, element = parse_subscript(subscript, pdesc.size())
    return Key(name, indices, element)
