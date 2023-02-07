
import typing
from collections.abc import Mapping, Sequence, Set
from types import UnionType
from typing import Any, Tuple


__all__ = [
    'validate_type'
]


def validate_type(value, type_):

    # print("value:", value)
    # print("type_:", type_)
    # print("get_origin(type_):", typing.get_origin(type_))
    # print("get_args(type_):", typing.get_args(type_))

    result = True
    args = typing.get_args(type_)
    origin = typing.get_origin(type_)

    # Type expected: leaf
    if origin is None:
        result = result and (type_ == Any or isinstance(value, type_))

    # Type expected: union
    elif issubclass(origin, UnionType):
        result = any(validate_type(value, arg) for arg in args)

    # Type expected: tuple
    elif issubclass(origin, Tuple):
        if isinstance(value, tuple) and (len(args) == len(value)):
            for val, arg in zip(value, args):
                result = result and validate_type(val, arg)
        else:
            result = False

    # Type expected: sequence
    elif issubclass(origin, Sequence):
        if isinstance(value, Sequence):
            for val in value:
                result = result and validate_type(val, args[0])
        else:
            result = False

    # Type expected: set
    elif issubclass(origin, Set):
        if isinstance(value, Set):
            for val in value:
                result = result and validate_type(val, args[0])
        else:
            result = False

    # Type expected: mapping
    elif issubclass(origin, Mapping):
        if isinstance(value, Mapping):
            for key, val in value.items():
                result = result and validate_type(key, args[0])
                result = result and validate_type(val, args[1])
        else:
            result = False

    # Type unsupported
    else:
        raise RuntimeError(
            f"the following type is not recognized: {origin.__qualname__}")

    return result
