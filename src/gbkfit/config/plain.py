"""
The plain values of configurations (see plain_values): those of YAML read
in round trip, or of Python, as dicts, lists and the values of Python.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from ruamel.yaml.scalarbool import ScalarBoolean

from gbkfit.utils import numutils


__all__ = [
    'plain_values'
]


def plain_values(value: Any) -> Any:
    """
    Return a value of a configuration as plain values, in containers of
    its own: mappings as dicts, lists and tuples as lists, numpy arrays
    and scalars as lists and values of Python (see nativify), and other
    numbers, bools and strings (e.g. those of ruamel.yaml, which keep
    their style) as those of Python; the others as they are (e.g. a
    Quantity, which some options take).
    """
    if isinstance(value, Mapping):
        return {plain_values(k): plain_values(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_values(item) for item in value]
    # (not the subclasses of arrays, e.g. Quantity)
    if isinstance(value, np.generic) or type(value) is np.ndarray:
        return numutils.nativify(value)
    # (the bools of ruamel.yaml with an anchor are ints)
    if isinstance(value, (bool, ScalarBoolean)):
        return bool(value)
    for type_ in (int, float, str):
        if isinstance(value, type_):
            return type_(value)
    return value
