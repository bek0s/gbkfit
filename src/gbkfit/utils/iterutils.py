
import collections.abc
import copy
from collections.abc import (
    Callable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence
)
from typing import Any, Literal

import numpy as np


def is_mapping(x: Any, strict: bool = True) -> bool:
    """
    Check if the given object is a mapping type.

    In strict mode, the function returns True only for `dict` and
    `collections.OrderedDict`. In relaxed mode, it returns True for any
    object that is an instance of `collections.abc.Mapping`.
    """
    mapping_types = (dict, collections.OrderedDict) if strict \
        else collections.abc.Mapping
    return isinstance(x, mapping_types)


def is_sequence(x: Any, strict: bool = True) -> bool:
    """
    Check if the given object is a sequence.

    In strict mode, the function returns True only for `list`, `tuple`,
    and 1-dimensional `numpy.ndarray`. In relaxed mode, it returns True
    for any object that is an instance of `collections.abc.Sequence`
    or a 1-dimensional `numpy.ndarray`.
    """
    is_list = isinstance(x, list)
    is_tuple = isinstance(x, tuple)
    is_array = isinstance(x, np.ndarray) and x.ndim == 1
    is_abc_sequence = isinstance(x, collections.abc.Sequence)
    return (is_list or is_tuple or is_array) \
        if strict else (is_abc_sequence or is_array)


def is_sequence_or_mapping(x: Any, strict: bool = True) -> bool:
    """
    Check if the given object is either a sequence or a mapping.

    See Also
    --------
    is_sequence : Check if an object is a sequence.
    is_mapping : Check if an object is a mapping.
    """
    return is_sequence(x, strict) or is_mapping(x, strict)


def listify(
        x: Any,
        none_is_val: bool = True,
        strict_sequence: bool = True
) -> list[Any]:
    """
    Convert an object to a list.

    If the object is already a sequence, it is converted to a list
    of its elements. If the object is not a sequence, it is wrapped
    into a single-item list.
    """
    if x is None and not none_is_val:
        return []
    return list([i for i in x] if is_sequence(x, strict_sequence) else [x])


def tuplify(
        x: Any,
        none_is_val: bool = True,
        strict_sequence: bool = True
) -> tuple[Any, ...]:
    """
    Convert an object to a tuple.

    If the object is already a sequence, it is converted into a tuple
    of its elements. If the object is not a sequence, it is wrapped
    into a single-item tuple.
    """
    return tuple(listify(x, none_is_val, strict_sequence))


def setify(
        x: Any,
        none_is_val: bool = True,
        strict_sequence: bool = True
) -> set[Any]:
    """
    Convert an object to a set.

    This function ensures that the input object is returned as a set.
    If the object is a set, a copy of it is returned. If the object is
    not a set or sequence, it is wrapped into a single-item set.
    """
    if isinstance(x, set):
        return set(x)
    return set(listify(x, none_is_val, strict_sequence))


def _make_seq(
        shape: Sequence[int],
        value: Any, type_: type[Any],
        deepcopy: bool
) -> Any:
    """
    Recursively creates a nested sequence based on the specified shape.
    """
    if shape:
        return type_([
            _make_seq(shape[1:], value, type_, deepcopy)
            for _ in range(shape[0])
        ])
    return copy.deepcopy(value) if deepcopy else value


def make_list(
        shape: int | Sequence[int],
        value: Any,
        deepcopy: bool = True
) -> list[Any]:
    """
    Create a nested list based on the specified shape and filled with
    the specified value.
    """
    shape = tuplify(shape, none_is_val=False)
    return _make_seq(shape, value, list, deepcopy)


def make_tuple(
        shape: int | Sequence[int],
        value: Any,
        deepcopy: bool = True
) -> tuple[Any]:
    """
    Create a nested tuple based on the specified shape and filled with
    the specified value.
    """
    shape = tuplify(shape, none_is_val=False)
    return _make_seq(shape, value, tuple, deepcopy)


def replace_item_in_sequence(
        x: MutableSequence[Any],
        old_value: Any,
        new_value: Any
) -> MutableSequence[Any]:
    """
    Replace all occurrences of old_value with new_value, in-place.
    """
    for i, value in enumerate(x):
        if value == old_value:
            x[i] = new_value
    return x


def rename_key_in_mapping(
        x: MutableMapping[Any, Any],
        old_key: Any,
        new_key: Any
) -> MutableMapping[Any, Any]:
    """
    Rename key old_key with the key new_key in a mapping, in-place.
    """
    x[new_key] = x.pop(old_key)
    return x


def remove_from_list_if(
        x: MutableSequence[Any],
        predicate: Callable[[Any], bool]
) -> MutableSequence[Any]:
    """
    Remove elements from the sequence in-place based on a predicate function.
    """
    x[:] = [i for i in x if not predicate(i)]
    return x


def remove_from_list(
        x: MutableSequence[Any],
        value: Any
) -> MutableSequence[Any]:
    """
    Remove all occurrences in-place of a specified value from the sequence.
    """
    return remove_from_list_if(x, lambda i: i == value)


def remove_from_mapping_if(
        x: MutableMapping[Any, Any],
        predicate: Callable[[Any, Any], bool]
) -> MutableMapping[Any, Any]:
    """
    Remove keys from the mapping in-place based on a predicate function.
    """
    for k in list(x.keys()):
        if predicate(k, x[k]):
            del x[k]
    return x


def remove_from_mapping_by_key(
        x: MutableMapping[Any, Any],
        key: Any
) -> MutableMapping[Any, Any]:
    """
    Remove a key from the mapping in-place.
    """
    return remove_from_mapping_if(x, lambda k, v: k == key)


def remove_from_mapping_by_value(
        x: MutableMapping[Any, Any],
        value: Any
) -> MutableMapping[Any, Any]:
    """
    Remove all key-value pairs from the mapping in-place based on a value.
    """
    return remove_from_mapping_if(x, lambda k, v: v == value)


def is_sorted(x: Sequence[Any], ascending: bool = True) -> bool:
    """
    Check if the given sequence is sorted in the specified order.
    """
    return not any(x[i-1] > x[i] if ascending else x[i-1] < x[i]
                   for i in range(1, len(x)))


def is_ascending(x: Sequence[Any]) -> bool:
    """
    Check if the elements in the sequence are in ascending order.
    """
    return is_sorted(x, ascending=True)


def is_descending(x: Sequence[Any]) -> bool:
    """
    Check if the elements in the sequence are in descending order.
    """
    return is_sorted(x, ascending=False)


def all_positive(x: Iterable[Any], include_zero: bool = True) -> bool:
    """
    Check if all elements in the iterable are positive.
    """
    return all(i >= 0 if include_zero else i > 0 for i in x)


def all_negative(x: Iterable[Any], include_zero: bool = False) -> bool:
    """
    Check if all elements in the iterable are negative.
    """
    return all(i <= 0 if include_zero else i < 0 for i in x)


def all_unique(x: Iterable[Any]) -> bool:
    """
    Check if all elements in an iterable are unique.
    """
    return len(get_duplicates(x)) == 0


def get_duplicates(x: Iterable[Any]) -> set[Any]:
    """
    Identify duplicate elements in an iterable.
    """
    seen = set()
    dupes = {i for i in x if i in seen or seen.add(i)}
    return dupes


def is_sequence_of_type(
        x: Sequence[Any],
        type_: type[Any],
        strict_sequence: bool = True
) -> bool:
    """
    Check if the object is a sequence and all elements are of a type.
    """
    is_sequence_ = is_sequence(x, strict_sequence)
    return is_sequence_ and all(isinstance(i, type_) for i in x)


def extract_subdict(
        x: Mapping[Any, Any],
        keys: Iterable[Any],
        on_missing_keys: Literal['ignore', 'none', 'raise'] = 'ignore'
) -> dict[Any, Any]:
    """
    Extract a subset of key-value pairs from a mapping as a dict.
    """
    result = {}
    for key in keys:
        if key in x:
            result[key] = x[key]
        elif on_missing_keys == 'none':
            result[key] = None
        elif on_missing_keys == 'raise':
            raise RuntimeError(f"missing key: {key}")
    return result


def validate_sequence_indices(
        indices: Sequence[int],
        length: int
) -> tuple[list[int], list[int]]:
    """
    Validate indices against the bounds of a sequence.
    """
    if length < 0:
        raise RuntimeError("length must be non-negative")
    valid_indices = []
    invalid_indices = []
    for i in indices:
        if -length <= i < length:
            valid_indices.append(i)
        else:
            invalid_indices.append(i)
    return valid_indices, invalid_indices


def unwrap_sequence_index(index: int, length: int) -> int:
    """
    Adjust an index to be within the bounds of a sequence.

    This function takes an index and a sequence length, and adjusts the
    index to ensure it is non-negative by adding the sequence length if
    the index is negative. The index is then wrapped within the bounds
    of the sequence, ensuring the resulting index is valid.
    """
    if length < 0:
        raise RuntimeError("length must be non-negative")
    index = index if index >= 0 else index + length
    if index < 0 or index >= length:
        raise RuntimeError("index is out of bounds")
    return index


def unwrap_sequence_indices(indices: Sequence[int], length: int) -> list[int]:
    """
    Adjust a list of indices to be within the bounds of a sequence.

    This function takes a sequence of indices and a sequence length,
    and adjusts each index to ensure it is non-negative by adding the
    sequence length if the index is negative. The indices are then
    wrapped to ensure each one is within the bounds of the sequence.
    """
    return [unwrap_sequence_index(i, length) for i in indices]


def sorted_sequence(
        x: Sequence[Any],
        value_order: Sequence[Any],
        on_missing_item: Literal[
            'raise', 'start', 'end', 'discard', 'preserve'] = 'raise'
) -> list[Any]:
    """
    Sort a sequence based on a custom value order.
    """

    # Check for duplicates in value_order
    if dupes := get_duplicates(value_order):
        raise RuntimeError(f"duplicate values in value_order: {dupes}")

    # Create position mapping (value => index) for value_order
    order_map = {value: index for index, value in enumerate(value_order)}

    # Raise exception on missing items in value_order
    if on_missing_item == 'raise':
        missing = set(x) - order_map.keys()
        if missing:
            raise RuntimeError(f"missing items in value_order: {missing}")

    # Discard items from x if not present in value_order
    if on_missing_item == 'discard':
        x = [item for item in x if item in order_map]

    def sort_key(item):
        if item in order_map:
            return 0, order_map[item]
        if on_missing_item == 'start':
            return -1, 0
        if on_missing_item == 'end':
            return 1, 0
        if on_missing_item == 'preserve':
            return 0, float('inf')
        raise RuntimeError("impossible")

    return sorted(x, key=sort_key)


def traverse_and_replace(
        x: Any,
        func: Callable[[Any], Any],
        strict: bool = True
) -> Any:
    """
    Recursively traverse a nested structure (sequence or mapping) and
    apply the given function to each element, but do not apply it to
    the sequences or mappings themselves.
    """
    if is_sequence(x, strict):
        return [traverse_and_replace(i, func, strict) for i in x]
    if is_mapping(x, strict):
        return {k: traverse_and_replace(v, func, strict) for k, v in x.items()}
    return func(x)


def nativify(x: Any) -> Any:
    """
    Recursively convert NumPy data types to native Python types.

    This function handles NumPy scalars (`np.integer`, `np.floating`),
    arrays (`np.ndarray`), as well as nested sequences and mappings.
    It ensures that all NumPy-specific types are converted to native
    Python types like `int`, `float`, and `list`.
    """
    if isinstance(x, np.integer):
        x = int(x)
    elif isinstance(x, np.floating):
        x = float(x)
    # Check first because is_sequence(np.ndarray) will also return true
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif is_sequence(x):
        x = [nativify(item) for item in x]
    elif is_mapping(x):
        x = {k: nativify(v) for k, v in x.items()}
    return x
