"""
Helpers for sequences, mappings and other iterables.

A sequence here is a list, a tuple or a one-dimensional numpy array (not
a string), and a mapping is a dict, as in configurations.
"""

import collections
import copy
import itertools
from collections.abc import (
    Callable, Iterable, Mapping, MutableMapping, MutableSequence, Sequence)
from typing import Any, Literal

import numpy as np


__all__ = [
    'all_unique',
    'duplicates',
    'extract_subdict',
    'extract_sublist',
    'is_ascending',
    'is_descending',
    'is_mapping',
    'is_sequence',
    'is_sequence_of_type',
    'is_sequence_or_mapping',
    'is_sorted',
    'listify',
    'make_list',
    'make_tuple',
    'merge_with_prefixes',
    'normalize_index',
    'normalize_indices',
    'remove_from_list',
    'remove_from_list_if',
    'remove_from_mapping_by_key',
    'remove_from_mapping_by_value',
    'remove_from_mapping_if',
    'rename_key',
    'replace_in_sequence',
    'setify',
    'sorted_by_order',
    'split_valid_indices',
    'traverse_and_replace',
    'tuplify'
]


def is_mapping(x: Any) -> bool:
    """
    Check whether an object is a mapping (a dict).

    Parameters
    ----------
    x : Any
        An object.

    Returns
    -------
    bool
        Whether it is a dict.
    """
    return isinstance(x, dict)


def is_sequence(x: Any) -> bool:
    """
    Check whether an object is a sequence.

    Parameters
    ----------
    x : Any
        An object.

    Returns
    -------
    bool
        Whether it is a list, a tuple or a one-dimensional numpy array.
    """
    is_array = isinstance(x, np.ndarray) and x.ndim == 1
    return isinstance(x, (list, tuple)) or is_array


def is_sequence_or_mapping(x: Any) -> bool:
    """
    Check whether an object is a sequence or a mapping.

    Parameters
    ----------
    x : Any
        An object.

    Returns
    -------
    bool
        Whether it is a sequence or a mapping (see is_sequence and
        is_mapping).
    """
    return is_sequence(x) or is_mapping(x)


def is_sequence_of_type(x: Any, type_: type) -> bool:
    """
    Check whether an object is a sequence of items of a type.

    Parameters
    ----------
    x : Any
        An object.
    type_ : type
        The type of the items.

    Returns
    -------
    bool
        Whether it is a sequence (see is_sequence) whose items are all
        instances of the type.
    """
    return is_sequence(x) and all(isinstance(item, type_) for item in x)


def listify(x: Any) -> list[Any]:
    """
    Return an object as a list.

    Parameters
    ----------
    x : Any
        An object.

    Returns
    -------
    list
        The items of a sequence (see is_sequence), no items for None, or
        else the object as the only item.
    """
    if x is None:
        return []
    return list(x) if is_sequence(x) else [x]


def tuplify(x: Any) -> tuple[Any, ...]:
    """
    Return an object as a tuple (see listify).

    Parameters
    ----------
    x : Any
        An object.

    Returns
    -------
    tuple
        The items of a sequence, no items for None, or else the object
        as the only item.
    """
    return tuple(listify(x))


def setify(x: Any) -> set[Any]:
    """
    Return an object as a set (see listify).

    Parameters
    ----------
    x : Any
        An object.

    Returns
    -------
    set
        The items of a set or of a sequence, no items for None, or else
        the object as the only item.
    """
    return set(x) if isinstance(x, set) else set(listify(x))


def _make_sequence(
        shape: tuple[int, ...],
        value: Any,
        type_: type,
        deepcopy: bool
) -> Any:
    """Return a nested sequence of the given shape, type and value."""
    if not shape:
        return copy.deepcopy(value) if deepcopy else value
    return type_(
        _make_sequence(shape[1:], value, type_, deepcopy)
        for _ in range(shape[0]))


def make_list(
        shape: int | Sequence[int],
        value: Any,
        deepcopy: bool = True
) -> list[Any]:
    """
    Return a list, nested for several dimensions, filled with a value.

    Parameters
    ----------
    shape : int or Sequence[int]
        The length of the list, or of each level of the nested lists.
    value : Any
        The value of the items.
    deepcopy : bool
        Whether each item is a deep copy of the value (e.g. a dict of its
        own), or the value itself.

    Returns
    -------
    list
        The list.
    """
    return _make_sequence(tuplify(shape), value, list, deepcopy)


def make_tuple(
        shape: int | Sequence[int],
        value: Any,
        deepcopy: bool = True
) -> tuple[Any, ...]:
    """
    Return a tuple, nested for several dimensions, filled with a value.

    Parameters
    ----------
    shape : int or Sequence[int]
        The length of the tuple, or of each level of the nested tuples.
    value : Any
        The value of the items.
    deepcopy : bool
        Whether each item is a deep copy of the value, or the value
        itself.

    Returns
    -------
    tuple
        The tuple.
    """
    return _make_sequence(tuplify(shape), value, tuple, deepcopy)


def replace_in_sequence(
        x: MutableSequence[Any],
        old_value: Any,
        new_value: Any
) -> MutableSequence[Any]:
    """
    Replace every item of a sequence equal to a value, in place.

    Parameters
    ----------
    x : MutableSequence
        A sequence.
    old_value : Any
        The value to replace.
    new_value : Any
        The value to replace it with.

    Returns
    -------
    MutableSequence
        The sequence.
    """
    for i, value in enumerate(x):
        if value == old_value:
            x[i] = new_value
    return x


def rename_key(
        x: MutableMapping[Any, Any],
        old_key: Any,
        new_key: Any
) -> MutableMapping[Any, Any]:
    """
    Rename a key of a mapping, in place.

    Parameters
    ----------
    x : MutableMapping
        A mapping.
    old_key : Any
        The key to rename (it must exist).
    new_key : Any
        Its new name.

    Returns
    -------
    MutableMapping
        The mapping.
    """
    x[new_key] = x.pop(old_key)
    return x


def remove_from_list_if(
        x: MutableSequence[Any],
        predicate: Callable[[Any], bool]
) -> MutableSequence[Any]:
    """
    Remove the items of a sequence that meet a condition, in place.

    Parameters
    ----------
    x : MutableSequence
        A sequence.
    predicate : Callable[[Any], bool]
        The condition, given an item.

    Returns
    -------
    MutableSequence
        The sequence.
    """
    x[:] = [item for item in x if not predicate(item)]
    return x


def remove_from_list(
        x: MutableSequence[Any],
        value: Any
) -> MutableSequence[Any]:
    """
    Remove the items of a sequence equal to a value, in place.

    Parameters
    ----------
    x : MutableSequence
        A sequence.
    value : Any
        The value to remove.

    Returns
    -------
    MutableSequence
        The sequence.
    """
    return remove_from_list_if(x, lambda item: item == value)


def remove_from_mapping_if(
        x: MutableMapping[Any, Any],
        predicate: Callable[[Any, Any], bool]
) -> MutableMapping[Any, Any]:
    """
    Remove the items of a mapping that meet a condition, in place.

    Parameters
    ----------
    x : MutableMapping
        A mapping.
    predicate : Callable[[Any, Any], bool]
        The condition, given a key and its value.

    Returns
    -------
    MutableMapping
        The mapping.
    """
    for key in list(x):
        if predicate(key, x[key]):
            del x[key]
    return x


def remove_from_mapping_by_key(
        x: MutableMapping[Any, Any],
        key: Any
) -> MutableMapping[Any, Any]:
    """
    Remove a key of a mapping, if it has it, in place.

    Parameters
    ----------
    x : MutableMapping
        A mapping.
    key : Any
        The key to remove.

    Returns
    -------
    MutableMapping
        The mapping.
    """
    x.pop(key, None)
    return x


def remove_from_mapping_by_value(
        x: MutableMapping[Any, Any],
        value: Any
) -> MutableMapping[Any, Any]:
    """
    Remove the items of a mapping whose value is equal to a value, in
    place.

    Parameters
    ----------
    x : MutableMapping
        A mapping.
    value : Any
        The value to remove.

    Returns
    -------
    MutableMapping
        The mapping.
    """
    return remove_from_mapping_if(x, lambda k, v: v == value)


def merge_with_prefixes(
        dicts: list[dict[str, Any]],
        prefixes: list[str]
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    """
    Merge dicts, with the keys of each prefixed by its prefix.

    Parameters
    ----------
    dicts : list[dict[str, Any]]
        The dicts.
    prefixes : list[str]
        The prefix of each dict.

    Returns
    -------
    dict[str, Any]
        The merged dict.
    tuple[dict[str, str], ...]
        For each dict, its keys and their prefixed keys.

    Raises
    ------
    RuntimeError
        If a prefixed key is repeated.
    """
    merged = {}
    mappings = []
    for item, prefix in zip(dicts, prefixes, strict=True):
        mapping = {key: f'{prefix}{key}' for key in item}
        repeated = sorted(set(mapping.values()) & merged.keys())
        if repeated:
            raise RuntimeError(f"names are repeated: {repeated}")
        merged |= {mapping[key]: value for key, value in item.items()}
        mappings.append(mapping)
    return merged, tuple(mappings)


def is_sorted(x: Sequence[Any], ascending: bool = True) -> bool:
    """
    Check whether a sequence is sorted (equal neighbours allowed).

    Parameters
    ----------
    x : Sequence
        A sequence.
    ascending : bool
        Whether in ascending (else descending) order.

    Returns
    -------
    bool
        Whether no item is after one greater (or, descending, smaller)
        than it.
    """
    if ascending:
        return all(a <= b for a, b in itertools.pairwise(x))
    return all(a >= b for a, b in itertools.pairwise(x))


def is_ascending(x: Sequence[Any]) -> bool:
    """
    Check whether a sequence is in ascending order (see is_sorted).

    Parameters
    ----------
    x : Sequence
        A sequence.

    Returns
    -------
    bool
        Whether it is in ascending order.
    """
    return is_sorted(x, ascending=True)


def is_descending(x: Sequence[Any]) -> bool:
    """
    Check whether a sequence is in descending order (see is_sorted).

    Parameters
    ----------
    x : Sequence
        A sequence.

    Returns
    -------
    bool
        Whether it is in descending order.
    """
    return is_sorted(x, ascending=False)




def duplicates(x: Iterable[Any]) -> set[Any]:
    """
    Return the items that appear more than once in an iterable.

    Parameters
    ----------
    x : Iterable
        An iterable of hashable items.

    Returns
    -------
    set
        The repeated items.
    """
    return {item for item, count in collections.Counter(x).items()
            if count > 1}


def all_unique(x: Iterable[Any]) -> bool:
    """
    Check whether no item appears more than once in an iterable.

    Parameters
    ----------
    x : Iterable
        An iterable of hashable items.

    Returns
    -------
    bool
        Whether all the items are unique.
    """
    return not duplicates(x)


def extract_sublist(
        x: Sequence[Any],
        items: Sequence[Any]
) -> tuple[list[Any], list[Any]]:
    """
    Split items into those in a sequence and those not in it.

    Parameters
    ----------
    x : Sequence
        A sequence.
    items : Sequence
        The items to look for.

    Returns
    -------
    list
        The items in the sequence, in their order.
    list
        The items not in the sequence, in their order.
    """
    found = [item for item in items if item in x]
    missing = [item for item in items if item not in x]
    return found, missing


def extract_subdict(
        x: Mapping[Any, Any],
        keys: Iterable[Any]
) -> tuple[dict[Any, Any], list[Any]]:
    """
    Return the items of a mapping with the given keys.

    Parameters
    ----------
    x : Mapping
        A mapping.
    keys : Iterable
        The keys to look for.

    Returns
    -------
    dict
        The keys found and their values, in the order of keys.
    list
        The keys not found, in their order.
    """
    keys = list(keys)
    found = {key: x[key] for key in keys if key in x}
    missing = [key for key in keys if key not in x]
    return found, missing


def split_valid_indices(
        indices: Sequence[int],
        length: int
) -> tuple[list[int], list[int]]:
    """
    Split indices into those valid for a sequence and the others.

    Parameters
    ----------
    indices : Sequence[int]
        The indices (negative indices count from the end).
    length : int
        The length of the sequence.

    Returns
    -------
    list[int]
        The valid indices, in their order.
    list[int]
        The invalid indices, in their order.

    Raises
    ------
    ValueError
        If the length is negative.
    """
    if length < 0:
        raise ValueError(f"the length must not be negative; it is {length}")
    valid = [i for i in indices if -length <= i < length]
    invalid = [i for i in indices if not -length <= i < length]
    return valid, invalid


def normalize_index(index: int, length: int) -> int:
    """
    Return the non-negative form of an index of a sequence.

    Parameters
    ----------
    index : int
        The index (negative indices count from the end).
    length : int
        The length of the sequence.

    Returns
    -------
    int
        The index, counted from the start.

    Raises
    ------
    ValueError
        If the length is negative.
    IndexError
        If the index is outside the sequence.
    """
    if length < 0:
        raise ValueError(f"the length must not be negative; it is {length}")
    if not -length <= index < length:
        raise IndexError(
            f"the index {index} is outside a sequence of length {length}")
    return index + length if index < 0 else index


def normalize_indices(indices: Sequence[int], length: int) -> list[int]:
    """
    Return the non-negative forms of indices of a sequence.

    Parameters
    ----------
    indices : Sequence[int]
        The indices (negative indices count from the end).
    length : int
        The length of the sequence.

    Returns
    -------
    list[int]
        The indices, counted from the start (see normalize_index).
    """
    return [normalize_index(i, length) for i in indices]


def sorted_by_order(
        x: Iterable[Any],
        order: Sequence[Any],
        on_missing: Literal['raise', 'start', 'end', 'discard'] = 'raise'
) -> list[Any]:
    """
    Return the items of an iterable sorted in a given order of values.

    Parameters
    ----------
    x : Iterable
        An iterable of hashable items.
    order : Sequence
        The values, in their order (each once).
    on_missing : {'raise', 'start', 'end', 'discard'}
        What to do with the items not in the order: raise ValueError, put
        them first or last (in their order), or leave them out.

    Returns
    -------
    list
        The sorted items.

    Raises
    ------
    ValueError
        If on_missing is unknown, if the order repeats values, or with
        'raise', if items are not in the order.
    """
    choices = ('raise', 'start', 'end', 'discard')
    if on_missing not in choices:
        raise ValueError(
            f"on_missing must be one of {choices}; it is {on_missing!r}")
    if repeated := duplicates(order):
        raise ValueError(f"the order repeats values: {list(repeated)}")
    positions = {value: i for i, value in enumerate(order)}
    x = list(x)
    missing = [item for item in x if item not in positions]
    if missing and on_missing == 'raise':
        raise ValueError(f"these items are not in the order: {missing}")
    if on_missing == 'discard':
        x = [item for item in x if item in positions]
    # (sorted is stable: the missing items keep their order)
    missing_key = (-1, 0) if on_missing == 'start' else (1, 0)
    return sorted(x, key=lambda item: (
        (0, positions[item]) if item in positions else missing_key))


def traverse_and_replace(x: Any, func: Callable[[Any], Any]) -> Any:
    """
    Return a nested structure with a function applied to its leaves.

    Parameters
    ----------
    x : Any
        A sequence or a mapping, possibly nested, or a leaf.
    func : Callable[[Any], Any]
        The function to apply to each leaf (an item that is neither a
        sequence nor a mapping).

    Returns
    -------
    Any
        The structure, of lists and dicts, with the leaves replaced.
    """
    if is_sequence(x):
        return [traverse_and_replace(item, func) for item in x]
    if is_mapping(x):
        return {k: traverse_and_replace(v, func) for k, v in x.items()}
    return func(x)
