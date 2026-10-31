"""
Helpers for sequences, mappings and other iterables.

A sequence here is a list, a tuple or a one-dimensional numpy array (not
a string), and a mapping is a dict, as in configurations.
"""

import collections
import copy
import itertools
from collections.abc import (
    Callable, Iterable, MutableMapping, MutableSequence, Sequence)
from typing import Any, Literal

import numpy as np


__all__ = [
    'all_unique',
    'duplicates',
    'is_ascending',
    'is_mapping',
    'is_sequence',
    'is_sequence_or_mapping',
    'is_sorted',
    'listify',
    'merge_with_prefixes',
    'normalize_index',
    'normalize_indices',
    'remove_from_list_if',
    'remove_from_mapping_if',
    'setify',
    'sorted_by_order',
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
