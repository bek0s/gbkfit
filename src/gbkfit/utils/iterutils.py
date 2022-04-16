
import collections.abc
import copy


def is_iterable(x, strict=True):
    return isinstance(x, (tuple, list, set, dict)) if strict \
        else isinstance(x, collections.abc.Iterable)


def is_mapping(x, strict=True):
    return isinstance(x, (dict,)) if strict \
        else isinstance(x, collections.abc.Mapping)


def is_sequence(x, strict=True):
    return isinstance(x, (tuple, list, set)) if strict \
        else isinstance(x, collections.abc.Sequence)


def listify(x, none_is_val=True):
    if x is None and not none_is_val:
        return []
    return list([i for i in x] if is_sequence(x) else [x])


def tuplify(x, none_is_val=True):
    return tuple(listify(x, none_is_val))


def _make_seq(s, v, t, copy_):
    s = tuplify(s)
    return t([_make_seq(s[1:], v, t, copy_) for _ in range(s[0])]) \
        if s else (copy.deepcopy(v) if copy_ else v)


def make_list(shape, v, copy_=True):
    return _make_seq(shape, v, list, copy_)


def make_tuple(shape, v, copy_=True):
    return _make_seq(shape, v, tuple, copy_)


def replace_items_in_place(x, old_value, new_value, copy_=False):
    for i, value in enumerate(x):
        if value == old_value:
            x[i] = copy.deepcopy(new_value) if copy_ else new_value


def replace_items_and_copy(x, old_value, new_value):
    return [new_value if item == old_value else item for item in x]


def is_sorted(x, ascending=True):
    return not any(x[i-1] > x[i] if ascending else x[i-1] < x[i]
                   for i in range(1, len(x)))


def is_ascending(x):
    return not any(x[i - 1] > x[i] for i in range(1, len(x)))


def is_descending(x):
    return not any(x[i - 1] < x[i] for i in range(1, len(x)))


def all_positive(x, include_zero=True):
    return all(i >= 0 if include_zero else i > 0 for i in x)


def all_negative(x, include_zero=False):
    return all(i <= 0 if include_zero else i < 0 for i in x)


def all_unique(x):
    seen = list()
    return not any(i in seen or seen.append(i) for i in x)


def is_iterable_of_type(x, type_):
    return is_iterable(x) and all(isinstance(i, type_) for i in x)


def is_sequence_of_type(x, type_):
    return is_sequence(x) and all(isinstance(i, type_) for i in x)


def extract_subdict(d, keys, on_missing_keys='ignore'):
    assert on_missing_keys in ['ignore', 'none', 'raise']
    result = {}
    keys_intersection = set(keys).intersection(d.keys())
    if on_missing_keys == 'raise' and len(keys) != len(keys_intersection):
        raise RuntimeError()
    for key in keys:
        if key not in d and on_missing_keys == 'ignore':
            continue
        result[key] = d.get(key)
    return result


def validate_sequence_indices(indices, length):
    def is_valid(i): return -length <= i < length
    def is_invalid(i): return not is_valid(i)
    valid_indices = set(filter(is_valid, indices))
    invalid_indices = set(filter(is_invalid, indices))
    return list(sorted(valid_indices)), list(sorted(invalid_indices))


def unwrap_sequence_indices(indices, length):
    return [i + length if i < 0 else i for i in indices]


def sorted_sequence(b, value_order):

    return sorted(b, key=lambda x: value_order.index(x))
