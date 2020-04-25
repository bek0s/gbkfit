
import collections
import collections.abc
import copy

import funcy
#import iteration_utilities


def is_iterable(x):
    return isinstance(x, collections.abc.Iterable)


def is_sequence(x):
    return isinstance(x, (list, tuple, set))


def listify(x):
    return list([i for i in x] if is_sequence(x) else [x])


def tuplify(x):
    return tuple([i for i in x] if is_sequence(x) else [x])


def _make_seq(s, v, t, copy_):
    return t([_make_seq(s[1:], v, t, copy_) for _ in range(s[0])]) \
        if s else (copy.deepcopy(v) if copy_ else v)


def make_list(shape, v, copy_=False):
    return _make_seq(shape, v, list, copy_)


def make_tuple(shape, v, copy_=False):
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


def all_negative(x, include_zero=True):
    return all(i <= 0 if include_zero else i < 0 for i in x)


def all_unique(x):
    seen = list()
    return not any(i in seen or seen.append(i) for i in x)


def is_sequence_of_type(x, type_):
    return all(isinstance(i, type_) for i in x)


def flatten(x):
    return funcy.flatten(x)

"""
def duplicates(x, key=None):
    return list(iteration_utilities.duplicates(x, key))
"""

def seq_to_string(x):
    return str(list(x))[1:-1]


def map_to_string(x):
    return str(x)[1:-1]
