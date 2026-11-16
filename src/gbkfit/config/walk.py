"""
The leaves of a configuration that name files or other parts of it (see
config_leaves), found as configurations are read (see from_config): by
the annotations of the options of the constructors they call, FilePath
and DataFile for files, and Serializable.config_references for the
options that name other parts of the configuration (e.g. the component
that a mass disk follows).
"""

import copy
import typing
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

import astropy.units

from .errors import ConfigError
from .serializable import (
    Serializable, _bare, _family_parser, _item_type, _takes_dicts,
    constructor_of, is_record)
from .types import config_form, option_types, union_members


__all__ = [
    'ConfigLeaf',
    'config_leaves',
    'with_files'
]


# The kind of the leaves that name files (see ConfigLeaf)
FILE = 'file'


class ConfigLeaf(NamedTuple):
    """
    A leaf of a configuration that names a file or another part of the
    configuration (see config_leaves).

    Attributes
    ----------
    path : tuple of str or int
        Its path in the configuration walked (of a DataFile given as a
        dict, that of its file).
    kind : str
        'file' for a file, or else the option of the configuration that
        holds it whose items it names (see
        Serializable.config_references; e.g. 'components').
    value : str
        The filename, as the configuration gives it, or the name.
    """
    path: tuple[str | int, ...]
    kind: str
    value: str


def config_leaves(value: Any, type_: Any) -> Iterator[ConfigLeaf]:
    """
    Yield the leaves of the configuration of an option of a type
    annotation (e.g. a galaxy, of Galaxy) that name files or other parts
    of the configuration (see ConfigLeaf), in the order of the
    configuration, as from_config reads it: the options of objects by the
    annotations of the constructors they call (of a family, chosen by
    their type), the items of sequences and the values of Mappings by
    theirs. What it cannot read (e.g. an unknown type) has none.

    Parameters
    ----------
    value : Any
        The configuration, as plain values (e.g. read from YAML).
    type_ : Any
        The annotation.

    Yields
    ------
    ConfigLeaf
        The leaves.
    """
    yield from _leaves(value, type_, ())


def with_files(
        value: Any, type_: Any, change: Callable[[str], str]
) -> Any:
    """
    Return a copy of the configuration of an option of a type annotation
    with the filenames that it gives changed (e.g. made relative to
    another directory; see config_leaves).
    """
    result = copy.deepcopy(value)
    for leaf in config_leaves(value, type_):
        if leaf.kind != FILE:
            continue
        if not leaf.path:
            return change(leaf.value)
        *parents, last = leaf.path
        holder = result
        for key in parents:
            holder = holder[key]
        holder[last] = change(leaf.value)
    return result


def _leaves(
        value: Any, type_: Any, path: tuple[str | int, ...]
) -> Iterator[ConfigLeaf]:
    """Yield the leaves of a value of an annotation, at a path."""
    if isinstance(value, Serializable):
        return
    members = tuple(
        member for x in union_members(type_)
        for member in union_members(_bare(x)))
    if isinstance(value, str):
        if any(config_form(m) in ('path', 'data_file') for m in members):
            yield ConfigLeaf(path, FILE, value)
    elif isinstance(value, Mapping):
        yield from _dict_leaves(value, members, path)
    elif isinstance(value, list | tuple):
        item_types = [x for x in map(_item_type, members) if x is not None]
        if item_types:
            for i, item in enumerate(value):
                yield from _leaves(item, item_types[0], (*path, i))


def _dict_leaves(
        value: Mapping[str, Any],
        members: tuple[Any, ...],
        path: tuple[str | int, ...]
) -> Iterator[ConfigLeaf]:
    """
    Yield the leaves of a dict of the members of a union annotation, as
    from_config chooses the member that reads it.
    """
    forms = [x for x in members if _takes_dicts(x)]
    parsers = [p for p in map(_family_parser, forms) if p is not None]
    others = [x for x in forms if _family_parser(x) is None]
    if parsers and ('type' in value or not others):
        cls = parsers[0].registered_classes().get(value.get('type'))
        if cls is not None:
            yield from _object_leaves(value, cls, path)
        return
    if not others:
        return
    member = others[0]
    if config_form(member) == 'data_file':
        if isinstance(value.get('file'), str):
            yield ConfigLeaf((*path, 'file'), FILE, value['file'])
    elif member is astropy.units.Quantity:
        return
    elif typing.get_origin(member) in (dict, Mapping):
        item_type = typing.get_args(member)[1]
        for key, item in value.items():
            yield from _leaves(item, item_type, (*path, key))
    elif is_record(member):
        yield from _options_leaves(value, member, None, path)
    else:
        yield from _object_leaves(value, member, path)


def _object_leaves(
        value: Mapping[str, Any],
        cls: type[Serializable],
        path: tuple[str | int, ...]
) -> Iterator[ConfigLeaf]:
    """
    Yield the leaves of the configuration of an object of a class, of the
    constructor it calls (none if it calls none; see constructor_of).
    """
    try:
        func, _ = constructor_of(cls, value)
    except ConfigError:
        return
    yield from _options_leaves(value, func, cls, path)


def _options_leaves(
        value: Mapping[str, Any],
        func: Callable,
        owner: type | None,
        path: tuple[str | int, ...]
) -> Iterator[ConfigLeaf]:
    """
    Yield the leaves of the options of a callable (of its parameters) of
    a class (None for a record): those that name other parts of the
    configuration (see Serializable.config_references), and those of
    the others, by their annotations.
    """
    hints = option_types(func)
    references = getattr(owner, 'config_references', {})
    for name, item in value.items():
        if name in references:
            names = [item] if isinstance(item, str) else \
                list(item) if isinstance(item, list) else []
            for i, named in enumerate(names):
                if isinstance(named, str):
                    yield ConfigLeaf(
                        (*path, name) if isinstance(item, str)
                        else (*path, name, i), references[name], named)
        elif name in hints:
            yield from _leaves(item, hints[name], (*path, name))
