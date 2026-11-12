"""
The parts of the rules of migrations (see Rule): their base, the changes
they make, and the helpers they share to rewrite configurations in
place, with the comments and the styles of YAML.
"""

import abc
import re
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from typing import Any, NamedTuple

from ruamel.yaml.comments import CommentedMap, CommentedSeq


__all__ = [
    'EXPRESSIONS',
    'Change',
    'PiecesView',
    'Rule',
    'ViewedPiece',
    'components',
    'flow',
    'insert_first',
    'items',
    'like',
    'rename',
    'renamed_params',
    'replaced'
]


# The options of params sections that hold names of parameters (others
# hold e.g. file names)
EXPRESSIONS = ('properties', 'derived', 'priors')


class Change(NamedTuple):
    """
    A change of a configuration by a migration: its path in the
    configuration (option names and list indices), and what changed, or
    what to change by hand where the migration cannot.
    """
    path: list[str | int]
    message: str


class PiecesView(dict):
    """
    The pieces of a project, together, as rules rewrite them (see
    migrate_project): a configuration whose sections are lists of the
    pieces. Not one setup: a rule that makes a choice for a whole setup
    (e.g. its cosmology, from those of its galaxies) leaves it to the
    setups, which it sees with their galaxies.
    """


class ViewedPiece(dict):
    """
    A copy of a piece that a setup names, as rules see it in the setup
    (see migrate_project): as it was, for what the setup takes of it
    (e.g. the parameters of a galaxy, or the likelihood of an
    observation). What rules change in it is not kept: the piece is
    rewritten with the other pieces, once for all the setups.

    Parameters
    ----------
    piece : str
        The name of the piece.
    content : Mapping
        The piece.
    """

    def __init__(self, piece: str, content: Mapping[str, Any]):
        super().__init__(content)
        self.piece = piece


class Rule(abc.ABC):
    """
    A change of the format of configurations: of those of the version
    before its own (see version) to its own.
    """

    @staticmethod
    @abc.abstractmethod
    def version() -> int:
        """Return the version of the format that it rewrites to."""

    @abc.abstractmethod
    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        """
        Rewrite the old forms of a configuration in the new form, in
        place.

        Parameters
        ----------
        config : MutableMapping
            The configuration. It can have only some of its sections
            (e.g. a piece of a project), and their items can be names of
            pieces, which are left as they are. A section can be a list of
            dictionaries (e.g. the pieces of a project).

        Returns
        -------
        list of Change
            The changes.
        """


def items(
        config: Mapping[str, Any], section: str
) -> Iterator[tuple[list[str | int], MutableMapping[str, Any]]]:
    """
    Yield the dictionaries of a section of a configuration with their
    paths: the section itself, if a dictionary, or each of its items
    that is one (not the names of pieces).
    """
    value = config.get(section)
    if isinstance(value, MutableMapping):
        yield [section], value
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, MutableMapping):
                yield [section, i], item


def components(
        galaxy: Mapping[str, Any]
) -> Iterator[tuple[list[str | int], MutableMapping[str, Any]]]:
    """
    Yield the components of a galaxy that are dictionaries, with their
    paths in it.
    """
    components = galaxy.get('components')
    if isinstance(components, MutableMapping):
        yield ['components'], components
    elif isinstance(components, list):
        for i, component in enumerate(components):
            if isinstance(component, MutableMapping):
                yield ['components', i], component


def rename(mapping: MutableMapping[str, Any], old: str, new: str) -> None:
    """
    Rename a key of a dictionary in place, at its position (with its
    comments, of YAML).
    """
    position = list(mapping).index(old)
    value = mapping.pop(old)
    if isinstance(mapping, CommentedMap):
        mapping.insert(position, new, value)
        if old in mapping.ca.items:
            mapping.ca.items[new] = mapping.ca.items.pop(old)
    else:
        pairs = list(mapping.items())
        pairs.insert(position, (new, value))
        mapping.clear()
        mapping.update(pairs)


def insert_first(
        mapping: MutableMapping[str, Any], key: str, value: Any
) -> None:
    """Set a key of a dictionary in place, as its first one."""
    mapping.pop(key, None)
    if isinstance(mapping, CommentedMap):
        mapping.insert(0, key, value)
    else:
        pairs = list(mapping.items())
        mapping.clear()
        mapping.update([(key, value)] + pairs)


def like(sequence: list[Any], items: list[Any]) -> list[Any]:
    """
    Return items as a list of the style of a sequence (of YAML: flow or
    block).
    """
    if not isinstance(sequence, CommentedSeq):
        return items
    result = CommentedSeq(items)
    if sequence.fa.flow_style():
        result.fa.set_flow_style()
    return result


def flow(mapping: dict[str, Any]) -> CommentedMap:
    """Return a dictionary as one of the flow style of YAML."""
    result = CommentedMap(mapping)
    result.fa.set_flow_style()
    return result


def replaced(
        value: Any,
        pattern: re.Pattern,
        new: str | Callable[[re.Match], str]
) -> tuple[Any, bool]:
    """
    Return a value with the matches of a pattern replaced in its strings
    and those of its items (in place for lists and dictionaries), and
    whether any was.
    """
    if isinstance(value, str):
        result = pattern.sub(new, value)
        return result, result != value
    found = False
    if isinstance(value, MutableMapping):
        for key in list(value):
            value[key], one = replaced(value[key], pattern, new)
            found |= one
    elif isinstance(value, list):
        for i, item in enumerate(value):
            value[i], one = replaced(item, pattern, new)
            found |= one
    return value, found


def renamed_params(
        params: MutableMapping[str, Any],
        path: list[str | int],
        renames: Mapping[str, str],
        what: str
) -> list[Change]:
    """
    Rename parameters of a params section by their names, in place, at
    once (so that renames do not chain): the keys of its properties and
    modes (e.g. a, a[1:]), and the names in its expressions; the function
    of its transforms is code, which the changes say to rename by hand.
    The changes of the expressions say whose parameters were renamed
    (what, e.g. 'halos').
    """
    changes = []
    for option in ('properties', 'modes'):
        mapping = params.get(option)
        if not isinstance(mapping, MutableMapping):
            continue
        for key in list(mapping):
            name = key.split('[')[0]
            if name in renames:
                new = renames[name] + key[len(name):]
                rename(mapping, key, new)
                changes.append(Change(path + [option], f"{key} -> {new}"))
    if 'transforms' in params:
        changes.append(Change(path + ['transforms'], (
            f"by hand: rename {', '.join(renames)} in the transforms "
            f"function")))
    pattern = re.compile(
        rf'(?<![\w.])({"|".join(re.escape(o) for o in renames)})(?!\w)')
    for key in [k for k in EXPRESSIONS if k in params]:
        params[key], found = replaced(
            params[key], pattern, lambda m: renames[m.group(1)])
        if found:
            changes.append(Change(path + [key], (
                f"the parameters of {what} renamed in its expressions")))
    return changes
