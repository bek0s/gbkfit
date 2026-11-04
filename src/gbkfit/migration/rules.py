"""
The parts of the rules of migrations (see Rule): their base, the changes
they make, and the helpers they share to rewrite configurations in
place, with the comments and the styles of YAML.
"""

import abc
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any, NamedTuple

from ruamel.yaml.comments import CommentedMap, CommentedSeq


__all__ = [
    'Change',
    'Rule',
    'components',
    'flow',
    'insert_first',
    'items',
    'like',
    'rename'
]


class Change(NamedTuple):
    """
    A change of a configuration by a migration: its path in the
    configuration (option names and list indices), and what changed, or
    what to change by hand where the migration cannot.
    """
    path: list[str | int]
    message: str


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
