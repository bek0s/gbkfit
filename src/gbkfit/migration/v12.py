"""
The rules of version 12 of the format (2026-11-14 on), which rewrite
configurations of version 11: the observable of an observation on the
sky is merged into it (its type is that of its observable, e.g.
pixel_spectra, and the options of its observable are its own).
"""

from collections.abc import Mapping, MutableMapping
from typing import Any

from ruamel.yaml.comments import CommentedMap

from .rules import Change, Rule, items


__all__ = [
    'RULES',
    'ObservableMerged'
]


class ObservableMerged(Rule):
    """
    The observable of an observation on the sky (2026-11-14) is merged
    into it: the observation takes the type of its observable, in the
    place of 'sky', and the options of the observable, at the place of
    its option observable.
    """

    @staticmethod
    def version() -> int:
        return 12

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, observation in items(config, 'observations'):
            observable = observation.get('observable')
            if observation.get('type', 'sky') != 'sky' \
                    or not isinstance(observable, Mapping):
                continue
            kind = observable.get('type')
            _splice(observation, 'observable', {
                key: value for key, value in observable.items()
                if key != 'type'}, observable)
            observation['type'] = kind
            changes.append(Change(
                path + ['observable'], f"merged: type sky -> {kind}"))
        return changes


def _splice(
        mapping: MutableMapping[str, Any],
        key: str,
        values: Mapping[str, Any],
        source: Mapping[str, Any]
) -> None:
    """
    Replace a key of a dictionary in place with the items of another, at
    its place, with their comments in the dictionary they come from
    (source), of YAML.
    """
    pairs = list(mapping.items())
    position = [name for name, _ in pairs].index(key)
    if not isinstance(mapping, CommentedMap):
        mapping.clear()
        mapping.update(
            pairs[:position] + list(values.items()) + pairs[position + 1:])
        return
    del mapping[key]
    for offset, (name, value) in enumerate(values.items()):
        mapping.insert(position + offset, name, value)
        if isinstance(source, CommentedMap) and name in source.ca.items:
            mapping.ca.items[name] = source.ca.items[name]


RULES: tuple[Rule, ...] = (
    ObservableMerged(),
)
