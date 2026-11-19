"""
The rules of version 19 of the format (2026-11-19 on), which rewrite
configurations of version 18: the parameters of a lens of one piece are
prefixed by the name of the piece too, as those of a lens of several.
"""

import re
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any

from .rules import EXPRESSIONS, Change, Rule, items, renamed_params


__all__ = [
    'RULES',
    'LensPieceNames'
]


class LensPieceNames(Rule):
    """
    The parameters of a lens of one piece were named after the lens alone
    ('lens_theta_e'); they are named after its piece too (2026-11-19:
    'lens_sie_theta_e', the piece named after its type by default), so
    that a piece added to a lens leaves the names of the others as they
    are. Renamed in the params sections (keys, modes and expressions) of
    the configuration: the names that start with the name of the lens and
    '_', but those of another item of a longer name (e.g. a galaxy
    'lens_b').
    """

    @staticmethod
    def version() -> int:
        return 19

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        singles = {}
        for _, lens in items(config, 'lenses'):
            pieces = lens.get('pieces')
            if isinstance(pieces, list):
                if len(pieces) != 1:
                    continue
                pieces = pieces[0]
            name = lens.get('name')
            if isinstance(name, str) and isinstance(pieces, Mapping):
                piece = pieces.get('name', pieces.get('type'))
                if isinstance(piece, str):
                    singles[name] = piece
        if not singles:
            return []
        others = [
            item['name'] for section in ('galaxies', 'lenses', 'observations')
            for _, item in items(config, section)
            if isinstance(item.get('name'), str)]
        changes = []
        for path, params in items(config, 'params'):
            renames = {}
            for name in _names(params):
                owner = max(
                    (o for o in others if name.startswith(f'{o}_')),
                    key=len, default=None)
                if owner in singles:
                    piece = singles[owner]
                    renames[name] = \
                        f'{owner}_{piece}_{name[len(owner) + 1:]}'
            if renames:
                changes += renamed_params(params, path, renames, 'lenses')
        return changes


def _names(params: Mapping[str, Any]) -> Iterator[str]:
    """
    Yield the names of the parameters of a params section: the keys of its
    properties and modes (without subscripts), and the names in its
    expressions.
    """
    found = set()
    for option in ('properties', 'modes'):
        if isinstance(mapping := params.get(option), Mapping):
            found |= {key.split('[')[0] for key in mapping}
    for key in EXPRESSIONS:
        found |= set(_identifiers(params.get(key)))
    yield from sorted(found)


def _identifiers(value: Any) -> Iterator[str]:
    """Yield the identifiers in the strings of a value and its items."""
    if isinstance(value, str):
        yield from re.findall(r'(?<![\w.])[A-Za-z_]\w*', value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _identifiers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _identifiers(item)


RULES: tuple[Rule, ...] = (
    LensPieceNames(),
)
