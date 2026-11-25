"""
The rules of version 22 of the format (2026-11-25 on), which rewrite
configurations of version 21: what an observation sees of the galaxies
leaves it (its options galaxies and components); the section seen of a
setup gives it, one item for each observation, in their order.
"""

from collections.abc import Mapping, MutableMapping
from typing import Any

from ruamel.yaml.comments import CommentedSeq

from .rules import Change, PiecesView, Rule, items
from .v10 import _insert_after


__all__ = [
    'RULES',
    'SeenSection'
]


# The options of an observation that said what it sees of the galaxies
_OPTIONS = ('galaxies', 'components')


class SeenSection(Rule):
    """
    What an observation sees of the galaxies is not an option of it
    (2026-11-25): the section seen of a setup, after its section
    observations, gives it, one item for each observation, in their
    order: null for everything, or names of galaxies, which it sees
    whole, and of components ('galaxy.component', or the component alone
    when there is one galaxy), which it sees alone of their galaxies.

    An observation that named galaxies (its option galaxies) sees them;
    one that named components (its option components: 'galaxy.component'
    when it saw several galaxies, the component alone when it saw one)
    sees those. A setup whose observations named neither has no section
    seen. In a project, the observation pieces lose both options, which
    move into each setup that names them.
    """

    @staticmethod
    def version() -> int:
        return 22

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        if isinstance(config, PiecesView):
            return _rewrite_pieces(config)
        return _rewrite_setup(config)


def _rewrite_pieces(view: PiecesView) -> list[Change]:
    """
    Rewrite the pieces of a project (see SeenSection), in place, and
    return the changes: the observation pieces lose their options
    galaxies and components.
    """
    changes = []
    for path, observation in items(view, 'observations'):
        for key in _OPTIONS:
            if key in observation:
                value = observation.pop(key)
                changes.append(Change(path + [key], (
                    f"removed (was {list(value)}): -> the section seen of "
                    f"the setups that name the piece")))
    return changes


def _rewrite_setup(config: MutableMapping[str, Any]) -> list[Change]:
    """
    Rewrite a setup, or a configuration file (see SeenSection), in place,
    and return the changes.
    """
    value = config.get('observations')
    observations = value if isinstance(value, list) \
        else [] if value is None else [value]
    galaxies = config.get('galaxies')
    galaxies = galaxies if isinstance(galaxies, list) \
        else [] if galaxies is None else [galaxies]
    # (the names of the galaxies; those of pieces that the rules do not
    # see are not known)
    names = [
        galaxy.get('name') if isinstance(galaxy, Mapping) else None
        for galaxy in galaxies]
    seen = []
    changes = []
    for i, observation in enumerate(observations):
        # (of an observation in place, or a piece as the setup sees it;
        # the pieces the rules do not see are not known)
        if not isinstance(observation, MutableMapping):
            seen.append(None)
            continue
        options = {key: observation.pop(key, None) for key in _OPTIONS}
        item, by_hand = _seen(names, **options)
        seen.append(item)
        if by_hand:
            changes.append(Change(['observations', i], by_hand))
    if all(item is None for item in seen):
        return changes
    section = CommentedSeq(seen)
    section.fa.set_flow_style()
    _insert_after(config, 'observations', 'seen', section)
    return changes + [Change(['seen'], (
        "what each observation sees of the galaxies, in their order, from "
        "their options galaxies and components"))]


def _seen(
        names: list[str | None],
        galaxies: list[str] | None,
        components: list[str] | None
) -> tuple[list[str] | None, str | None]:
    """
    Return the item of the section seen of an observation of the given
    options galaxies and components, in a setup of galaxies of the given
    names (None where not known), and what to do by hand, if anything.
    """
    if components is None:
        return (None if galaxies is None else list(galaxies)), None
    components = list(components)
    # (components alone of one galaxy seen, 'galaxy.component' of several)
    seen = list(galaxies) if galaxies is not None else names
    if len(seen) != 1:
        return components, None
    if len(names) == 1:
        return components, None
    if seen[0] is None:
        return components, (
            "by hand: name the components it sees in the section seen as "
            "'galaxy.component' (the name of its galaxy is not known)")
    return [f'{seen[0]}.{name}' for name in components], None


RULES: tuple[Rule, ...] = (
    SeenSection(),
)
