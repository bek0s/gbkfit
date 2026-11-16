"""
The rules of version 13 of the format (2026-11-16 on), which rewrite
configurations of version 12: the light of stars is light in bands, and a
spectral observation sees the velocities of the continuum by its option
continuum.
"""

from collections.abc import MutableMapping
from typing import Any

from .rules import Change, Rule, components, items, like
from .v4 import _insert_after


__all__ = [
    'RULES',
    'ContinuumNotStars'
]


class ContinuumNotStars(Rule):
    """
    The emission 'stars' is gone (2026-11-16): the light of stars is
    light in bands (type band), and a spectral observation sees the
    velocities of the continuum of the components that emit in bands and
    move by its option continuum (was the name 'stars' among its lines,
    which is dropped from them; lines: [] where it was the only one, as
    lines left out are all of them).
    """

    @staticmethod
    def version() -> int:
        return 13

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in items(config, 'galaxies'):
            for path, component in components(galaxy):
                emission = component.get('emission')
                if isinstance(emission, MutableMapping) \
                        and emission.get('type') == 'stars':
                    emission['type'] = 'band'
                    changes.append(Change(
                        galaxy_path + path + ['emission'],
                        "type stars -> band"))
        for path, observation in items(config, 'observations'):
            lines = observation.get('lines')
            if not isinstance(lines, list) or 'stars' not in lines:
                continue
            observation['lines'] = like(
                lines, [line for line in lines if line != 'stars'])
            _insert_after(observation, 'lines', 'continuum', True)
            changes.append(Change(
                path + ['lines'], "'stars' -> continuum: true"))
        return changes


RULES: tuple[Rule, ...] = (
    ContinuumNotStars(),
)
