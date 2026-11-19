"""
The rules of version 18 of the format (2026-11-19 on), which rewrite
configurations of version 17: the noise of the mock data of an
observation is its own option noise, not one of the section mock.
"""

from collections.abc import Mapping, MutableMapping
from typing import Any

from .rules import Change, Rule, ViewedPiece, items
from .v4 import _insert_after


__all__ = [
    'RULES',
    'ObservationNoise'
]


class ObservationNoise(Rule):
    """
    The noise of mock data (the option noise of the section mock: one for
    every observation, or one for each by observation name) is the option
    noise of each observation (2026-11-19), after its dataset; the section
    mock keeps nmocks, nsamples and seed. A setup of a project that gives
    a noise to an observation piece it names stops the migration: other
    setups may name the piece too, with other noises, so its noise is to
    be given by hand.
    """

    @staticmethod
    def version() -> int:
        return 18

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        mock = config.get('mock')
        if not isinstance(mock, MutableMapping) or 'noise' not in mock:
            return []
        noise = mock['noise']
        # (by observation name: a dictionary without a type, see
        # ObservationGroup.per_observation)
        by_name = isinstance(noise, Mapping) and 'type' not in noise
        changes = []
        for path, observation in items(config, 'observations'):
            name = observation.get('name')
            if by_name and name not in noise:
                continue
            if isinstance(observation, ViewedPiece):
                raise ValueError(
                    f"the setup gives the noise of the mock data of the "
                    f"observation piece {observation.piece!r}, which other "
                    f"setups can name too: give the piece its noise by hand "
                    f"(its option noise), and take it out of the section mock")
            _insert_after(
                observation, 'dataset', 'noise',
                noise[name] if by_name else noise)
            changes.append(Change(
                path + ['noise'], "the noise of its mock data, from mock.noise"))
        del mock['noise']
        changes.append(Change(
            ['mock', 'noise'], "moved into the observations (their noise)"))
        return changes


RULES: tuple[Rule, ...] = (
    ObservationNoise(),
)
