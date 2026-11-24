"""
The rules of version 20 of the format (2026-11-21 on), which rewrite
configurations of version 19: the footprint of a dataset (where the
instrument looked) is one, an option of the dataset, not of each map of
moments.
"""

from collections.abc import Iterator, MutableMapping
from typing import Any

from gbkfit.config import format_location
from .rules import Change, Rule, items


__all__ = [
    'RULES',
    'DatasetFootprint'
]


# The data items of the datasets of moments (see gbkfit.moments)
_MOMENT_ITEMS = (*(f'moment{order}' for order in range(8)), 'h3', 'h4')


class DatasetFootprint(Rule):
    """
    The footprint of a dataset of moment maps (pixel_moments) was an
    option of each map; it is one, of the dataset (2026-11-21), as that of
    the other datasets on grids. The footprint of its maps, the same for
    all, becomes that of the dataset; maps of different footprints stop
    the migration (which field the instrument saw is then to be said by
    hand). Those of the maps of moments of regions (region_moments), which
    have no pixels on the sky, go.
    """

    @staticmethod
    def version() -> int:
        return 20

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, dataset in _datasets(config):
            maps = {
                key: item for key in _MOMENT_ITEMS
                if isinstance(item := dataset.get(key), MutableMapping)
                and 'footprint' in item}
            if not maps:
                continue
            if dataset.get('type') == 'pixel_moments':
                footprints = [item['footprint'] for item in maps.values()]
                if any(f != footprints[0] for f in footprints[1:]):
                    raise ValueError(
                        f"the maps {sorted(maps)} of the dataset at "
                        f"{format_location(path)} have different "
                        f"footprints: give the dataset one footprint by "
                        f"hand (its option footprint), and take those of "
                        f"its maps out")
                dataset['footprint'] = footprints[0]
                changes.append(Change(
                    path + ['footprint'],
                    f"one for all the maps, from those of {sorted(maps)}"))
            else:
                changes += [
                    Change(path + [key, 'footprint'],
                           "removed: moments of regions have none")
                    for key in maps]
            for item in maps.values():
                del item['footprint']
        return changes


def _datasets(
        config: MutableMapping[str, Any]
) -> Iterator[tuple[list[str | int], MutableMapping[str, Any]]]:
    """
    Yield the datasets of a configuration with their paths: the dataset
    pieces of a project (its section 'datasets', see PiecesView), and
    those that observations give in place.
    """
    yield from items(config, 'datasets')
    for path, observation in items(config, 'observations'):
        dataset = observation.get('dataset')
        if isinstance(dataset, MutableMapping):
            yield path + ['dataset'], dataset


RULES: tuple[Rule, ...] = (
    DatasetFootprint(),
)
