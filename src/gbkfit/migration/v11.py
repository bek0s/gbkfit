"""
The rules of version 11 of the format (2026-11-14 on), which rewrite
configurations of version 10: an observation on the sky no longer
chooses its line of sight (the components it sees decide how it is
evaluated along it), and its depth axis, with the options of radiative
transfer, is its option depth; and the dataset of an observation is its
option dataset (it was data).
"""

import json
from collections.abc import Mapping, MutableMapping
from typing import Any

from ruamel.yaml.comments import CommentedMap

from .rules import Change, Rule, flow, items, rename


__all__ = [
    'RULES',
    'LineOfSightDepth',
    'ObservationDataset'
]


class LineOfSightDepth(Rule):
    """
    The option line_of_sight of an observation leaves (2026-11-14): the
    components it sees decide how it is evaluated along the line of sight
    (on the plane of the sky, on a depth axis, or by radiative transfer;
    see Scene.evaluation). The depth axis of a grid or of radiative
    transfer (its option depth), and the options slab and subchannels of
    radiative transfer, are its option depth (see Depth), in its place.
    Radiative transfer is now only that of line opacity and of galaxies
    that dim one another: a Monte Carlo disk that a line of sight of
    transfer saw without them is seen on the depth axis, of the lines of
    its clouds (transfer merged those of each voxel).
    """

    @staticmethod
    def version() -> int:
        return 11

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, observation in items(config, 'observations'):
            if 'line_of_sight' not in observation:
                continue
            old = observation['line_of_sight']
            depth = _depth(old)
            shown = json.dumps(old, default=str)
            if depth:
                rename(observation, 'line_of_sight', 'depth')
                observation['depth'] = depth
                message = f"{shown} -> depth {json.dumps(depth)}"
            else:
                del observation['line_of_sight']
                message = f"{shown} removed"
            message += (
                "; the components seen decide how the observation is "
                "evaluated along the line of sight")
            if _type(old) == 'transfer':
                message += (
                    " (radiative transfer only for line opacity, or galaxies "
                    "that dim one another)")
            changes.append(Change(path + ['line_of_sight'], message))
        return changes


def _type(line_of_sight: Any) -> str | None:
    """Return the type of a line of sight (a dictionary, or its type)."""
    if isinstance(line_of_sight, Mapping):
        return line_of_sight.get('type')
    return line_of_sight if isinstance(line_of_sight, str) else None


def _depth(line_of_sight: Any) -> MutableMapping[str, Any] | None:
    """
    Return the options of the depth axis of a line of sight, which the
    option depth of the observation takes: those of its depth axis, and
    the options of radiative transfer (slab, subchannels) of one of
    radiative transfer; None if there are none.
    """
    if not isinstance(line_of_sight, Mapping):
        return None
    depth = line_of_sight.get('depth')
    depth = depth if isinstance(depth, CommentedMap) \
        else flow(dict(depth or {}))
    for key in ('slab', 'subchannels'):
        if line_of_sight.get(key) is not None:
            depth[key] = line_of_sight[key]
    return depth or None


class ObservationDataset(Rule):
    """
    The dataset of an observation (2026-11-14; of the sky or of profiles,
    in place or the name of a piece) is its option dataset, in the place
    of its option data: the values of each data item keep their option
    data.
    """

    @staticmethod
    def version() -> int:
        return 11

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, observation in items(config, 'observations'):
            if 'data' in observation:
                rename(observation, 'data', 'dataset')
                changes.append(Change(path + ['data'], "-> dataset"))
        return changes


RULES: tuple[Rule, ...] = (
    LineOfSightDepth(),
    ObservationDataset(),
)
