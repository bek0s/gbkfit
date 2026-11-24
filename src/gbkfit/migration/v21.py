"""
The rules of version 21 of the format (2026-11-24 on), which rewrite
configurations of version 20: the dataset leaves the observation (the
section datasets of a setup gives the data of its observations, one for
each, in their order).
"""

import copy
import logging
from collections.abc import Mapping, MutableMapping
from typing import Any

from ruamel.yaml.comments import CommentedSeq

import gbkfit.units
from gbkfit.config import ConfigError, paths_relative_to, plain_values
from gbkfit.dataset import Dataset, dataset_parser
from gbkfit.observation import observation_parser
from .rules import (
    Change, PiecesView, Rule, flow, insert_first, items)
from .v10 import _insert_after


__all__ = [
    'RULES',
    'DatasetPairing'
]


_log = logging.getLogger(__name__)

# The errors of data and observations that cannot be loaded (e.g. files
# that are missing), which leave an observation piece as it is
_LOAD_ERRORS = (ConfigError, OSError, ValueError, TypeError, KeyError)


class DatasetPairing(Rule):
    """
    The data of an observation are not an option of it (2026-11-24): the
    section datasets of a setup, after its section observations, gives
    the dataset of each observation, in their order (a dataset in place,
    the name of a piece of a project, or null for an observation without
    data); a setup whose observations have no data has none. The dataset
    of a profile, which had no type in it, has the type profile.

    In a project, the observation pieces lose their datasets, which move
    into each setup that names them (as the piece was). A piece stands
    alone (it renders, and draws mock data): it takes the options that
    its data gave it (see Observation.data_options, and the unit of its
    light), but those it gives itself (and of an option of options, e.g.
    its spectral axis, those of its own), its data read (their headers)
    as the library reads them; data that cannot be read leave it without
    them, with a warning, as does a piece that does not stand alone with
    them (e.g. regions in a file, which are not copied). Untyped dataset
    pieces (of profiles) take the type profile.
    """

    @staticmethod
    def version() -> int:
        return 21

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        if isinstance(config, PiecesView):
            return _rewrite_pieces(config)
        return _rewrite_setup(config)


def _rewrite_pieces(view: PiecesView) -> list[Change]:
    """
    Rewrite the pieces of a project (see DatasetPairing), in place, and
    return the changes: untyped dataset pieces take the type profile, and
    the observation pieces lose their datasets, taking the options that
    their data gave them.
    """
    changes = []
    for path, dataset in items(view, 'datasets'):
        if 'type' not in dataset:
            insert_first(dataset, 'type', 'profile')
            changes.append(Change(
                path + ['type'], "profile (a dataset of a profile)"))
    names = view.names.get('observations', [])
    for path, observation in items(view, 'observations'):
        dataset = observation.pop('dataset', None)
        if dataset is None:
            continue
        shown = repr(dataset) if isinstance(dataset, str) else "in place"
        changes.append(Change(path + ['dataset'], (
            f"removed (was {shown}): -> the section datasets of the setups "
            f"that name the piece")))
        # (of the pieces; not of the copies of observations of setups,
        # whose changes are not kept)
        if path[1] < len(names):
            changes += _sampled(view, path, names[path[1]], observation,
                                dataset)
    return changes


def _sampled(
        view: PiecesView,
        path: list[str | int],
        name: str,
        observation: MutableMapping[str, Any],
        dataset: Any
) -> list[Change]:
    """
    Give an observation piece of a project (of a name, at a path) the
    options that its dataset (in place, or the name of a dataset piece)
    gave it, but those it gives itself, in place, and return the changes;
    with a warning (and a change to make by hand) where its data cannot
    be read, or it does not stand alone with them.
    """
    project = view.project
    if project is None:
        return []
    if isinstance(dataset, str):
        dataset = view.piece('datasets', dataset)
        if dataset is None:
            return _by_hand(path, name, "its dataset piece is missing")
    info = plain_values(dataset)
    if observation.get('type') == 'profile':
        info.setdefault('type', 'profile')
    try:
        with paths_relative_to(project.path):
            data = dataset_parser.load(info)
            options = _options_of_data(view, observation, data)
    except _LOAD_ERRORS as e:
        return _by_hand(path, name, f"its data do not give them: {e}")
    copied = _with_options(observation, options)
    unit = copied.pop('unit', None)
    # (with the unit of its data, if it stands alone with it: one per beam
    # needs a PSF)
    for candidate in ([copied | dict(unit=unit)] if unit else []) + [copied]:
        try:
            with paths_relative_to(project.path):
                observation_parser.load(
                    _resolved(view, dict(observation) | candidate))
        except _LOAD_ERRORS as e:
            error = e
            continue
        break
    else:
        return _by_hand(path, name, (
            f"it does not stand alone with the options of its data: "
            f"{error}"))
    changes = []
    position = 'name' if 'name' in observation else 'type'
    for key, value in candidate.items():
        if key in observation and observation[key] == value:
            continue
        changes.append(Change(path + [key], "from its data"))
        if key in observation:
            observation[key] = value
        else:
            _insert_after(observation, position, key, _inline(value))
            position = key
    return changes


def _inline(value: Any) -> Any:
    """
    Return an option of plain values in the flow style of YAML (e.g. size:
    [48, 48]).
    """
    if isinstance(value, Mapping):
        return flow({key: _inline(item) for key, item in value.items()})
    if isinstance(value, list):
        result = CommentedSeq(_inline(item) for item in value)
        result.fa.set_flow_style()
        return result
    return value


def _options_of_data(
        view: PiecesView,
        observation: Mapping[str, Any],
        data: Dataset
) -> dict[str, Any]:
    """
    Return the options that data give an observation piece (see
    Observation.data_options, and the unit of the light of its data, if
    they have one), as a configuration: those of the data alone, not
    those in files.

    Raises
    ------
    ConfigError, ValueError, TypeError
        If the observation or its data cannot be loaded.
    """
    kind = observation_parser.registered_class(observation.get('type'))
    options = kind.data_options(data)
    paired = observation_parser.load(
        _resolved(view, observation), dataset=data)
    light = paired.light_items()
    unit = None if not light else paired.model_unit(light[0])[0]
    if unit is not None:
        options['unit'] = gbkfit.units.format_unit(unit)
    return options


def _with_options(
        observation: Mapping[str, Any], options: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Return the options that an observation piece takes of its data: those
    it does not give itself, and of those that are options of options
    (e.g. a spectral axis), the whole of those it gives, with the options
    of the data that it does not give.
    """
    copied = {}
    for key, value in options.items():
        own = observation.get(key)
        if own is None:
            copied[key] = value
        elif isinstance(own, Mapping) and isinstance(value, Mapping) \
                and set(value) - set(own):
            merged = copy.deepcopy(own)
            for sub, item in value.items():
                merged.setdefault(sub, item)
            copied[key] = merged
    return copied


def _resolved(
        view: PiecesView, observation: Mapping[str, Any]
) -> dict[str, Any]:
    """
    Return an observation piece as plain values, with the pieces that it
    names in place (its noise, PSF and LSF; see gbkfit.project.resolve),
    as the rules rewrote them so far.
    """
    resolved = plain_values(observation)
    if isinstance(name := resolved.get('noise'), str):
        resolved['noise'] = plain_values(view.piece('noise', name))
    instrument = resolved.get('instrument')
    if isinstance(instrument, dict):
        for key in ('psf', 'lsf'):
            if isinstance(name := instrument.get(key), str):
                instrument[key] = plain_values(view.piece(key, name))
    return resolved


def _by_hand(path: list[str | int], name: str, why: str) -> list[Change]:
    """
    Warn that an observation piece (of a name, at a path) does not take
    the options of its data, and return the change to make by hand.
    """
    message = (
        f"by hand: give the piece the options its data gave it (e.g. its "
        f"grid), which it needs alone; {why}")
    _log.warning("observation piece %r: %s", name, message)
    return [Change(path, message)]


def _rewrite_setup(config: MutableMapping[str, Any]) -> list[Change]:
    """
    Rewrite a setup, or a configuration file (see DatasetPairing), in
    place, and return the changes.
    """
    value = config.get('observations')
    observations = value if isinstance(value, list) \
        else [] if value is None else [value]
    datasets = []
    for observation in observations:
        # (of an observation in place, or a piece as the setup sees it;
        # the pieces the rules do not see are not known)
        dataset = observation.pop('dataset', None) \
            if isinstance(observation, MutableMapping) else None
        if _type_of(observation) == 'profile' \
                and isinstance(dataset, MutableMapping) \
                and 'type' not in dataset:
            # (that of a profile, whose observation gave its type)
            insert_first(dataset, 'type', 'profile')
        datasets.append(dataset)
    if all(dataset is None for dataset in datasets):
        return []
    # (names of pieces on one line, datasets in place one under another)
    section = CommentedSeq(datasets)
    if all(not isinstance(dataset, Mapping) for dataset in datasets):
        section.fa.set_flow_style()
    _insert_after(config, 'observations', 'datasets', section)
    return [Change(['datasets'], (
        "the datasets of the observations, in their order, from them"))]


def _type_of(observation: Any) -> Any:
    """Return the type of an observation, if it is a dictionary."""
    return observation.get('type') \
        if isinstance(observation, Mapping) else None


RULES: tuple[Rule, ...] = (
    DatasetPairing(),
)
