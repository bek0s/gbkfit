"""
The rules of version 10 of the format (2026-11-12 on), which rewrite
configurations of version 9: the likelihoods leave the observations for
the section 'likelihoods' of their setup, and the options given for each
observation (the noise of mock data, the schemes of a bootstrap) are by
observation name instead of by position.
"""

import json
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from ruamel.yaml.comments import CommentedMap

from .rules import Change, PiecesView, Rule, ViewedPiece, insert_first, items


__all__ = [
    'RULES',
    'BootstrapSchemesByName',
    'MockLikelihood',
    'MockNoisesByName',
    'ObservationLikelihoods'
]


# The likelihood of an observation with data that gives none (see
# Likelihoods)
_DEFAULT = {'type': 'gauss'}


def _observations(config: Mapping[str, Any]) -> list[tuple[list, Any]]:
    """
    Return the observations of a configuration with their paths: the
    items of its section 'observations' (a list, or one), dictionaries or
    names of pieces.
    """
    value = config.get('observations')
    if isinstance(value, list):
        return [(['observations', i], item) for i, item in enumerate(value)]
    return [] if value is None else [(['observations'], value)]


def _has_data(observation: Any) -> bool | None:
    """
    Return whether an observation (a dictionary, or the name of a piece)
    has data; None if not known (the name of a piece the rules do not
    see).
    """
    if not isinstance(observation, Mapping):
        return None
    return observation.get('data') is not None


def _shown(value: Any) -> str:
    """Return a value of a configuration as JSON, for the changes."""
    return json.dumps(value, default=str)


def _names(
        config: MutableMapping[str, Any],
        indices: Sequence[int],
        what: str,
        path: list[str | int]
) -> tuple[dict[int, str], list[Change]]:
    """
    Return the names of the observations of a configuration of the given
    indices, which options given by observation name need (e.g. their
    likelihoods; what, for the changes), and the changes. If one has no
    name, every observation is named (they have names, or none has): those
    in place by their labels ('observation{i}', the ids of their outputs,
    which stay). Those that cannot be named here (pieces, which are
    rewritten once for all the setups that name them) are left out, with
    changes to make by hand at the path of the option (those of the
    observations of pieces are not the setup's).
    """
    observations = _observations(config)

    def name_of(observation: Any) -> str | None:
        return observation.get('name') \
            if isinstance(observation, Mapping) else None
    names, changes = {}, []
    unnamed = any(name_of(observations[i][1]) is None for i in indices)
    for i, (observation_path, observation) in enumerate(observations):
        name = name_of(observation)
        if name is not None or not unnamed:
            names[i] = name
            continue
        label = f'observation{i}'
        if isinstance(observation, MutableMapping) \
                and not isinstance(observation, ViewedPiece):
            insert_first(observation, 'name', label)
            names[i] = label
            changes.append(Change(observation_path, (
                f"name {label} (its label, the id of its outputs), for the "
                f"{what} given by observation name")))
            continue
        piece = observation.piece if isinstance(observation, ViewedPiece) \
            else observation
        changes.append(Change(path, (
            f"by hand: name the observation piece {piece!r} (e.g. "
            f"{label!r}), and give its {what} by that name")))
    return {i: names[i] for i in indices if names.get(i) is not None}, \
        changes


def _by_name(values: Mapping[int, Any], names: Mapping[int, str]) -> Any:
    """
    Return values of observations (by index) as an option given by
    observation name, of those named, in the style of YAML.
    """
    return CommentedMap(
        (names[i], value) for i, value in values.items() if i in names)


def _one_or_by_name(
        config: MutableMapping[str, Any],
        values: Mapping[int, Any],
        what: str,
        path: list[str | int]
) -> tuple[Any, bool, list[Change]]:
    """
    Return values of observations of a configuration (by index) as an
    option given by observation (at the path), one for all if they are
    all the same, or else by observation name (see _names), whether it is
    by name, and the changes of the names.
    """
    first = next(iter(values.values()))
    if all(value == first for value in values.values()):
        return first, False, []
    names, changes = _names(config, list(values), what, path)
    return _by_name(values, names), True, changes


def _insert_after(
        mapping: MutableMapping[str, Any],
        after: str,
        key: str,
        value: Any
) -> None:
    """
    Set a key of a dictionary in place, after another key (at the end if
    the other is not in it).
    """
    keys = list(mapping)
    position = keys.index(after) + 1 if after in keys else len(keys)
    if isinstance(mapping, CommentedMap):
        mapping.insert(position, key, value)
        return
    pairs = list(mapping.items())
    pairs.insert(position, (key, value))
    mapping.clear()
    mapping.update(pairs)


class ObservationLikelihoods(Rule):
    """
    The likelihoods of the observations are the section 'likelihoods' of
    their setup (2026-11-12; was the option likelihood of each
    observation): one likelihood for every observation, if each
    observation with data had the same one, or else likelihoods by
    observation name, of those that gave one (the others keep the
    Gaussian, the default). An observation without a name is named by
    its label, 'observation{i}' (the id of its outputs, which stays). In
    a project, the observation pieces lose their likelihood, which moves
    into each setup that names them; a piece without a name that needs
    one is to be named by hand.
    """

    @staticmethod
    def version() -> int:
        return 10

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        if isinstance(config, PiecesView):
            return [
                Change(path + ['likelihood'], (
                    "-> the section likelihoods of the setups that name the "
                    "piece"))
                for path, observation in items(config, 'observations')
                if observation.pop('likelihood', None) is not None]
        observations = _observations(config)
        given = {}
        changes = []
        for i, (path, observation) in enumerate(observations):
            if isinstance(observation, MutableMapping) \
                    and 'likelihood' in observation:
                likelihood = observation.pop('likelihood')
                # (not of the pieces, whose changes are their own)
                if not isinstance(observation, ViewedPiece):
                    changes.append(Change(
                        path + ['likelihood'], "-> section likelihoods"))
                if likelihood is not None:
                    given[i] = likelihood
        if not given:
            return changes
        # (the likelihood of each observation with data: given, or the
        # default; those given by name need a name of each)
        fitted = {
            i: given.get(i, _DEFAULT)
            for i, (_, observation) in enumerate(observations)
            if _has_data(observation) or i in given}
        first = next(iter(given.values()))
        if all(likelihood == first for likelihood in fitted.values()):
            section, named = first, []
            message = "one likelihood for every observation"
        else:
            names, named = _names(
                config, list(fitted), 'likelihood', ['likelihoods'])
            section = _by_name(given, names)
            message = "the likelihoods of the observations, by name"
        _insert_after(config, 'observations', 'likelihoods', section)
        return changes + named + [Change(['likelihoods'], message)]


def _fitted(
        config: Mapping[str, Any], nobservations: int
) -> list[Any]:
    """
    Return the likelihood that a fit gives each observation of a
    configuration of version 10 (see Likelihoods): of its section
    likelihoods, or the default.
    """
    section = config.get('likelihoods')
    observations = _observations(config)
    if section is None:
        return [_DEFAULT] * nobservations
    if isinstance(section, Mapping) and 'type' in section:
        return [section] * nobservations
    return [
        section.get(observation.get('name'), _DEFAULT)
        if isinstance(observation, Mapping) else _DEFAULT
        for _, observation in observations]


class MockLikelihood(Rule):
    """
    The option likelihood of mock data is no more (2026-11-12): the fits
    of mock data compare them with the models under the likelihoods of
    their setup (see ObservationLikelihoods), which a setup of mock data
    can give before it has data. A setup without data takes it as its
    likelihoods (one for all, or by observation name; null, that of the
    observation, the Gaussian without data); one with data drops it, and
    if it differs from the likelihoods of the fit, the change is to make
    by hand (e.g. a setup of mocks of its own).
    """

    @staticmethod
    def version() -> int:
        return 10

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        observations = _observations(config)
        for path, mock in items(config, 'mock'):
            if 'likelihood' not in mock:
                continue
            option = mock.pop('likelihood')
            path = path + ['likelihood']
            # (one, or one (or null) for each observation)
            if isinstance(option, list):
                if len(option) != len(observations):
                    changes.append(Change(path, (
                        f"by hand: dropped {_shown(option)}, a list of "
                        f"{len(option)} likelihoods for "
                        f"{len(observations)} observations; give the "
                        f"likelihoods of the fits in the section "
                        f"likelihoods")))
                    continue
                per = dict(enumerate(option))
            else:
                per = dict.fromkeys(range(len(observations)), option)
            data = [_has_data(observation) for _, observation in observations]
            if not any(data) and None not in data \
                    and 'likelihoods' not in config:
                changes += self._as_likelihoods(config, per, path)
                continue
            # (the likelihood of the fit of each mock that it gave: its own,
            # or else that of the data of the observation, which the
            # setup now gives, or the default without data)
            fitted = _fitted(config, len(observations))
            mocked = {
                i: value if value is not None
                else fitted[i] if data[i] else _DEFAULT
                for i, value in per.items()}
            if all(value == fitted[i] for i, value in mocked.items()):
                changes.append(Change(path, (
                    "dropped: the fits of mock data take the likelihoods of "
                    "the setup, which are the same")))
            else:
                changes.append(Change(path, (
                    f"by hand: dropped {_shown(option)}, which differs from "
                    f"the likelihoods of the setup, which the fits of mock "
                    f"data now take (give those of the mocks in a setup of "
                    f"their own)")))
        return changes

    @staticmethod
    def _as_likelihoods(
            config: MutableMapping[str, Any],
            per: Mapping[int, Any],
            path: list[str | int]
    ) -> list[Change]:
        """
        Make the likelihoods of the mock data of each observation (by
        index; null, the default) the section likelihoods of a setup
        without data, and return the changes.
        """
        given = {i: value for i, value in per.items() if value is not None}
        if not given:
            return [Change(path, "dropped: the default (Gaussian)")]
        first = next(iter(given.values()))
        if len(given) == len(per) and all(
                value == first for value in given.values()):
            section, named = first, []
        else:
            names, named = _names(
                config, list(per), 'likelihood', ['likelihoods'])
            section = _by_name(given, names)
        _insert_after(config, 'observations', 'likelihoods', section)
        return named + [Change(path, (
            "-> section likelihoods (of the fits of the mock data, which "
            "the setup prepares)"))]


def _keyed_list(
        config: MutableMapping[str, Any],
        options: MutableMapping[str, Any],
        key: str,
        path: list[str | int],
        indices: Sequence[int],
        what: str
) -> list[Change]:
    """
    Rewrite an option given for each observation as a list (key, of
    options, e.g. the noise of mock data), by position, as one value for
    all if they are all the same, or else by observation name (of the
    observations of the given indices, those the items are of), in place,
    and return the changes.
    """
    value = options.get(key)
    if not isinstance(value, list):
        return []
    if len(value) != len(indices):
        return [Change(path + [key], (
            f"by hand: a list of {len(value)} for {len(indices)} "
            f"observations; give one for all, or one for each by "
            f"observation name"))]
    rewritten, by_name, named = _one_or_by_name(
        config, dict(zip(indices, value)), what, path + [key])
    options[key] = rewritten
    return named + [Change(path + [key], (
        "by observation name, not by position" if by_name
        else "one for every observation"))]


class MockNoisesByName(Rule):
    """
    The noises of mock data of each observation are given by observation
    name (2026-11-12; were a list, by position), unless one is for all: a
    list of the same noises is that noise. An observation without a name
    is named by its label (see ObservationLikelihoods).
    """

    @staticmethod
    def version() -> int:
        return 10

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        indices = range(len(_observations(config)))
        for path, mock in items(config, 'mock'):
            changes += _keyed_list(
                config, mock, 'noise', path, indices, 'noise of mock data')
        return changes


class BootstrapSchemesByName(Rule):
    """
    The schemes of a bootstrap of each observation are given by
    observation name (2026-11-12; were a list, by position, of every
    observation, which had data), unless one is for all: a list of the
    same schemes is that scheme. An observation without a name is named
    by its label (see ObservationLikelihoods).
    """

    @staticmethod
    def version() -> int:
        return 10

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        indices = range(len(_observations(config)))
        for path, bootstrap in items(config, 'bootstrap'):
            changes += _keyed_list(
                config, bootstrap, 'scheme', path, indices,
                'scheme of a bootstrap')
        return changes


# The rules, in the order of the changes: MockLikelihood reads the section
# likelihoods that ObservationLikelihoods writes
RULES: tuple[Rule, ...] = (
    ObservationLikelihoods(),
    MockLikelihood(),
    MockNoisesByName(),
    BootstrapSchemesByName(),
)
