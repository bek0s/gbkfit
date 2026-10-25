"""
Migration of configurations to the current format.

The format changes by rules (see Rule, RULES), each of which finds its
old form in a configuration and rewrites it in the new form. The
loaders of configurations refuse the old forms, with an error that
names the new form (see refuse_old_forms); migrate rewrites them in a
configuration, migrate_file in a file (keeping its comments), and
migrate_project in the pieces and setups of a project (see
gbkfit.project).
"""

import abc
import json
import os
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any, NamedTuple

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap

import gbkfit.project
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Change',
    'RULES',
    'Rule',
    'migrate',
    'migrate_file',
    'migrate_project',
    'refuse_old_forms'
]


# The sections whose items are pieces of projects
_SECTIONS = ('galaxies', 'observations', 'params', 'fitter')

# (round trip, keeping comments; flow collections are rewritten within
# the width, and null stays null)
_yaml = ruamel.yaml.YAML()
_yaml.representer.add_representer(
    type(None),
    lambda representer, _: representer.represent_scalar(
        'tag:yaml.org,2002:null', 'null'))


class Change(NamedTuple):
    """
    A change of a configuration by a migration: its path in the
    configuration (option names and list indices), and what changed, or
    what to change by hand where the migration cannot.
    """
    path: list[str | int]
    message: str


class Rule(abc.ABC):
    """A change of the format of configurations."""

    @abc.abstractmethod
    def find(
            self, section: str, item: Mapping[str, Any]
    ) -> ConfigError | None:
        """
        Find the old form in an item of a section of a configuration.

        Parameters
        ----------
        section : str
            The section (e.g. 'galaxies').
        item : Mapping
            The item: one of a list of them (e.g. a galaxy), or the
            section itself (e.g. 'params').

        Returns
        -------
        ConfigError or None
            An error that names the new form, with its path in the item;
            None if the item has no old form.
        """

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


class _FluxScales(Rule):
    """
    Flux scales replaced the calibration of observations and the
    amplitudes of spectra of their likelihoods (2026-10-25): calibration
    is flux_scale 'one', whose parameter is '{name}_flux_scale' (was
    '{name}_calibration'), and amplitudes are a solved flux scale
    'per_spectrum' ('per_row' of slitless spectra).
    """

    def find(
            self, section: str, item: Mapping[str, Any]
    ) -> ConfigError | None:
        if section != 'observations':
            return None
        if 'calibration' in item:
            return ConfigError(
                "option 'calibration' is now flux_scale: 'one', whose "
                "parameter is '{name}_flux_scale' (was "
                "'{name}_calibration')", ['calibration'])
        likelihood = item.get('likelihood')
        if isinstance(likelihood, Mapping) and (
                {'amplitudes', 'amplitude_sigma'} & set(likelihood)):
            sigma = likelihood.get('amplitude_sigma')
            return ConfigError(
                f"the amplitudes of the likelihood are now a solved flux "
                f"scale of the observation: its option flux_scale: "
                f"per_spectrum (or per_row, of slitless spectra), and in "
                f"the parameters '{{name}}_flux_scale': {{solve: {{type: "
                f"gaussian, mean: 1, sigma: {sigma}, minimum: 0}}}}",
                ['likelihood'])
        return None

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in _items(config, 'observations'):
            if 'calibration' in item:
                changes += self._calibration(config, path, item)
            likelihood = item.get('likelihood')
            if isinstance(likelihood, MutableMapping) and (
                    {'amplitudes', 'amplitude_sigma'} & set(likelihood)):
                changes += self._amplitudes(config, path, item, likelihood)
        return changes

    @staticmethod
    def _calibration(
            config: MutableMapping[str, Any],
            path: list[str | int],
            item: MutableMapping[str, Any]
    ) -> list[Change]:
        """Rewrite the calibration of an observation as its flux scale."""
        if not item['calibration']:
            del item['calibration']
            return [Change(path, "calibration: false removed")]
        _rename(item, 'calibration', 'flux_scale')
        item['flux_scale'] = 'one'
        changes = [Change(path, "calibration -> flux_scale: one")]
        if (name := item.get('name')) is None:
            return changes + [Change(
                path, "by hand: name the observation (its flux scale is "
                      "the parameter '{name}_flux_scale')")]
        old, new = f'{name}_calibration', f'{name}_flux_scale'
        for params_path, params in _items(config, 'params'):
            if old in params:
                _rename(params, old, new)
                changes.append(Change(params_path, f"{old} -> {new}"))
        if len(changes) == 1:
            changes.append(Change(
                path, f"by hand: rename the parameter {old} to {new}"))
        return changes

    @staticmethod
    def _amplitudes(
            config: MutableMapping[str, Any],
            path: list[str | int],
            item: MutableMapping[str, Any],
            likelihood: MutableMapping[str, Any]
    ) -> list[Change]:
        """
        Rewrite the amplitudes of the likelihood of an observation as its
        solved flux scale.
        """
        amplitudes = likelihood.pop('amplitudes', None)
        sigma = likelihood.pop('amplitude_sigma', None)
        if amplitudes is None:
            return [Change(
                path + ['likelihood'], "amplitude_sigma removed (no "
                "amplitudes)")]
        observable = item.get('observable')
        slitless = isinstance(observable, Mapping) and \
            observable.get('type') == 'slitless_spectra'
        item['flux_scale'] = 'per_row' if slitless else 'per_spectrum'
        changes = [Change(
            path, f"likelihood amplitudes -> flux_scale: "
                  f"{item['flux_scale']}")]
        name = item.get('name')
        param = f'{name}_flux_scale'
        params = list(_items(config, 'params'))
        if sigma is None or name is None or len(params) != 1:
            changes.append(Change(
                path, f"by hand: add to the parameters {param}: {{solve: "
                      f"{{type: gaussian, mean: 1, sigma: {sigma}, "
                      f"minimum: 0}}}}"))
            return changes
        params_path, params = params[0]
        params[param] = dict(solve=dict(
            type='gaussian', mean=1, sigma=sigma, minimum=0))
        changes.append(Change(params_path, f"{param} added (solved)"))
        return changes


class _DiskTypes(Rule):
    """
    One disk replaced the smooth disk and the disk of clouds, and the cone
    the smooth cone (2026-10-25): the component types 'smdisk' and
    'smcone' are 'disk' and 'cone', and 'mcdisk' is 'disk' with its cflux
    and seed in its option clouds.
    """

    _NEW = dict(smdisk='disk', mcdisk='disk', smcone='cone')

    def find(
            self, section: str, item: Mapping[str, Any]
    ) -> ConfigError | None:
        if section != 'galaxies':
            return None
        for path, component in _components(item):
            if (old := component.get('type')) not in self._NEW:
                continue
            new = f"type '{self._NEW[old]}'"
            if old == 'mcdisk':
                clouds = dict(cflux=component.get('cflux'))
                if 'seed' in component:
                    clouds['seed'] = component['seed']
                new += f" with clouds: {clouds}"
            return ConfigError(
                f"type '{old}' is now {new}", path + ['type'])
        return None

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in _items(config, 'galaxies'):
            for path, component in _components(galaxy):
                if (old := component.get('type')) not in self._NEW:
                    continue
                component['type'] = self._NEW[old]
                message = f"type {old} -> {self._NEW[old]}"
                if old == 'mcdisk' and 'cflux' in component:
                    clouds = dict(cflux=component['cflux'])
                    if 'seed' in component:
                        clouds['seed'] = component.pop('seed')
                    _rename(component, 'cflux', 'clouds')
                    component['clouds'] = clouds
                    message += ", cflux and seed -> clouds"
                changes.append(Change(galaxy_path + path, message))
        return changes


class _OwnGeometries(Rule):
    """
    The components of a galaxy share its geometry by default (2026-10-25),
    whose parameters are named plainly (e.g. posa); they had their own
    (e.g. disk_posa). The components that would change get the option
    geometry 'own', which keeps the model and the names of its
    parameters: those of galaxies of several components, those with
    names (which prefix their parameters) and the warped ones (loose or
    tilted, their own warps). There is no old form to find: the
    configuration of the old default is valid (its parameters are then
    unknown, and named so).
    """

    def find(
            self, section: str, item: Mapping[str, Any]
    ) -> ConfigError | None:
        return None

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in _items(config, 'galaxies'):
            components = list(_components(galaxy))
            for path, component in components:
                if 'geometry' in component:
                    continue
                if len(components) > 1 or 'name' in component \
                        or component.get('loose') or component.get('tilted'):
                    component['geometry'] = 'own'
                    changes.append(Change(
                        galaxy_path + path, "geometry: own (its own, as "
                        "before; the default is now that of its galaxy)"))
        return changes


class _ObservationRenames(Rule):
    """
    Two options of observations were renamed (2026-10-25): scale is
    oversampling (an observation has a flux scale too), and weights is
    pixel_weights (its likelihood has weights too).
    """

    _NEW = dict(scale='oversampling', weights='pixel_weights')

    def find(
            self, section: str, item: Mapping[str, Any]
    ) -> ConfigError | None:
        if section != 'observations':
            return None
        for old, new in self._NEW.items():
            if old in item:
                return ConfigError(f"option '{old}' is now {new}", [old])
        return None

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in _items(config, 'observations'):
            for old, new in self._NEW.items():
                if old in item:
                    _rename(item, old, new)
                    changes.append(Change(path, f"{old} -> {new}"))
        return changes


# The rules, in the order of the changes
RULES: tuple[Rule, ...] = (
    _FluxScales(),
    _DiskTypes(),
    _OwnGeometries(),
    _ObservationRenames(),
)


def refuse_old_forms(section: str, item: Any) -> None:
    """
    Refuse the old forms in an item of a section of a configuration (see
    Rule.find).

    Parameters
    ----------
    section : str
        The section (e.g. 'galaxies').
    item : Any
        The item: one of a list of them (e.g. a galaxy), or the section
        itself (e.g. 'params'). Items that are not dictionaries have no
        old forms.

    Raises
    ------
    ConfigError
        If the item has an old form: an error that names the new form.
    """
    if not isinstance(item, Mapping):
        return
    for rule in RULES:
        if (error := rule.find(section, item)) is not None:
            raise ConfigError(
                f"{error.message} (gbkfit-cli migrate rewrites it)",
                error.path, error.context)


def migrate(config: MutableMapping[str, Any]) -> list[Change]:
    """
    Rewrite the old forms of a configuration in the current format, in
    place (see Rule.rewrite).

    Parameters
    ----------
    config : MutableMapping
        The configuration, or some of its sections.

    Returns
    -------
    list of Change
        The changes, in order.
    """
    return [change for rule in RULES for change in rule.rewrite(config)]


def migrate_file(path: str, output: str | None = None) -> list[Change]:
    """
    Rewrite the old forms of a configuration file (YAML or JSON) in the
    current format, keeping the comments of YAML (see migrate).

    Parameters
    ----------
    path : str
        The file.
    output : str, optional
        The file to write; by default the file itself, if it changes.

    Returns
    -------
    list of Change
        The changes, in order.

    Raises
    ------
    ConfigError
        If the file cannot be read.
    """
    config = _read(path)
    changes = migrate(config) if isinstance(config, MutableMapping) else []
    if output is not None or changes:
        _write(output or path, config)
    return changes


def migrate_project(path: str) -> dict[str, list[Change]]:
    """
    Rewrite the old forms of the pieces and the setups of a project in
    the current format, in place, and mark it of the current version of
    the format of projects (see gbkfit.project.FORMAT_VERSION).

    The pieces are migrated together, as one configuration with all of
    them (so that a rule can change a piece of observations and one of
    parameters, as one); each setup is migrated alone.

    Parameters
    ----------
    path : str
        The folder of the project.

    Returns
    -------
    dict of str to list of Change
        The changes of each file that changed (by its path in the folder
        of the project), with their paths in the piece or the setup.

    Raises
    ------
    ValueError
        If the folder is not a project of a version that this one reads.
    ConfigError
        If a piece or a setup cannot be read.
    """
    project = gbkfit.project.open(path)
    files = {
        section: [project.piece_file(section, name)
                  for name in project.names(_folder(section))]
        for section in _SECTIONS}
    pieces = {
        section: [_read(file) for file in section_files]
        for section, section_files in files.items()}
    by_piece: dict[tuple[str, int], list[Change]] = {}
    for change in migrate(pieces):
        section, index, *rest = change.path
        by_piece.setdefault((section, index), []).append(
            Change(rest, change.message))
    changes = {}
    for (section, index), piece_changes in by_piece.items():
        _write(files[section][index], pieces[section][index])
        changes[files[section][index]] = piece_changes
    for name in project.names('setups'):
        file = os.path.join(project.path, 'setups', f'{name}.yaml')
        if setup_changes := migrate_file(file):
            changes[file] = setup_changes
    _mark_current(project.path)
    return {
        os.path.relpath(file, project.path): file_changes
        for file, file_changes in changes.items()}


def _items(
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


def _components(
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


def _rename(mapping: MutableMapping[str, Any], old: str, new: str) -> None:
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
        items = list(mapping.items())
        items.insert(position, (new, value))
        mapping.clear()
        mapping.update(items)


def _folder(section: str) -> str:
    """Return the folder of the pieces of a section of a project."""
    return 'fitters' if section == 'fitter' else section


def _read(path: str) -> Any:
    """
    Read a configuration file (YAML or JSON), with its comments; raise
    ConfigError if it cannot be read.
    """
    try:
        with open(path) as f:
            return _yaml.load(f)
    except (OSError, ruamel.yaml.YAMLError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e


def _write(path: str, config: Any) -> None:
    """Write a configuration file: JSON if it ends with .json, else YAML."""
    with open(path, 'w') as f:
        if path.endswith('.json'):
            json.dump(config, f, indent=2)
            f.write('\n')
        else:
            _yaml.dump(config, f)


def _mark_current(path: str) -> None:
    """Mark a project of the current version of the format of projects."""
    marker = Path(path, gbkfit.project.MARKER)
    with marker.open('w') as f:
        ruamel.yaml.YAML(typ='safe').dump(dict(
            format=gbkfit.project.FORMAT,
            version=gbkfit.project.FORMAT_VERSION), f)
