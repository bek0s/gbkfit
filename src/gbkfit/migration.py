"""
Migration of configurations to the current format.

Configurations carry the version of their format (see
gbkfit.project.FORMAT_VERSION): configuration files their section
'version', and the pieces and setups of a project its marker. The format
changes by rules (see Rule, RULES), each of which rewrites the old form
of a configuration of an older version in the form of its own; migrate
rewrites a configuration, migrate_file a file (keeping its comments), and
migrate_project the pieces and setups of a project (see gbkfit.project).
Configurations of the current version are left as they are.
"""

import abc
import json
import os
from collections.abc import Iterator, Mapping, MutableMapping
from pathlib import Path
from typing import Any, NamedTuple

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq

import gbkfit.project
from gbkfit.project import FORMAT_VERSION
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'Change',
    'RULES',
    'Rule',
    'migrate',
    'migrate_file',
    'migrate_project'
]


# The sections whose items are pieces of projects
_SECTIONS = ('galaxies', 'observations', 'params', 'fitter')

# (round trip, keeping comments; a flow collection is rewritten on one
# line, as ruamel breaks lines between keys and their values, and null
# stays null)
_yaml = ruamel.yaml.YAML()
_yaml.width = 4096
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
    """
    A change of the format of configurations: of those of the version
    before its own (see version) to its own.
    """

    @staticmethod
    @abc.abstractmethod
    def version() -> int:
        """Return the version of the format that it rewrites to."""

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

    @staticmethod
    def version() -> int:
        return 2

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

    @staticmethod
    def version() -> int:
        return 2

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
    tilted, their own warps).
    """

    @staticmethod
    def version() -> int:
        return 2

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

    @staticmethod
    def version() -> int:
        return 2

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in _items(config, 'observations'):
            for old, new in self._NEW.items():
                if old in item:
                    _rename(item, old, new)
                    changes.append(Change(path, f"{old} -> {new}"))
        return changes


class _TypedKinematics(Rule):
    """
    The kinematics of every component are typed (2026-10-28): those of
    disks and cones, by traits, are of the type 'traits', as those of
    spheroids were, and the vertical equilibrium of a disk is one typed
    option of them: equilibrium: {type: <equilibrium>, density:
    <equilibrium_density>, direction: <equilibrium_direction>}.
    """

    _OPTIONS = dict(
        equilibrium_density='density', equilibrium_direction='direction')

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for galaxy_path, galaxy in _items(config, 'galaxies'):
            for path, component in _components(galaxy):
                kinematics = component.get('kinematics')
                if not isinstance(kinematics, MutableMapping) \
                        or 'type' in kinematics:
                    continue
                _insert_first(kinematics, 'type', 'traits')
                message = "type: traits"
                if 'equilibrium' in kinematics:
                    kinematics['equilibrium'] = self._equilibrium(kinematics)
                    message += ", equilibrium options -> equilibrium: {type}"
                changes.append(Change(
                    galaxy_path + path + ['kinematics'], message))
        return changes

    def _equilibrium(
            self, kinematics: MutableMapping[str, Any]
    ) -> dict[str, Any] | None:
        """
        Remove the options of the vertical equilibrium of kinematics, and
        return it in its new form.
        """
        options = {
            new: kinematics.pop(old) for old, new in self._OPTIONS.items()
            if old in kinematics}
        mode = kinematics['equilibrium']
        return None if mode is None else dict(type=mode) | options


class _SpectralAxes(Rule):
    """
    The spectral axis of an observable and its data is one option,
    spectral (2026-10-28): the options spec_* of moments, spectra in
    regions and slitless spectra; the last axis of the grid (size, step,
    rpix, rval), and rest, medium and convention, of spectral cubes and
    slit spectra, and of their data. Its convention no longer needs a
    rest, and one is required without data: an axis without one gets the
    convention it had, 'relativistic'.
    """

    # The observables of spec_* options, and those of a spectral axis that
    # is the last axis of their grid, with the number of their axes on the
    # sky
    _SPEC = ('pixel_moments', 'region_moments', 'region_spectra',
             'slitless_spectra')
    _CUBES = dict(pixel_spectra=2, slit_spectra=1)
    _OPTIONS = ('size', 'step', 'rpix', 'rval', 'convention', 'rest',
                'medium')

    @staticmethod
    def version() -> int:
        return 3

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in _items(config, 'observations'):
            observable = item.get('observable')
            if not isinstance(observable, MutableMapping):
                continue
            kind = observable.get('type')
            data = item.get('data')
            if kind in self._SPEC and self._spec(observable):
                changes.append(Change(
                    path + ['observable'], "spec_* -> spectral"))
            elif kind in self._CUBES and self._cube(
                    observable, self._CUBES[kind], True):
                changes.append(Change(
                    path + ['observable'],
                    "the spectral axis of the grid, rest, medium and "
                    "convention -> spectral"))
            if not isinstance(data, MutableMapping):
                continue
            if kind == 'region_spectra':
                moved = self._cube(data, 0, False)
            elif kind in self._CUBES:
                moved = self._cube(data, self._CUBES[kind], False)
            else:
                moved = False
            if moved:
                changes.append(Change(
                    path + ['data'], "the world coordinates of the spectral "
                    "axis, rest and medium -> spectral"))
        return changes

    def _spec(self, observable: MutableMapping[str, Any]) -> bool:
        """
        Move the options spec_* of an observable into its option spectral;
        return whether it had any.
        """
        spectral = {
            key: observable.pop(f'spec_{key}') for key in self._OPTIONS
            if f'spec_{key}' in observable}
        if not spectral:
            return False
        self._with_convention(spectral)
        observable['spectral'] = _flow(spectral)
        return True

    def _cube(
            self, options: MutableMapping[str, Any], nspatial: int,
            model: bool
    ) -> bool:
        """
        Move the spectral axis of the grid of an observable (model) or of
        its data, the axis after its nspatial axes on the sky, and its
        rest, medium and convention, into its option spectral; return
        whether it had any.
        """
        spectral = {}
        for key in ('size', 'step', 'rpix', 'rval'):
            if key not in options:
                continue
            value = options[key]
            if isinstance(value, list) and len(value) == nspatial + 1:
                spectral[key] = value[-1]
                options[key] = _like(value, value[:-1])
            elif not isinstance(value, list):
                spectral[key] = value
            if not nspatial or options[key] == []:
                del options[key]
        for key in ('convention', 'rest', 'medium'):
            if key in options:
                spectral[key] = options.pop(key)
        if model and 'size' in spectral:
            self._with_convention(spectral)
        if not spectral:
            return False
        options['spectral'] = _flow(spectral)
        return True

    @staticmethod
    def _with_convention(spectral: dict[str, Any]) -> None:
        """
        Give a spectral axis without a rest or a convention the
        convention its velocities had: relativistic.
        """
        if spectral.get('rest') is None \
                and spectral.get('convention') is None:
            spectral['convention'] = 'relativistic'


# The rules, in the order of the changes
RULES: tuple[Rule, ...] = (
    _FluxScales(),
    _DiskTypes(),
    _OwnGeometries(),
    _ObservationRenames(),
    _TypedKinematics(),
    _SpectralAxes(),
)


def migrate(
        config: MutableMapping[str, Any], from_version: int | None = None
) -> list[Change]:
    """
    Rewrite a configuration of an older version of the format in the
    current one, in place (see Rule.rewrite), with its section 'version'
    set to the current version.

    Parameters
    ----------
    config : MutableMapping
        The configuration.
    from_version : int, optional
        The version of its format, if it has no section 'version' (those
        of before version 3 had none): 1, before 2026-10-25; 2, before
        2026-10-28.

    Returns
    -------
    list of Change
        The changes, in order; none if it is of the current version.

    Raises
    ------
    ConfigError
        If its version is not given, is not one of the format, or is newer
        than the current one, or from_version is not its version.
    """
    version = _version(config.get('version'), from_version)
    if version == FORMAT_VERSION:
        return []
    changes = _rewrite(config, version)
    _insert_first(config, 'version', FORMAT_VERSION)
    return changes + [Change(['version'], f"{version} -> {FORMAT_VERSION}")]


def migrate_file(
        path: str, output: str | None = None, from_version: int | None = None
) -> list[Change]:
    """
    Rewrite a configuration file (YAML or JSON) of an older version of the
    format in the current one, keeping the comments of YAML (see
    migrate).

    Parameters
    ----------
    path : str
        The file.
    output : str, optional
        The file to write; by default the file itself, if it changes.
    from_version : int, optional
        The version of its format, if it has no section 'version' (see
        migrate).

    Returns
    -------
    list of Change
        The changes, in order.

    Raises
    ------
    ConfigError
        If the file cannot be read, or its version is not valid (see
        migrate).
    """
    config = _read(path)
    if not isinstance(config, MutableMapping):
        raise ConfigError(f"{path} is not a configuration (a dictionary)")
    changes = migrate(config, from_version)
    if output is not None or changes:
        _write(output or path, config)
    return changes


def migrate_project(path: str) -> dict[str, list[Change]]:
    """
    Rewrite the pieces and the setups of a project of an older version of
    the format in the current one, in place, and mark it of the current
    version (see gbkfit.project.FORMAT_VERSION).

    The pieces are migrated together, as one configuration with all of
    them (so that a rule can change a piece of observations and one of
    parameters, as one); each setup is migrated alone. The marker of the
    project gives their version; they have no section 'version'.

    Parameters
    ----------
    path : str
        The folder of the project.

    Returns
    -------
    dict of str to list of Change
        The changes of each file that changed (by its path in the folder
        of the project), with their paths in the piece or the setup; none
        for a project of the current version.

    Raises
    ------
    ValueError
        If the folder is not a project of a version that this one reads.
    ConfigError
        If a piece or a setup cannot be read.
    """
    project = gbkfit.project.open(path)
    if (version := project.version()) == FORMAT_VERSION:
        return {}
    files = {
        section: [project.piece_file(section, name)
                  for name in project.names(_folder(section))]
        for section in _SECTIONS}
    pieces = {
        section: [_read(file) for file in section_files]
        for section, section_files in files.items()}
    by_piece: dict[tuple[str, int], list[Change]] = {}
    for change in _rewrite(pieces, version):
        section, index, *rest = change.path
        by_piece.setdefault((section, index), []).append(
            Change(rest, change.message))
    changes = {}
    for (section, index), piece_changes in by_piece.items():
        _write(files[section][index], pieces[section][index])
        changes[files[section][index]] = piece_changes
    for name in project.names('setups'):
        file = os.path.join(project.path, 'setups', f'{name}.yaml')
        setup = _read(file)
        if isinstance(setup, MutableMapping) \
                and (setup_changes := _rewrite(setup, version)):
            _write(file, setup)
            changes[file] = setup_changes
    _mark_current(project.path)
    return {
        os.path.relpath(file, project.path): file_changes
        for file, file_changes in changes.items()}


def _version(version: Any, from_version: int | None) -> int:
    """
    Return the version of the format of a configuration: its section
    'version', or else from_version (see migrate).

    Raises
    ------
    ConfigError
        If neither is given, the version is not one of the format or is
        newer than the current one, or they differ.
    """
    if version is None:
        if from_version is None:
            raise ConfigError(
                "the configuration has no version: give that of its format "
                "(1, before 2026-10-25; 2, before 2026-10-28)")
        version = from_version
    elif from_version is not None and from_version != version:
        raise ConfigError(
            f"the configuration is of version {version}; not "
            f"{from_version}", ['version'])
    if isinstance(version, bool) or version not in range(
            1, FORMAT_VERSION + 1):
        raise ConfigError(
            f"the versions of the format are 1 to {FORMAT_VERSION} (that "
            f"of this gbkfit); the configuration is of version "
            f"{version!r}", ['version'])
    return version


def _rewrite(
        config: MutableMapping[str, Any], version: int
) -> list[Change]:
    """
    Rewrite a configuration of a version in the current one, in place, by
    the rules of the versions after it.
    """
    return [
        change for rule in RULES if rule.version() > version
        for change in rule.rewrite(config)]


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


def _insert_first(
        mapping: MutableMapping[str, Any], key: str, value: Any
) -> None:
    """Set a key of a dictionary in place, as its first one."""
    mapping.pop(key, None)
    if isinstance(mapping, CommentedMap):
        mapping.insert(0, key, value)
    else:
        items = list(mapping.items())
        mapping.clear()
        mapping.update([(key, value)] + items)


def _like(sequence: list[Any], items: list[Any]) -> list[Any]:
    """
    Return items as a list of the style of a sequence (of YAML: flow or
    block).
    """
    if not isinstance(sequence, CommentedSeq):
        return items
    result = CommentedSeq(items)
    if sequence.fa.flow_style():
        result.fa.set_flow_style()
    return result


def _flow(mapping: dict[str, Any]) -> CommentedMap:
    """Return a dictionary as one of the flow style of YAML."""
    result = CommentedMap(mapping)
    result.fa.set_flow_style()
    return result


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
    """Mark a project of the current version of the format."""
    marker = Path(path, gbkfit.project.MARKER)
    with marker.open('w') as f:
        ruamel.yaml.YAML(typ='safe').dump(dict(
            format=gbkfit.project.FORMAT,
            version=gbkfit.project.FORMAT_VERSION), f)
