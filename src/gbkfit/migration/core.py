"""
The migration of configurations: the versions of their format, the
chain of the rules of the versions after theirs, and the rewriting of
files and of projects.
"""

import contextlib
import copy
import json
import os
import shutil
import tempfile
from collections.abc import Generator, Mapping, MutableMapping
from typing import Any, TextIO

import ruamel.yaml

from gbkfit.config import FORMAT_VERSION, ConfigError
import gbkfit.project
from . import v2, v3, v4, v5, v6, v7, v8, v9
from .rules import Change, PiecesView, Rule, insert_first


__all__ = [
    'RULES',
    'migrate',
    'migrate_file',
    'migrate_project'
]


# The sections and options of configurations whose items are pieces of
# projects (see gbkfit.project.piece_folder)
_SECTIONS = (
    'galaxies', 'observations', 'data', 'psf', 'lsf', 'noise', 'params',
    'fitter')

# The sections of setups whose pieces the rules see with them, as they
# were (rules that rename parameters take the galaxies and the
# observations that the parameters are of: see _setup_view)
_VIEWED = ('galaxies', 'observations')

# The umask of the process, for the files it makes (see _replacing):
# there is no way to read it but to set it, which is done once here
_UMASK = os.umask(0)
os.umask(_UMASK)


class _Representer(ruamel.yaml.representer.RoundTripRepresenter):
    """The representer of round trips, which writes null as null."""


_Representer.add_representer(
    type(None),
    lambda representer, _: representer.represent_scalar(
        'tag:yaml.org,2002:null', 'null'))


def _yaml() -> ruamel.yaml.YAML:
    """
    Return a new YAML of round trips, keeping comments: a flow collection
    is rewritten on one line, as ruamel breaks lines between keys and
    their values, and null stays null.
    """
    yaml = ruamel.yaml.YAML()
    yaml.Representer = _Representer
    yaml.width = 4096
    return yaml


# The rules of all the versions, in order (see Rule)
RULES: tuple[Rule, ...] = (
    *v2.RULES, *v3.RULES, *v4.RULES, *v5.RULES, *v6.RULES, *v7.RULES,
    *v8.RULES, *v9.RULES)


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
    insert_first(config, 'version', FORMAT_VERSION)
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
    OSError
        If the file cannot be read.
    ConfigError
        If it is not YAML, or its version is not valid (see migrate).
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
    version (see gbkfit.config.FORMAT_VERSION).

    The pieces are migrated together, as one configuration with all of
    them (so that a rule can change a piece of observations and one of
    parameters, as one; see PiecesView); each setup is migrated alone,
    with copies of the galaxy and observation pieces it names. The marker
    of the project gives their version; they have no section 'version'.
    Nothing is written until all are rewritten: a rule that refuses a
    setup leaves the project as it was.

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
    OSError
        If a piece or a setup cannot be read.
    ConfigError
        If a piece or a setup is not YAML.
    """
    project = gbkfit.project.open(path)
    if (version := project.version()) == FORMAT_VERSION:
        return {}
    files = {
        section: [
            project.piece_file(section, name)
            for name in project.names(gbkfit.project.piece_folder(section))]
        for section in _SECTIONS}
    pieces = {
        section: [_read(file) for file in section_files]
        for section, section_files in files.items()}
    setup_files = [
        os.path.join(project.path, 'setups', f'{name}.yaml')
        for name in project.names('setups')]
    setups = [_read(file) for file in setup_files]
    # The galaxy and observation pieces as they were, by name, for the
    # setups that name them; and the galaxies and observations inline in
    # the setups, whose parameters the params pieces can have (rules that
    # rename parameters take the galaxies and the observations they are
    # with: see _setup_view)
    originals = {
        section: dict(zip(
            project.names(gbkfit.project.piece_folder(section)),
            copy.deepcopy(pieces[section])))
        for section in _VIEWED}
    inline = {
        section: [
            copy.deepcopy(item) for setup in setups
            for item in _listed(setup, section) if isinstance(item, Mapping)]
        for section in _VIEWED}
    by_piece: dict[tuple[str, int], list[Change]] = {}
    view = PiecesView(pieces, **{
        section: pieces[section] + inline[section] for section in _VIEWED})
    for change in _rewrite(view, version):
        section, index, *rest = change.path
        # (not the copies of the galaxies and observations of the setups)
        if index < len(files[section]):
            by_piece.setdefault((section, index), []).append(
                Change(rest, change.message))
    changes = {
        files[section][index]: piece_changes
        for (section, index), piece_changes in by_piece.items()}
    rewritten = {
        files[section][index]: pieces[section][index]
        for section, index in by_piece}
    for file, setup in zip(setup_files, setups):
        if not isinstance(setup, MutableMapping):
            continue
        view, named = _setup_view(setup, originals)
        setup_changes = [
            change for change in _rewrite(view, version)
            if not (len(change.path) > 1 and change.path[0] in named
                    and change.path[1] in named[change.path[0]])]
        if view is not setup:
            _write_back(setup, view, named)
        if setup_changes:
            rewritten[file] = setup
            changes[file] = setup_changes
    # (written only once all are rewritten: a rule that refuses one
    # leaves the project as it was)
    for file, config in rewritten.items():
        _write(file, config)
    _mark_current(project.path)
    return {
        os.path.relpath(file, project.path): file_changes
        for file, file_changes in changes.items()}


def _listed(config: Any, section: str) -> list[Any]:
    """Return the items of a section of a configuration, as a list."""
    if not isinstance(config, Mapping):
        return []
    value = config.get(section)
    return value if isinstance(value, list) else [] if value is None \
        else [value]


def _setup_view(
        setup: MutableMapping[str, Any],
        originals: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, set[int]]]:
    """
    Return a view of a setup with its galaxies and observations that name
    pieces replaced by copies of the pieces as they were (originals, by
    section and name), so that the rules that rename parameters know them
    (the rest is the setup itself, rewritten in place), and the positions
    of those items, by section (whose changes are those of the pieces,
    not of the setup).
    """
    named = {
        section: {
            i for i, item in enumerate(_listed(setup, section))
            if isinstance(item, str) and item in originals[section]}
        for section in _VIEWED}
    named = {section: found for section, found in named.items() if found}
    if not named:
        return setup, {}
    view = dict(setup)
    for section, found in named.items():
        view[section] = [
            copy.deepcopy(originals[section][item]) if i in found else item
            for i, item in enumerate(_listed(setup, section))]
    return view, named


def _write_back(
        setup: MutableMapping[str, Any],
        view: Mapping[str, Any],
        named: Mapping[str, set[int]]
) -> None:
    """
    Take into a setup, in place, what the rules rewrote in its view (see
    _setup_view): its options, and the items of its galaxies and
    observations but those that name pieces.
    """
    for key in [k for k in setup if k not in view]:
        del setup[key]
    for key, value in view.items():
        if key not in named:
            setup[key] = value
    for section, found in named.items():
        if isinstance(listed := setup[section], list):
            for i, item in enumerate(view[section]):
                if i not in found:
                    listed[i] = item


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


def _read(path: str) -> Any:
    """
    Read a configuration file (YAML or JSON), with its comments; raise
    ConfigError if it is neither (and its OSError if it cannot be read).
    """
    with open(path) as f:
        try:
            return _yaml().load(f)
        except ruamel.yaml.YAMLError as e:
            raise ConfigError(f"{path} is not YAML: {e}") from e


def _write(path: str, config: Any) -> None:
    """Write a configuration file: JSON if it ends with .json, else YAML."""
    with _replacing(path) as f:
        if path.endswith('.json'):
            json.dump(config, f, indent=2)
            f.write('\n')
        else:
            _yaml().dump(config, f)


def _mark_current(path: str) -> None:
    """Mark a project of the current version of the format."""
    with _replacing(os.path.join(path, gbkfit.project.MARKER)) as f:
        ruamel.yaml.YAML(typ='safe').dump(dict(
            format=gbkfit.project.FORMAT,
            version=FORMAT_VERSION), f)


@contextlib.contextmanager
def _replacing(path: str) -> Generator[TextIO]:
    """
    Write a file whole or not at all: into a new file next to it, which
    replaces it once written, so that an error or a crash meanwhile
    leaves the file as it was (a migration rewrites the files of users).
    """
    folder, name = os.path.split(os.path.abspath(path))
    fd, temp = tempfile.mkstemp(prefix=f'.{name}.', dir=folder)
    try:
        with os.fdopen(fd, 'w') as f:
            yield f
        if os.path.exists(path):
            shutil.copymode(path, temp)
        else:
            os.chmod(temp, 0o666 & ~_UMASK)
        os.replace(temp, path)
    except BaseException:
        os.unlink(temp)
        raise
