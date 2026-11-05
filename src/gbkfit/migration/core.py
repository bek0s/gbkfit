"""
The migration of configurations: the versions of their format, the
chain of the rules of the versions after theirs, and the rewriting of
files and of projects.
"""

import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Generator, MutableMapping
from typing import Any, TextIO

import ruamel.yaml

from gbkfit.config import ConfigError
import gbkfit.project
from gbkfit.project import FORMAT_VERSION
from . import v2, v3, v4, v5
from .rules import Change, Rule, insert_first


__all__ = [
    'RULES',
    'migrate',
    'migrate_file',
    'migrate_project'
]


# The sections whose items are pieces of projects
_SECTIONS = ('galaxies', 'observations', 'params', 'fitter')

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
    *v2.RULES, *v3.RULES, *v4.RULES, *v5.RULES)


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
    OSError
        If a piece or a setup cannot be read.
    ConfigError
        If a piece or a setup is not YAML.
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


def _folder(section: str) -> str:
    """Return the folder of the pieces of a section of a project."""
    return 'fitters' if section == 'fitter' else section


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
            version=gbkfit.project.FORMAT_VERSION), f)


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
