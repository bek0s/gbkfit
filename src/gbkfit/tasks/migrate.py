"""
The migrate task: rewrite a configuration file, or the pieces and setups
of a project, in the current format of configurations (see
gbkfit.migration), and log the changes.
"""

import logging
import os

import gbkfit.migration
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'migrate'
]


_log = logging.getLogger(__name__)


def migrate(
        path: str,
        output: str | None = None,
        from_version: int | None = None
) -> None:
    """
    Rewrite a configuration file, or the pieces and setups of a project,
    of an older version of the format in the current one, and log the
    changes.

    Parameters
    ----------
    path : str
        The configuration file (YAML or JSON), or the folder of a project
        (rewritten in place).
    output : str, optional
        The file to write a configuration file to; by default the file
        itself, if it changes. Not for a project.
    from_version : int, optional
        The version of the format of a configuration file without the
        section 'version' (see gbkfit.migration.migrate). Not for a
        project, whose marker gives it.

    Raises
    ------
    ConfigError
        If a file cannot be read, the version of a file is not valid, the
        folder is not a project of a version that this one reads, or an
        output or a version is given for a project.
    """
    if os.path.isdir(path):
        if output is not None or from_version is not None:
            raise ConfigError(
                "a project is migrated in place, from the version of its "
                "marker; it takes no output and no from_version")
        try:
            changes = gbkfit.migration.migrate_project(path)
        except ValueError as e:
            raise ConfigError(str(e)) from e
    else:
        changes = {path: gbkfit.migration.migrate_file(
            path, output, from_version)}
    for file, file_changes in changes.items():
        for change in file_changes:
            where = '.'.join(map(str, change.path))
            _log.info(f"{file}: {where}: {change.message}" if where
                      else f"{file}: {change.message}")
    if not any(changes.values()):
        _log.info(f"{path}: nothing to migrate")
