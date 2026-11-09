"""
Migration of configurations to the current format.

Configurations carry the version of their format (see
gbkfit.config.FORMAT_VERSION): configuration files their section
'version', and the pieces and setups of a project its marker. The format
changes by rules (see Rule, RULES), each of which rewrites the old form
of a configuration of an older version in the form of its own; migrate
rewrites a configuration, migrate_file a file (keeping its comments), and
migrate_project the pieces and setups of a project (see gbkfit.project).
Configurations of the current version are left as they are.

The rules of each version are in a module of their own (e.g. v2, those
that rewrite configurations of version 1), and the machinery that
chains them is in core.
"""

from .core import *
from .rules import Change, Rule


__all__ = [
    'RULES',
    'Change',
    'Rule',
    'migrate',
    'migrate_file',
    'migrate_project'
]
