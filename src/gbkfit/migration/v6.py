"""
The rules of version 6 of the format (2026-11-08 on), which rewrite
configurations of version 5.
"""

from collections.abc import MutableMapping
from typing import Any

from .rules import Change, Rule, flow, items


__all__ = [
    'RULES',
    'TableFiles'
]


class TableFiles(Rule):
    """
    The file of a table of an option that varies along the spectral axis
    (see SpectralTable.from_file) is given as other files are
    (2026-11-08): the HDU of a FITS file is in it, {file, hdu}, not an
    option of the table.
    """

    @staticmethod
    def version() -> int:
        return 6

    def rewrite(self, config: MutableMapping[str, Any]) -> list[Change]:
        changes = []
        for path, item in items(config, 'observations'):
            changes += self._walk(item, path)
        return changes

    def _walk(self, value: Any, path: list[str | int]) -> list[Change]:
        """Rewrite the tables of a value and of its items, in place."""
        changes = []
        if isinstance(value, MutableMapping):
            if {'file', 'points', 'column', 'hdu'} <= value.keys():
                hdu = value.pop('hdu')
                if hdu is not None:
                    value['file'] = flow(dict(file=value['file'], hdu=hdu))
                changes.append(Change(path, "file, hdu -> file {file, hdu}"))
            for key, item in value.items():
                changes += self._walk(item, path + [key])
        elif isinstance(value, list):
            for i, item in enumerate(value):
                changes += self._walk(item, path + [i])
        return changes


# The rules, in the order of the changes
RULES: tuple[Rule, ...] = (
    TableFiles(),
)
