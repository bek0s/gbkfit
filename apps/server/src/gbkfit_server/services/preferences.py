"""
The preferences of the user of the pages (e.g. the theme of the Studio),
in their folder of GBKFIT, the same whatever the folder served or the
port: unlike the storage of a browser, which is that of an address, and
so of a port.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import Invalid


__all__ = [
    'Preferences',
    'user_folder'
]


# The file of the preferences, in the folder of the user
_FILE = 'preferences.json'


def user_folder() -> Path:
    """
    Return the folder of GBKFIT of the user: ~/.gbkfit on every system
    (as ~/.jupyter or ~/.astropy are; on Windows, in the folder of the
    user, %USERPROFILE%), or that of $GBKFIT_HOME.
    """
    return Path(os.environ.get('GBKFIT_HOME') or Path.home() / '.gbkfit')


class Preferences:
    """
    The preferences of the user: the values of their names (JSON), in a
    file of JSON (made as they are first set).

    Parameters
    ----------
    folder : str or Path, optional
        The folder of the file; by default that of the user (see
        user_folder).

    Attributes
    ----------
    file : Path
        The file.
    """

    def __init__(self, folder: str | Path | None = None):
        self.file = Path(folder if folder is not None else user_folder()) \
            / _FILE

    def read(self) -> dict[str, Any]:
        """
        Return the preferences (none if the file is not there).

        Raises
        ------
        Invalid
            If the file is not an object of JSON (e.g. edited by hand):
            it is kept, for the user to mend or delete.
        """
        try:
            text = self.file.read_text(encoding='utf-8')
        except FileNotFoundError:
            return {}
        try:
            values = json.loads(text)
        except json.JSONDecodeError as e:
            raise Invalid(
                f"the preferences in {self.file} are not valid JSON ({e}); "
                f"mend or delete the file") from None
        if not isinstance(values, dict):
            raise Invalid(
                f"the preferences in {self.file} are not an object of JSON; "
                f"mend or delete the file")
        return values

    def update(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        """
        Set preferences (a value of None removes its name), keeping the
        others; return them all.

        The file is written whole, then moved over the old one, so that
        it is never left half written.
        """
        values = self.read()
        for name, value in changes.items():
            if value is None:
                values.pop(name, None)
            else:
                values[name] = value
        self.file.parent.mkdir(parents=True, exist_ok=True)
        written = self.file.with_name(f'{_FILE}.new')
        written.write_text(
            json.dumps(values, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8')
        os.replace(written, self.file)
        return values
