"""
The pages that the server serves (e.g. the Studio): those installed,
which declare themselves by entry points of the group gbkfit_server.pages
(their name, and a function that returns the folder of their built page).
"""

import logging
from importlib.metadata import entry_points


__all__ = [
    'installed_pages'
]


_log = logging.getLogger(__name__)


def installed_pages() -> dict[str, str]:
    """
    Return the folders of the installed pages, by name; a page whose
    folder cannot be found is left out, with a warning.
    """
    pages = {}
    for entry in sorted(
            entry_points(group='gbkfit_server.pages'), key=lambda e: e.name):
        try:
            pages[entry.name] = entry.load()()
        except Exception as e:
            _log.warning(f"the page '{entry.name}' is not served: {e}")
    return pages
