"""
gbkfit-studio: the Studio of GBKFIT, a page that the server serves (see
gbkfit-server). The page is built from web/ (npm run build) into page/,
and the server finds it by its entry point (gbkfit_server.pages).
"""

import os


__all__ = [
    'page'
]


def page() -> str:
    """
    Return the folder of the built page.

    Raises
    ------
    FileNotFoundError
        If the page is not built.
    """
    folder = os.path.join(os.path.dirname(__file__), 'page')
    if not os.path.isfile(os.path.join(folder, 'index.html')):
        raise FileNotFoundError(
            f"the page of the Studio is not built in {folder} (npm run build "
            f"in apps/studio)")
    return folder
