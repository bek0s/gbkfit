"""
gbkfit-studio: start the server of a folder (gbkfit-server, which serves
the Studio), and open the Studio in the browser.

    gbkfit-studio [FOLDER] [the options of gbkfit-server]
"""

import os
import sys


def main() -> None:
    """Run gbkfit-server with the options given, opening the browser."""
    command = [sys.executable, '-m', 'gbkfit_server', *sys.argv[1:], '--open']
    os.execv(sys.executable, command)
