"""
The console of the tasks: stderr, where the log of the command-line tool
goes too. On a terminal the log goes through it (see
apps.cli.configure_logging), so that it shows above the line of the
progress of a fit (see _progress.console_progress).
"""

import rich.console


__all__ = [
    'CONSOLE'
]


CONSOLE = rich.console.Console(stderr=True)
