"""
The console of the tasks: stderr, where the log of the programs that run
them goes too. On a terminal the log goes through it (see log_handler),
so that it shows above the line of the progress of a fit (see
_progress.console_progress).
"""

import logging
import sys

import rich.console
import rich.highlighter
import rich.logging


__all__ = [
    'log_handler'
]


CONSOLE = rich.console.Console(stderr=True)


def log_handler() -> logging.Handler:
    """
    Return a handler of log records that writes their messages to the
    console of the tasks (stderr), for the programs that run them.

    On a terminal, the messages go through the console, so that they show
    above the line of the progress of a fit; elsewhere (a file, a pipe),
    they are written as they are.

    Returns
    -------
    logging.Handler
        The handler.
    """
    if not CONSOLE.is_terminal:
        return logging.StreamHandler(sys.stderr)
    return rich.logging.RichHandler(
        console=CONSOLE, show_time=False, show_level=False, show_path=False,
        highlighter=rich.highlighter.NullHighlighter())
