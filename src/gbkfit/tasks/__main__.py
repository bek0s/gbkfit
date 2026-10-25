"""
Run a task of GBKFIT in a process of its own, for the programs that start
tasks (e.g. a server that runs jobs):

    python -m gbkfit.tasks [--lenient] [--verbose] TASK [OPTIONS]

TASK is the name of a task function: eval, fit, resume, bootstrap, plot,
migrate, or one of prep_* (see gbkfit.tasks.prep). OPTIONS are its
arguments by name, as a JSON object. The log goes to stderr (and to the
run of the task). An error in a configuration is reported without a
traceback (with one if --verbose), with exit status 1; invalid arguments
exit with status 2.
"""

import argparse
import importlib
import inspect
import json
import logging
import sys
from collections.abc import Callable
from typing import Any

from gbkfit.tasks import log_handler
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError

# (not __name__, which is __main__, outside the logger 'gbkfit')
_log = logging.getLogger('gbkfit.tasks')

# The tasks by name: their modules in gbkfit.tasks (imported when run) and
# their functions
_TASKS = dict(
    eval=('eval', 'eval_'),
    fit=('fit', 'fit'),
    resume=('fit', 'resume'),
    bootstrap=('fit', 'bootstrap'),
    plot=('plot', 'plot'),
    migrate=('migrate', 'migrate'),
) | {name: ('prep', name) for name in (
    'prep_lsf_muse', 'prep_pixel_brightness', 'prep_pixel_moments',
    'prep_pixel_spectra', 'prep_psf_casa_beams', 'prep_psf_muse',
    'prep_region_moments', 'prep_region_spectra', 'prep_slit_spectra')}


def _json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object (the options of a task)."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"invalid JSON: {e}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return value


def _make_parser() -> argparse.ArgumentParser:
    """Return the parser of the command line."""
    parser = argparse.ArgumentParser(
        prog='python -m gbkfit.tasks',
        description="Run a task of GBKFIT.")
    parser.add_argument(
        '--lenient', action='store_true',
        help="ignore unknown configuration options, sections and "
             "parameters with a warning, instead of failing")
    parser.add_argument(
        '--verbose', action='store_true', help="enable verbose logging")
    parser.add_argument(
        'task', choices=_TASKS, metavar='TASK',
        help=f"the task: {', '.join(_TASKS)}")
    parser.add_argument(
        'options', type=_json_object, nargs='?', default={},
        help="the arguments of the task by name, as a JSON object")
    return parser


def _bind(
        parser: argparse.ArgumentParser,
        task: str,
        options: dict[str, Any]
) -> Callable[[], None]:
    """
    Return the function of a task bound to its options; exit with an error
    if they do not match its arguments.
    """
    module, name = _TASKS[task]
    function = getattr(importlib.import_module(f'gbkfit.tasks.{module}'), name)
    try:
        bound = inspect.signature(function).bind(**options)
    except TypeError as e:
        parser.error(f"invalid options of task {task}: {e}")
    return lambda: function(*bound.args, **bound.kwargs)


def _configure_logging(level: int) -> None:
    """Log GBKFIT to the console of the tasks (stderr) at a level."""
    logger = logging.getLogger('gbkfit')
    logger.setLevel(level)
    logger.addHandler(log_handler())
    logger.propagate = False


def main() -> None:
    """
    Run a task: parse the command line, and run the task with its options.
    """
    parser = _make_parser()
    args = parser.parse_args()
    task = _bind(parser, args.task, args.options)
    _configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    _log.debug(f"running task {args.task} with options {args.options}")
    try:
        with parseutils.strict_mode(not args.lenient):
            task()
    except ConfigError as e:
        if args.verbose:
            raise
        _log.error(f"error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
