"""
Run a task of GBKFIT in a process of its own, for the programs that start
tasks (e.g. a server that runs jobs):

    python -m gbkfit.tasks [--lenient] [--verbose] TASK [OPTIONS]

TASK is the name of a task function: eval, fit, resume, bootstrap, plot,
migrate, or one of prep_* (see gbkfit.tasks.prep). OPTIONS are its
arguments by name, as a JSON object. The log goes to stderr (and to the
run of the task). An error in a configuration or in the options, or of
files and folders, is reported without a traceback (with one if
--verbose), with exit status 1; an invalid command line exits with
status 2.
"""

import argparse
import json
import logging
from typing import Any

from gbkfit.tasks import TASKS, exit_on_user_errors, log_to_console, run

# (not __name__, which is __main__, outside the logger 'gbkfit')
_log = logging.getLogger('gbkfit.tasks')


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
        'task', choices=TASKS, metavar='TASK',
        help=f"the task: {', '.join(TASKS)}")
    parser.add_argument(
        'options', type=_json_object, nargs='?', default={},
        help="the arguments of the task by name, as a JSON object")
    return parser


def main() -> None:
    """
    Run a task: parse the command line, and run the task with its options.
    """
    parser = _make_parser()
    args = parser.parse_args()
    log_to_console(logging.DEBUG if args.verbose else logging.INFO)
    _log.debug(f"running task {args.task} with options {args.options}")
    with exit_on_user_errors(args.verbose):
        run(args.task, args.options, args.lenient)


if __name__ == '__main__':
    main()
