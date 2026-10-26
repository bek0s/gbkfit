"""
Check a configuration without running its task: set it up as its task
would (see gbkfit.load; a fit if it has the section 'fitter'), and
collect its problems: its warnings, its unknown options (errors, all of
them: the check loads past them, which a task does not), and its error,
if any, each with its path in the configuration (e.g. to show them in
place in an editor).
"""

import logging
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

import gbkfit.setup
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError, UnknownNamesError


__all__ = [
    'Checked',
    'Problem',
    'check'
]


class Problem(NamedTuple):
    """
    A problem of a configuration: an error or a warning, its path in the
    configuration (option names and list indices; e.g. ['galaxies', 0]),
    and its message (with the type of the part, if known).
    """
    severity: Literal['error', 'warning']
    path: list[str | int]
    message: str


class Checked(NamedTuple):
    """
    The problems of a configuration (see check), and its sources (the
    files of the pieces it names; see gbkfit.project.Resolved), if it
    could be read.
    """
    problems: list[Problem]
    sources: dict[str, Any]


def check(config: str, texts: Mapping[str, str] | None = None) -> Checked:
    """
    Check a configuration: set it up as its task would, and return its
    problems.

    Parameters
    ----------
    config : str
        The configuration file.
    texts : Mapping of str to str, optional
        Texts to read in place of files, by their absolute paths: of the
        configuration, or of the pieces of its project (e.g. edits not
        saved yet).

    Returns
    -------
    Checked
        The warnings and the unknown names, then the error, if any.
    """
    warnings = _Warnings()
    logger = logging.getLogger('gbkfit')
    logger.addHandler(warnings)
    problems, sources = [], {}
    try:
        # (lenient, to list every unknown name)
        with parseutils.strict_mode(False):
            file = gbkfit.setup.read_config(config, texts=texts)
            sources = file.sources
            with gbkfit.setup.load(file) as setup:
                _check_task_sections(file.config, setup)
    except ConfigError as e:
        problems.append(_problem('error', e))
    finally:
        logger.removeHandler(warnings)
    return Checked(warnings.problems + problems, sources)


class _Warnings(logging.Handler):
    """
    The warnings about a configuration logged (see parseutils.warn), and
    its unknown names, as errors (see parseutils.report_unknown).
    """

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.problems: list[Problem] = []

    def emit(self, record: logging.LogRecord) -> None:
        if (error := getattr(record, 'config_problem', None)) is not None:
            severity = 'error' if isinstance(error, UnknownNamesError) \
                else 'warning'
            self.problems.append(_problem(severity, error))


def _problem(
        severity: Literal['error', 'warning'], error: ConfigError
) -> Problem:
    """Return the problem of an error (or of a warning)."""
    message = error.message if error.context is None \
        else f"[{error.context}] {error.message}"
    return Problem(severity, list(error.path), message)


def _check_task_sections(
        config: dict[str, Any], setup: gbkfit.setup.Setup
) -> None:
    """
    Load the sections of a configuration that only tasks read (the mock
    data of eval, the bootstrap of fit), as they load them, so that their
    errors show with those of the rest.
    """
    from gbkfit.tasks import eval as eval_task, fit as fit_task
    if 'mock' in config:
        with parseutils.config_path('mock'):
            eval_task._load_mock(
                config['mock'], setup.group_plan.group().nobservations())
    if setup.objective is not None:
        fit_task._load_bootstrap(config, setup)
