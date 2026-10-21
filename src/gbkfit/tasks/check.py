"""
Check a configuration without running its task: set it up as its task
would (a fit if it has the section 'fitter', else an evaluation; see
gbkfit.tasks.fit and gbkfit.tasks.eval), and collect its problems: its
warnings (e.g. unknown options) and its error, if any, each with its
path in the configuration (e.g. to show them in place in an editor).
"""

import logging
from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail, eval as eval_task, fit as fit_task


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
        The warnings, then the error, if any.
    """
    warnings = _Warnings()
    logger = logging.getLogger('gbkfit')
    logger.addHandler(warnings)
    problems, sources = [], {}
    try:
        with parseutils.strict_mode(False):
            file = _detail.read_config(
                config, eval_task._REQUIRED,
                eval_task._OPTIONAL + eval_task._FIT, texts)
            sources = file.sources
            if 'fitter' in file.config:
                fit_task._set_up(file.config, file.root)
            else:
                with parseutils.paths_relative_to(file.root):
                    eval_task._set_up(file.config)
    except ConfigError as e:
        problems.append(_problem('error', e))
    finally:
        logger.removeHandler(warnings)
    return Checked(warnings.problems + problems, sources)


class _Warnings(logging.Handler):
    """The warnings about a configuration logged (see parseutils.warn)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.problems: list[Problem] = []

    def emit(self, record: logging.LogRecord) -> None:
        if (error := getattr(record, 'config_problem', None)) is not None:
            self.problems.append(_problem('warning', error))


def _problem(
        severity: Literal['error', 'warning'], error: ConfigError
) -> Problem:
    """Return the problem of an error (or of a warning)."""
    message = error.message if error.context is None \
        else f"[{error.context}] {error.message}"
    return Problem(severity, list(error.path), message)
