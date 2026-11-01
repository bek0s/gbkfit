"""
Check a configuration without running its task, and without devices:
describe it (see gbkfit.setup.describe), and load the sections that only
tasks read (the mock data of eval, the bootstrap of fit) as they load
them; collect its problems: its warnings, its unknown names (errors, all
of them: the check loads past them, which a task does not), and its
error, if any, each with its path in the configuration (e.g. to show
them in place in an editor).
"""

import os
from collections.abc import Mapping
from typing import Any, NamedTuple

import gbkfit.fitting
import gbkfit.observation
import gbkfit.setup
from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import Problem


__all__ = [
    'Checked',
    'Problem',
    'check'
]


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
    Check a configuration: describe it, load the sections of its tasks,
    and return its problems.

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
    description = gbkfit.setup.describe(config, texts)
    problems = list(description.problems)
    if description.params is not None:
        with parseutils.strict_mode(False), \
                parseutils.collect_problems() as task_problems:
            _check_task_sections(description)
        problems += task_problems
    return Checked(problems, description.sources)


def _check_task_sections(description: gbkfit.setup.Description) -> None:
    """
    Load the sections of a configuration that only tasks read (the mock
    data of eval, the bootstrap of a fit, which needs data), as they load
    them, so that their errors show with those of the rest.
    """
    config, group = description.config, description.group
    # (their paths relative to the root of the configuration, as eval
    # takes them, not to the current directory)
    with parseutils.paths_relative_to(description.root or os.curdir):
        if 'mock' in config:
            with parseutils.config_path('mock'):
                gbkfit.observation.MockOptions.load(
                    config['mock']).per_observation(group.nobservations())
        if 'bootstrap' in config and group.has_data():
            with parseutils.config_path('bootstrap'):
                gbkfit.fitting.BootstrapOptions.load(
                    config['bootstrap']).check(group, description.params)
