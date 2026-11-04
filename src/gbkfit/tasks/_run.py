"""
Running the tasks by name, as the programs that run them do (gbkfit-cli,
and `python -m gbkfit.tasks`; see __main__).
"""

import contextlib
import importlib
import inspect
import logging
import sys
from collections.abc import Generator, Mapping
from typing import Any

from gbkfit.config import ConfigError, strict_mode


__all__ = [
    'TASKS',
    'exit_on_user_errors',
    'run'
]


_log = logging.getLogger(__name__)

# The tasks by name: their modules in gbkfit.tasks (imported when run) and
# their functions
TASKS = dict(
    eval=('eval', 'eval_'),
    fit=('fit', 'fit'),
    resume=('fit', 'resume'),
    bootstrap=('fit', 'bootstrap'),
    plot=('plot', 'plot'),
    migrate=('migrate', 'migrate'),
) | {name: ('prep', name) for name in (
    'prep_gauss_hermite_moments', 'prep_lsf_muse', 'prep_pixel_brightness',
    'prep_pixel_moments', 'prep_pixel_spectra', 'prep_psf_casa_beams',
    'prep_psf_muse', 'prep_region_moments', 'prep_region_spectra',
    'prep_slit_spectra')}


def run(task: str, options: Mapping[str, Any], lenient: bool = False) -> None:
    """
    Run a task by name with its arguments by name: strict about unknown
    options of configurations, sections and parameters, unless lenient
    (see strict_mode).

    Parameters
    ----------
    task : str
        The name of the task (see TASKS).
    options : Mapping
        The arguments of its function, by name.
    lenient : bool, optional
        Whether unknown names of configurations are warnings.

    Raises
    ------
    ConfigError
        If the options are not the arguments of the function of the task.
    """
    module, name = TASKS[task]
    function = getattr(importlib.import_module(f'gbkfit.tasks.{module}'), name)
    try:
        bound = inspect.signature(function).bind(**options)
    except TypeError as e:
        raise ConfigError(f"invalid options of task {task}: {e}") from None
    with strict_mode(not lenient):
        function(*bound.args, **bound.kwargs)


@contextlib.contextmanager
def exit_on_user_errors(verbose: bool) -> Generator[None]:
    """
    In the block, end the program (status 1) at an error that a user
    fixes, logged without a traceback (with one if verbose): of a
    configuration (ConfigError), or of files and folders (OSError, e.g.
    one not found, or an output directory that exists).
    """
    try:
        yield
    except (ConfigError, OSError) as e:
        if verbose:
            raise
        _log.error(f"error: {e}")
        sys.exit(1)
