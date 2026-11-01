"""
Helpers shared by the tasks: the directory of a run, its log, and the
configurations written into it (gbkfit.setup sets configurations up, and
gbkfit.outputs writes the outputs of evaluations).
"""

import contextlib
import copy
import logging
import os
from collections.abc import Iterator, Mapping
from typing import Any

import gbkfit.outputs
import gbkfit.project
from gbkfit.setup import ConfigFile


# The log of a run, and the form of its messages
_LOG = 'logs/gbkfit.log'
_LOG_FORMAT = (
    '{asctime}:{processName}({process}):{threadName}({thread}):'
    '{levelname}:{name}:{funcName}:{lineno}:{message}')


def run_dir(output_dir: str | None, config: ConfigFile) -> str:
    """
    Return the directory of the run of a configuration: the one given, or
    by default the folder of the runs of its setup in its project (see
    gbkfit.project.Project.run_path), or else 'output'.
    """
    if output_dir is not None:
        return output_dir
    if config.setup is not None:
        return config.project.run_path(config.setup)
    return 'output'


def write_setup(run: gbkfit.outputs.RunWriter, config: ConfigFile) -> None:
    """
    Write the sources of a configuration in a project as a product of a
    run (setup.json, kind 'setup'; see gbkfit.project.Resolved), with the
    folder of the project relative to the run ('project').
    """
    if config.project is None:
        return
    run.write(
        'setup.json', 'setup',
        dict(project=os.path.relpath(config.project.path, run.path))
        | config.sources)


@contextlib.contextmanager
def run_log(run: gbkfit.outputs.RunWriter) -> Iterator[None]:
    """
    While the block runs, copy the log of GBKFIT (the logger 'gbkfit', at
    its level) to the log of a run (logs/gbkfit.log, kind 'log'; appended
    to, e.g. when a fit resumes), with the time, the process, the thread
    and the place of each message.
    """
    handler = logging.FileHandler(run.filename(_LOG))
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, style='{'))
    run.record(_LOG, 'log', 'log')
    logger = logging.getLogger('gbkfit')
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


def write_config(
        run: gbkfit.outputs.RunWriter,
        path: str,
        kind: str,
        cfg: Mapping[str, Any],
        root: str,
        **ids: str
) -> None:
    """
    Write a configuration as a product of a run (YAML), with the version
    of its format and its root (the directory its relative paths are
    relative to) relative to the directory of its file.
    """
    directory = os.path.dirname(os.path.join(run.path, path))
    run.write(
        path, kind,
        dict(version=gbkfit.project.FORMAT_VERSION,
             root=os.path.relpath(root, directory)) | copy.deepcopy(cfg),
        **ids)
