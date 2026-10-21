"""
The models of the requests and responses of the API (their schemas are
in its contract, openapi.json).
"""

from typing import Any, Literal

from pydantic import BaseModel

from gbkfit_server.services.jobs import TASKS


__all__ = [
    'JobInfo',
    'JobRequest',
    'ObservationInfo',
    'ProjectInfo',
    'RunInfo',
    'ServerInfo',
    'SessionInfo',
    'SessionRequest'
]


class ServerInfo(BaseModel):
    """
    The server: its name and version, the path of its API, and the names
    of the pages it serves (at /<name>/).
    """
    name: str
    version: str
    api: str
    pages: list[str]


class ProjectInfo(BaseModel):
    """
    A project: its path, the names of its pieces and setups by folder
    (e.g. 'galaxies'), and the paths of its runs.
    """
    path: str
    pieces: dict[str, list[str]]
    runs: list[str]


class RunInfo(BaseModel):
    """A run: its manifest (see gbkfit.outputs)."""
    path: str
    task: str
    options: dict[str, Any]
    status: str
    gbkfit_version: str
    created: str
    updated: str
    products: list[dict[str, Any]]


class JobRequest(BaseModel):
    """
    A job to run: its task, and the options of the task (see
    gbkfit.tasks), with their paths relative to the workspace. eval and
    fit need a configuration ('config'), and write their run in the runs
    of its setup, unless they give its folder ('output_dir'); resume
    needs the folder of the run of the fit ('output_dir').
    """
    task: Literal[TASKS]
    options: dict[str, Any] = {}


class JobInfo(BaseModel):
    """
    A job: its id, task and options, its state, its run, when it was
    created, started and ended, the exit status of its process, and the
    end of its log.
    """
    id: str
    task: str
    options: dict[str, Any]
    created: str
    state: Literal[
        'queued', 'running', 'complete', 'failed', 'stopped', 'cancelled']
    run: str | None
    started: str | None
    ended: str | None
    status: int | None
    output: str


class SessionRequest(BaseModel):
    """A live session to open: its configuration (its path)."""
    config: str


class ObservationInfo(BaseModel):
    """An observation of a session: its label, and its data items."""
    label: str
    items: list[str]


class SessionInfo(BaseModel):
    """
    A live session: its id, its configuration, its observations, and its
    free parameters (by element name), with their properties (e.g. their
    values, min and max).
    """
    id: str
    config: str
    observations: list[ObservationInfo]
    free: dict[str, dict[str, Any]]
