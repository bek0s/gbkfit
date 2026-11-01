"""
The models of the requests and responses of the API (their schemas are
in its contract, openapi.json).
"""

from typing import Any, Literal

from pydantic import BaseModel

from gbkfit_server.services.jobs import TASKS


__all__ = [
    'ArrayInfo',
    'CheckRequest',
    'PieceRequest',
    'ProjectRequest',
    'CheckResult',
    'Diagnostic',
    'FileEntry',
    'FolderInfo',
    'ImportRequest',
    'JobInfo',
    'JobRequest',
    'ProjectInfo',
    'RunInfo',
    'ServerInfo',
    'SessionInfo',
    'SessionRequest',
    'SpectrumView'
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


class FileEntry(BaseModel):
    """A file of a folder: its name, and its size (bytes)."""
    name: str
    size: int


class FolderInfo(BaseModel):
    """
    A folder: its path, and its folders and files (but the hidden ones),
    by name.
    """
    path: str
    folders: list[str]
    files: list[FileEntry]


class ArrayInfo(BaseModel):
    """
    An array of a file of data (e.g. an HDU of a FITS file), from its
    header: its index and name in the file, its shape and the types of
    its axes, in the order of the model (x, y, then the spectral axis),
    the unit of its values, the size of its pixels in that order (arcsec,
    and km/s along a velocity axis; none if its coordinates cannot be
    read), its beam (bmaj and bmin in arcsec, bpa in degrees; lists for
    a beam of each channel), the noise of its values that its header
    states (keyword RMS or NOISE), whether its header leaves it open
    whether its velocities are radio or optical (a dataset of it gives
    their convention), and what a dataset of it refuses or needs given.
    """
    index: int
    name: str
    shape: list[int]
    axes: list[str]
    unit: str | None
    step: list[float] | None
    beam: dict[str, float | list[float]] | None
    rms: float | None
    ambiguous_velocity: bool
    problems: list[str]


class SpectrumView(BaseModel):
    """
    The spectrum of a pixel of a cube: its values (none where it has
    none), and the coordinates of its channels along its spectral axis
    and their unit (km/s for velocities, GHz for frequencies, micron for
    wavelengths; none if its header does not give them).
    """
    values: list[float | None]
    spectral: list[float] | None
    unit: str | None


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
    needs the folder of the run of the fit ('output_dir'); prep needs the
    folder of the prepared files ('output_dir'), where they are kept as
    imported files are (an identical file is reused; a name taken is
    numbered).
    """
    task: Literal[TASKS]
    options: dict[str, Any] = {}


class JobInfo(BaseModel):
    """
    A job: its id, task and options, its state, its run, the files that
    a job of prep prepared (by the options of their files, e.g. file_d;
    of moment maps, a list, one for each map, None for a file not
    prepared), when it was created, started and ended, the exit status
    of its process, and the end of its log.
    """
    id: str
    task: str
    options: dict[str, Any]
    created: str
    state: Literal[
        'queued', 'running', 'complete', 'failed', 'stopped', 'cancelled']
    run: str | None
    files: dict[str, str | list[str | None]] | None
    started: str | None
    ended: str | None
    status: int | None
    output: str


class SessionRequest(BaseModel):
    """A live session to open: its configuration (its path)."""
    config: str


class SessionInfo(BaseModel):
    """
    A live session: its id, its configuration, its free parameters (by
    element name), with their properties (e.g. their values, min and
    max), and whether it has data (and so residuals and a chi-square).
    """
    id: str
    config: str
    free: dict[str, dict[str, Any]]
    data: bool


class CheckRequest(BaseModel):
    """The text of a file to check (e.g. not saved yet), and its path."""
    path: str
    text: str


class Diagnostic(BaseModel):
    """A problem of a text: its line (from 1), severity and message."""
    line: int
    severity: Literal['error', 'warning']
    message: str


class CheckResult(BaseModel):
    """The problems of a text (see the checker of the server)."""
    diagnostics: list[Diagnostic]


class ProjectRequest(BaseModel):
    """A project to create: its folder (its path)."""
    path: str


class ImportRequest(BaseModel):
    """
    A file of data of the workspace to copy into a project: the project,
    and the path of the file.
    """
    project: str
    path: str


class PieceRequest(BaseModel):
    """
    A piece (or a setup) to create: its project, its folder (e.g.
    galaxies, psfs or setups), its name (its path in the folder, without
    .yaml), and its content (its options); by default, that of the
    template of its folder.
    """
    project: str
    folder: str
    name: str
    content: dict[str, Any] | None = None
