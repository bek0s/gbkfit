"""
The jobs: the tasks of the library that the server runs for its clients
(see gbkfit.tasks), in a queue, by an executor (see executors). A job is
its run (see gbkfit.outputs): its folder is chosen as the job starts, and
the state of the job follows its process.
"""

import asyncio
import collections
import dataclasses
import datetime
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import gbkfit.outputs
import gbkfit.project

from .errors import Invalid, NotFound
from .events import Events
from .executors import Executor, Process
from .workspace import Workspace


__all__ = [
    'Job',
    'Jobs',
    'TASKS'
]


# The tasks of the jobs: those that make a run, and those that continue
# one (in the folder of their option output_dir)
_MAKE_RUN = ('eval', 'fit')
_CONTINUE_RUN = ('resume',)
TASKS = _MAKE_RUN + _CONTINUE_RUN

# How often the run of a job is read, for its events (seconds)
_POLL = 0.5

State = Literal[
    'queued', 'running', 'complete', 'failed', 'stopped', 'cancelled']


@dataclasses.dataclass
class Job:
    """
    A job: its id, task and options (those of the task, with its paths
    relative to the workspace), its state, its run (relative to the
    workspace), when it was created, started and ended (ISO 8601, UTC),
    the exit status of its process, and the end of its log.
    """
    id: str
    task: str
    options: dict[str, Any]
    created: str
    state: State = 'queued'
    run: str | None = None
    started: str | None = None
    ended: str | None = None
    status: int | None = None
    output: str = ''

    def info(self) -> dict[str, Any]:
        """Return it as JSON values."""
        return dataclasses.asdict(self)


class Jobs:
    """
    The jobs of the server.

    Parameters
    ----------
    workspace : Workspace
        The workspace, where the processes of the jobs run.
    executor : Executor
        What runs the tasks.
    events : Events
        Where the events of the jobs go: 'job' (the job, when its state
        changes), 'run' (its run, when its manifest changes) and
        'progress' (each report of the progress of a fit; see
        gbkfit.tasks.fit).
    """

    def __init__(
            self, workspace: Workspace, executor: Executor, events: Events):
        self._workspace = workspace
        self._executor = executor
        self._events = events
        self._jobs: dict[str, Job] = {}
        self._queue: collections.deque[Job] = collections.deque()
        self._processes: dict[str, Process] = {}
        self._stopping: set[str] = set()
        self._running: set[asyncio.Task[None]] = set()

    def list(self) -> list[Job]:
        """Return the jobs, oldest first."""
        return list(self._jobs.values())

    def get(self, id_: str) -> Job:
        """
        Return a job.

        Raises
        ------
        NotFound
            If there is no job of the id.
        """
        if id_ not in self._jobs:
            raise NotFound(f"no job '{id_}'")
        return self._jobs[id_]

    def submit(self, task: str, options: Mapping[str, Any]) -> Job:
        """
        Queue a job, and start it if there is room.

        Raises
        ------
        Invalid
            If the task is not one of TASKS, or an option it needs is
            missing or out of the workspace.
        """
        if task not in TASKS:
            raise Invalid(
                f"the task of a job is one of {TASKS}; it is {task!r}")
        key = 'config' if task in _MAKE_RUN else 'output_dir'
        if not isinstance(options.get(key), str):
            raise Invalid(f"the task {task} needs the option '{key}'")
        self._workspace.path(options[key])
        job = Job(uuid.uuid4().hex[:12], task, dict(options), _now())
        self._jobs[job.id] = job
        self._queue.append(job)
        self._publish(job)
        self._start_queued()
        return job

    def stop(self, id_: str) -> Job:
        """
        Stop a job: cancel it if queued, or interrupt its process if
        running, as Ctrl-C does (e.g. a fit ends with its best so far).

        Raises
        ------
        NotFound
            If there is no job of the id.
        """
        job = self.get(id_)
        if job.state == 'queued':
            self._queue.remove(job)
            job.state, job.ended = 'cancelled', _now()
            self._publish(job)
        elif job.id in self._processes:
            self._stopping.add(job.id)
            self._processes[job.id].interrupt()
        return job

    async def close(self) -> None:
        """Cancel the queued jobs, and stop the running ones."""
        for job in list(self._queue):
            self.stop(job.id)
        for id_ in list(self._processes):
            self.stop(id_)
        await asyncio.gather(*self._running)

    def _start_queued(self) -> None:
        """Start the queued jobs that the executor has room for."""
        while self._queue and len(self._running) < self._executor.capacity:
            task = asyncio.create_task(self._run(self._queue.popleft()))
            self._running.add(task)
            task.add_done_callback(self._ended)

    def _ended(self, task: asyncio.Task[None]) -> None:
        """Start the next jobs when one ends."""
        self._running.discard(task)
        self._start_queued()

    async def _run(self, job: Job) -> None:
        """Run a job, and follow its run until it ends."""
        try:
            options = self._prepare(job)
            process = await self._executor.start(
                job.task, options, self._workspace.root)
        except (Invalid, OSError) as e:
            job.state, job.ended, job.output = 'failed', _now(), str(e)
            self._publish(job)
            return
        self._processes[job.id] = process
        job.state, job.started = 'running', _now()
        self._publish(job)
        watching = asyncio.create_task(self._watch(job))
        job.status = await process.wait()
        watching.cancel()
        del self._processes[job.id]
        job.output = process.output()
        job.state = 'stopped' if job.id in self._stopping \
            else 'complete' if job.status == 0 else 'failed'
        job.ended = _now()
        self._publish(job)

    def _prepare(self, job: Job) -> dict[str, Any]:
        """
        Return the options of the task of a job, with absolute paths, and
        set its run: for a task that makes a run, a new folder (by default
        in the runs of its setup; see gbkfit.project), which the task
        writes into.

        Raises
        ------
        Invalid
            If the configuration is not a setup of a project and the
            options give no folder, or the folder cannot be made.
        """
        options = dict(job.options)
        if job.task in _CONTINUE_RUN:
            options['output_dir'] = self._workspace.path(options['output_dir'])
            job.run = self._workspace.relative(options['output_dir'])
            return options
        config = self._workspace.path(options['config'])
        folder = options.get('output_dir')
        if folder is None:
            folder = _setup_runs(config)
        try:
            path = gbkfit.outputs.make_directory(
                self._workspace.path(folder),
                options.get('output_dir_mode', 'unique'))
        except OSError as e:
            raise Invalid(f"cannot make the folder of the run: {e}") from e
        job.run = self._workspace.relative(path)
        return options | dict(
            config=config, output_dir=path, output_dir_mode='overwrite')

    async def _watch(self, job: Job) -> None:
        """
        Publish the changes of the run of a running job: of its manifest,
        and the new reports of its progress.
        """
        path = self._workspace.path(job.run)
        updated, offset = None, 0
        while True:
            await asyncio.sleep(_POLL)
            try:
                run = gbkfit.outputs.open(path)
            except ValueError:
                continue
            if run.updated != updated:
                updated = run.updated
                self._events.publish('run', dict(
                    job=job.id, run=job.run, status=run.status,
                    updated=run.updated))
            for product in run.products('progress'):
                offset = self._publish_progress(
                    job, run.filename(product), offset)

    def _publish_progress(self, job: Job, filename: str, offset: int) -> int:
        """
        Publish the reports of progress of a file (JSON lines) after an
        offset; return the offset after the last whole line.
        """
        with open(filename, 'rb') as f:
            f.seek(offset)
            text = f.read()
        end = text.rfind(b'\n') + 1
        for line in text[:end].splitlines():
            self._events.publish('progress', dict(
                job=job.id, report=json.loads(line)))
        return offset + end

    def _publish(self, job: Job) -> None:
        """Publish the state of a job."""
        self._events.publish('job', job.info())


def _setup_runs(config: str) -> str:
    """
    Return the folder of the runs of a setup of a project (see
    gbkfit.project.Project.run_path).

    Raises
    ------
    Invalid
        If the configuration is not a setup of a project.
    """
    try:
        project = gbkfit.project.find(config)
    except ValueError as e:
        raise Invalid(str(e)) from e
    name = None if project is None else project.setup_name(config)
    if name is None:
        raise Invalid(
            f"{Path(config).name} is not a setup of a project (in its folder "
            f"setups): give the folder of the run (output_dir)")
    return project.run_path(name)


def _now() -> str:
    """Return the time now (ISO 8601, UTC)."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')
