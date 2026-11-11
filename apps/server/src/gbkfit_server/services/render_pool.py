"""
The renders of the server (see Renders), each in a worker process of a
pool (see render_worker): the server stays free while they work, and a
render can be stopped, its worker ended and another started in its
place (native code cannot be interrupted in a thread). A render that runs
longer than a moment is announced to the pages (events 'render'), with
the stage it reaches, so that they can show it and cancel it; a quick one
is not, so that the renders of a slider do not crowd the pages.

A render whose client goes away (e.g. it asked for a newer one) is
cancelled: dropped if it waits; if it runs, its result is dropped once it
comes, but a render that has run longer than the moment is stopped.
"""

import asyncio
import dataclasses
import logging
import pickle
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from . import frames
from .errors import Cancelled, Failed, Invalid, NotFound, ServiceError
from .events import Events
from .executors import OutputTail
from .workspace import Workspace


__all__ = [
    'Render',
    'RenderPool'
]


_log = logging.getLogger(__name__)

# The errors of a render, by the names its worker gives them
_ERRORS: dict[str, type[Exception]] = dict(
    Invalid=Invalid, NotFound=NotFound, Failed=Failed)


@dataclasses.dataclass(eq=False)
class Render:
    """
    A render: its id, what it renders (for people, e.g. "galaxy
    ngc2403"), its kind and arguments (see render_worker), the stage it
    has reached ('waiting' for a worker, then those of the render), when
    it started (time.monotonic), whether it has been announced (see the
    module), and its result to come.
    """
    id: str
    what: str
    kind: str
    args: dict[str, Any]
    result: asyncio.Future[Any]
    stage: str = 'waiting'
    started: float = dataclasses.field(default_factory=time.monotonic)
    announced: bool = False

    def elapsed(self) -> float:
        """Return the seconds since it started."""
        return time.monotonic() - self.started

    def info(self) -> dict[str, Any]:
        """
        Return it for the pages: its id, what it renders, its kind, its
        stage and the seconds since it started.
        """
        return dict(
            id=self.id, what=self.what, kind=self.kind, stage=self.stage,
            elapsed=round(self.elapsed(), 1))


class _Worker:
    """A worker process, the tail of its log, and the render it does."""

    def __init__(
            self,
            process: asyncio.subprocess.Process,
            output: OutputTail,
            logging: asyncio.Task[None]
    ):
        self.process = process
        self.output = output
        self.logging = logging
        self.render: Render | None = None
        self.reading: asyncio.Task[None] | None = None


class RenderPool:
    """
    The pool of the workers of the renders of a workspace.

    Parameters
    ----------
    workspace : Workspace
        The workspace, whose galaxies the workers render.
    events : Events
        The events of the server, where long renders are announced.
    size : int, optional
        The most workers at once (renders at once).
    moment : float, optional
        The seconds after which a render is announced, and after which it
        is stopped when its client goes away (see the module).
    """

    def __init__(
            self,
            workspace: Workspace,
            events: Events,
            size: int = 2,
            moment: float = 1.0
    ):
        self._workspace = workspace
        self._events = events
        self._size = size
        self._moment = moment
        self._workers: set[_Worker] = set()
        self._starting = 0
        self._idle: asyncio.Queue[_Worker] = asyncio.Queue()
        self._renders: dict[str, Render] = {}
        self._closed = False

    async def render(
            self,
            kind: str,
            args: Mapping[str, Any],
            what: str,
            gone: Callable[[], Awaitable[bool]] | None = None
    ) -> Any:
        """
        Render in a worker (kind and args: see render_worker), and return
        the result; cancel the render if its client goes away (gone()
        becomes true; see the module).

        Raises
        ------
        Invalid, NotFound, Failed
            The errors of the render (e.g. an invalid galaxy), or Failed
            if its worker ended.
        Cancelled
            If the render was cancelled (see cancel).
        """
        loop = asyncio.get_running_loop()
        render = Render(
            uuid.uuid4().hex[:12], what, kind, dict(args),
            loop.create_future())
        self._renders[render.id] = render
        tasks = [
            asyncio.create_task(self._dispatch(render)),
            asyncio.create_task(self._announce_later(render))]
        try:
            while True:
                done, _ = await asyncio.wait({render.result}, timeout=0.25)
                if done:
                    return render.result.result()
                if gone is not None and await gone():
                    self._cancel(render, stop=False)
        finally:
            for task in tasks:
                task.cancel()
            del self._renders[render.id]
            if render.announced:
                self._events.publish('render', render.info() | dict(
                    state=_state(render.result)))

    def running(self) -> list[dict[str, Any]]:
        """
        Return the renders that have been announced (see Render.info), in
        the order they started.
        """
        return [r.info() for r in self._renders.values() if r.announced]

    def cancel(self, id_: str) -> None:
        """
        Cancel a render: drop it if it waits, and stop it if it runs.

        Raises
        ------
        NotFound
            If no render of the id waits or runs.
        """
        if id_ not in self._renders:
            raise NotFound(f"no render '{id_}'")
        self._cancel(self._renders[id_], stop=True)

    async def close(self) -> None:
        """Cancel every render, and end the workers."""
        self._closed = True
        for render in list(self._renders.values()):
            self._cancel(render, stop=True)
        for worker in list(self._workers):
            await self._end(worker)

    def _cancel(self, render: Render, stop: bool) -> None:
        """
        Cancel a render (see the module): drop it if it waits; if it runs,
        drop its result, and stop it if stop, or if it has run longer than
        the moment.
        """
        if not render.result.done():
            render.result.set_exception(
                Cancelled(f"the render of {render.what} was cancelled"))
        worker = next(
            (w for w in self._workers if w.render is render), None)
        if worker is not None and (stop or render.elapsed() > self._moment):
            worker.process.kill()

    async def _dispatch(self, render: Render) -> None:
        """
        Give a render to a worker, once one is free; fail it if no worker
        can start.
        """
        try:
            worker = await self._take()
        except ServiceError as e:
            if not render.result.done():
                render.result.set_exception(e)
            return
        if render.result.done():
            # (cancelled while it waited)
            self._idle.put_nowait(worker)
            return
        worker.render = render
        render.stage = 'starting'
        self._spare()
        worker.process.stdin.write(frames.encode(dict(
            type='render', id=render.id, kind=render.kind,
            args=render.args)))
        await worker.process.stdin.drain()

    async def _take(self) -> _Worker:
        """
        Return a free worker: an idle one that has not ended, or a new one
        if there is room (started to the end even if the caller is
        cancelled meanwhile, so that no process is left behind).

        Raises
        ------
        Failed
            If a new worker cannot start.
        """
        while True:
            if self._idle.empty() \
                    and len(self._workers) + self._starting < self._size:
                await asyncio.shield(self._start())
            worker = await self._idle.get()
            if worker in self._workers:
                return worker

    def _spare(self) -> None:
        """Start a worker in the background, if there is room for one."""
        if not self._closed and self._idle.empty() \
                and len(self._workers) + self._starting < self._size:
            asyncio.create_task(self._start_quietly())

    async def _start_quietly(self) -> None:
        """Start a worker (see _start), logging why if it cannot."""
        try:
            await self._start()
        except Failed as e:
            _log.warning("%s", e)

    async def _start(self) -> None:
        """
        Start a worker, and once it is ready, make it idle.

        Raises
        ------
        Failed
            If it cannot start.
        """
        self._starting += 1
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'gbkfit_server.services.render_worker',
                self._workspace.root, cwd=self._workspace.root,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True)
            output = OutputTail()
            worker = _Worker(
                process, output,
                asyncio.create_task(output.read(process.stderr)))
            try:
                ready, _ = await frames.read_async(process.stdout)
            except asyncio.IncompleteReadError:
                ready = dict(type='error')
        finally:
            self._starting -= 1
        if ready['type'] != 'ready':
            await process.wait()
            await worker.logging
            raise Failed(
                f"a worker of the renders could not start:\n"
                f"{output.text()}")
        self._workers.add(worker)
        worker.reading = asyncio.create_task(self._read(worker))
        self._idle.put_nowait(worker)

    async def _read(self, worker: _Worker) -> None:
        """
        Read the frames of a worker, for the render it does, until it
        ends; then start another in its place if a render waits.
        """
        try:
            while True:
                header, data = await frames.read_async(worker.process.stdout)
                render = worker.render
                if render is None or header.get('id') != render.id:
                    continue
                if header['type'] == 'progress':
                    render.stage = header['stage']
                    if render.announced:
                        self._events.publish('render', render.info() | dict(
                            state='running'))
                    continue
                if not render.result.done():
                    if header['type'] == 'result':
                        render.result.set_result(pickle.loads(data))
                    else:
                        render.result.set_exception(
                            _ERRORS[header['error']](header['message']))
                worker.render = None
                self._idle.put_nowait(worker)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        # (the worker ended: stopped, or it failed)
        await self._end(worker)
        render = worker.render
        if render is not None and not render.result.done():
            render.result.set_exception(Failed(
                f"the worker of the render of {render.what} ended:\n"
                f"{worker.output.text()}"))
        if not self._closed and any(
                r.stage == 'waiting' for r in self._renders.values()):
            self._spare()

    async def _end(self, worker: _Worker) -> None:
        """End a worker (if it has not ended), and forget it."""
        self._workers.discard(worker)
        if worker.process.returncode is None:
            worker.process.kill()
        await worker.process.wait()
        await worker.logging

    async def _announce_later(self, render: Render) -> None:
        """Announce a render once it has run longer than the moment."""
        await asyncio.sleep(self._moment)
        if not render.result.done():
            render.announced = True
            self._events.publish(
                'render', render.info() | dict(state='running'))


def _state(result: asyncio.Future[Any]) -> str:
    """
    Return the state of the result of a render that has ended: 'done',
    'cancelled' or 'failed'.
    """
    error = result.exception() if result.done() else None
    if error is None:
        return 'done'
    return 'cancelled' if isinstance(error, Cancelled) else 'failed'
