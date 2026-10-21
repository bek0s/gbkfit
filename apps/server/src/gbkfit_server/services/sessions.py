"""
The live sessions: a configuration set up once in a process of its own
(see worker), and evaluated at the values of its free parameters that a
client sends (e.g. as it moves a slider), in memory, without files.
"""

import asyncio
import sys
import uuid
from collections.abc import Mapping
from typing import Any

from . import frames
from .errors import Failed, Invalid, NotFound
from .executors import OutputTail
from .workspace import Workspace


__all__ = [
    'Latest',
    'Session',
    'Sessions'
]


class Latest:
    """
    The latest of the values sent while others are evaluated (e.g. by a
    slider): those sent in between are dropped, so that the evaluations
    follow the sender without falling behind it.
    """

    def __init__(self) -> None:
        self._value: Any = None
        self._ready = asyncio.Event()

    def put(self, value: Any) -> None:
        """Send values, replacing those not taken yet."""
        self._value = value
        self._ready.set()

    async def get(self) -> Any:
        """Wait for values, and take the latest."""
        await self._ready.wait()
        self._ready.clear()
        return self._value


class Session:
    """
    A live session: the worker of a configuration (see Sessions.open),
    which evaluates one set of values at a time.
    """

    def __init__(
            self,
            id_: str,
            config: str,
            process: asyncio.subprocess.Process,
            output: OutputTail,
            reading: asyncio.Task[None],
            ready: Mapping[str, Any],
            data: bytes
    ):
        self._id = id_
        self._config = config
        self._process = process
        self._output = output
        self._reading = reading
        self._ready = dict(ready)
        self._data = frames.encode(self._ready, data)
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        """The id of the session."""
        return self._id

    def info(self) -> dict[str, Any]:
        """
        Return the session: its id, its configuration (relative to the
        workspace), its free parameters (by element name), with their
        properties (e.g. their values, min and max), and whether it has
        data (and so residuals and a chi-square; see worker).
        """
        return dict(
            id=self._id, config=self._config, free=self._ready['free'],
            data=self._ready['data'])

    def data(self) -> bytes:
        """
        Return the frame of the data of the configuration (see worker),
        which do not change with the values.
        """
        return self._data

    async def evaluate(self, values: Mapping[str, float]) -> bytes:
        """
        Evaluate the models at values of the free parameters (by element
        name; the others keep their latest values), and return the frame
        of the result for the client: the models and the residuals, with
        the chi-square (see worker), or an error ('message').

        Raises
        ------
        Failed
            If the worker has ended.
        """
        async with self._lock:
            try:
                self._process.stdin.write(frames.encode(dict(
                    type='evaluate', values=dict(values))))
                await self._process.stdin.drain()
                header, data = await frames.read_async(self._process.stdout)
            except (asyncio.IncompleteReadError, ConnectionError) as e:
                raise Failed(
                    f"the session of {self._config} has ended:\n"
                    f"{self._output.text()}") from e
        return frames.encode(header, data)

    async def close(self) -> None:
        """End the worker (when its input ends)."""
        if self._process.returncode is None:
            self._process.stdin.close()
        await self._process.wait()
        await self._reading


class Sessions:
    """
    The live sessions of the server.

    Parameters
    ----------
    workspace : Workspace
        The workspace, where the workers run.
    """

    def __init__(self, workspace: Workspace):
        self._workspace = workspace
        self._sessions: dict[str, Session] = {}

    async def open(self, config: str) -> Session:
        """
        Open a session of a configuration: start its worker, and wait for
        it to set the configuration up.

        Raises
        ------
        Invalid
            If the configuration cannot be set up (e.g. it is invalid).
        """
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'gbkfit_server.services.worker',
            self._workspace.path(config), cwd=self._workspace.root,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True)
        output = OutputTail()
        reading = asyncio.create_task(output.read(process.stderr))
        try:
            ready, data = await frames.read_async(process.stdout)
        except asyncio.IncompleteReadError:
            ready, data = dict(type='error', message=None), b''
        if ready['type'] != 'ready':
            await process.wait()
            await reading
            raise Invalid(
                f"cannot set up {config}: "
                f"{ready['message'] or output.text()}")
        session = Session(
            uuid.uuid4().hex[:12], config, process, output, reading, ready,
            data)
        self._sessions[session.id] = session
        return session

    def get(self, id_: str) -> Session:
        """
        Return a session.

        Raises
        ------
        NotFound
            If there is no session of the id.
        """
        if id_ not in self._sessions:
            raise NotFound(f"no session '{id_}'")
        return self._sessions[id_]

    async def close(self, id_: str) -> None:
        """
        Close a session.

        Raises
        ------
        NotFound
            If there is no session of the id.
        """
        session = self.get(id_)
        del self._sessions[id_]
        await session.close()

    async def close_all(self) -> None:
        """Close every session."""
        await asyncio.gather(
            *(self.close(id_) for id_ in list(self._sessions)))
