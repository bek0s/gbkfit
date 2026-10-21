"""
The executors of the jobs: they start the tasks of the library (see
gbkfit.tasks) in processes of their own, on this machine (LocalExecutor)
or elsewhere (e.g. a cluster, later).
"""

import asyncio
import collections
import json
import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol


__all__ = [
    'Executor',
    'LocalExecutor',
    'LocalProcess',
    'OutputTail',
    'Process'
]


class Process(Protocol):
    """The process of a task, which an executor started."""

    async def wait(self) -> int:
        """Wait for it to end; return its exit status."""

    def interrupt(self) -> None:
        """Interrupt it as Ctrl-C does (e.g. a fit ends with its best)."""

    def output(self) -> str:
        """Return the end of its output (its log)."""


class Executor(Protocol):
    """What starts the processes of the tasks."""

    @property
    def capacity(self) -> int:
        """The number of tasks it runs at once."""

    async def start(
            self, task: str, options: Mapping[str, Any], cwd: str
    ) -> Process:
        """Start a task with its options, in a folder."""


class OutputTail:
    """
    The last lines of a stream (e.g. the stderr of a process), read while
    it lasts (see read).
    """

    def __init__(self, nlines: int = 200):
        self._lines: collections.deque[str] = collections.deque(
            maxlen=nlines)

    async def read(self, stream: asyncio.StreamReader) -> None:
        """Read a stream to its end."""
        async for line in stream:
            self._lines.append(line.decode(errors='replace'))

    def text(self) -> str:
        """Return the lines read."""
        return ''.join(self._lines)


class LocalProcess:
    """
    The process of a task on this machine (see LocalExecutor), and a
    function to call when it ends.
    """

    def __init__(
            self,
            process: asyncio.subprocess.Process,
            on_end: Callable[[], None]
    ):
        self._process = process
        self._on_end = on_end
        self._output = OutputTail()
        self._reading = asyncio.create_task(
            self._output.read(process.stderr))

    async def wait(self) -> int:
        status = await self._process.wait()
        await self._reading
        self._on_end()
        return status

    def interrupt(self) -> None:
        if self._process.returncode is None:
            self._process.send_signal(signal.SIGINT)

    def output(self) -> str:
        return self._output.text()


class LocalExecutor:
    """
    Run tasks on this machine (python -m gbkfit.tasks), a number of them
    at once, each on a GPU of its own if there are GPUs.

    Parameters
    ----------
    capacity : int
        The number of tasks at once.
    gpus : Sequence of str, optional
        The GPUs (as CUDA_VISIBLE_DEVICES names them), at least as many
        as the tasks at once.

    Raises
    ------
    ValueError
        If there are fewer GPUs than tasks at once.
    """

    def __init__(self, capacity: int = 1, gpus: Sequence[str] = ()):
        if gpus and len(gpus) < capacity:
            raise ValueError(
                f"{capacity} tasks at once need as many GPUs; there are "
                f"{len(gpus)}")
        self._capacity = capacity
        self._free_gpus = list(gpus)

    @property
    def capacity(self) -> int:
        return self._capacity

    async def start(
            self, task: str, options: Mapping[str, Any], cwd: str
    ) -> LocalProcess:
        env = dict(os.environ)
        gpu = self._free_gpus.pop(0) if self._free_gpus else None
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = gpu
        # (in a session of its own: the Ctrl-C of the server's terminal is
        # not the task's)
        process = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'gbkfit.tasks', task, json.dumps(options),
            cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE, start_new_session=True)

        def release() -> None:
            if gpu is not None:
                self._free_gpus.append(gpu)
        return LocalProcess(process, release)
