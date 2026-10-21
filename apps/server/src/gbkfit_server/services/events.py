"""
The events of the server (e.g. a job that ends, the progress of a fit),
which its clients follow.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any, NamedTuple


__all__ = [
    'Event',
    'Events'
]


class Event(NamedTuple):
    """An event: its kind (e.g. 'job'), and its data (JSON values)."""
    kind: str
    data: dict[str, Any]


class Events:
    """The events of the server, to each of its subscribers."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[Event]] = set()

    def publish(self, kind: str, data: dict[str, Any]) -> None:
        """Send an event to every subscriber."""
        for queue in self._queues:
            queue.put_nowait(Event(kind, data))

    async def subscribe(self) -> AsyncIterator[Event]:
        """Yield the events from now on, until the caller stops."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._queues.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._queues.discard(queue)
