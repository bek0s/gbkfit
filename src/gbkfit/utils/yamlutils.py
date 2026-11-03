"""
The readers and writers of YAML that threads share.

A ruamel.yaml.YAML keeps the state of the document it reads or writes
(its reader, scanner, parser and the rest), so two threads that use one at
once (e.g. the requests that a server answers in threads) spoil each
other's: errors, documents read wrong, or files written wrong. A YAML of
this module is one that a thread at a time uses.
"""

import threading
from typing import Any

import ruamel.yaml


__all__ = [
    'YAML'
]


class YAML(ruamel.yaml.YAML):
    """A ruamel.yaml.YAML that a thread at a time reads or writes with."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # (reentrant: a load of a path loads the file it opens, and a dump
        # to one dumps to it)
        self._lock = threading.RLock()

    def load(self, stream: Any) -> Any:
        with self._lock:
            return super().load(stream)

    def load_all(self, stream: Any) -> Any:
        # (all its documents read before another thread can use it)
        with self._lock:
            return list(super().load_all(stream))

    def dump(
            self, data: Any, stream: Any = None, *, transform: Any = None
    ) -> Any:
        with self._lock:
            return super().dump(data, stream, transform=transform)

    def dump_all(
            self, documents: Any, stream: Any, *, transform: Any = None
    ) -> Any:
        with self._lock:
            return super().dump_all(documents, stream, transform=transform)
