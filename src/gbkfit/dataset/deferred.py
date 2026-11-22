"""
Arrays read when first used: of a shape known before (e.g. from the
header of a file), read once (see Deferred).
"""

from collections.abc import Callable
from typing import Any

import numpy as np


__all__ = [
    'Deferred'
]


class Deferred:
    """
    An array of a known shape whose values are read the first time they
    are asked for, and kept (e.g. a footprint, or the valid pixels of
    regions, of files opened from their headers alone).

    A copy for another process (a pickle) has the values, read first: the
    reading need not be one that can be sent.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array (numpy order).
    read : Callable
        Reads the array, of that shape.
    """

    def __init__(self, shape: tuple[int, ...], read: Callable[[], np.ndarray]):
        self._shape = tuple(shape)
        self._read: Callable[[], np.ndarray] | None = read
        self._values: np.ndarray | None = None

    @classmethod
    def of(cls, values: np.ndarray) -> 'Deferred':
        """Return one of values already read."""
        deferred = cls(np.shape(values), lambda: values)
        deferred._values, deferred._read = values, None
        return deferred

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array (numpy order)."""
        return self._shape

    def values(self) -> np.ndarray:
        """
        Return the array, read now if not yet.

        Raises
        ------
        ValueError
            If it cannot be read, or is not of its shape.
        """
        if self._values is None:
            values = np.asarray(self._read())
            if values.shape != self._shape:
                raise ValueError(
                    f"the array read is of shape {values.shape}; it was "
                    f"opened of shape {self._shape}")
            self._values, self._read = values, None
        return self._values

    def __getstate__(self) -> dict[str, Any]:
        return dict(shape=self._shape, values=self.values())

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._shape, self._values, self._read = \
            state['shape'], state['values'], None
