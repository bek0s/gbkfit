"""
The modes of parameters: how the values of a vector parameter are given.

A parameter with a mode is given (fixed, free or tied) by coded values,
e.g. offsets from one of its elements, which the mode decodes into the
values of the parameter. The models and the expressions that read the
parameter see its decoded values, and a fit varies its coded ones.
"""

import abc

import numpy as np

from gbkfit.utils import numutils, parseutils
from gbkfit.utils.parseutils import ConfigError


__all__ = [
    'ParamMode',
    'ParamModeIncrements',
    'ParamModeOffsets',
    'param_mode_parser'
]


def _check_origin(origin: int) -> None:
    """Raise ConfigError unless the origin is an integer."""
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise ConfigError(f"origin must be an integer; it is {origin!r}")


def _check_origin_in(origin: int, size: int) -> None:
    """Raise ConfigError unless the origin is an index of size elements."""
    if not -size <= origin < size:
        raise ConfigError(
            f"origin must be an index of the {size} elements of the "
            f"parameter; it is {origin}")


class ParamMode(parseutils.TypedSerializable, abc.ABC):
    """
    How the values of a vector parameter are given: by coded values,
    which the mode decodes.
    """

    @abc.abstractmethod
    def check(self, size: int) -> None:
        """
        Check that the mode suits a vector parameter.

        Parameters
        ----------
        size : int
            The number of elements of the parameter.

        Raises
        ------
        ConfigError
            If the mode does not suit the parameter.
        """
        pass

    @abc.abstractmethod
    def decode(self, values: np.ndarray) -> np.ndarray:
        """
        Decode the coded values of a parameter.

        Parameters
        ----------
        values : np.ndarray
            The coded values.

        Returns
        -------
        np.ndarray
            The values of the parameter (a new array).
        """
        pass


class ParamModeOffsets(ParamMode):
    """
    Values given as offsets from the value of one element: the element at
    the origin is given as its value, and each other one as its offset
    from it (e.g. the position angles of the rings of a warp as offsets
    from that of the innermost ring).

    Parameters
    ----------
    origin : int, optional
        The index of the element whose value is given (negative indices
        count from the end).

    Raises
    ------
    ConfigError
        If the origin is not an integer.
    """

    @staticmethod
    def type() -> str:
        return 'offsets'

    def dump(self) -> dict[str, str | int]:
        return dict(type=self.type(), origin=self._origin)

    def __init__(self, origin: int = 0):
        _check_origin(origin)
        self._origin = origin

    def origin(self) -> int:
        """Return the index of the element whose value is given."""
        return self._origin

    def check(self, size: int) -> None:
        _check_origin_in(self._origin, size)

    def decode(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        decoded = values + values[self._origin]
        decoded[self._origin] = values[self._origin]
        return decoded


class ParamModeIncrements(ParamMode):
    """
    Values given as increments outward from one element: the element at
    the origin is given as its value, and each other one as its increment
    over its neighbour towards the origin (e.g. the change of the position
    angle from each ring of a warp to the next).

    Parameters
    ----------
    origin : int, optional
        The index of the element whose value is given (negative indices
        count from the end).

    Raises
    ------
    ConfigError
        If the origin is not an integer.
    """

    @staticmethod
    def type() -> str:
        return 'increments'

    def dump(self) -> dict[str, str | int]:
        return dict(type=self.type(), origin=self._origin)

    def __init__(self, origin: int = 0):
        _check_origin(origin)
        self._origin = origin

    def origin(self) -> int:
        """Return the index of the element whose value is given."""
        return self._origin

    def check(self, size: int) -> None:
        _check_origin_in(self._origin, size)

    def decode(self, values: np.ndarray) -> np.ndarray:
        return numutils.cumsum_from(
            np.asarray(values, dtype=float), self._origin)


param_mode_parser = parseutils.TypedParser(ParamMode, [
    ParamModeIncrements,
    ParamModeOffsets])
