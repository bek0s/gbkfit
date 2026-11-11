"""
The blocks of the objective: solved elements that it solves together
(see Objective).
"""

import abc


__all__ = [
    'Block'
]


class Block(abc.ABC):
    """
    Solved elements that the objective solves together: those of the flux
    scale of an observation (ScaleBlock), amplitudes of the light of the
    models (AmplitudeBlock), or the weights of orbits (OrbitBlock).
    """

    @abc.abstractmethod
    def elements(self) -> tuple[str, ...]:
        """Return the names of the elements, in their order."""

    @abc.abstractmethod
    def likelihood(self) -> str:
        """
        Return what the log-likelihood takes of their values (see
        Solve.likelihood).
        """

    def close(self) -> None:
        """Release its memory, if any; the next solve allocates it again."""
