"""
The blocks of the objective: solved elements that it solves together
(see Objective).
"""

import abc

from gbkfit.params import ParamValues, Solution


__all__ = [
    'BasisBlock',
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


class BasisBlock(Block):
    """
    Solved elements that the models are linear in, solved before the
    models are evaluated at their values (see Objective): amplitudes of
    the light (AmplitudeBlock) or the weights of orbits (OrbitBlock).
    """

    @abc.abstractmethod
    def solve(self, params: ParamValues) -> Solution:
        """
        Solve the elements at the values of the parameters, and return
        them. It overwrites the residuals of the comparisons of its
        observations.
        """

    @abc.abstractmethod
    def solved(
            self, params: ParamValues, solution: Solution
    ) -> ParamValues:
        """
        Return the values of the parameters with the elements at their
        values in a solution of them (copies of those it changes).
        """
