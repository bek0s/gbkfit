"""
The values of the parameters that a stage of evaluations reads
(ParamView), and the last output of the stage (StageCache), which it
gives again while those values and the stages it takes do not change.
"""

from collections.abc import Callable, Iterable
from typing import Any, Generic, NoReturn, TypeVar

import numpy as np

from .space import ParamValues


__all__ = [
    'ParamView',
    'StageCache'
]


_Output = TypeVar('_Output')


class ParamView(dict[str, float | np.ndarray]):
    """
    The values of the parameters that a stage of evaluations reads (see
    StageCache), by name: those alone, so that a stage that reads another
    fails, with a KeyError that names it, rather than gives an output
    that a change of it does not change.
    """

    def __missing__(self, name: str) -> NoReturn:
        raise KeyError(
            f"{name!r} is not among the parameters that the stage reads; "
            f"declare it")


class StageCache(Generic[_Output]):
    """
    The last output of a stage of evaluations (e.g. the mass of a galaxy,
    or the model of an observation): the values of the parameters that it
    reads, the versions of the caches of the stages that it takes (whose
    outputs it takes), its output and its version. The stage is evaluated
    again only when one of those values or versions changes (see
    evaluate), and on the values of the parameters it reads alone (see
    ParamView); each output it keeps is a new version, which the stages
    that take it see.

    Two values are the same when equal, element by element for arrays;
    NaN is never the same as itself.

    Parameters
    ----------
    reads : Iterable of str
        The names of the parameters that the stage reads.
    takes : Iterable of StageCache, optional
        The caches of the stages whose outputs it takes.
    """

    def __init__(
            self,
            reads: Iterable[str],
            takes: Iterable['StageCache[Any]'] = ()
    ):
        self._reads = tuple(reads)
        self._takes = tuple(takes)
        self._version = 0
        # The values that the stage read, the versions of the stages it
        # took and its output, at its last evaluation
        self._kept: tuple[ParamValues, tuple[int, ...], _Output] | None = \
            None

    def reads(self) -> tuple[str, ...]:
        """Return the names of the parameters that its stage reads."""
        return self._reads

    def takes(self) -> tuple['StageCache[Any]', ...]:
        """Return the caches of the stages that its stage takes."""
        return self._takes

    def version(self) -> int:
        """
        Return its version: the number of the outputs it kept and of the
        times it forgot one (0 at first).
        """
        return self._version

    def evaluate(
            self,
            params: ParamValues,
            stage: Callable[[ParamView], _Output]
    ) -> _Output:
        """
        Return the output of the stage at the values of the parameters:
        that of its last evaluation if the values of the parameters it
        reads and the versions of the stages it takes are the same; else
        that of the stage evaluated again (see refresh).

        Parameters
        ----------
        params : ParamValues
            The values of the parameters, those it reads among them.
        stage : callable
            The stage, of the values of the parameters it reads.

        Returns
        -------
        object
            The output of the stage.
        """
        if self._kept is not None:
            values, versions, output = self._kept
            if versions == self._taken() and all(
                    _same(params[name], value)
                    for name, value in values.items()):
                return output
        return self.refresh(params, stage)

    def refresh(
            self,
            params: ParamValues,
            stage: Callable[[ParamView], _Output]
    ) -> _Output:
        """
        Evaluate the stage at the values of the parameters it reads, keep
        its output, as a new version, with copies of those values, and
        return it: e.g. with extra outputs, which the stage gives only when
        evaluated.

        Parameters
        ----------
        params : ParamValues
            The values of the parameters, those it reads among them.
        stage : callable
            The stage, of the values of the parameters it reads.

        Returns
        -------
        object
            The output of the stage.
        """
        view = self.view(params)
        output = stage(view)
        values = {
            name: value if np.isscalar(value) else np.array(value)
            for name, value in view.items()}
        self._kept = (values, self._taken(), output)
        self._version += 1
        return output

    def view(self, params: ParamValues) -> ParamView:
        """
        Return the values of the parameters that its stage reads, of those
        of the parameters (see ParamView).
        """
        return ParamView((name, params[name]) for name in self._reads)

    def forget(self) -> None:
        """
        Forget its output, as a new version (e.g. its memory was
        overwritten): the next evaluation evaluates the stage again.
        """
        self._kept = None
        self._version += 1

    def _taken(self) -> tuple[int, ...]:
        """Return the versions of the stages that its stage takes."""
        return tuple(cache.version() for cache in self._takes)


def _same(a: float | np.ndarray, b: float | np.ndarray) -> bool:
    """
    Check whether two values of a parameter are the same: equal, element
    by element for arrays; NaN is not the same as itself.
    """
    if np.isscalar(a) and np.isscalar(b):
        return bool(a == b)
    return np.array_equal(a, b)
