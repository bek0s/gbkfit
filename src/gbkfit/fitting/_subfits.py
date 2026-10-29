"""
Fits made of sub-fits, each a fit of its own (e.g. the refits of a
bootstrap, or the starts of multistart): run at once on workers, or one
after the other, stopped, logged and resumed alike (see run_subfits).
"""

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from gbkfit.fitting.control import FitControl, FitStopped
from gbkfit.fitting.evaluation_log import SubFitLog
from gbkfit.objective import Objective

if TYPE_CHECKING:
    # (the workers import fitting, for its control)
    from gbkfit.parallel import WorkerPool


__all__ = [
    'Outcome',
    'run_subfits'
]


_Task = TypeVar('_Task')

# The outcome of a sub-fit: a dictionary of JSON values (see SubFitLog)
Outcome: TypeAlias = dict[str, Any]


def run_subfits(
        objective: Objective,
        function: Callable[[Objective, _Task, FitControl], Outcome],
        make: Callable[[int], _Task],
        count: int,
        pool: 'WorkerPool | None' = None,
        control: FitControl | None = None,
        log: SubFitLog | None = None,
        on_outcome: Callable[[int, Outcome, bool], None] | None = None
) -> list[Outcome]:
    """
    Run the sub-fits of a fit: function(objective, make(i), control) for
    i < count, each a fit of its own on the objective (or a copy of it)
    that returns its outcome.

    The sub-fits whose outcomes the log has (of a fit that resumes) are
    taken from it; the others run on the workers at once, each on its
    copy of the objective (see WorkerPool.map_lazily), or else one after
    the other on the objective, and are logged as each finishes. Each
    outcome goes to on_outcome, with whether it came from the log, one at
    a time, as it comes.

    A sub-fit takes a control that stops it once the given control is
    stopped (see FitControl): then no sub-fit starts any more, and it
    raises FitStopped once those running have stopped. Those that
    finished are logged: the fit resumes from there.

    Parameters
    ----------
    objective : Objective
        The objective of the fit.
    function : Callable
        A sub-fit: of an objective, a task and a control, its outcome.
    make : Callable
        The task of each sub-fit, by index: made when it starts.
    count : int
        The number of sub-fits.
    pool : WorkerPool, optional
        Workers of copies of the objective.
    control : FitControl, optional
        The control of the fit.
    log : SubFitLog, optional
        The log of the outcomes.
    on_outcome : Callable, optional
        What takes each outcome: of its index, the outcome, and whether it
        came from the log.

    Returns
    -------
    list of Outcome
        The outcome of each sub-fit, in order.

    Raises
    ------
    FitStopped
        If the fit was asked to stop before all the sub-fits finished.
    """
    outcomes = {} if log is None else {
        i: outcome for i, outcome in log.outcomes().items() if i < count}
    lock = threading.Lock()
    if on_outcome is not None:
        for i, outcome in sorted(outcomes.items()):
            on_outcome(i, outcome, True)
    todo = [i for i in range(count) if i not in outcomes]

    def finished(j: int, outcome: Outcome) -> None:
        with lock:
            if log is not None:
                log.append(todo[j], outcome)
            if on_outcome is not None:
                on_outcome(todo[j], outcome, False)
    if pool is not None:
        new = pool.map_lazily(
            function, lambda j: make(todo[j]), len(todo), finished, control)
    else:
        own = FitControl() if control is None else control
        new = []
        for j, i in enumerate(todo):
            if own.stopped():
                raise FitStopped("the fit was asked to stop")
            new.append(function(objective, make(i), own))
            finished(j, new[-1])
    outcomes.update(zip(todo, new))
    return [outcomes[i] for i in range(count)]
