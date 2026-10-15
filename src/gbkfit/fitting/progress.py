"""
The progress of fits, which the fit problem reports as they run (see
FitProblem): to observers that the caller of Fitter.fit passes, which
decide what to show and where (e.g. the fit task: a line on the
terminal, the log, a file of JSON lines). The fitters need not report
anything: every one of them evaluates the models through the problem.
"""

import dataclasses
from collections.abc import Callable


__all__ = [
    'FitProgress',
    'ProgressObserver'
]


@dataclasses.dataclass(frozen=True)
class FitProgress:
    """
    The state of a fit, as the fit problem reports it.

    Attributes
    ----------
    stage : str
        What the fit is doing: 'fit', a stage of the fitter (e.g. 'start 3
        of 16' for multistart), or 'uncertainty' (the uncertainty of the
        solutions, after the fitter).
    nevals : int
        The model evaluations so far.
    elapsed : float
        The time since the fit started (s).
    chi2 : float
        The chi-square of the last evaluation.
    best_chi2 : float
        The lowest chi-square so far.
    best : dict of str to float, optional
        The free values at the lowest chi-square, by element name (None
        before a finite chi-square).
    final : bool
        Whether this is the last report of the fit.
    """
    stage: str
    nevals: int
    elapsed: float
    chi2: float
    best_chi2: float
    best: dict[str, float] | None
    final: bool = False


# A function that receives the progress of a fit
ProgressObserver = Callable[[FitProgress], None]
