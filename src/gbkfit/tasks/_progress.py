"""
How the tasks show the progress of fits (see FitProgress): on a terminal,
a line at the bottom of the console, rewritten in place, with the log
above it (see _console and console_progress); elsewhere (a file, a pipe),
where rewritten lines would pile up, a line in the log now and then; every
report as a line of JSON in a file (JSON Lines), for programs to read;
and the best values so far, now and then (e.g. as a configuration to fit
again from, if the fit is stopped).
"""

import contextlib
import json
import logging
import math
import time
from collections.abc import Callable, Iterator

import rich.live
import rich.text

from gbkfit.fitting import FitProgress, ProgressObserver
from ._console import CONSOLE


__all__ = [
    'BestValues',
    'JsonLinesProgress',
    'LogProgress',
    'TerminalProgress',
    'console_progress'
]


@contextlib.contextmanager
def console_progress(logger: logging.Logger) -> Iterator[ProgressObserver]:
    """
    Show the progress of a fit on the console while the block runs: on a
    terminal, a line at its bottom (see TerminalProgress), else lines in
    the log (see LogProgress).
    """
    if not CONSOLE.is_terminal:
        yield LogProgress(logger)
        return
    with rich.live.Live(console=CONSOLE, auto_refresh=False) as live:
        yield TerminalProgress(live)


class TerminalProgress:
    """
    The progress on the last line of a terminal, rewritten with each
    report; the last report of each stage stays above it.
    """

    def __init__(self, live: rich.live.Live):
        self._live = live
        self._last: FitProgress | None = None

    def __call__(self, progress: FitProgress) -> None:
        if self._last is not None and progress.stage != self._last.stage:
            self._live.console.print(_summary(self._last))
        self._last = progress
        self._live.update(rich.text.Text(_summary(progress)), refresh=True)


class LogProgress:
    """
    The progress in a log: a line at most every interval seconds, and at
    each new stage and the end (unless the line would repeat the last).
    """

    def __init__(self, logger: logging.Logger, interval: float = 30.0):
        self._logger = logger
        self._interval = interval
        self._last: FitProgress | None = None
        self._time = -math.inf

    def __call__(self, progress: FitProgress) -> None:
        now = time.perf_counter()
        last = self._last
        new_stage = last is None or progress.stage != last.stage
        if progress.final and last is not None \
                and progress.nevals == last.nevals and not new_stage:
            return
        if new_stage or progress.final or now - self._time >= self._interval:
            self._last = progress
            self._time = now
            self._logger.info(_summary(progress))


class JsonLinesProgress:
    """
    Every report as a line of JSON in a file (written anew), with the free
    values at the best chi-square; infinite chi-squares are null.
    """

    def __init__(self, filename: str):
        self._filename = filename
        open(filename, 'w').close()

    def __call__(self, progress: FitProgress) -> None:
        record = dict(
            stage=progress.stage, nevals=progress.nevals,
            elapsed=round(progress.elapsed, 3), chi2=_finite(progress.chi2),
            best_chi2=_finite(progress.best_chi2), best=progress.best,
            final=progress.final)
        with open(self._filename, 'a') as f:
            f.write(json.dumps(record) + '\n')


class BestValues:
    """
    The free values at the best chi-square so far, given to write (e.g. a
    configuration to fit again from) at most every interval seconds when
    they change, and at the end.
    """

    def __init__(
            self,
            write: Callable[[dict[str, float]], None],
            interval: float = 10.0
    ):
        self._write = write
        self._interval = interval
        self._written: float | None = None
        self._time = -math.inf

    def __call__(self, progress: FitProgress) -> None:
        if progress.best is None or progress.best_chi2 == self._written:
            return
        now = time.perf_counter()
        if progress.final or now - self._time >= self._interval:
            self._written = progress.best_chi2
            self._time = now
            self._write(progress.best)


def _summary(progress: FitProgress) -> str:
    """Return the progress in one line of text."""
    minutes, seconds = divmod(int(progress.elapsed), 60)
    rate = progress.nevals / progress.elapsed if progress.elapsed > 0 else 0
    return (
        f"{progress.stage}: {progress.nevals} evaluations, "
        f"{minutes}:{seconds:02d} ({rate:.0f}/s), chi2 {progress.chi2:.7g}, "
        f"best {progress.best_chi2:.7g}")


def _finite(value: float) -> float | None:
    """Return a value, or None if it is not finite (not valid JSON)."""
    return value if math.isfinite(value) else None
