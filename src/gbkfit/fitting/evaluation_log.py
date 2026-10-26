"""
The log of the model evaluations of a fit, from which a stopped or killed
fit resumes (see EvaluationLog).
"""

import json
import logging
import os
import time
from collections.abc import Sequence
from typing import BinaryIO

import numpy as np

from gbkfit.objective import Merits


__all__ = [
    'EvaluationLog'
]


_log = logging.getLogger(__name__)

# The first line of the files of a log (of the version of their format)
_MAGIC = b'GBKFIT EVALUATIONS 2\n'

# The numbers of each evaluation
_NNUMBERS = len(Merits._fields)

# The least time between two flushes of the files (s)
_FLUSH_INTERVAL = 1.0


class EvaluationLog:
    """
    A file of the model evaluations of a fit, written as it runs: the free
    values of each point the fitter asks about, with its numbers (see
    Merits). A fit that resumes from it (see resume) runs again
    with the same configuration and seed, so that its fitter asks for the
    same points: they are answered from the log while they match it
    exactly, then evaluated (and logged). The result is that of the fit
    had it not stopped; the replay costs only the fitter's own work.

    The fit problem logs the evaluations of fitters that ask for numbers
    (samplers, CMA-ES, the ranking of multistart). Least squares, which
    ask for residuals (too many to log), are not logged: they restart
    from their best point (e.g. best.yaml of the fit task).

    Beside it, '<name>-chain<ext>' holds the steps of the chain of an MCMC
    sampler (see record_chain), written as it runs; a resumed fit writes
    it again as its replay runs the chain again.

    The files: a first line naming the format, a line of JSON (the names
    of the free elements; of a chain, also its walkers), then rows of
    float64. A row cut short (a fit killed while writing it) is ignored.

    Parameters
    ----------
    filename : str
        The file, written anew.
    """

    def __init__(self, filename: str):
        self._filename = filename
        self._chain_filename = '{0}-chain{1}'.format(
            *os.path.splitext(filename))
        self._rows = np.empty((0, 0))
        self._header = 0
        self._next = 0
        self._replaying = False
        self._names: tuple[str, ...] | None = None
        self._file: BinaryIO | None = None
        self._chain_file: BinaryIO | None = None
        self._flushed = time.perf_counter()

    @classmethod
    def resume(cls, filename: str) -> 'EvaluationLog':
        """
        Return the log of a fit to resume: its evaluations are replayed,
        and the new ones are appended to it.

        Raises
        ------
        ValueError
            If the file is not a log of evaluations (of this version).
        """
        log = cls(filename)
        names, rows, header = _read(filename)
        log._names = names
        log._rows = rows.reshape(-1, len(names) + _NNUMBERS)
        log._header = header
        log._replaying = len(log._rows) > 0
        return log

    def filename(self) -> str:
        """Return the file of the log."""
        return self._filename

    def nreplayed(self) -> int:
        """Return the number of evaluations answered from the log so far."""
        return self._next

    def begin(self, names: Sequence[str]) -> None:
        """
        Open the files for a fit of the free elements of the given names
        (see FitProblem).

        Raises
        ------
        ValueError
            If the log of a fit to resume has other free elements.
        """
        names = tuple(names)
        if self._names is not None and self._names != names:
            raise ValueError(
                f"{self._filename}: the log is of a fit of other free "
                f"elements: {list(self._names)}, not {list(names)}")
        self._names = names
        if self._replaying:
            self._file = open(self._filename, 'r+b')
        else:
            self._file = open(self._filename, 'wb')
            self._file.write(_MAGIC)
            self._file.write(json.dumps(dict(names=names)).encode() + b'\n')
            self._header = self._file.tell()
        self._chain_file = None

    def replay(self, x: np.ndarray) -> Merits | None:
        """
        Return the numbers of the next evaluation of the log, if it is at
        the given free values; else None, and the replay ends (the rest of
        the log is dropped).
        """
        if not self._replaying:
            return None
        if self._next < len(self._rows):
            row = self._rows[self._next]
            if np.array_equal(row[:-_NNUMBERS], x):
                self._next += 1
                return Merits(*row[-_NNUMBERS:].tolist())
            _log.warning(
                f"{self._filename}: the fit asked for another point than "
                f"the log at evaluation {self._next + 1} (of "
                f"{len(self._rows)} logged): it goes on from there without "
                f"the log")
        else:
            _log.info(
                f"{self._filename}: {self._next} evaluations replayed; the "
                f"fit goes on")
        self._end_replay()
        return None

    def append(self, x: np.ndarray, merits: Merits) -> None:
        """Log an evaluation."""
        assert self._file is not None and not self._replaying
        self._file.write(np.concatenate(
            [np.asarray(x, dtype=np.float64), merits]).tobytes())
        self._flush_now_and_then()

    def record_chain(
            self,
            positions: np.ndarray,
            log_probs: np.ndarray
    ) -> None:
        """
        Write a step of the chain of an MCMC sampler: the position of each
        walker (one per row) and its log-probability.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if self._chain_file is None:
            self._chain_file = open(self._chain_filename, 'wb')
            self._chain_file.write(_MAGIC)
            self._chain_file.write(json.dumps(dict(
                names=self._names, nwalkers=len(positions))).encode() + b'\n')
        self._chain_file.write(np.column_stack(
            [positions, np.asarray(log_probs, dtype=np.float64)]).tobytes())
        self._flush_now_and_then()

    def close(self) -> None:
        """Flush and close the files."""
        for file in (self._file, self._chain_file):
            if file is not None:
                file.close()
        self._file = self._chain_file = None

    def _end_replay(self) -> None:
        """Drop the rest of the log, and append to it from here."""
        assert self._file is not None
        self._replaying = False
        self._file.seek(self._header + self._next * self._rows.shape[1] * 8)
        self._file.truncate()

    def _flush_now_and_then(self) -> None:
        """Flush the files if the last flush is old enough."""
        now = time.perf_counter()
        if now - self._flushed >= _FLUSH_INTERVAL:
            for file in (self._file, self._chain_file):
                if file is not None:
                    file.flush()
            self._flushed = now


def _read(filename: str) -> tuple[tuple[str, ...], np.ndarray, int]:
    """
    Return the names of the free elements of a log, its values (the
    complete rows, flattened), and the size of its header (bytes).
    """
    with open(filename, 'rb') as file:
        if file.readline() != _MAGIC:
            raise ValueError(
                f"{filename} is not a log of evaluations of this version of "
                f"GBKFIT")
        names = tuple(json.loads(file.readline())['names'])
        header = file.tell()
        values = file.read()
    width = len(names) + _NNUMBERS
    values = np.frombuffer(values[:len(values) // 8 * 8], dtype=np.float64)
    return names, values[:len(values) // width * width], header
