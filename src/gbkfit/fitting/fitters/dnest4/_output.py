"""
The output of DNest4: its files, and the posterior and the evidence from
them (the postprocess of DNest4's Python package, dnest4/classic.py).
"""

import dataclasses
import os
import warnings

import numpy as np


__all__ = [
    'DNest4Files',
    'DNest4Output',
    'DNest4Posterior',
    'posterior'
]


@dataclasses.dataclass(frozen=True)
class DNest4Output:
    """
    What a run of DNest4 wrote: its levels (log X, log-likelihood,
    tiebreaker, accepts, tries, exceeds and visits of each, one per
    row), and the information (level, log-likelihood, tiebreaker,
    particle) and the coordinates of each saved particle (one per row;
    see the module _dnest4).
    """
    levels: np.ndarray
    info: np.ndarray
    samples: np.ndarray


@dataclasses.dataclass(frozen=True)
class DNest4Files:
    """The files of a run of DNest4, in a directory."""
    directory: str

    def sample(self) -> str:
        """Return the file of the coordinates of the saved particles."""
        return os.path.join(self.directory, 'sample.txt')

    def sample_info(self) -> str:
        """Return the file of the information of the saved particles."""
        return os.path.join(self.directory, 'sample_info.txt')

    def levels(self) -> str:
        """Return the file of the levels."""
        return os.path.join(self.directory, 'levels.txt')

    def read(self) -> DNest4Output:
        """Return what the files have (of a run done, or running)."""
        return DNest4Output(
            _load(self.levels()), _load(self.sample_info()),
            _load(self.sample()))


@dataclasses.dataclass(frozen=True)
class DNest4Posterior:
    """
    The posterior of the saved particles of DNest4: the weight of each
    (they sum to 1), the log of the evidence and its error, and the
    information (nats).
    """
    weights: np.ndarray
    log_z: float
    log_z_error: float
    information: float

    def ess(self) -> float:
        """Return the effective sample size of the weights."""
        w = self.weights[self.weights > 0]
        return float(np.exp(-np.sum(w * np.log(w))))


def posterior(
        output: DNest4Output,
        rng: np.random.Generator | None
) -> DNest4Posterior:
    """
    Return the posterior of the saved particles of DNest4, as its
    postprocess works it out: each particle lies between two levels,
    whose log X are known; the log X of the particles between them, in
    order of likelihood, are drawn uniformly in X (resampled 20 times,
    whose spread gives the error of the evidence; without rng, once, at
    even steps); and the prior mass of each particle is half the gap
    between its neighbours.
    """
    levels, info = output.levels, output.info
    log_l = info[:, 1]
    resamples = 1 if rng is None else 20
    log_p = np.empty((resamples, len(info)))
    above = _above(info, levels)
    for z in range(resamples):
        for i in range(len(levels)):
            which = np.flatnonzero(above == i)
            log_x_max = levels[i, 0]
            log_x_min = levels[i + 1, 0] if i < len(levels) - 1 else -np.inf
            u_min = np.exp(log_x_min - log_x_max)
            u = np.linspace(1, len(which), len(which)) / (len(which) + 1) \
                if rng is None else rng.uniform(size=len(which))
            # (in order of likelihood, decreasing log X)
            log_x = np.sort(log_x_max + np.log(u_min + (1 - u_min) * u))[::-1]
            right = np.concatenate([[log_x_max], log_x[:-1]])
            left = np.concatenate([log_x[1:], [log_x_min]])
            order = which[np.lexsort((info[which, 2], log_l[which]))]
            log_p[z, order] = np.log(0.5) + _log_diff_exp(right, left)
    log_p -= _log_sum_exp(log_p)
    log_post = log_p + log_l
    log_z = _log_sum_exp(log_post)
    p = np.exp(log_post - log_z)
    information = -log_z[:, 0] + np.sum(p * log_l, axis=1)
    weights = p.mean(axis=0)
    return DNest4Posterior(
        weights=weights / weights.sum(),
        log_z=float(log_z.mean()),
        log_z_error=float(log_z.std()),
        information=float(information.mean()))


def _above(info: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """
    Return the highest level that each saved particle reaches (its
    log-likelihood, then its tiebreaker, at least the level's).
    """
    log_l, tie = info[:, 1:2], info[:, 2:3]
    level_log_l, level_tie = levels[1:, 1], levels[1:, 2]
    reaches = (log_l > level_log_l) \
        | ((log_l == level_log_l) & (tie >= level_tie))
    return np.maximum(info[:, 0].astype(int), reaches.sum(axis=1))


def _log_sum_exp(values: np.ndarray) -> np.ndarray:
    """Return the log of the sum of the exp of each row (kept as column)."""
    biggest = values.max(axis=1, keepdims=True)
    return np.log(np.exp(values - biggest).sum(axis=1, keepdims=True)) \
        + biggest


def _log_diff_exp(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Return log(exp(x1) - exp(x2)), for x1 >= x2."""
    with np.errstate(divide='ignore'):
        return x1 + np.log1p(-np.exp(x2 - x1))


def _load(path: str) -> np.ndarray:
    """
    Return the numbers of a file of DNest4, one row per line (none if
    it has none yet).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        rows = np.loadtxt(path, ndmin=2)
    return rows
