"""
The information that a fit gains on a parameter: the Kullback-Leibler
divergence of its posterior from its prior (as geko reports).
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy.stats


__all__ = [
    'information_gain'
]


class _LogDensity(Protocol):
    """A prior: anything with the log of its density (e.g. fitting.Prior)."""

    def log_pdf(self, x: npt.ArrayLike) -> np.ndarray: ...


def information_gain(
        samples: npt.ArrayLike,
        prior: _LogDensity,
        weights: npt.ArrayLike | None = None
) -> float:
    """
    Return the information gained on a parameter, the Kullback-Leibler
    divergence D(posterior || prior) = E_posterior[ln p(x) - ln pi(x)]
    (nats; bits: divided by ln 2): 0 if the data teach nothing, ln(range
    / width) roughly of a posterior much narrower than its prior. The
    posterior density p is a Gaussian kernel density estimate of the
    samples (Scott's bandwidth; it rounds sharp edges, e.g. a posterior
    against the bound of a uniform prior).

    Parameters
    ----------
    samples : ArrayLike
        The posterior samples of the parameter.
    prior : Prior
        Its prior (anything with log_pdf, e.g. gbkfit.fitting.Prior).
    weights : ArrayLike, optional
        The weights of the samples (of nested samplers).

    Returns
    -------
    float
        The information gain (nats).

    Raises
    ------
    ValueError
        If there are fewer than two samples, or they do not vary.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    weights = None if weights is None else np.asarray(weights, float).ravel()
    if len(samples) < 2 or np.ptp(samples) == 0:
        raise ValueError(
            "the information gain needs at least two different samples")
    density = scipy.stats.gaussian_kde(samples, weights=weights)
    gain = np.log(density(samples)) - np.asarray(
        prior.log_pdf(samples), dtype=float)
    return float(np.average(gain, weights=weights))
