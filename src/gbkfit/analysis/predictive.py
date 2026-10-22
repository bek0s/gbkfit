"""
The predictive accuracy of fits, for comparing models: the expected log
predictive density of new data, estimated from the log-likelihood of
each data point under posterior samples, by leave-one-out cross
validation with Pareto-smoothed importance sampling (PSIS-LOO; Vehtari,
Gelman & Gabry 2017) or by WAIC (Watanabe 2010).
"""

import dataclasses
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.special

from gbkfit.objective import Objective
from gbkfit.params import ParamSpace


__all__ = [
    'PredictiveScore',
    'compare',
    'loo',
    'pointwise_log_likelihoods',
    'waic'
]


@dataclasses.dataclass(frozen=True)
class PredictiveScore:
    """
    The expected log predictive density of a fit (see loo and waic).

    Attributes
    ----------
    elpd : float
        The expected log predictive density of all points (higher:
        better predictions).
    elpd_error : float
        Its standard error.
    p_eff : float
        The effective number of parameters (lppd - elpd).
    pointwise : np.ndarray
        The elpd of each point (or block of points).
    pareto_k : np.ndarray or None
        Of PSIS-LOO, the shape of the Pareto tail of the importance
        ratios of each point: its elpd is reliable below 0.7.
    """
    elpd: float
    elpd_error: float
    p_eff: float
    pointwise: np.ndarray
    pareto_k: np.ndarray | None = None


def pointwise_log_likelihoods(
        objective: Objective,
        params: ParamSpace,
        samples: npt.ArrayLike
) -> np.ndarray:
    """
    Return the log-likelihood of each data point under each posterior
    sample (samples, points; see Objective.pointwise_log_likelihood): one
    evaluation of the models per sample.

    Parameters
    ----------
    objective : Objective
        The objective of the fit.
    params : ParamSpace
        Its parameters.
    samples : ArrayLike
        The free values of the samples, one per row (of equal weight:
        resample those of nested samplers by their weights).

    Returns
    -------
    np.ndarray
        The log-likelihoods.
    """
    return np.array([
        objective.pointwise_log_likelihood(params.evaluate(x))
        for x in np.atleast_2d(np.asarray(samples, dtype=float))])


def waic(
        log_likelihoods: npt.ArrayLike,
        blocks: npt.ArrayLike | None = None
) -> PredictiveScore:
    """
    Return the WAIC estimate of the expected log predictive density of a
    fit: of each point, log of the mean of its likelihood over the
    samples, less the variance of its log-likelihood.

    Parameters
    ----------
    log_likelihoods : ArrayLike
        The log-likelihood of each point under each sample (samples,
        points; see pointwise_log_likelihoods).
    blocks : ArrayLike, optional
        The block of each point (integers from 0): the points of a block
        count as one (e.g. pixels that a PSF correlates).

    Returns
    -------
    PredictiveScore
        The score.
    """
    ll = _blocked(log_likelihoods, blocks)
    lppd = scipy.special.logsumexp(ll, axis=0) - np.log(len(ll))
    pointwise = lppd - np.var(ll, axis=0, ddof=1)
    return _score(pointwise, lppd, None)


def loo(
        log_likelihoods: npt.ArrayLike,
        blocks: npt.ArrayLike | None = None
) -> PredictiveScore:
    """
    Return the PSIS-LOO estimate of the expected log predictive density
    of a fit (Vehtari, Gelman & Gabry 2017): of each point, its likelihood
    averaged over the samples with the importance ratios that leave it
    out, 1 / p(y_i | theta), whose largest values (the larger of 20% and
    3 sqrt(S) of the S samples) are replaced by the order statistics of
    the generalised Pareto distribution fitted to them (Zhang & Stephens
    2009, with the weak prior of Vehtari et al.), none above the largest
    ratio. The shape k of each fit tells how reliable the estimate of the
    point is (below 0.7).

    Parameters
    ----------
    log_likelihoods : ArrayLike
        The log-likelihood of each point under each sample (samples,
        points; see pointwise_log_likelihoods).
    blocks : ArrayLike, optional
        The block of each point (see waic): blocks are left out whole.

    Returns
    -------
    PredictiveScore
        The score.
    """
    ll = _blocked(log_likelihoods, blocks)
    nsamples = len(ll)
    lppd = scipy.special.logsumexp(ll, axis=0) - np.log(nsamples)
    tail = int(np.ceil(min(0.2 * nsamples, 3 * np.sqrt(nsamples))))
    pointwise = np.empty(ll.shape[1])
    pareto_k = np.empty(ll.shape[1])
    for i in range(ll.shape[1]):
        log_weights, pareto_k[i] = _smoothed_log_weights(-ll[:, i], tail)
        pointwise[i] = scipy.special.logsumexp(log_weights + ll[:, i])
    return _score(pointwise, lppd, pareto_k)


def compare(a: PredictiveScore, b: PredictiveScore) -> tuple[float, float]:
    """
    Return the difference of the elpd of two fits of the same data points
    (a - b; positive: a predicts better), and its standard error, from
    the differences of their points.

    Raises
    ------
    ValueError
        If the fits are of different numbers of points.
    """
    if a.pointwise.shape != b.pointwise.shape:
        raise ValueError("the fits must be of the same data points")
    difference = a.pointwise - b.pointwise
    return float(difference.sum()), _error(difference)


def _blocked(
        log_likelihoods: npt.ArrayLike,
        blocks: npt.ArrayLike | None
) -> np.ndarray:
    """
    Return the log-likelihoods (samples, points), summed over the points
    of each block (if any).
    """
    ll = np.atleast_2d(np.asarray(log_likelihoods, dtype=float))
    if blocks is None:
        return ll
    blocks = np.asarray(blocks, dtype=int)
    summed = np.zeros((len(ll), blocks.max() + 1))
    np.add.at(summed, (slice(None), blocks), ll)
    return summed


def _score(
        pointwise: np.ndarray,
        lppd: np.ndarray,
        pareto_k: np.ndarray | None
) -> PredictiveScore:
    """Return the score of the elpd and the lppd of each point."""
    return PredictiveScore(
        elpd=float(pointwise.sum()), elpd_error=_error(pointwise),
        p_eff=float(np.sum(lppd - pointwise)), pointwise=pointwise,
        pareto_k=pareto_k)


def _error(pointwise: np.ndarray) -> float:
    """Return the standard error of the sum of values of points."""
    return float(np.sqrt(len(pointwise) * np.var(pointwise, ddof=1)))


def _smoothed_log_weights(
        log_ratios: np.ndarray, tail: int) -> tuple[np.ndarray, float]:
    """
    Return the Pareto-smoothed log importance weights of log ratios
    (normalised; see loo), and the shape k of the fit of their tail (inf
    if the tail is too short to fit).
    """
    log_ratios = log_ratios - log_ratios.max()
    order = np.argsort(log_ratios)
    cutoff = max(log_ratios[order[-tail - 1]], np.log(np.finfo(float).tiny))
    in_tail = np.flatnonzero(log_ratios > cutoff)
    k = np.inf
    if len(in_tail) > 4:
        in_tail = in_tail[np.argsort(log_ratios[in_tail])]
        exceedances = np.exp(log_ratios[in_tail]) - np.exp(cutoff)
        k, sigma = _fit_generalised_pareto(exceedances)
        if np.isfinite(k):
            p = (np.arange(len(in_tail)) + 0.5) / len(in_tail)
            quantiles = sigma * np.expm1(-k * np.log1p(-p)) / k \
                if k != 0 else -sigma * np.log1p(-p)
            log_ratios[in_tail] = np.log(quantiles + np.exp(cutoff))
            # (none above the largest ratio)
            log_ratios = np.minimum(log_ratios, 0)
    return log_ratios - scipy.special.logsumexp(log_ratios), float(k)


def _fit_generalised_pareto(x: Sequence[float]) -> tuple[float, float]:
    """
    Return the shape k and the scale sigma of the generalised Pareto
    distribution of the sorted positive values x, by the empirical Bayes
    estimate of Zhang & Stephens (2009), with k shrunk towards 0.5 as by
    a prior of 10 values (Vehtari et al. 2017).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = 30 + int(np.sqrt(n))
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (3 * x[int(n / 4 + 0.5) - 1]) + 1 / x[-1]
    k = np.mean(np.log1p(-b[:, None] * x), axis=1)
    log_likelihood = n * (np.log(-b / k) - k - 1)
    weights = 1 / np.sum(
        np.exp(log_likelihood[None, :] - log_likelihood[:, None]), axis=1)
    weights /= weights.sum()
    b_post = np.sum(b * weights)
    k_post = np.mean(np.log1p(-b_post * x))
    sigma = -k_post / b_post
    return (n * k_post + 10 * 0.5) / (n + 10), float(sigma)
