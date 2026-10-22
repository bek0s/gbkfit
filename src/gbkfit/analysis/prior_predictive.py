"""
Prior predictive checks (as geko's): data that the models give at
parameters drawn from their priors, against the data, before a fit.
"""

import dataclasses

import numpy as np

from gbkfit.fitting import FitParamPrior
from gbkfit.objective import Objective
from gbkfit.params import ParamEvaluationError, ParamSpace


__all__ = [
    'PriorPredictive',
    'prior_predictive'
]


@dataclasses.dataclass(frozen=True)
class PriorPredictive:
    """
    A prior predictive check (see prior_predictive).

    Attributes
    ----------
    chi2 : np.ndarray
        The chi-square of the models at each draw (inf where the
        parameters cannot be evaluated).
    bands : list of dict of str to np.ndarray
        Of each observation, by data item: the 5th, 50th and 95th
        percentiles of the data the draws predict at each value (3, *the
        shape of the data).
    coverage : list of dict of str to float
        Of each observation, by data item: the share of the values of the
        data within the 5-95 band (low: the priors do not reach the data).
    """
    chi2: np.ndarray
    bands: list[dict[str, np.ndarray]]
    coverage: list[dict[str, float]]


def prior_predictive(
        objective: Objective,
        params: ParamSpace,
        ndraws: int = 100,
        seed: int | None = None
) -> PriorPredictive:
    """
    Return a prior predictive check of a fit before it runs: the models at
    ndraws values of the free elements drawn from their priors (their
    free properties, as samplers read them: see FitParamPrior), each with
    noise of the errors of the data (none without errors), as data they
    predict; their percentiles at each value of the data, and the share of
    the data within them.

    Parameters
    ----------
    objective : Objective
        The objective of the fit (its data, and the models of its
        observations).
    params : ParamSpace
        Its parameters, each free element with a prior (or min and max).
    ndraws : int, optional
        The number of draws.
    seed : int, optional
        The seed of the random numbers.

    Returns
    -------
    PriorPredictive
        The check.

    Raises
    ------
    ConfigError
        If a free element has no prior.
    ValueError
        If no draw can be evaluated.
    """
    priors = [FitParamPrior.load(info).prior()
              for info in params.free_properties().values()]
    rng = np.random.default_rng(seed)
    datasets = objective.datasets()
    chi2 = np.full(ndraws, np.inf)
    predicted: list[dict[str, list[np.ndarray]]] = [{} for _ in datasets]
    for k in range(ndraws):
        x = np.array([float(p.ppf(rng.uniform())) for p in priors])
        try:
            values = params.evaluate(x)
        except ParamEvaluationError:
            continue
        chi2[k] = objective.chi2(values)
        models = objective.group().model_h(values)
        for i, (dataset, model) in enumerate(zip(datasets, models)):
            for key, item in model.items():
                error = dataset[key].error()
                noise = 0 if error is None else rng.normal(0, 1, error.shape)
                predicted[i].setdefault(key, []).append(
                    item['d'] + noise * (0 if error is None else error))
    if not np.isfinite(chi2).any():
        raise ValueError("the parameters cannot be evaluated at any draw")
    bands, coverage = [], []
    for dataset, items in zip(datasets, predicted):
        bands.append({
            key: np.nanpercentile(draws, [5, 50, 95], axis=0)
            for key, draws in items.items()})
        coverage.append({
            key: _coverage(dataset[key].data(), band)
            for key, band in bands[-1].items()})
    return PriorPredictive(chi2=chi2, bands=bands, coverage=coverage)


def _coverage(data: np.ndarray, band: np.ndarray) -> float:
    """Return the share of the finite data within the 5-95 band."""
    good = np.isfinite(data) & np.isfinite(band[0]) & np.isfinite(band[2])
    inside = (data[good] >= band[0][good]) & (data[good] <= band[2][good])
    return float(inside.mean()) if good.any() else float('nan')
