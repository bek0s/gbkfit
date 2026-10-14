"""
The problem that fitters solve: the objective as functions of the free
values.
"""

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from gbkfit.fitting.priors import Prior
from gbkfit.objective import Objective
from gbkfit.params import ParamEvaluationError, ParamSpace, ParamValues
from gbkfit.utils import threadutils


__all__ = [
    'FitProblem'
]


class FitProblem:
    """
    The objective of a fit as functions of the free values: a vector x
    in the order of ParamSpace.free_properties. Fitters call these
    functions; the problem counts the model evaluations and keeps the
    best point seen.

    Values for which the parameters cannot be evaluated (e.g. outside
    their valid range; see ParamEvaluationError) give a chi-square of inf
    and a log-likelihood of -inf; residuals raises the error.

    The batch functions take several points (one per row). They evaluate
    the models one after the other, for now.

    Parameters
    ----------
    objective : Objective
        The objective.
    params : ParamSpace
        The parameters of the models of the objective.
    priors : Sequence of Prior, optional
        The prior of each free element, in order.

    Raises
    ------
    ValueError
        If the parameter space lacks parameters of the objective, nothing
        is free, or the number of priors is wrong.
    """

    def __init__(
            self,
            objective: Objective,
            params: ParamSpace,
            priors: Sequence[Prior] | None = None
    ):
        if missing := set(objective.pdescs()) - set(params.pdescs()):
            raise ValueError(
                f"these parameters of the objective are missing from the "
                f"parameter space: {sorted(missing)}")
        names = tuple(params.free_properties())
        if not names:
            raise ValueError("no parameter is free")
        if priors is not None and len(priors) != len(names):
            raise ValueError(
                f"expected {len(names)} priors, one per free element, got "
                f"{len(priors)}")
        self._objective = objective
        self._params = params
        self._names = names
        self._priors = None if priors is None else tuple(priors)
        self._precision = max(
            float(np.finfo(obs.dtype()).eps)
            for obs in objective.group().observations())
        self._nevals = 0
        self._best_x: np.ndarray | None = None
        self._best_chi2 = math.inf

    def objective(self) -> Objective:
        """Return the objective."""
        return self._objective

    def params(self) -> ParamSpace:
        """Return the parameter space."""
        return self._params

    def names(self) -> tuple[str, ...]:
        """Return the names of the free elements, in order."""
        return self._names

    def ndim(self) -> int:
        """Return the number of free elements."""
        return len(self._names)

    def priors(self) -> tuple[Prior, ...] | None:
        """Return the priors of the free elements, if any."""
        return self._priors

    def precision(self) -> float:
        """
        Return the relative precision of the models: the machine epsilon
        of the least precise observation (e.g. 1.2e-7 for float32). It
        sets the steps of finite differences.
        """
        return self._precision

    def nevals(self) -> int:
        """Return the number of model evaluations so far."""
        return self._nevals

    def best(self) -> tuple[np.ndarray, float] | None:
        """Return the point with the lowest chi-square so far, and it."""
        if self._best_x is None:
            return None
        return self._best_x.copy(), self._best_chi2

    def evaluate(self, x: npt.ArrayLike) -> ParamValues:
        """
        Return the values of all parameters at the given free values.

        Raises
        ------
        ParamEvaluationError
            If the parameters cannot be evaluated there.
        """
        return self._params.evaluate(np.asarray(x, dtype=float))

    def residuals(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Return the weighted residuals at the given free values, in a new
        array (see Objective.residuals).

        Raises
        ------
        ParamEvaluationError
            If the parameters cannot be evaluated there.
        """
        residuals = self._objective.residuals(self.evaluate(x))
        # (a multithreaded dot product leaves the BLAS threads spinning
        # into the next evaluation)
        with threadutils.single_threaded_blas():
            cost = float(residuals @ residuals)
        self._record(x, cost)
        return residuals

    def chi2(self, x: npt.ArrayLike) -> float:
        """Return the chi-square at the given free values."""
        try:
            values = self.evaluate(x)
        except ParamEvaluationError:
            return math.inf
        chi2 = self._objective.chi2(values)
        self._record(x, chi2)
        return chi2

    def log_likelihood(self, x: npt.ArrayLike) -> float:
        """
        Return the log-likelihood at the given free values (see
        Objective.log_likelihood).
        """
        try:
            values = self.evaluate(x)
        except ParamEvaluationError:
            return -math.inf
        chi2, log_likelihood = self._objective.chi2_and_log_likelihood(
            values)
        self._record(x, chi2)
        return log_likelihood

    def log_prior(self, x: npt.ArrayLike) -> float:
        """
        Return the log of the prior density at the given free values.

        Raises
        ------
        RuntimeError
            If the problem has no priors.
        """
        priors = self._require_priors()
        return float(sum(
            p.log_pdf(v) for p, v in zip(priors, np.asarray(x, dtype=float))))

    def log_posterior(self, x: npt.ArrayLike) -> float:
        """
        Return the log of the posterior density (up to a constant) at the
        given free values. Points outside the support of the priors are
        not evaluated.
        """
        log_prior = self.log_prior(x)
        if not math.isfinite(log_prior):
            return -math.inf
        return log_prior + self.log_likelihood(x)

    def prior_transform(self, u: npt.ArrayLike) -> np.ndarray:
        """
        Return the free values of points of the unit cube, by the ppf of
        each prior (the transform of nested samplers).

        Parameters
        ----------
        u : ArrayLike
            One point, or one per row.

        Returns
        -------
        np.ndarray
            The free values, of the shape of u.

        Raises
        ------
        RuntimeError
            If the problem has no priors.
        """
        priors = self._require_priors()
        u = np.asarray(u, dtype=float)
        x = np.empty_like(u)
        for i, prior in enumerate(priors):
            x[..., i] = prior.ppf(u[..., i])
        return x

    def chi2_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """Return the chi-square of each point (one per row)."""
        return np.array([self.chi2(x) for x in np.atleast_2d(xs)])

    def log_likelihood_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """Return the log-likelihood of each point (one per row)."""
        return np.array(
            [self.log_likelihood(x) for x in np.atleast_2d(xs)])

    def log_posterior_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """Return the log-posterior of each point (one per row)."""
        return np.array([self.log_posterior(x) for x in np.atleast_2d(xs)])

    def _require_priors(self) -> tuple[Prior, ...]:
        """Return the priors, or raise RuntimeError if there are none."""
        if self._priors is None:
            raise RuntimeError("the problem has no priors")
        return self._priors

    def _record(self, x: npt.ArrayLike, chi2: float) -> None:
        """Count an evaluation, and keep the point if it is the best."""
        self._nevals += 1
        if chi2 < self._best_chi2:
            self._best_chi2 = chi2
            self._best_x = np.array(x, dtype=float)
