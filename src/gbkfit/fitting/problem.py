"""
The problem that fitters solve: the objective as functions of the free
values.
"""

import math
import time
import typing
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from gbkfit.fitting.control import FitControl, FitStopped
from gbkfit.fitting.evaluation_log import EvaluationLog
from gbkfit.fitting.priors import Prior
from gbkfit.fitting.progress import FitProgress, ProgressObserver
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

    Free elements may have priors: samplers need one for each, optimizers
    take any (then they minimise the chi-square plus the priors', see
    cost, and the residuals include those of the priors).

    The batch functions take several points (one per row). They evaluate
    the models one after the other, for now.

    The problem reports the progress of the fit to the given observers
    (see FitProgress): at most every interval seconds as it evaluates,
    when the stage changes (see set_stage), and at the end (see finish).

    Every evaluation raises FitStopped once the control (if any) is
    stopped. The numbers it evaluates (chi-squares, log-likelihoods, not
    residuals) go to the log (if any), and those of a fit that resumes
    come from it while it lasts (see EvaluationLog).

    Parameters
    ----------
    objective : Objective
        The objective.
    params : ParamSpace
        The parameters of the models of the objective.
    priors : Sequence of Prior or None, optional
        The prior of each free element, in order (None for an element
        without one).
    observers : Sequence of ProgressObserver, optional
        Who receives the progress of the fit.
    interval : float, optional
        The least time between two reports as it evaluates (s).
    control : FitControl, optional
        The control of the fit, to stop it.
    log : EvaluationLog, optional
        The log of the evaluations (of a fit that resumes, or a new one).

    Raises
    ------
    ValueError
        If the parameter space lacks parameters of the objective, nothing
        is free, the number of priors is wrong, or the log is of a fit of
        other free elements.
    """

    def __init__(
            self,
            objective: Objective,
            params: ParamSpace,
            priors: Sequence[Prior | None] | None = None,
            observers: Sequence[ProgressObserver] = (),
            interval: float = 2.0,
            control: FitControl | None = None,
            log: EvaluationLog | None = None
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
        if self._priors is not None and set(self._priors) == {None}:
            self._priors = None
        self._precision = max(
            float(np.finfo(obs.dtype()).eps)
            for obs in objective.group().observations())
        self._nevals = 0
        self._best_x: np.ndarray | None = None
        self._best_chi2 = math.inf
        self._observers = tuple(observers)
        self._interval = interval
        self._stage = 'fit'
        self._start = time.perf_counter()
        self._last_chi2 = math.inf
        self._last_report = self._start
        self._control = control
        self._log = log
        if log is not None:
            log.begin(names)

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

    def priors(self) -> tuple[Prior | None, ...] | None:
        """
        Return the priors of the free elements (None for one without),
        if any has one.
        """
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
        """
        Return the point with the lowest chi-square so far, and it (of
        optimizers: the lowest cost, see cost).
        """
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
        array (see Objective.residuals), followed by those of the priors
        (see prior_residuals).

        Raises
        ------
        ParamEvaluationError
            If the parameters cannot be evaluated there.
        TypeError
            If a prior has no residual (see Prior.has_residual).
        FitStopped
            If the fit was asked to stop.
        """
        self._check_stop()
        residuals = self._objective.residuals(self.evaluate(x))
        if self._priors is not None:
            residuals = np.concatenate(
                [residuals, self.prior_residuals(x, strict=True)])
        # (a multithreaded dot product leaves the BLAS threads spinning
        # into the next evaluation)
        with threadutils.single_threaded_blas():
            cost = float(residuals @ residuals)
        self._record(x, cost)
        return residuals

    def chi2(self, x: npt.ArrayLike) -> float:
        """Return the chi-square at the given free values."""
        evaluated = self._chi2_and_log_likelihood(x)
        if evaluated is None:
            return math.inf
        self._record(x, evaluated[0])
        return evaluated[0]

    def cost(self, x: npt.ArrayLike) -> float:
        """
        Return what optimizers minimise at the given free values: the
        chi-square plus that of the priors (see Prior.chi2; -2 log of the
        posterior, up to a constant), or the chi-square without priors.
        """
        evaluated = self._chi2_and_log_likelihood(x)
        if evaluated is None:
            return math.inf
        cost = evaluated[0] + self.prior_chi2(x)
        self._record(x, cost)
        return cost

    def prior_chi2(self, x: npt.ArrayLike) -> float:
        """
        Return the chi-square of the priors at the given free values (see
        Prior.chi2; 0 without priors).
        """
        return float(sum(
            prior.chi2(value) for prior, value in self._given_priors(x)))

    def prior_residuals(
            self,
            x: npt.ArrayLike,
            strict: bool = False
    ) -> np.ndarray:
        """
        Return the residuals of the priors that have them at the given
        free values (see Prior.residual; empty without priors), in the
        order of the elements; with strict, raise TypeError for a prior
        without one.
        """
        return np.array([
            prior.residual(value) for prior, value in self._given_priors(x)
            if strict or prior.has_residual()], dtype=float)

    def log_likelihood(self, x: npt.ArrayLike) -> float:
        """
        Return the log-likelihood at the given free values (see
        Objective.log_likelihood).
        """
        evaluated = self._chi2_and_log_likelihood(x)
        if evaluated is None:
            return -math.inf
        self._record(x, evaluated[0])
        return evaluated[1]

    def log_prior(self, x: npt.ArrayLike) -> float:
        """
        Return the log of the prior density at the given free values.

        Raises
        ------
        RuntimeError
            If the problem has no priors.
        """
        if self._priors is None:
            raise RuntimeError("the problem has no priors")
        return float(sum(
            prior.log_pdf(value) for prior, value in self._given_priors(x)))

    def log_posterior(self, x: npt.ArrayLike) -> float:
        """
        Return the log of the posterior density (up to a constant) at the
        given free values. Points outside the support of the priors are
        not evaluated.

        Raises
        ------
        FitStopped
            If the fit was asked to stop.
        """
        self._check_stop()
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

    def cost_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """Return the cost of each point (one per row; see cost)."""
        return np.array([self.cost(x) for x in np.atleast_2d(xs)])

    def log_likelihood_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """Return the log-likelihood of each point (one per row)."""
        return np.array(
            [self.log_likelihood(x) for x in np.atleast_2d(xs)])

    def log_posterior_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """Return the log-posterior of each point (one per row)."""
        return np.array([self.log_posterior(x) for x in np.atleast_2d(xs)])

    def _require_priors(self) -> tuple[Prior, ...]:
        """
        Return the priors, or raise RuntimeError unless every free
        element has one.
        """
        if self._priors is None or None in self._priors:
            raise RuntimeError("not every free element has a prior")
        return typing.cast(tuple[Prior, ...], self._priors)

    def _given_priors(
            self,
            x: npt.ArrayLike
    ) -> list[tuple[Prior, float]]:
        """Return the priors there are, with their free values."""
        if self._priors is None:
            return []
        return [
            (prior, value) for prior, value in zip(
                self._priors, np.asarray(x, dtype=float).tolist())
            if prior is not None]

    def record_chain(
            self,
            positions: np.ndarray,
            log_probs: np.ndarray
    ) -> None:
        """
        Write a step of the chain of an MCMC sampler to the log, if any
        (see EvaluationLog.record_chain).
        """
        if self._log is not None:
            self._log.record_chain(positions, log_probs)

    def close(self) -> None:
        """Close the log, if any."""
        if self._log is not None:
            self._log.close()

    def _check_stop(self) -> None:
        """Raise FitStopped if the fit was asked to stop."""
        if self._control is not None and self._control.stopped():
            raise FitStopped("the fit was asked to stop")

    def _chi2_and_log_likelihood(
            self,
            x: npt.ArrayLike
    ) -> tuple[float, float] | None:
        """
        Return the chi-square and log-likelihood at the given free values,
        or None where the parameters cannot be evaluated: from the log
        while a fit that resumes replays it, else from the objective (and
        logged, None as inf and -inf).

        Raises
        ------
        FitStopped
            If the fit was asked to stop.
        """
        self._check_stop()
        x = np.asarray(x, dtype=float)
        logged = None if self._log is None else self._log.replay(x)
        if logged is None:
            try:
                logged = self._objective.chi2_and_log_likelihood(
                    self.evaluate(x))
            except ParamEvaluationError:
                logged = math.inf, -math.inf
            if self._log is not None:
                self._log.append(x, *logged)
        if logged == (math.inf, -math.inf):
            return None
        return logged

    def set_stage(self, stage: str) -> None:
        """
        Set what the fit is doing (see FitProgress.stage), and report it.
        """
        self._stage = stage
        self._report()

    def finish(self) -> None:
        """Report the end of the fit."""
        self._report(final=True)

    def _record(self, x: npt.ArrayLike, chi2: float) -> None:
        """
        Count an evaluation, keep the point if it is the best, and report
        the progress if the interval has passed.
        """
        self._nevals += 1
        self._last_chi2 = chi2
        if chi2 < self._best_chi2:
            self._best_chi2 = chi2
            self._best_x = np.array(x, dtype=float)
        if self._observers \
                and time.perf_counter() - self._last_report >= self._interval:
            self._report()

    def _report(self, final: bool = False) -> None:
        """Report the progress to the observers."""
        now = time.perf_counter()
        self._last_report = now
        if not self._observers:
            return
        best = None if self._best_x is None \
            else dict(zip(self._names, self._best_x.tolist()))
        progress = FitProgress(
            self._stage, self._nevals, now - self._start, self._last_chi2,
            self._best_chi2, best, final)
        for observer in self._observers:
            observer(progress)
