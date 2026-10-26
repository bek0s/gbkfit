"""
The problem that fitters solve: the objective as functions of the free
values.
"""

import logging
import math
import time
import typing
from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from gbkfit.fitting.control import FitControl, FitStopped
from gbkfit.fitting.evaluation_log import EvaluationLog
from gbkfit.params.joint_priors import joint_prior_residuals
from gbkfit.fitting.pool import EvaluationPool
from gbkfit.params.priors import Prior
from gbkfit.fitting.progress import FitProgress, ProgressObserver
from gbkfit.objective import Merits, Objective
from gbkfit.params import (
    ModelEvaluationError, ParamEvaluationError, ParamSpace, ParamValues)
from gbkfit.utils import threadutils


_log = logging.getLogger(__name__)


__all__ = [
    'MERITS',
    'FitProblem'
]


_T = TypeVar('_T')

# What optimizers may minimise (see FitProblem.cost): the fields of Merits
MERITS = ('chi2', 'aicp')

# The numbers of a point whose parameters cannot be evaluated
_UNEVALUATED = Merits(math.inf, -math.inf, math.inf)


class FitProblem:
    """
    The objective of a fit as functions of the free values: a vector x
    in the order of ParamSpace.free_properties. Fitters call these
    functions; the problem counts the model evaluations and keeps the
    best point seen.

    Values for which the parameters cannot be evaluated (e.g. outside
    their valid range; see ParamEvaluationError) give a chi-square of inf
    and a log-likelihood of -inf; residuals raises the error. So do the
    values for which the priors on several parameters cannot be (e.g.
    where a derived value that they read is not finite; see
    ParamSpace.expression): a cost of inf, a log-prior and what nested
    samplers integrate of -inf.

    Optimizers minimise its merit (see cost): the chi-square, or the AICp
    (see Objective.aicp; e.g. of models whose solved values are
    regularised). Free elements may have priors: samplers need one for
    each, optimizers take any (then they minimise the merit plus the
    chi-square of the priors, see cost, and the residuals include those
    of the priors). The parameters may also have priors on several of
    them at once (see ParamSpace and JointPrior), which add to the priors
    of the elements the same way.

    The batch functions take several points (one per row). With a pool
    (see EvaluationPool), they evaluate them at once, on copies of the
    objective; the problem counts, keeps, logs and reports them in their
    order, as without one.

    The problem reports the progress of the fit to the given observers
    (see FitProgress): at most every interval seconds as it evaluates,
    when the stage changes (see set_stage), and at the end (see finish).
    An observer that raises is logged and dropped: the fit goes on.

    A model evaluation whose chi-square or log-likelihood is not a number
    (e.g. of a model that is NaN where the data have values) counts as
    one whose parameters cannot be evaluated (inf and -inf); residuals
    with NaN go to the fitter as they are (least-squares libraries reject
    or refuse them). Both are counted (see nan_nevals), and the first is
    logged.

    Every model evaluation raises FitStopped once the control (if any) is
    stopped: those of a batch before it count, without a pool (with one,
    none of the batch do). The numbers it evaluates (see Merits; not
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
    pool : EvaluationPool, optional
        Copies of the objective that evaluate the batches at once.
    merit : {'chi2', 'aicp'}, optional
        What optimizers minimise (see cost).

    Raises
    ------
    ValueError
        If the parameter space lacks parameters of the objective, the
        objective does not solve its solved elements (see
        Objective.solves), nothing is free, the number of priors is
        wrong, the log is of a fit of other free elements, the pool is of
        another objective, or the merit is unknown.
    ConfigError
        If a prior on several parameters is invalid.
    """

    def __init__(
            self,
            objective: Objective,
            params: ParamSpace,
            priors: Sequence[Prior | None] | None = None,
            observers: Sequence[ProgressObserver] = (),
            interval: float = 2.0,
            control: FitControl | None = None,
            log: EvaluationLog | None = None,
            pool: EvaluationPool | None = None,
            merit: str = 'chi2'
    ):
        if missing := set(objective.pdescs()) - set(params.pdescs()):
            raise ValueError(
                f"these parameters of the objective are missing from the "
                f"parameter space: {sorted(missing)}")
        if objective.solves() != params.solves():
            raise ValueError(
                "the objective must solve the solved elements of the "
                "parameters, with their solves: make it with "
                "Objective(group_plan, params.solves())")
        names = tuple(params.free_properties())
        if not names:
            raise ValueError("no parameter is free")
        if priors is not None and len(priors) != len(names):
            raise ValueError(
                f"expected {len(names)} priors, one per free element, got "
                f"{len(priors)}")
        if pool is not None and pool.objective() is not objective:
            raise ValueError("the pool is of copies of another objective")
        if merit not in MERITS:
            raise ValueError(
                f"the merit must be one of {list(MERITS)}; it is {merit!r}")
        self._objective = objective
        self._params = params
        self._names = names
        self._priors = None if priors is None else tuple(priors)
        if self._priors is not None and set(self._priors) == {None}:
            self._priors = None
        self._joint = joint_prior_residuals(params)
        self._precision = max(
            float(np.finfo(obs.dtype()).eps)
            for obs in objective.group().observations())
        self._nevals = 0
        self._nan_nevals = 0
        self._failed_nevals = 0
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
        self._pool = pool
        self._merit = merit
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

    def merit(self) -> str:
        """Return what optimizers minimise (see cost)."""
        return self._merit

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

    def nan_nevals(self) -> int:
        """
        Return the number of model evaluations so far whose chi-square
        was not a number.
        """
        return self._nan_nevals

    def failed_nevals(self) -> int:
        """
        Return the number of model evaluations so far where the model or a
        solve failed (see ModelEvaluationError).
        """
        return self._failed_nevals

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
        return self.residuals_batch(np.atleast_2d(x))[0]

    def residuals_batch(self, xs: npt.ArrayLike) -> list[np.ndarray]:
        """
        Return the residuals at each point (one per row; see residuals).
        """
        xs = _points(xs)
        values = [self.evaluate(x) for x in xs]
        evaluated = self.map_objective(Objective.residuals, values)
        batch = []
        for x, residuals in zip(xs, evaluated):
            if self._priors is not None or self._joint:
                residuals = np.concatenate(
                    [residuals, self.prior_residuals(x, strict=True)])
            # (a multithreaded dot product leaves the BLAS threads
            # spinning into the next evaluation)
            with threadutils.single_threaded_blas():
                cost = float(residuals @ residuals)
            if math.isnan(cost):
                self._count_nan(x)
            self._record(x, cost)
            batch.append(residuals)
        return batch

    def chi2(self, x: npt.ArrayLike) -> float:
        """Return the chi-square at the given free values."""
        return float(self.chi2_batch(np.atleast_2d(x))[0])

    def cost(self, x: npt.ArrayLike) -> float:
        """
        Return what optimizers minimise at the given free values: the merit
        (see Merits: the chi-square, or the AICp) plus the chi-square of
        the priors (see Prior.chi2; with the chi-square, -2 log of the
        posterior, up to a constant), or the merit without priors.
        """
        return float(self.cost_batch(np.atleast_2d(x))[0])

    def prior_chi2(self, x: npt.ArrayLike) -> float:
        """
        Return the chi-square of the priors at the given free values (see
        Prior.chi2), with that of the priors on several parameters (the
        sum of the squares of their residuals; see JointPrior); 0 without
        priors.

        Raises
        ------
        ParamEvaluationError
            If there are priors on several parameters, and the parameters
            cannot be evaluated there.
        """
        joint = self._joint_residuals(x)
        return float(sum(
            prior.chi2(value) for prior, value in self._given_priors(x))
            + joint @ joint)

    def prior_residuals(
            self,
            x: npt.ArrayLike,
            strict: bool = False
    ) -> np.ndarray:
        """
        Return the residuals of the priors that have them at the given
        free values (see Prior.residual), in the order of the elements,
        then those of the priors on several parameters (see JointPrior);
        empty without priors. With strict, raise TypeError for a prior
        without one.

        Raises
        ------
        ParamEvaluationError
            If there are priors on several parameters, and the parameters
            cannot be evaluated there.
        """
        return np.concatenate([np.array([
            prior.residual(value) for prior, value in self._given_priors(x)
            if strict or prior.has_residual()], dtype=float),
            self._joint_residuals(x)])

    def log_likelihood(self, x: npt.ArrayLike) -> float:
        """
        Return the log-likelihood at the given free values (see
        Objective.log_likelihood).
        """
        return float(self.log_likelihood_batch(np.atleast_2d(x))[0])

    def log_prior(self, x: npt.ArrayLike) -> float:
        """
        Return the log of the prior density at the given free values: with
        priors on several parameters, up to a constant (their -chi2 / 2;
        see JointPrior); -inf where the parameters cannot be evaluated.

        Raises
        ------
        RuntimeError
            If the problem has no priors.
        """
        if self._priors is None:
            raise RuntimeError("the problem has no priors")
        log_prior = float(sum(
            prior.log_pdf(value) for prior, value in self._given_priors(x)))
        if not (self._joint and math.isfinite(log_prior)):
            return log_prior
        try:
            joint = self._joint_residuals(x)
        except ParamEvaluationError:
            return -math.inf
        return log_prior - 0.5 * float(joint @ joint)

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
        return float(self.log_posterior_batch(np.atleast_2d(x))[0])

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
        chi2 = []
        for x, merits in self._merits(_points(xs)):
            if merits is None:
                chi2.append(math.inf)
                continue
            self._record(x, merits.chi2)
            chi2.append(merits.chi2)
        return np.array(chi2)

    def cost_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """Return the cost of each point (one per row; see cost)."""
        costs = []
        for x, merits in self._merits(_points(xs)):
            if merits is None:
                costs.append(math.inf)
                continue
            try:
                cost = getattr(merits, self._merit) + self.prior_chi2(x)
            except ParamEvaluationError:
                cost = math.inf
            self._record(x, cost)
            costs.append(cost)
        return np.array(costs)

    def log_likelihood_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """Return the log-likelihood of each point (one per row)."""
        log_likelihoods = []
        for x, merits in self._merits(_points(xs)):
            if merits is None:
                log_likelihoods.append(-math.inf)
                continue
            self._record(x, merits.chi2)
            log_likelihoods.append(merits.log_likelihood)
        return np.array(log_likelihoods)

    def log_posterior_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """
        Return the log-posterior of each point (one per row; see
        log_posterior).
        """
        self._check_stop()
        xs = _points(xs)
        log_prior = np.array([self.log_prior(x) for x in xs])
        inside = np.isfinite(log_prior)
        log_posterior = np.full(len(xs), -math.inf)
        log_posterior[inside] = \
            log_prior[inside] + self.log_likelihood_batch(xs[inside])
        return log_posterior

    def has_joint_priors(self) -> bool:
        """
        Return whether the parameters have priors on several of them (see
        JointPrior).
        """
        return bool(self._joint)

    def nested_log_likelihood(self, x: npt.ArrayLike) -> float:
        """
        Return what nested samplers integrate over the priors of the
        elements at the given free values (see
        nested_log_likelihood_batch).
        """
        return float(self.nested_log_likelihood_batch(np.atleast_2d(x))[0])

    def nested_log_likelihood_batch(self, xs: npt.ArrayLike) -> np.ndarray:
        """
        Return what nested samplers integrate over the priors of the
        elements (see prior_transform) at each point (one per row): the
        log-likelihood (see log_likelihood_batch), plus the log of the
        priors on several parameters, -chi2 / 2 (see JointPrior), which a
        prior transform cannot draw from (it draws each element alone).

        Their evidence is then that of these priors unnormalised, C times
        the evidence: see joint_prior_log_density_batch and
        fitters._detail.corrected_evidence.
        """
        xs = _points(xs)
        log_likelihood = self.log_likelihood_batch(xs)
        if self._joint:
            for i, x in enumerate(xs):
                if np.isfinite(log_likelihood[i]):
                    log_likelihood[i] += self.joint_prior_log_density(x)
        return log_likelihood

    def joint_prior_log_density(self, x: npt.ArrayLike) -> float:
        """
        Return the log of the priors on several parameters at the given
        free values (see joint_prior_log_density_batch).
        """
        return float(self.joint_prior_log_density_batch(
            np.atleast_2d(x))[0])

    def joint_prior_log_density_batch(
            self,
            xs: npt.ArrayLike
    ) -> np.ndarray:
        """
        Return the log of the priors on several parameters, -chi2 / 2 (see
        JointPrior), at each point (one per row; -inf where the parameters
        or the priors cannot be evaluated), without the models of the
        observations (with the functions of the models that the derived
        values they read take, see ParamSpace.derived_values): nested
        samplers integrate it over the priors of the elements, which gives
        the normalisation that their evidence lacks (see
        nested_log_likelihood_batch). It does not count, keep or log the
        points.
        """
        log_density = []
        for x in _points(xs):
            try:
                joint = self._joint_residuals(x)
            except ParamEvaluationError:
                log_density.append(-math.inf)
                continue
            log_density.append(-0.5 * float(joint @ joint))
        return np.array(log_density)

    def map_objective(
            self,
            function: Callable[[Objective, ParamValues], _T],
            values: Sequence[ParamValues]
    ) -> Iterator[_T]:
        """
        Yield function(objective, v) for each of the given parameter
        values v (see evaluate), in order: with a pool, on its copies of
        the objective at once (see EvaluationPool); else on the objective,
        one after the other as they are taken. It does not count, keep or
        log the evaluations (the other functions do).

        Raises
        ------
        FitStopped
            At an evaluation, if the fit was asked to stop.
        """
        function = self._stoppable(function)
        if self._pool is not None:
            yield from self._pool.map(function, values)
        else:
            for v in values:
                yield function(self._objective, v)

    def _require_priors(self) -> tuple[Prior, ...]:
        """
        Return the priors, or raise RuntimeError unless every free
        element has one.
        """
        if self._priors is None or None in self._priors:
            raise RuntimeError("not every free element has a prior")
        return typing.cast(tuple[Prior, ...], self._priors)

    def _joint_residuals(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Return the residuals of the priors on several parameters at the
        given free values (empty without them).
        """
        if not self._joint:
            return np.empty(0)
        values = self.evaluate(x)
        return np.concatenate([residuals(values) for residuals in self._joint])

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

    def _stoppable(
            self,
            function: Callable[[Objective, ParamValues], _T]
    ) -> Callable[[Objective, ParamValues], _T]:
        """
        Return the function of an evaluation, which raises FitStopped first
        if the fit was asked to stop.
        """
        def stoppable(objective: Objective, values: ParamValues) -> _T:
            self._check_stop()
            return function(objective, values)
        return stoppable

    def _merits(
            self,
            xs: np.ndarray
    ) -> Iterator[tuple[np.ndarray, Merits | None]]:
        """
        Yield each point (one per row) with its numbers (see Merits), or
        None where the parameters cannot be evaluated, the model or a solve
        fails (counted; see failed_nevals), or the numbers are not
        numbers: from the log while a fit that resumes replays it,
        else from the objective (see map_objective), and logged (None as
        infinities, numbers that are not as they are).

        Raises
        ------
        FitStopped
            If the fit was asked to stop.
        """
        # (the log replays the first points of a batch at most: its replay
        # ends at the first point it does not have)
        new = []
        for x in xs:
            logged = None if self._log is None else self._log.replay(x)
            if logged is not None:
                yield x, self._evaluated(x, logged)
                continue
            try:
                new.append((x, self.evaluate(x)))
            except ParamEvaluationError:
                new.append((x, None))
        evaluated = self.map_objective(
            _merits_or_failure,
            [values for _, values in new if values is not None])
        for x, values in new:
            merits = _UNEVALUATED if values is None else next(evaluated)
            if isinstance(merits, ModelEvaluationError):
                self._count_failed(x, merits)
                merits = _UNEVALUATED
            if self._log is not None:
                self._log.append(x, merits)
            yield x, self._evaluated(x, merits)

    def _evaluated(self, x: np.ndarray, merits: Merits) -> Merits | None:
        """
        Return the numbers of a point, or None for those of a point whose
        parameters could not be evaluated (infinities), or that are not
        numbers (counted).
        """
        if any(math.isnan(number) for number in merits):
            self._count_nan(x)
            return None
        return None if merits == _UNEVALUATED else merits

    def _count_nan(self, x: np.ndarray) -> None:
        """
        Count a model evaluation whose chi-square is not a number, and log
        the first.
        """
        if not self._nan_nevals:
            _log.warning(
                "the chi-square is not a number at the free values %s (e.g. "
                "a model that is NaN where the data have values); see the "
                "diagnostics 'nan_nevals' of the result",
                dict(zip(self._names, np.asarray(x).tolist())))
        self._nan_nevals += 1

    def _count_failed(
            self, x: np.ndarray, failure: ModelEvaluationError
    ) -> None:
        """
        Count a model evaluation where the model or a solve failed, and log
        the first.
        """
        if not self._failed_nevals:
            _log.warning(
                "the model cannot be evaluated at the free values %s (%s): "
                "such points count as ones whose parameters cannot be "
                "evaluated; see the diagnostics 'failed_nevals' of the "
                "result",
                dict(zip(self._names, np.asarray(x).tolist())), failure)
        self._failed_nevals += 1

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
        failed = []
        for observer in self._observers:
            try:
                observer(progress)
            except Exception:
                _log.exception(
                    "a progress observer failed; the fit goes on without it")
                failed.append(observer)
        self._observers = tuple(
            observer for observer in self._observers
            if observer not in failed)


def _merits_or_failure(
        objective: Objective, values: ParamValues
) -> Merits | ModelEvaluationError:
    """
    Return the merits of an objective at the values of the parameters, or
    the failure of its model or a solve there (see ModelEvaluationError),
    which a point of a batch must not raise for the others.
    """
    try:
        return objective.merits(values)
    except ModelEvaluationError as e:
        return e


def _points(xs: npt.ArrayLike) -> np.ndarray:
    """Return points of free values, one per row."""
    return np.atleast_2d(np.asarray(xs, dtype=float))

