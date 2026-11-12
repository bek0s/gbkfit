"""
The setups of GBKFIT: galaxies, their observations and their parameters
set up on drivers (see Setup), and their evaluations.
"""

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Self

from gbkfit.control import StopControl
from gbkfit.driver import Driver
from gbkfit.fitting import (
    Bootstrap, BootstrapOptions, EvaluationLog, Fitter, FitResult,
    ProgressObserver, SubFitLog, run_bootstrap)
from gbkfit.objective import Objective, ObjectiveEvaluation
from gbkfit.observation import (
    GroupPlan, MockOptions, Observation, ObservationGroup, draw_mocks)
from gbkfit.outputs import Output
from gbkfit.parallel import WorkerPool, Workers
from gbkfit.params import ParamSpace, ParamValues
from ._outputs import data_outputs, evaluation_outputs


__all__ = [
    'Setup',
    'SetupEvaluation'
]


_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SetupEvaluation:
    """
    An evaluation of a setup at values of its free elements (see
    Setup.evaluate).

    Attributes
    ----------
    values : ParamValues
        The values of all parameters, with those of the solved elements
        that the objective solves there (without an objective, their
        values without data; see Solve.default).
    objective : ObjectiveEvaluation or None
        The evaluation of the objective there (its merits, the parts of
        its chi-square, and the solved values), or None without an
        objective.
    """
    values: ParamValues
    objective: ObjectiveEvaluation | None


class Setup:
    """
    Galaxies, their observations and their parameters, set up on drivers:
    the plan of the group (see ObservationGroup.plan), the parameters, the
    objective if every observation has data, the fitter if given, workers
    of copies of the objective if given and it has one (see WorkerPool),
    and the options of mock data and of a bootstrap if given. load sets up
    a configuration as one. A fitter, workers or a bootstrap need no data
    until a fit (e.g. a configuration of a fit evaluated to draw mocks).

    It holds the memory of its objective and its workers until it is
    closed (see close; e.g. at the end of a with block).

    Parameters
    ----------
    group : ObservationGroup
        The galaxies and their observations.
    params : ParamSpace
        The parameters, of the group: of its parameters, constants and
        functions (see ObservationGroup.pdescs, constants and
        function_names).
    fitter : Fitter, optional
        The fitter of its fits (see fit).
    driver : Driver, optional
        The driver of the galaxies and the observations; the host by
        default.
    placement : Mapping of str to Driver, optional
        Other drivers of some of them, by their labels; the galaxies
        without mass that it leaves out follow the observations that see
        them (see ObservationGroup.drivers).
    workers : Workers, optional
        The workers of copies of the objective, which evaluate the batches
        of points of fits at once (see Workers.start).
    mock_options : MockOptions, optional
        The options of mock data of the observations (see mock).
    bootstrap_options : BootstrapOptions, optional
        The options of a bootstrap of a fit (see bootstrap).

    Raises
    ------
    ValueError
        If the parameters are not of the group, the fitter cannot fit
        them (see Fitter.check), the options of mock data are not of the
        observations (see MockOptions.check), the options of a
        bootstrap do not suit the data (see BootstrapOptions.check), or
        the group cannot be planned (see ObservationGroup.plan).

    Examples
    --------
    >>> group = ObservationGroup(galaxy, observation)
    >>> params = ParamSpace(
    ...     group.pdescs(), dict(incl=60, vpt_vt=FreeElement(min=50, max=300)),
    ...     constants=group.constants())
    >>> with Setup(group, params, fitter=FitterScipyLeastSquares(),
    ...            driver=DriverCuda()) as setup:
    ...     result = setup.fit()
    """

    def __init__(
            self,
            group: ObservationGroup,
            params: ParamSpace,
            fitter: Fitter | None = None,
            driver: Driver | None = None,
            placement: Mapping[str, Driver] | None = None,
            workers: Workers | None = None,
            mock_options: MockOptions | None = None,
            bootstrap_options: BootstrapOptions | None = None
    ):
        if missing := [n for n in group.pdescs() if n not in params.pdescs()]:
            raise ValueError(
                f"the parameters are not those of the group: they lack "
                f"{missing}")
        if unknown := sorted(
                set(params.function_names()) - set(group.function_names())):
            raise ValueError(
                f"the galaxies have no functions {unknown}; they have "
                f"{list(group.function_names())}")
        if fitter is not None:
            fitter.check(params, group)
        if mock_options is not None:
            mock_options.check(group)
        if bootstrap_options is not None and group.has_data():
            bootstrap_options.check(group, params)
        self._group_plan = group.plan(driver, placement)
        self._params = params
        self._objective = None
        self._fitter = fitter
        self._mock_options = mock_options
        self._bootstrap_options = bootstrap_options
        self._pool = None
        try:
            if group.has_data():
                _log.info("setting up objective...")
                self._objective = Objective(
                    self._group_plan, params.solves())
            if workers is not None and self._objective is not None:
                self._pool = workers.start(self._objective)
        # (of any kind, e.g. an interrupt: no one can close a setup that
        # was not made, and the plan holds the memory of devices)
        except BaseException:
            self.close()
            raise

    def group(self) -> ObservationGroup:
        """Return the group: the galaxies and their observations."""
        return self._group_plan.group()

    def group_plan(self) -> GroupPlan:
        """Return the plan of the group."""
        return self._group_plan

    def params(self) -> ParamSpace:
        """Return the parameters."""
        return self._params

    def objective(self) -> Objective | None:
        """Return the objective, or None if an observation has no data."""
        return self._objective

    def fitter(self) -> Fitter | None:
        """Return the fitter, if given."""
        return self._fitter

    def pool(self) -> WorkerPool | None:
        """
        Return the workers of copies of the objective, if given and it has
        one.
        """
        return self._pool

    def mock_options(self) -> MockOptions | None:
        """Return the options of mock data, if given."""
        return self._mock_options

    def bootstrap_options(self) -> BootstrapOptions | None:
        """Return the options of a bootstrap, if given."""
        return self._bootstrap_options

    def evaluate(
            self, free: Mapping[str, float] | Sequence[float] | None = None
    ) -> SetupEvaluation:
        """
        Evaluate the setup at values of its free elements: the values of
        all parameters, with those of the solved elements, and the
        evaluation of its objective there, if it has one.

        Parameters
        ----------
        free : Mapping of str to float or Sequence of float, optional
            The values of the free elements (see ParamSpace.evaluate); by
            default their option 'value'.

        Returns
        -------
        SetupEvaluation
            The evaluation.

        Raises
        ------
        ValueError
            If values of free elements are missing or unknown.
        ParamEvaluationError
            If the parameters cannot be evaluated there.
        """
        values = self._params.evaluate(free)
        if self._objective is None:
            return SetupEvaluation(values, None)
        evaluation = self._objective.evaluate(values)
        return SetupEvaluation(
            self._params.evaluate(free, evaluation.solved), evaluation)

    def outputs(
            self,
            evaluation: SetupEvaluation,
            data: bool = False,
            extra: bool = False
    ) -> dict[str, Output]:
        """
        Return the outputs of an evaluation, as the tasks write them, by
        their paths in a run (see evaluation_outputs).

        Parameters
        ----------
        evaluation : SetupEvaluation
            The evaluation (see evaluate).
        data : bool, optional
            Whether to add the data of the observations ('data', 'error'
            and 'data_image'; see data_outputs), which do not change with
            the values.
        extra : bool, optional
            Whether to add the extra outputs of the models ('extra'; e.g.
            their cubes at high resolution).

        Returns
        -------
        dict
            The outputs: the models ('model', and their images,
            'model_image'), and of an evaluation of the objective the
            residuals too ('residual', 'wresidual', 'chi2_residual' and
            'residual_image'; see evaluation_outputs).
        """
        residuals = evaluation.objective is not None
        outputs = evaluation_outputs(
            self._group_plan, self._objective, evaluation.values, True,
            residuals, extra)
        if data:
            outputs |= data_outputs(self._group_plan)
        return outputs

    def fit(
            self,
            observers: Sequence[ProgressObserver] = (),
            control: StopControl | None = None,
            log: EvaluationLog | None = None
    ) -> FitResult:
        """
        Fit the data of its objective with its fitter, on its workers if
        it has them (see Fitter.fit).

        Parameters
        ----------
        observers : Sequence of ProgressObserver, optional
            Who receives the progress of the fit as it runs (see
            FitProgress).
        control : StopControl, optional
            The control of the fit, to stop it.
        log : EvaluationLog, optional
            The log of the evaluations: a new one, or that of a fit to
            resume (with the same objective, parameters and fitter).

        Returns
        -------
        FitResult
            The result.

        Raises
        ------
        ValueError
            If it has no fitter, or no objective (an observation has no
            data), or the fit cannot start (see Fitter.fit).
        """
        if self._fitter is None:
            raise ValueError("a fit needs a fitter (section 'fitter')")
        if self._objective is None:
            raise ValueError("a fit needs data for every observation")
        return self._fitter.fit(
            self._objective, self._params, observers, control, log,
            self._pool)

    def bootstrap(
            self,
            start: Mapping[str, float],
            control: StopControl | None = None,
            log: SubFitLog | None = None
    ) -> Bootstrap:
        """
        Bootstrap a fit from its best fit by its options, on its workers if
        it has them (see run_bootstrap): the refits use the fitter of the
        options, or else its own.

        Parameters
        ----------
        start : Mapping of str to float
            The best fit: the value of every element (see
            FitSolution.elements).
        control : StopControl, optional
            The control of the bootstrap, to stop it.
        log : SubFitLog, optional
            The log of the refits: a new one, or that of a bootstrap to
            resume.

        Returns
        -------
        Bootstrap
            The bootstrap.

        Raises
        ------
        ValueError
            If it has no options of a bootstrap, no objective (an
            observation has no data) or no fitter, or the bootstrap cannot
            run (see run_bootstrap).
        Stopped
            If it was asked to stop (see control).
        """
        options = self._bootstrap_options
        if options is None:
            raise ValueError(
                "a bootstrap needs its options (section 'bootstrap')")
        if self._objective is None:
            raise ValueError("a bootstrap needs data for every observation")
        if self._fitter is None and options.fitter() is None:
            raise ValueError(
                "a bootstrap needs a fitter (section 'fitter', or its "
                "option 'fitter')")
        return run_bootstrap(
            self._objective, self._params, start, self._fitter, options,
            self._pool, control, log)

    def mock(self, values: ParamValues) -> list[list[Observation]]:
        """
        Draw mock data of its observations at values of the parameters, by
        its options (see MockOptions): their models measured from what
        their instruments record of the models plus noise, with their
        errors (see draw_mocks).

        Parameters
        ----------
        values : ParamValues
            The values of the parameters (see evaluate).

        Returns
        -------
        list of list of Observation
            The observations with their mock data, and the likelihoods of
            the options: those of each mock.

        Raises
        ------
        ValueError
            If it has no options of mock data, or the data cannot be drawn
            (see draw_mocks).
        """
        options = self._mock_options
        if options is None:
            raise ValueError("mock data need their options (section 'mock')")
        group = self.group()
        _, likelihoods = options.per_observation(group.nobservations())
        return [
            [observation.with_data(dataset, likelihood)
             for observation, dataset, likelihood in zip(
                 group.observations(), datasets, likelihoods)]
            for datasets in draw_mocks(self._group_plan, values, options)]

    def close(self) -> None:
        """
        Release the memory of its objective and its plan (allocated again by
        their next evaluation; see Objective.close and GroupPlan.close),
        and stop its workers.
        """
        if self._pool is not None:
            self._pool.close()
        if self._objective is not None:
            self._objective.close()
        self._group_plan.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
