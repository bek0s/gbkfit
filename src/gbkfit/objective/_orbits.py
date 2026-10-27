"""
The solves of the weights of orbits of an objective (OrbitSolve): the
weights of the orbits of the stars of a spheroid (see
Component.orbit_weights), solved to fit the observations that see them,
whose models are linear in them (see OrbitRecording).
"""

from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.driver import DeviceArray
from gbkfit.observation import GroupPlan, OrbitRecording, OrbitSeries
from gbkfit.params import ParamValues, Solution, Solve, solve_together_rows
from gbkfit.params._keys import element_name
from gbkfit.utils.parseutils import ConfigError
from .comparisons import Comparison


__all__ = [
    'OrbitSolve',
    'orbit_solves'
]


class OrbitSolve:
    """
    The weights of the orbits of a recording of a group plan (see
    OrbitRecording), solved together to fit the moments of the bins of
    its observations (see Objective).

    The model of each moment of a bin is linear in the weights f (see
    OrbitSeries): base + slope (moments @ f). Its comparison gives its
    residual c (model - data) (see Comparison.check_pointwise; c is the
    residual of the model data + 1): the rows of the data, c slope
    moments, and their targets, c (data - base). The constraints of the
    light of the orbits give the rows of constraints, which count with the
    prior of the weights (see solve_together_rows).

    Parameters
    ----------
    recording : OrbitRecording
        The recording.
    group_plan : GroupPlan
        The plan of its observations, each with data.
    comparisons : Sequence of Comparison
        The comparison of each observation of the group.
    elements : Sequence of str
        The names of the elements of its weights, in their order.
    solve : Solve
        Their solve.

    Raises
    ------
    ConfigError
        If an observation of it does not compare each value of its data
        with its model alone (see Comparison.check_pointwise).
    """

    def __init__(
            self,
            recording: OrbitRecording,
            group_plan: GroupPlan,
            comparisons: Sequence[Comparison],
            elements: Sequence[str],
            solve: Solve
    ):
        indices = recording.observations()
        observations = [group_plan.group().observations()[i] for i in indices]
        self._recording = recording
        self._comparisons = tuple(comparisons[i] for i in indices)
        for comparison, observation in zip(self._comparisons, observations):
            comparison.check_pointwise(observation.keys())
        self._drivers = tuple(
            group_plan.observation_drivers()[i] for i in indices)
        self._dtypes = tuple(
            observation.dtype() for observation in observations)
        # (the data of each observation, 0 where missing, by key)
        self._data = tuple(
            {key: np.nan_to_num(np.asarray(dataset[key].data(), float))
             for key in dataset}
            for dataset in (
                observation.data() for observation in observations))
        self._elements = tuple(elements)
        self._solve = solve
        self._ones: list[dict[str, dict[str, DeviceArray | None]]] = []

    def recording(self) -> OrbitRecording:
        """Return the recording."""
        return self._recording

    def elements(self) -> tuple[str, ...]:
        """Return the names of the elements of its weights."""
        return self._elements

    def close(self) -> None:
        """Release its memory; the next solve allocates it again."""
        self._ones = []

    def solve(self, series: OrbitSeries, params: ParamValues) -> Solution:
        """
        Solve the weights at the values of the parameters, of the orbits
        of its recording in the series of the bins (see
        GroupPlan.orbit_series), and return them, with the residuals of
        the constraints of the light before those of their prior.
        """
        if not self._ones:
            self._prepare()
        rows, targets = [], []
        for comparison, ones, data, driver, i in zip(
                self._comparisons, self._ones, self._data, self._drivers,
                self._recording.observations()):
            comparison.evaluate(ones, params, True, {})
            factors = {
                key: driver.mem_copy_d2h(residual).astype(np.float64)
                for key, residual in comparison.residual_nddata_d().items()}
            for key, moment in series.moments[i].items():
                factor = factors[key]
                rows.append((factor * moment.slope)[:, None] * moment.moments)
                targets.append(factor * (data[key] - moment.base))
        rows = np.vstack((*rows, series.light_rows))
        targets = np.concatenate((*targets, series.light_targets))
        ndata = len(rows) - len(series.light_rows)
        solution = solve_together_rows(
            [self._solve], [len(self._elements)], rows, targets, ndata)
        residuals = rows[ndata:] @ solution.values - targets[ndata:]
        return Solution(
            solution.values,
            np.concatenate((residuals, solution.residuals)),
            solution.log_term, solution.m_eff)

    def _prepare(self) -> None:
        """
        Allocate its memory: the models of the data plus 1 of each
        observation (whose residuals are the factors of its comparison).
        """
        self._ones = [
            {key: dict(
                d=driver.mem_copy_h2d((value + 1).astype(dtype)),
                m=None, w=None)
             for key, value in data.items()}
            for data, driver, dtype in zip(
                self._data, self._drivers, self._dtypes)]


def orbit_solves(
        group_plan: GroupPlan,
        comparisons: Sequence[Comparison],
        solves: Mapping[str, Solve]
) -> tuple[OrbitSolve, ...]:
    """
    Return the solves of the weights of the orbits of the recordings of a
    group plan (see OrbitSolve and GroupPlan.orbit_recordings), of the
    comparison of each observation, and the solve of each solved element
    (by element name).

    Raises
    ------
    ConfigError
        If the weights of the orbits of a recording are not solved, or not
        by one solve, or an orbit solve cannot be made (see OrbitSolve).
    """
    pdescs = group_plan.group().pdescs()
    orbits = []
    for recording in group_plan.orbit_recordings():
        param = recording.weights()
        elements = [
            element_name(param, k) for k in range(pdescs[param].size())]
        if not all(element in solves for element in elements):
            raise ConfigError(
                f"the weights of the orbits of the stars of "
                f"{recording.spheroid().name()!r} ({param!r}) are solved to "
                f"fit the data: give them a solve, e.g. {param}: {{solve: "
                f"{{type: gaussian, mean: 1, sigma: 1, minimum: 0, "
                f"likelihood: profiled}}}}")
        if len({id(solves[element]) for element in elements}) > 1:
            raise ConfigError(
                f"the weights of the orbits {param!r} are solved by one "
                f"solve")
        orbits.append(OrbitSolve(
            recording, group_plan, comparisons, elements,
            solves[elements[0]]))
    return tuple(orbits)
