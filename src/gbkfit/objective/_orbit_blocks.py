"""
The solves of the weights of orbits of an objective (OrbitBlock): the
weights of the orbits of the stars of a spheroid (see
Component.orbit_weights), solved to fit the observations that see them,
whose models are linear in them (see OrbitRecording).
"""

import functools
from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.observation import GroupPlan
from gbkfit.observation.orbits import OrbitRecording, OrbitSeries
from gbkfit.params import (
    ParamValues, ParamView, Solution, Solve, StageCache, element_names,
    solve_together_rows)
from gbkfit.utils import threadutils
from ._block import BasisBlock
from ._comparisons import Comparison


__all__ = [
    'OrbitBlock',
    'orbit_blocks'
]


# The most threads of BLAS in the solves of the weights of orbits (see
# OrbitBlock.solve): more lose to their overheads (an AICp solve of 20000
# weights on 1, 4, 8 and 22 threads of a 16-core laptop: 0.84, 0.48, 0.52
# and 2.1 s for 305 rows; 29, 12.4, 12.8 and 18.7 s for 2495)
_BLAS_THREADS = 4


class OrbitBlock(BasisBlock):
    """
    The weights of the orbits of a recording of a group plan (see
    OrbitRecording), solved together to fit the moments of the bins of
    its observations (see Objective).

    The model of each moment of a bin is linear in the weights f (see
    OrbitSeries): base + slope (moments @ f). Its comparison gives its
    residual c (model - data) (see Comparison.pointwise_factors): the
    rows of the data, c slope moments, and their targets, c (data -
    base). The constraints of the light of the orbits give the rows of
    constraints, which count with the prior of the weights (see
    solve_together_rows).

    The factors c read the parameters of the likelihoods of the
    observations (e.g. a fitted noise) alone: a solve keeps its solution
    while they and the recording keep theirs (see solve), e.g. while
    another observation of the group moves.

    Parameters
    ----------
    recording : OrbitRecording
        The recording.
    group_plan : GroupPlan
        The plan of its observations, each with data.
    comparisons : Mapping of int to Comparison
        The comparison of each observation with data of the group, by
        index.
    elements : Sequence of str
        The names of the elements of its weights, in their order.
    solve : Solve
        Their solve.

    Raises
    ------
    ValueError
        If the weighted residuals of the comparison of an observation of
        it are not pointwise (see Comparison.pointwise_form).
    """

    def __init__(
            self,
            recording: OrbitRecording,
            group_plan: GroupPlan,
            comparisons: Mapping[int, Comparison],
            elements: Sequence[str],
            solve: Solve
    ):
        indices = recording.observations()
        observations = [group_plan.group().observations()[i] for i in indices]
        self._recording = recording
        self._comparisons = tuple(comparisons[i] for i in indices)
        for i, comparison, observation in zip(
                indices, self._comparisons, observations):
            if reason := comparison.pointwise_form(observation.keys()):
                raise ValueError(
                    f"the weights of the orbits {recording.weights()!r} "
                    f"cannot be solved from observation {i}: {reason}")
        # (the data of each observation, 0 where missing, by key)
        self._data = tuple(
            {key: np.nan_to_num(np.asarray(dataset[key].data(), float))
             for key in dataset}
            for dataset in (
                observation.dataset() for observation in observations))
        self._elements = tuple(elements)
        self._solve = solve
        self._group_plan = group_plan
        # The cache of its solutions: of the parameters of the likelihoods
        # of its observations, taking the orbits of its recording
        self._index = k = group_plan.orbit_recordings().index(recording)
        self._cache: StageCache[Solution] = StageCache(
            (name for comparison in self._comparisons
             for name in comparison.likelihood().pdescs()),
            (group_plan.orbit_caches()[k],))

    def recording(self) -> OrbitRecording:
        """Return the recording."""
        return self._recording

    def elements(self) -> tuple[str, ...]:
        """Return the names of the elements of its weights."""
        return self._elements

    def likelihood(self) -> str:
        """Return what the log-likelihood takes of their values (see Solve)."""
        return self._solve.likelihood()

    def solve(self, params: ParamValues) -> Solution:
        """
        Solve the weights at the values of the parameters, of the orbits
        of its recording in the series of the bins (those that the plan
        gives at these values, see GroupPlan.orbit_series), with BLAS on
        at most _BLAS_THREADS threads, and return them, with the
        residuals of the constraints of the light before those of their
        prior: those of its last solve if the parameters of the
        likelihoods of its observations and the recording are the same
        (see the class).
        """
        plan = self._group_plan
        with plan.timers().measure('objective_orbits'), \
                threadutils.blas_limit(_BLAS_THREADS):
            series = plan.orbit_series(params)[self._index]
            return self._cache.evaluate(
                params, functools.partial(self._solved, series))

    def solved(
            self, params: ParamValues, solution: Solution
    ) -> ParamValues:
        return params | {self._recording.weights(): solution.values}

    def _solved(self, series: OrbitSeries, params: ParamView) -> Solution:
        """
        Solve the weights (see solve), at the values of the parameters of
        the likelihoods of its observations.
        """
        rows, targets = [], []
        for comparison, data, i in zip(
                self._comparisons, self._data,
                self._recording.observations()):
            factors = comparison.pointwise_factors(
                series.moments[i].keys(), params)
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


def orbit_blocks(
        group_plan: GroupPlan,
        comparisons: Mapping[int, Comparison],
        solves: Mapping[str, Solve]
) -> tuple[OrbitBlock, ...]:
    """
    Return the solves of the weights of the orbits of the recordings of a
    group plan (see OrbitBlock and GroupPlan.orbit_recordings: of the
    observations with data), of the comparison of each observation with
    data (by index), and the solve of each solved element (by element
    name).

    Raises
    ------
    ValueError
        If the weights of the orbits of a recording are not solved, or not
        by one solve, or an orbit solve cannot be made (see OrbitBlock).
    """
    pdescs = group_plan.group().pdescs()
    blocks = []
    for recording in group_plan.orbit_recordings():
        param = recording.weights()
        elements = element_names(param, range(pdescs[param].size()))
        if not all(element in solves for element in elements):
            raise ValueError(
                f"the weights of the orbits of the stars of "
                f"{recording.spheroid().name()!r} ({param!r}) are solved to "
                f"fit the data: give them a solve, e.g. {param}: {{solve: "
                f"{{type: gauss, mean: 1, sigma: 1, min: 0, "
                f"likelihood: profiled}}}}")
        if len({id(solves[element]) for element in elements}) > 1:
            raise ValueError(
                f"the weights of the orbits {param!r} are solved by one "
                f"solve")
        blocks.append(OrbitBlock(
            recording, group_plan, comparisons, elements,
            solves[elements[0]]))
    return tuple(blocks)
