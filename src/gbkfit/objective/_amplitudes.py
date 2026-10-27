"""
The solves of the amplitudes of the light of an objective
(AmplitudeSolve): amplitudes of the light of components (see
Component.light_amplitudes) solved together to fit the observations whose
models are linear in them (see GroupPlan.amplitude_pieces).
"""

from collections.abc import Collection, Mapping, Sequence
from typing import NamedTuple

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.observation import LinearPieces, ObservationGroup
from gbkfit.params import (
    ParamValues, Solution, Solve, check_together, solve_together)
from gbkfit.params._keys import element_name
from gbkfit.utils.parseutils import ConfigError
from .comparisons import Comparison


__all__ = [
    'AmplitudeSolve',
    'amplitude_solves'
]


class _Sums(NamedTuple):
    """
    The memory of the sums of the amplitudes of an observation (see
    AmplitudeSolve), in float64: the rows R and the residuals r of the
    base, on its device, and N = R R^T and R r on the host (h) and the
    device (d).
    """
    rows: DeviceArray
    residuals: DeviceArray
    h_normal: np.ndarray
    d_normal: DeviceArray
    h_rhs: np.ndarray
    d_rhs: DeviceArray


class AmplitudeSolve:
    """
    Amplitudes of the light of the models (see Component.light_amplitudes)
    solved together (see solve_together), where their parts of the model
    fall on the same data: in one or several observations (see Objective).

    The plan of the group gives the models of the observations as linear
    in them (see GroupPlan.amplitude_pieces): the base, with them at 0,
    and the piece of each. The comparison of each observation weighs the
    pieces, the rows R (see Comparison.weigh), and gives the residuals r
    of the base (see Comparison.evaluate): the sums over the observations
    of N = R R^T and b = -R r are the normal equations of the amplitudes.

    Parameters
    ----------
    amplitudes : Sequence of tuple
        The elements: each the name of its parameter and its index (None
        for a scalar), in runs of one solve.
    runs : Sequence of tuple
        The runs of the elements of one solve: the solve, and the number
        of its elements.
    observations : Sequence of int
        The observations whose models take them, by index.
    comparisons : Sequence of Comparison
        The comparison of each observation of the group.
    drivers : Sequence of Driver
        The driver of each observation of the group.
    """

    def __init__(
            self,
            amplitudes: Sequence[tuple[str, int | None]],
            runs: Sequence[tuple[Solve, int]],
            observations: Sequence[int],
            comparisons: Sequence[Comparison],
            drivers: Sequence[Driver]
    ):
        self._amplitudes = tuple(amplitudes)
        self._elements = tuple(
            element_name(*element) for element in amplitudes)
        self._runs = tuple(runs)
        self._observations = tuple(observations)
        self._comparisons = tuple(comparisons[i] for i in observations)
        self._drivers = tuple(drivers[i] for i in observations)
        self._sums: list[_Sums] = []

    def amplitudes(self) -> tuple[tuple[str, int | None], ...]:
        """
        Return the elements: each the name of its parameter and its index
        (None for a scalar).
        """
        return self._amplitudes

    def observations(self) -> tuple[int, ...]:
        """Return the observations whose models take them, by index."""
        return self._observations

    def elements(self) -> tuple[str, ...]:
        """Return the names of the elements, in their order."""
        return self._elements

    def likelihood(self) -> str:
        """
        Return what the log-likelihood takes of their values (see Solve;
        one for all, see check_together).
        """
        return self._runs[0][0].likelihood()

    def close(self) -> None:
        """Release its memory; the next solve allocates it again."""
        self._sums = []

    def solve(
            self,
            pieces: Mapping[int, LinearPieces],
            params: ParamValues
    ) -> Solution:
        """
        Solve the amplitudes from the pieces of the models of their
        observations (see GroupPlan.amplitude_pieces), by index, at the
        values of the parameters, and return them. It overwrites the
        residuals of the comparisons of the observations.
        """
        if not self._sums:
            self._prepare()
        size = len(self._amplitudes)
        normal = np.zeros((size, size))
        rhs = np.zeros(size)
        for i, comparison, driver, sums in zip(
                self._observations, self._comparisons, self._drivers,
                self._sums):
            comparison.weigh(pieces[i], params, sums.rows)
            comparison.evaluate(pieces[i].base, params, True, {})
            driver.mem_copy_d2d(comparison.residual_vector_d(), sums.residuals)
            driver.math_matmul(sums.rows, sums.rows.T, out=sums.d_normal)
            driver.math_matmul(sums.rows, sums.residuals, out=sums.d_rhs)
            driver.mem_copy_d2h(sums.d_normal, sums.h_normal)
            driver.mem_copy_d2h(sums.d_rhs, sums.h_rhs)
            normal += sums.h_normal
            rhs -= sums.h_rhs
        return solve_together(
            [solve for solve, _ in self._runs],
            [count for _, count in self._runs], normal, rhs)

    def _prepare(self) -> None:
        """Allocate the memory of the sums of each observation."""
        size = len(self._amplitudes)
        self._sums = []
        for comparison, driver in zip(self._comparisons, self._drivers):
            nvalues = comparison.residual_vector_d().size
            self._sums.append(_Sums(
                driver.mem_alloc_d((size, nvalues), np.float64),
                driver.mem_alloc_d(nvalues, np.float64),
                *driver.mem_alloc_s((size, size), np.float64),
                *driver.mem_alloc_s(size, np.float64)))


def amplitude_solves(
        group: ObservationGroup,
        comparisons: Sequence[Comparison],
        drivers: Sequence[Driver],
        solves: Mapping[str, Solve],
        scaled: Collection[int]
) -> tuple[AmplitudeSolve, ...]:
    """
    Return the solves of the amplitudes of the light of a group (see
    AmplitudeSolve), of the comparison and the driver of each observation,
    and the solve of each solved element (by element name; all must be
    amplitudes of the light): those of parameters whose models fall on the
    same observations together.

    Raises
    ------
    ConfigError
        If an element is not an amplitude of the light, or one that a model
        is not linear in (with why), its observation has a solved flux
        scale (in the given observations: the model is not linear in
        both), the weighted residuals of its comparison are not linear in
        the model (see Comparison.linear_form), or its solves cannot be
        solved together (see check_together).
    """
    if not solves:
        return ()
    pdescs = group.pdescs()
    # The observations that take each amplitude of the light, and why
    # their models are not linear in it, if not
    takers: dict[str, list[int]] = {}
    reasons: dict[str, str] = {}
    for i in range(group.nobservations()):
        for param, reason in group.light_amplitudes(i).items():
            takers.setdefault(param, []).append(i)
            if reason is not None:
                reasons.setdefault(param, reason)
    elements: dict[str, list[tuple[str, int | None]]] = {}
    for param, pdesc in pdescs.items():
        names = [param] if pdesc.is_scalar() else [
            element_name(param, k) for k in range(pdesc.size())]
        for k, name in enumerate(names):
            if name in solves:
                elements.setdefault(param, []).append(
                    (param, None if pdesc.is_scalar() else k))
    if unknown := sorted(set(solves) - {
            element_name(*e) for es in elements.values() for e in es}):
        raise ConfigError(f"these elements are not parameters: {unknown}")
    for param in elements:
        if param not in takers:
            raise ConfigError(
                f"{param} cannot be solved: only the flux scales of "
                f"observations and the amplitudes of the light of components "
                f"seen by observations can be solved (the model is linear "
                f"in them)")
        if param in reasons:
            raise ConfigError(
                f"{param} cannot be solved: {reasons[param]}, so the model "
                f"is not linear in it; fit it instead")
        for i in takers[param]:
            if i in scaled:
                raise ConfigError(
                    f"{param} cannot be solved with the solved flux scale of "
                    f"observation {i}: the model is not linear in both")
            keys = group.observations()[i].keys()
            if reason := comparisons[i].linear_form(keys):
                raise ConfigError(
                    f"{param} cannot be solved from observation {i}: "
                    f"{reason}")
    # (the parameters whose models fall on the same observations, together)
    groups: list[tuple[list[str], set[int]]] = []
    for param in elements:
        observations = set(takers[param])
        joined = [g for g in groups if g[1] & observations]
        merged = ([param], observations)
        for g in joined:
            merged[0][:0] = g[0]
            merged[1].update(g[1])
            groups.remove(g)
        groups.append(merged)
    result = []
    for params, observations in groups:
        members = [e for param in params for e in elements[param]]
        runs: list[tuple[Solve, int]] = []
        for element in members:
            solve = solves[element_name(*element)]
            if runs and runs[-1][0] is solve:
                runs[-1] = (solve, runs[-1][1] + 1)
            else:
                runs.append((solve, 1))
        check_together(
            [solve for solve, _ in runs], [count for _, count in runs])
        result.append(AmplitudeSolve(
            members, runs, sorted(observations), comparisons, drivers))
    return tuple(result)
