"""
The solves of the flux scales of an objective (ScaleBlock): elements of
the flux scale of an observation (see ObservationSky), which multiplies
its model per item, spectrum or row (see FluxScale): the model at scale 1
is their piece, and the sums of its comparison solve each alone.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.driver import DeviceArray
from gbkfit.observation import (
    FluxScale, GroupPlan, ModelData, ObservationGroup)
from gbkfit.params import ParamValues, Solution, Solve, element_names
from ._block import Block
from ._comparisons import Comparison


__all__ = [
    'FluxScales',
    'ScaleBlock',
    'scale_blocks'
]


class ScaleBlock(Block):
    """
    Elements of the flux scale of an observation solved by one solve (see
    Objective): each scales the model of its observation where the scale
    has it (see FluxScale), so that the model at scale 1 is its piece, and
    each is solved alone (see Solve.solve) from the sums of the comparison
    of the observation (see Comparison.scale_sums).

    Parameters
    ----------
    observation : int
        The observation, by index.
    param : str
        The parameter of its flux scale.
    indices : np.ndarray, optional
        The indices of the elements (None for a scalar parameter).
    scale : FluxScale
        How the scale scales the model of the observation.
    solve : Solve
        The solve of the elements.
    comparison : Comparison
        The comparison of the observation.
    """

    def __init__(
            self,
            observation: int,
            param: str,
            indices: np.ndarray | None,
            scale: FluxScale,
            solve: Solve,
            comparison: Comparison
    ):
        self._observation = observation
        self._param = param
        self._indices = indices
        self._scale = scale
        self._solve = solve
        self._comparison = comparison
        self._elements = tuple(element_names(param, indices))

    def observation(self) -> int:
        """Return the observation, by index."""
        return self._observation

    def elements(self) -> tuple[str, ...]:
        """Return the names of the elements, in their order."""
        return self._elements

    def likelihood(self) -> str:
        """Return what the log-likelihood takes of their values (see Solve)."""
        return self._solve.likelihood()

    def neutral(self, params: ParamValues) -> ParamValues:
        """
        Return the values of the parameters with the elements at 1, where
        the model of the observation is their piece (copies of those it
        changes).
        """
        values = dict(params)
        if self._indices is None:
            values[self._param] = 1.0
        else:
            values[self._param] = np.array(values[self._param], dtype=float)
            values[self._param][self._indices] = 1.0
        return values

    def solve(self, model: ModelData, params: ParamValues) -> Solution:
        """
        Solve the elements from the sums of the comparison of their
        observation (see Comparison.scale_sums), of its model at scale 1,
        and return them.
        """
        total_a = total_b = 0.0
        axis = self._scale.axis
        for key in self._scale.items:
            a, b = self._comparison.scale_sums(
                model, params, key, 0 if axis is None else axis)
            if axis is None:
                a, b = np.sum(a), np.sum(b)
            total_a, total_b = total_a + a, total_b + b
        a, b = np.ravel(total_a), np.ravel(total_b)
        if self._indices is not None:
            a, b = a[self._indices], b[self._indices]
        return self._solve.solve(a, b)

    def place(self, values: np.ndarray, scales: np.ndarray) -> None:
        """
        Write their values to the scales of the observation (an array of
        the shape of the items of the scale, but 1 along its axis).
        """
        flat = scales.reshape(-1)
        flat[slice(None) if self._indices is None else self._indices] = values


def scale_blocks(
        group: ObservationGroup,
        comparisons: Mapping[int, Comparison],
        solves: Mapping[str, Solve]
) -> tuple[ScaleBlock, ...]:
    """
    Return the solves of the solved elements of the flux scales of the
    observations of a group, of the comparison of each observation with
    data (by index) and the solve of each solved element (by element name;
    those of other parameters are left): those of one parameter and one
    solve together.

    Raises
    ------
    ValueError
        If an observation without data has solved elements of its flux
        scale, or the weighted residuals of the comparison of an
        observation are not pointwise (see Comparison.pointwise_form).
    """
    pdescs = group.pdescs()
    result = []
    for i, observation in enumerate(group.observations()):
        if (flux_scale := observation.flux_scale()) is None:
            continue
        param, scale = flux_scale
        pdesc = pdescs[param]
        names = element_names(
            param, None if pdesc.is_scalar() else range(pdesc.size()))
        indices_of: dict[int, list[int]] = {}
        for k, name in enumerate(names):
            if name in solves:
                indices_of.setdefault(id(solves[name]), []).append(k)
        if indices_of and i not in comparisons:
            raise ValueError(
                f"the flux scale of observation {i} ({param}) cannot be "
                f"solved: the observation has no data to solve it from")
        if indices_of and (
                reason := comparisons[i].pointwise_form(scale.items)):
            raise ValueError(
                f"the flux scale of observation {i} ({param}) cannot be "
                f"solved: {reason}")
        for indices in indices_of.values():
            result.append(ScaleBlock(
                i, param, None if pdesc.is_scalar() else np.array(indices),
                scale, solves[names[indices[0]]], comparisons[i]))
    return tuple(result)


class FluxScales:
    """
    The solved flux scales of the observations of a group plan (see
    ScaleBlock), solved after their models: the models are evaluated with
    them at 1 (see neutral), their blocks solve them from the models, and
    the comparisons compare the models times them (see solve).

    Parameters
    ----------
    group_plan : GroupPlan
        The plan of the observations.
    blocks : Sequence of ScaleBlock
        The blocks of the solved elements of their scales.
    """

    def __init__(self, group_plan: GroupPlan, blocks: Sequence[ScaleBlock]):
        self._group = group_plan.group()
        self._drivers = group_plan.observation_drivers()
        self._blocks = tuple(blocks)
        self._observations = tuple(sorted(
            {block.observation() for block in blocks}))
        # The scales of each observation with solved ones, on the host (h)
        # and the device (d), by index (see prepare)
        self._h_scales: dict[int, np.ndarray] = {}
        self._d_scales: dict[int, DeviceArray] = {}

    def blocks(self) -> tuple[ScaleBlock, ...]:
        """Return the blocks of the solved elements of the scales."""
        return self._blocks

    def observations(self) -> tuple[int, ...]:
        """Return the observations with solved scales, by index."""
        return self._observations

    def prepare(self) -> None:
        """
        Allocate the memory of the scales of the observations: of the
        shape of the data of each, but 1 along the axis of its scale.
        """
        for i in self._observations:
            observation = self._group.observations()[i]
            _, scale = observation.flux_scale()
            shape = tuple(
                1 if scale.axis in (None, k) else n
                for k, n in enumerate(observation.dataset().shape()))
            self._h_scales[i], self._d_scales[i] = \
                self._drivers[i].mem_alloc_s(shape, observation.dtype())

    def close(self) -> None:
        """Release its memory; prepare allocates it again."""
        self._h_scales = {}
        self._d_scales = {}

    def neutral(self, params: ParamValues) -> ParamValues:
        """
        Return the values of the parameters with the solved elements of
        the scales at 1 (copies of those of their parameters).
        """
        values = params
        for block in self._blocks:
            values = block.neutral(values)
        return values

    def solve(
            self,
            models: Sequence[ModelData],
            params: ParamValues
    ) -> tuple[dict[Block, Solution], dict[int, dict[str, DeviceArray]]]:
        """
        Solve the elements of the scales from the models of the
        observations (see ScaleBlock), with the scales at 1, and return
        the solution of each block, and the scales of the data items of
        each observation with solved scales, by index, on its device, by
        name (see Comparison.evaluate).
        """
        solutions: dict[Block, Solution] = {}
        for i in self._observations:
            self._h_scales[i][...] = 1
        for block in self._blocks:
            i = block.observation()
            solutions[block] = solution = block.solve(models[i], params)
            block.place(solution.values, self._h_scales[i])
        scales = {}
        for i in self._observations:
            self._drivers[i].mem_copy_h2d(self._h_scales[i], self._d_scales[i])
            _, scale = self._group.observations()[i].flux_scale()
            scales[i] = dict.fromkeys(scale.items, self._d_scales[i])
        return solutions, scales
