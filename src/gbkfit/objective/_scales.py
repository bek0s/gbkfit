"""
The solves of the flux scales of an objective (ScaleSolve): elements of
the flux scale of an observation (see ObservationSky), which multiplies
its model per item, spectrum or row (see FluxScale): the model at scale 1
is their piece, and the sums of its comparison solve each alone.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from gbkfit.observation import FluxScale, ModelData, ObservationGroup
from gbkfit.params import ParamValues, Solution, Solve
from gbkfit.params._keys import element_name
from gbkfit.utils.parseutils import ConfigError
from .comparisons import Comparison


__all__ = [
    'ScaleSolve',
    'scale_solves'
]


class ScaleSolve:
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
        self._elements = (param,) if indices is None else tuple(
            element_name(param, int(k)) for k in indices)

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


def scale_solves(
        group: ObservationGroup,
        comparisons: Sequence[Comparison],
        solves: Mapping[str, Solve]
) -> tuple[ScaleSolve, ...]:
    """
    Return the solves of the solved elements of the flux scales of the
    observations of a group, of the comparison of each observation and the
    solve of each solved element (by element name; those of other
    parameters are left): those of one parameter and one solve together.

    Raises
    ------
    ConfigError
        If the weighted residuals of the comparison of an observation are
        not pointwise (see Comparison.pointwise_form).
    """
    pdescs = group.pdescs()
    result = []
    for i, observation in enumerate(group.observations()):
        if (flux_scale := observation.flux_scale()) is None:
            continue
        param, scale = flux_scale
        pdesc = pdescs[param]
        names = [param] if pdesc.is_scalar() else [
            element_name(param, k) for k in range(pdesc.size())]
        indices_of: dict[int, list[int]] = {}
        for k, name in enumerate(names):
            if name in solves:
                indices_of.setdefault(id(solves[name]), []).append(k)
        if indices_of and (
                reason := comparisons[i].pointwise_form(scale.items)):
            raise ConfigError(
                f"the flux scale of observation {i} ({param}) cannot be "
                f"solved: {reason}")
        for indices in indices_of.values():
            result.append(ScaleSolve(
                i, param, None if pdesc.is_scalar() else np.array(indices),
                scale, solves[names[indices[0]]], comparisons[i]))
    return tuple(result)
