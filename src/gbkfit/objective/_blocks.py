"""
The blocks of the solved elements of an objective (see make_blocks): of
the weights of orbits, of the flux scales of observations, and of the
amplitudes of the light.
"""

from collections.abc import Iterable, Mapping

from gbkfit.observation import GroupPlan
from gbkfit.params import Solve
from ._amplitude_blocks import amplitude_blocks
from ._block import BasisBlock, Block
from ._comparisons import Comparison
from ._orbit_blocks import orbit_blocks
from ._scale_blocks import FluxScales, scale_blocks


__all__ = [
    'make_blocks'
]


def make_blocks(
        group_plan: GroupPlan,
        comparisons: Mapping[int, Comparison],
        solves: Mapping[str, Solve]
) -> tuple[tuple[BasisBlock, ...], FluxScales]:
    """
    Return the blocks of the solved elements of an objective of a group
    plan, of the comparison of each observation with data (by index) and
    the solve of each solved element (by element name): those that the
    models are linear in, solved before the models in their order (the
    amplitudes of the light, then the weights of orbits), and the flux
    scales, solved after them. Each kind claims its elements from those
    the kinds before it left: the weights of orbits, the flux scales, then
    the amplitudes of the light, which every element left must be.

    Raises
    ------
    ValueError
        If the blocks of a kind cannot be made (see orbit_blocks,
        scale_blocks and amplitude_blocks).
    """
    orbits = orbit_blocks(group_plan, comparisons, solves)
    scales = FluxScales(group_plan, scale_blocks(
        group_plan.group(), comparisons, _unclaimed(solves, orbits)))
    amplitudes = amplitude_blocks(
        group_plan, comparisons,
        _unclaimed(solves, (*orbits, *scales.blocks())),
        scales.observations())
    return (*amplitudes, *orbits), scales


def _unclaimed(
        solves: Mapping[str, Solve], blocks: Iterable[Block]
) -> dict[str, Solve]:
    """
    Return the solves of the solved elements that none of the blocks
    solves, by element name.
    """
    claimed = {element for block in blocks for element in block.elements()}
    return {name: solve for name, solve in solves.items()
            if name not in claimed}
