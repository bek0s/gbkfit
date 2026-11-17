"""
The depth order of the galaxies of a group (see DepthOrder): the galaxies
along the line of sight, alone or in groups of galaxies that pass through
each other (see DepthGroup), and the parameters of the separations of
the galaxies of a group.
"""

from collections.abc import Sequence
from typing import NamedTuple

from gbkfit.config import check_name
from gbkfit.model import DepthUnit
from gbkfit.params import ParamDesc, ParamScalarDesc


__all__ = [
    'DepthGroup',
    'DepthOrder'
]


class DepthGroup(NamedTuple):
    """
    Galaxies that can pass through each other along the line of sight
    (e.g. a merger), by name, which radiative transfer evaluates together
    (see TransferPlan), and the name of the group (the id 'group' of its
    extra outputs). Each galaxy after the first has a parameter of its
    position along the line of sight from the first, '{galaxy}_separation'
    (arcsec; positive behind it, away from the viewer).
    """
    name: str
    galaxies: tuple[str, ...]


class DepthOrder:
    """
    The depth order of the galaxies of a group (see ObservationGroup:
    depth_order), of the names of the galaxies and of the observations,
    and the prefixes of the parameters of each galaxy: its items, each the
    name of its group (None for a galaxy alone) and the indices of its
    galaxies, and the parameter of the separation of each galaxy of a
    group after its first ('{prefix}separation').

    Raises
    ------
    ValueError
        If a name is unknown or repeats, or a group has fewer than two
        galaxies or the name of a galaxy or an observation.
    """

    def __init__(
            self,
            order: Sequence['str | DepthGroup'],
            galaxies: Sequence[str | None],
            observations: Sequence[str | None],
            prefixes: Sequence[str]
    ):
        self._items = _items(order, galaxies, observations)
        self._separations = {
            g: f'{prefixes[g]}separation'
            for name, members in self._items
            if name is not None for g in members[1:]}

    def items(self) -> tuple[tuple[str | None, tuple[int, ...]], ...]:
        """
        Return its items, front to back: each the name of its group (None
        for a galaxy alone) and the indices of its galaxies.
        """
        return self._items

    def separations(self) -> dict[int, str]:
        """
        Return the parameter of the separation of each galaxy of a group
        after its first, by its index.
        """
        return dict(self._separations)

    def pdescs(self) -> dict[str, ParamDesc]:
        """Return the parameters of the separations, by name."""
        return {
            separation: ParamScalarDesc(
                title='Separation along the line of sight',
                desc="the position of the galaxy along the line of sight "
                     "from the first of its group (away from the viewer)",
                unit='arcsec', default=0)
            for separation in self._separations.values()}

    def seen(self, seen: Sequence[int]) -> tuple[DepthUnit, ...]:
        """
        Return the order of the galaxies of the given indices, as an
        observation that sees them sees it (see SeenPlane.order): of
        each item, the galaxies of it that it sees, by their positions
        among them (a group of which it sees one galaxy is that galaxy).
        """
        order = []
        for name, members in self._items:
            visible = [g for g in members if g in seen]
            galaxies = tuple(seen.index(g) for g in visible)
            if len(visible) == 1:
                order.append(DepthUnit(galaxies))
            elif visible:
                order.append(DepthUnit(galaxies, name, tuple(
                    self._separations.get(g) for g in visible)))
        return tuple(order)


def _items(
        order: Sequence[str | DepthGroup],
        galaxies: Sequence[str | None],
        observations: Sequence[str | None]
) -> tuple[tuple[str | None, tuple[int, ...]], ...]:
    """
    Return the items of a depth order (see ObservationGroup: depth_order)
    as the name of each group (None for a galaxy) and the indices of its
    galaxies, among the names of the galaxies. Raise ValueError if a name
    is unknown or repeats, or a group has fewer than two galaxies or the
    name of a galaxy or an observation.
    """
    items = []
    for item in order:
        if isinstance(item, DepthGroup):
            check_name(item.name)
            if len(item.galaxies) < 2:
                raise ValueError(
                    f"the group {item.name!r} of depth_order needs at least "
                    f"two galaxies; it has {list(item.galaxies)}")
            if item.name in [*galaxies, *observations]:
                raise ValueError(
                    f"the group {item.name!r} of depth_order has the name of "
                    f"a galaxy or an observation")
            items.append((item.name, tuple(item.galaxies)))
        else:
            items.append((None, (item,)))
    names = [name for _, members in items for name in members]
    if unknown := [name for name in names if name not in galaxies]:
        raise ValueError(
            f"depth_order names unknown galaxies {unknown}; the galaxies are "
            f"{list(galaxies)}")
    if len(set(names)) != len(names):
        raise ValueError(f"depth_order repeats galaxies: {names}")
    return tuple(
        (name, tuple(galaxies.index(galaxy) for galaxy in members))
        for name, members in items)
