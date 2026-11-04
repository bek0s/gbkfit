"""
The radial nodes of components (the rings of a disk, the shells of a
cone): their option rnodes, a list of radii or radii evenly spaced
(EvenNodes).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.config import Serializable, check_options
from gbkfit.utils import iterutils
from ._detail import RnMax, RnSep


__all__ = [
    'EvenNodes'
]


class EvenNodes(Serializable):
    """
    Radii of nodes from 0 to a maximum, evenly spaced: every sep, to the
    first at or beyond max (see every), or len of them (see count). The
    components that take them keep them in this form (see their dump),
    so that their configurations keep the form they were given in.

    Raises
    ------
    ValueError
        If the options are not valid.
    """

    # The two forms of configurations: {max, sep} and {max, len}
    config_constructors = ('every', 'count')

    @classmethod
    def every(cls, max: RnMax, sep: RnSep) -> 'EvenNodes':
        """
        Make nodes from 0 every sep (arcsec), to the first at or beyond
        max (arcsec).

        Raises
        ------
        ValueError
            Unless 0 < sep <= max.
        """
        check_options(EvenNodes.every, locals())
        return cls(max, sep, None)

    @classmethod
    def count(cls, max: RnMax, len: int) -> 'EvenNodes':
        """
        Make len nodes from 0 to max (arcsec).

        Raises
        ------
        ValueError
            Unless 0 < max and 2 <= len.
        """
        check_options(EvenNodes.count, locals())
        return cls(max, None, len)

    def dump(self) -> dict[str, Any]:
        return dict(max=self._max) | (
            dict(sep=self._sep) if self._sep is not None
            else dict(len=self._len))

    def __init__(self, max: float, sep: float | None, len: int | None):
        # The constructors of the forms check the options; this one only
        # their values, given in Python by them
        if not 0 < max:
            raise ValueError(f"max must be greater than 0; it is {max}")
        if sep is not None and not 0 < sep <= max:
            raise ValueError(
                f"sep must be greater than 0 and at most max ({max}); it is "
                f"{sep}")
        if len is not None and not 2 <= len:
            raise ValueError(f"len must be at least 2; it is {len}")
        self._max = max
        self._sep = sep
        self._len = len

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EvenNodes) and self.dump() == other.dump()

    def __repr__(self) -> str:
        options = ', '.join(f'{k}={v!r}' for k, v in self.dump().items())
        return f'EvenNodes({options})'

    def radii(self) -> tuple[float, ...]:
        """Return the radii of the nodes."""
        if self._sep is None:
            return tuple(np.linspace(0, self._max, self._len).tolist())
        # To the first node at or beyond max; the count tolerates the
        # rounding of max / sep, which made np.arange to max + sep add a
        # node (e.g. 2.4 for 2.2 in 0.2s)
        count = int(np.ceil(self._max / self._sep - 1e-9)) + 1
        return tuple((self._sep * np.arange(count)).tolist())


def node_radii(
        nodes: Sequence[float] | EvenNodes | None
) -> tuple[float, ...]:
    """
    Return the radii of the radial nodes of a component, of its option
    rnodes. Raise ValueError if they are not given, or are not at least
    two, from 0, ascending and unique.
    """
    if nodes is None:
        raise ValueError(
            "rnodes must be given: a list of radii from 0, or radii evenly "
            "spaced, {max, sep} or {max, len}")
    radii = nodes.radii() if isinstance(nodes, EvenNodes) else tuple(nodes)
    if len(radii) < 2:
        raise ValueError("at least two rnodes must be provided")
    # From the centre: a component has no hole of nothing at all, which
    # was a sharp inner edge; one of little density has 0 at its nodes
    if radii[0] != 0:
        raise ValueError(f"rnodes must start at 0; they start at {radii[0]}")
    if not iterutils.is_ascending(radii):
        raise ValueError("rnodes must be ascending")
    if not iterutils.all_unique(radii):
        raise ValueError("rnodes must be unique")
    return radii


def dump_nodes(
        nodes: Sequence[float] | EvenNodes | None,
        radii: Sequence[float]
) -> list[float] | dict[str, Any]:
    """
    Return the option rnodes of the configuration of a component, in the
    form it was given (nodes): its radii, or EvenNodes as such.
    """
    return nodes.dump() if isinstance(nodes, EvenNodes) else list(radii)
