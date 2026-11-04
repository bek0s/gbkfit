"""
The radial nodes of components (the rings of a disk, the shells of a
cone, the nodes of the traits of a spheroid): their option rnodes, the
radii of Nodes.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from gbkfit.config import Serializable, check_options
from gbkfit.utils import iterutils
from ._detail import RNodes, RnMax, RnSep


__all__ = [
    'Nodes'
]


class Nodes(Serializable):
    """
    The radii of the radial nodes of a component, from 0 (arcsec): given
    (Nodes), or evenly spaced, every sep to the first at or beyond max
    (see every), or len of them to max (see count). They keep the form
    they were made in, which their dump gives: their radii, {max, sep} or
    {max, len}; so the configuration of a component keeps the form it was
    given in.

    In configurations, the option rnodes of a component is a list of
    radii or one of the forms of the evenly spaced ones (see
    config_constructors); in Python, Nodes or the list.

    Parameters
    ----------
    radii : Sequence of float
        The radii: at least two, from 0, ascending and unique.

    Raises
    ------
    ValueError
        If the radii, or the options of a form, are not valid.
    """

    # The forms of configurations that are dicts, {max, sep} and
    # {max, len}; that of the radii is a list, which the components that
    # take Nodes take too (see as_nodes)
    config_constructors = ('every', 'count')

    @classmethod
    def every(cls, max: RnMax, sep: RnSep) -> 'Nodes':
        """
        Make nodes from 0 every sep, to the first at or beyond max.

        Raises
        ------
        ValueError
            Unless 0 < sep <= max.
        """
        check_options(Nodes.every, locals())
        if not 0 < sep <= max:
            raise ValueError(
                f"sep must be greater than 0 and at most max ({max}); it is "
                f"{sep}")
        # The count tolerates the rounding of max / sep, which made
        # np.arange to max + sep add a node (e.g. 2.4 for 2.2 in 0.2s)
        count = int(np.ceil(max / sep - 1e-9)) + 1
        nodes = cls((sep * np.arange(count)).tolist())
        nodes._form = dict(max=max, sep=sep)
        return nodes

    @classmethod
    def count(cls, max: RnMax, len: int) -> 'Nodes':
        """
        Make len nodes from 0 to max.

        Raises
        ------
        ValueError
            Unless 0 < max and 2 <= len.
        """
        check_options(Nodes.count, locals())
        if not 0 < max:
            raise ValueError(f"max must be greater than 0; it is {max}")
        if not 2 <= len:
            raise ValueError(f"len must be at least 2; it is {len}")
        nodes = cls(np.linspace(0, max, len).tolist())
        nodes._form = dict(max=max, len=len)
        return nodes

    def dump(self) -> list[float] | dict[str, Any]:  # type: ignore[override]
        # The list of the radii is a form of its own, as configurations
        # give it
        return list(self._radii) if self._form is None else dict(self._form)

    def __init__(self, radii: RNodes):
        check_options(Nodes.__init__, locals())
        radii = tuple(radii)
        if len(radii) < 2:
            raise ValueError("at least two rnodes must be provided")
        # From the centre: a component has no hole of nothing at all, which
        # was a sharp inner edge; one of little density has 0 at its nodes
        if radii[0] != 0:
            raise ValueError(
                f"rnodes must start at 0; they start at {radii[0]}")
        if not iterutils.is_ascending(radii):
            raise ValueError("rnodes must be ascending")
        if not iterutils.all_unique(radii):
            raise ValueError("rnodes must be unique")
        self._radii = radii
        # The options of the form it was made in, if evenly spaced
        self._form: dict[str, Any] | None = None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Nodes) and self.dump() == other.dump()

    def __hash__(self) -> int:
        return hash(self._radii)

    def __repr__(self) -> str:
        if self._form is None:
            return f'Nodes({list(self._radii)!r})'
        method = 'every' if 'sep' in self._form else 'count'
        options = ', '.join(f'{k}={v!r}' for k, v in self._form.items())
        return f'Nodes.{method}({options})'

    def radii(self) -> tuple[float, ...]:
        """Return the radii."""
        return self._radii


def as_nodes(rnodes: 'Nodes | Sequence[float] | None') -> Nodes:
    """
    Return the Nodes of the option rnodes of a component: Nodes, or a
    list of radii. Raise ValueError if it is not given, or not valid.
    """
    if rnodes is None:
        raise ValueError(
            "rnodes must be given: a list of radii from 0, or radii evenly "
            "spaced, {max, sep} or {max, len}")
    return rnodes if isinstance(rnodes, Nodes) else Nodes(rnodes)
