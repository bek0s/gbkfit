"""
Geometries that components share: their centre, orientation and systemic
velocity, possibly warped.
"""

from collections.abc import Sequence
from typing import Any

from gbkfit.utils import parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _detail


__all__ = [
    'GEOMETRY_PARAMS',
    'Geometry',
    'geometry_parser'
]


# The geometric parameters of the components, and the warp that makes each
# of them node-wise (one value for each ring): loose (the centre and the
# systemic velocity) or tilted (the position angle and the inclination)
GEOMETRY_PARAMS = dict(
    vsys='loose', xpos='loose', ypos='loose', posa='tilted', incl='tilted')


class Geometry(parseutils.Serializable):
    """
    A geometry that components share: some or all of their geometric
    parameters (see GEOMETRY_PARAMS), which are then parameters of the
    geometry, named after it (e.g. disk_posa), instead of each component.

    The components that use a geometry (see Component.geometry) take its
    warps and, if it is warped, its rings: their geometric parameters
    that it does not share are their own, but also node-wise on its rings
    if it is warped. A component of one centre (e.g. a point) cannot use
    a loose geometry. Geometries are values: components share the
    geometry of a name, and must not have different ones of one name.

    Parameters
    ----------
    name : str
        Its name, which prefixes its parameters.
    params : Sequence of str, optional
        The geometric parameters it shares; by default, all those of the
        components that use it.
    loose : bool, optional
        Whether the centre and the systemic velocity are node-wise.
    tilted : bool, optional
        Whether the position angle and the inclination are node-wise.
    rnmin, rnmax, rnsep, rnlen, rnodes : optional
        The radii of the rings of a warped geometry (one of: rnodes;
        rnmin, rnmax and rnsep; rnmin, rnmax and rnlen); none without
        warps, when each component has its own rings.

    Raises
    ------
    ConfigError
        If the name is not valid, the parameters are not geometric or
        repeat, or the rings are given without warps or not given with
        them.
    """

    def dump(self) -> dict[str, Any]:
        params = {} if self._params is None else dict(
            params=list(self._params))
        rnodes = {} if self._rnodes is None else dict(
            rnodes=list(self._rnodes))
        return dict(name=self._name) | params | dict(
            loose=self._loose, tilted=self._tilted) | rnodes

    def __init__(
            self,
            name: str,
            params: Sequence[str] | None = None,
            loose: bool = False,
            tilted: bool = False,
            rnmin: float | None = None,
            rnmax: float | None = None,
            rnsep: float | None = None,
            rnlen: int | None = None,
            rnodes: Sequence[float] | None = None
    ):
        parseutils.check_name(name)
        if name is None:
            raise ConfigError("a geometry needs a name")
        if params is not None:
            params = tuple(params)
            if not params:
                raise ConfigError("a geometry needs at least one parameter")
            if unknown := [p for p in params if p not in GEOMETRY_PARAMS]:
                raise ConfigError(
                    f"the parameters of a geometry are some of "
                    f"{list(GEOMETRY_PARAMS)}; {unknown} are not")
            if len(set(params)) != len(params):
                raise ConfigError(
                    f"the parameters of a geometry repeat: {list(params)}")
        given = [rnmin, rnmax, rnsep, rnlen, rnodes]
        if loose or tilted:
            rnodes = _detail.parse_nodes(
                'r', rnmin, rnmax, rnsep, rnlen, rnodes)
        elif any(value is not None for value in given):
            raise ConfigError(
                "the rings of a geometry are those of its warps; without "
                "warps (loose or tilted), each component has its own")
        self._name = name
        self._params = params
        self._loose = loose
        self._tilted = tilted
        self._rnodes = rnodes

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Geometry) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def _key(self) -> tuple:
        """Return what makes the geometry: its name and options."""
        return (
            self._name, self._params, self._loose, self._tilted,
            self._rnodes)

    def name(self) -> str:
        """Return its name."""
        return self._name

    def params(self) -> tuple[str, ...] | None:
        """Return the geometric parameters it shares, if given."""
        return self._params

    def loose(self) -> bool:
        """Return whether the centre and the systemic velocity are warped."""
        return self._loose

    def tilted(self) -> bool:
        """Return whether the position angle and inclination are warped."""
        return self._tilted

    def rnodes(self) -> tuple[float, ...] | None:
        """Return the radii of its rings, if it is warped."""
        return self._rnodes


geometry_parser = parseutils.BasicParser(Geometry)
