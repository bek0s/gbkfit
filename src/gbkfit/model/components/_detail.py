"""
Helpers shared by the components.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

import numpy as np

from gbkfit.config import Unit
from gbkfit.params import ParamValues
from gbkfit.utils import iterutils
from ._units import BandFactors, LineFactors

if TYPE_CHECKING:
    from .base import Component


# The interpolations of the values of node-wise parameters between nodes
INTERPOLATIONS = ('linear', 'akima', 'pchip')

# The options of the radii of the nodes of a component (its rings or
# shells; see parse_nodes), of its subnodes (rstep), and of the
# interpolation between them, as the components take them: one definition
# of each
RnMax: TypeAlias = Annotated[float, Unit('arcsec')]
RnSep: TypeAlias = Annotated[float, Unit('arcsec')]
RNodes: TypeAlias = Annotated[Sequence[float], Unit('arcsec')]
RStep: TypeAlias = Annotated[float, Unit('arcsec')]
Interp: TypeAlias = Literal[INTERPOLATIONS]


def disk_to_sky(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | float,
        xpos: np.ndarray | float,
        ypos: np.ndarray | float,
        posa: np.ndarray | float,
        incl: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the positions on the sky (x and y, arcsec, as xpos and ypos)
    and the depths along the line of sight (arcsec, away from the viewer)
    of points of a disk: at (x, y) in its plane (x along the azimuth 0, y
    along the azimuth 90 degrees) and the height z above it, of its
    centre (xpos, ypos), position angle posa and inclination incl
    (degrees), as the kernels place them (transform_cpos_posa_incl_inverse
    in gmodels.hpp). The arguments broadcast together.
    """
    incl = np.radians(incl)
    # The inclination about the major axis (x)...
    y_incl = y * np.cos(incl) - z * np.sin(incl)
    depth = y * np.sin(incl) + z * np.cos(incl)
    # ... then the position angle and the centre
    theta = -np.radians(posa) - np.pi
    sky_x = -x * np.sin(theta) + y_incl * np.cos(theta)
    sky_y = -x * np.cos(theta) - y_incl * np.sin(theta)
    return sky_x + xpos, sky_y + ypos, depth


class Fluxes:
    """
    The fluxes of the rows of the light of a component that an
    observation sees (see ComponentPlan.line_values): 1 for its first line
    (or band), and for the others their flux ratio parameters, by row
    (ratios), times the factors that turn the unit of its light into that
    of the data (see _units), if any: those of its lines or of its band,
    at its systemic velocity (see systemic_velocity).
    """

    def __init__(
            self,
            nrows: int,
            ratios: Sequence[tuple[int, str]] = (),
            factors: LineFactors | BandFactors | None = None
    ):
        self._nrows = nrows
        self._ratios = tuple(ratios)
        self._factors = factors

    def __call__(self, params: ParamValues) -> np.ndarray:
        """Return the flux of each row, with the given parameters."""
        fluxes = np.ones(self._nrows)
        for row, name in self._ratios:
            fluxes[row] *= params[name]
        if self._factors is not None:
            fluxes *= self._factors(systemic_velocity(params))
        return fluxes


def systemic_velocity(params: ParamValues) -> float:
    """
    Return the systemic velocity of a component (km/s) with the given
    parameters: its parameter vsys (the mean of its values at the nodes,
    if node-wise), or 0 without one (e.g. a disk without velocity
    traits).
    """
    return float(np.mean(params.get('vsys', 0.0)))


def dump_name(component: 'Component') -> dict[str, Any]:
    """Return the name option of a component: none if it has no name."""
    name = component.name()
    return dict(name=name) if name is not None else {}


def parse_nodes(
        prefix: str,
        nmax: float | None,
        nsep: float | None,
        nlen: int | None,
        nodes: Any
) -> tuple[float, ...]:
    """
    Return the nodes given by one of: nodes; nmax and nsep (from 0 every
    nsep, to the first node at or beyond nmax); nmax and nlen (nlen nodes
    from 0 to nmax). The prefix names the options in messages (e.g. 'r'
    for rnodes). Raise ValueError unless exactly one of them is given
    (alone), or the nodes are not at least two, from 0, ascending and
    unique.
    """
    # (exactly one set, alone: an option beside another set would be
    # ignored silently)
    given = {name for name, value in (
        ('nodes', nodes), ('nmax', nmax), ('nsep', nsep), ('nlen', nlen))
        if value is not None}
    if given not in ({'nodes'}, {'nmax', 'nsep'}, {'nmax', 'nlen'}):
        raise ValueError(
            f"exactly one of the following sets of options must be given, "
            f"alone: (1) {prefix}nodes; (2) {prefix}nmax, {prefix}nsep; "
            f"(3) {prefix}nmax, {prefix}nlen; given: "
            f"{sorted(prefix + name for name in given)}")
    nodes_arange = given == {'nmax', 'nsep'}
    nodes_linspace = given == {'nmax', 'nlen'}
    if (nodes_arange or nodes_linspace) and not 0 < nmax:
        raise ValueError(
            f"the following expression must be true: 0 < {prefix}nmax")
    if nodes_arange and not (0 < nsep <= nmax):
        raise ValueError(
            f"the following expression must be true: "
            f"0 < {prefix}nsep <= {prefix}nmax")
    if nlen is not None and not 2 <= nlen:
        raise ValueError(
            f"the following expression must be true: "
            f"2 <= {prefix}nlen")
    if nodes_arange:
        # From 0 every nsep, to the first node at or beyond nmax; the
        # count tolerates the rounding of nmax / nsep, which made
        # np.arange to nmax + nsep add a node (e.g. 2.4 for 2.2 in 0.2s)
        count = int(np.ceil(nmax / nsep - 1e-9)) + 1
        nodes = (nsep * np.arange(count)).tolist()
    elif nodes_linspace:
        nodes = np.linspace(0, nmax, nlen).tolist()
    nodes = tuple(nodes)
    if len(nodes) < 2:
        raise ValueError(f"at least two {prefix}nodes must be provided")
    # (from the centre: a component has no hole of nothing at all, which
    # was a sharp inner edge; one of little density has 0 at its nodes)
    if nodes[0] != 0:
        raise ValueError(
            f"{prefix}nodes must start at 0; they start at {nodes[0]}")
    if not iterutils.is_ascending(nodes):
        raise ValueError(f"{prefix}nodes must be ascending")
    if not iterutils.all_unique(nodes):
        raise ValueError(f"{prefix}nodes must be unique")
    return nodes
