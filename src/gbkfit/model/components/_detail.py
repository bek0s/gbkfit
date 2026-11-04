"""
Helpers shared by the components.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

import numpy as np

from gbkfit.config import Unit
from gbkfit.params import ParamValues
from ._units import BandFactors, LineFactors

if TYPE_CHECKING:
    from .base import Component


# The interpolations of the values of node-wise parameters between nodes
INTERPOLATIONS = ('linear', 'akima', 'pchip')

# The options of the radii of the nodes of a component (its rings or
# shells; see nodes.node_radii), of its subnodes (rstep), and of the
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
