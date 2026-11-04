"""
The cone component: shells around an apex, with traits.
"""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from gbkfit.config import check_options
from gbkfit.driver import Driver
from gbkfit.grid import Grid
from gbkfit.params import ParamDesc
from .. import traits
from .._native import make_bands, make_lines
from .._detail import Interp, RNodes, RStep
from .._units import DataUnit
from ..base import Component, ComponentPlan, FrameParams
from ..lines import Band
from ..nodes import EvenNodes
from ..parts import ConeTraitsKinematics, Density
from ..roles import STARS, Emission, Opacity
from . import _cone, _detail


__all__ = [
    'Cone'
]


class Cone(Component):
    """
    A cone, evaluated at every voxel of a grid with a depth axis.
    Its walls are around the mid-surface at the angle wall (a parameter)
    from its axis, on one side of its apex or both; its traits are
    functions of the distance from the apex and the azimuth around the
    axis (polar traits), or of the distance from the mid-surface across
    the walls (height traits; see traits). Its density traits are the
    density per unit area of the mid-surface, and its density height
    traits its profile across the walls (uniform: a wall of hard edges).

    Parameters
    ----------
    density : Density
        Its density, with height traits.
    kinematics : ConeTraitsKinematics, optional
        How it moves; required to emit lines. Its velocity traits are
        around the axis ('tan') or out of the apex ('out').
    emission : Emission, optional
        The light it emits.
    opacity : DustOpacity, optional
        The dust that removes the light behind it.
    sptraits : SPTrait or Sequence of SPTrait, optional
        Its selection polar traits: the parts of the cone that are kept
        (e.g. a range of radii), for all its roles.
    sides : str, optional
        Its sides: 'both', 'positive' (along the positive side of its axis,
        away from the viewer for inclinations below 90 degrees) or
        'negative'.
    rnodes : Sequence of float or EvenNodes
        The radii of its shells (arcsec), from its apex (0): a list, or
        evenly spaced (see EvenNodes), which its configuration keeps.
        Beyond its last radius, it goes on as a disk does beyond its last
        ring (see Disk).
    rstep : float, optional
        The width of the shells it is evaluated on, at most half the
        smallest separation of its radii; by default that, or 1 if less.
    interp : str, optional
        How its node-wise parameters are interpolated between its radii:
        'linear', 'akima' or 'pchip'.
    name : str, optional
        Its name (see Component).

    Raises
    ------
    ValueError
        If the options are not valid (e.g. the shells), the cone does not
        take one of the traits, or its parts and roles do not go together
        (e.g. lines without kinematics, or a density without height
        traits).
    """

    @staticmethod
    def type() -> str:
        return 'cone'

    def dump(self) -> dict[str, Any]:
        return _detail.dump_cone_component(
            self, self._cone, self._rnodes, self._density, self._kinematics)

    def __init__(
            self,
            density: Density,
            kinematics: ConeTraitsKinematics | None = None,
            emission: Emission | None = None,
            opacity: Opacity | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            sides: Literal[_cone.SIDES] = 'both',
            rnodes: RNodes | EvenNodes | None = None,
            rstep: RStep | None = None,
            interp: Interp = 'linear',
            name: str | None = None
    ):
        check_options(Cone.__init__, locals())
        super().__init__(name, emission, opacity)
        self._density = density
        self._kinematics = kinematics
        # As given, for its configuration (see dump)
        self._rnodes = rnodes
        self._cone = _detail.make_cone(
            _cone.SmoothCone, density, kinematics, emission, opacity,
            sptraits, sides, rnodes=rnodes, rstep=rstep, interp=interp)
        self._lines = make_lines(self._cone.pdescs(), emission)
        self._bands = make_bands(self._cone.pdescs(), emission)

    def density(self) -> Density:
        """Return its density."""
        return self._density

    def check_lines(self, lines: tuple[str, ...] | None) -> None:
        if lines is not None and STARS in lines and self.kinematics() is None:
            raise ValueError(
                "a spectral cube sees the stars of a cone with kinematics "
                "only (its stars have no velocities to place in channels)")

    def kinematics(self) -> ConeTraitsKinematics | None:
        """Return its kinematics, if any."""
        return self._kinematics

    def sides(self) -> str:
        """Return its sides."""
        return self._cone.sides()

    def thick(self, seen_by: 'type[Emission]') -> bool:
        return True

    def frame_params(self) -> FrameParams:
        return self._cone.frame_params()

    def pdescs(self) -> dict[str, ParamDesc]:
        lines = self._lines.pdescs() if self._lines is not None else {}
        bands = self._bands.pdescs() if self._bands is not None else {}
        return self._cone.pdescs() | lines | bands

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._cone.rnodes(), subrnodes=self._cone.subrnodes())

    def plan(
            self,
            driver: Driver,
            spectral: Grid,
            dtype: np.dtype,
            lines: tuple[str, ...] | None,
            band: Band | None,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            data_unit: DataUnit | None,
            redshift: float
    ) -> ComponentPlan:
        emission = self.emission()
        return _detail.cone_plan(
            self._cone, self._lines, self._bands,
            emission.unit() if emission is not None else None, driver,
            spectral, dtype, lines, band, emits, opacity_ratios, data_unit,
            redshift)
