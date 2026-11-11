"""
The cone component: shells around an apex, with traits.
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal

import numpy as np

from gbkfit.config import Choices, check_options
from gbkfit.driver import Driver
from gbkfit.grid import Grid
from gbkfit.params import ParamDesc
from .. import traits
from .._native import make_bands, make_lines
from .._common import Interp, RNodes, RStep
from .._units import DataUnit
from ..base import Component, ComponentPlan, FrameParams, SceneTarget
from ..lines import Band
from ..nodes import Nodes, as_nodes
from ..parts import KinematicTraits, density_traits, kinematic_traits
from ..roles import STARS, Emission, Opacity
from . import _build, _nodal


__all__ = [
    'Cone'
]


class Cone(Component):
    """
    A hollow cone, on one side of its apex or both (e.g. a biconical
    outflow). It is evaluated at every voxel of a grid with a depth axis.
    Its walls are around the mid-surface at the angle wall (a parameter)
    from its axis, on one side of its apex or both; its traits are
    functions of the distance from the apex and the azimuth around the
    axis (polar traits), or of the distance from the mid-surface across
    the walls (height traits; see traits). Its density traits are the
    density per unit area of the mid-surface, and its density height
    traits its profile across the walls (uniform: a wall of hard edges).

    Parameters
    ----------
    rptraits : RPTrait or Sequence of RPTrait
        Its density polar traits: the density per unit area of its
        mid-surface (parameters 'rpt_*').
    rhtraits : RHTrait or Sequence of RHTrait
        Its density height traits, its profile across the walls: one that
        every polar trait shares, or one for each (see ThickDisk).
    vptraits : VPTrait or Sequence of VPTrait, optional
        Its velocity polar traits (parameters 'vpt_*'), around the axis
        ('tan') or out of the apex ('out'). A cone with vptraits and
        dptraits moves; one that emits lines must.
    dptraits : DPTrait or Sequence of DPTrait, optional
        Its velocity dispersion polar traits (parameters 'dpt_*').
    vhtraits, dhtraits : optional
        Its velocity and dispersion height traits, one for each polar
        trait (see parts.KinematicTraits).
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
    rnodes : Nodes or Sequence of float
        The radii of its shells (arcsec), from its apex (0): given, or
        evenly spaced (see Nodes), which its configuration keeps.
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
        (e.g. lines without kinematics).
    """

    @staticmethod
    def type() -> str:
        return 'cone'

    @staticmethod
    def type_title() -> str:
        return 'Cone'

    def dump(self) -> dict[str, Any]:
        return _build.dump_cone_component(
            self, self._cone, self._rnodes, self._kinematics)

    def __init__(
            self,
            rptraits: traits.RPTrait | Sequence[traits.RPTrait],
            rhtraits: traits.RHTrait | Sequence[traits.RHTrait],
            vptraits: traits.VPTrait | Sequence[traits.VPTrait] | None = None,
            dptraits: traits.DPTrait | Sequence[traits.DPTrait] | None = None,
            vhtraits:
            traits.VHTrait | Sequence[traits.VHTrait | None] | None = None,
            dhtraits:
            traits.DHTrait | Sequence[traits.DHTrait | None] | None = None,
            emission: Emission | None = None,
            opacity: Opacity | None = None,
            sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None = None,
            sides: Annotated[Literal[_nodal.SIDES], Choices(
                both='Both sides', positive='Positive side',
                negative='Negative side')] = 'both',
            rnodes: Nodes | RNodes | None = None,
            rstep: RStep | None = None,
            interp: Interp = 'linear',
            name: str | None = None
    ):
        check_options(Cone.__init__, locals())
        super().__init__(name, emission, opacity)
        rptraits, rhtraits, _ = density_traits(rptraits, rhtraits, True)
        self._kinematics = kinematics = kinematic_traits(
            vptraits, dptraits, vhtraits, dhtraits)
        # In the form it was given in, for its configuration (see dump)
        self._rnodes = rnodes = as_nodes(rnodes)
        self._cone = _build.make_cone(
            _nodal.SmoothCone, rptraits, rhtraits, kinematics, emission,
            opacity, sptraits, sides, rnodes=rnodes, rstep=rstep,
            interp=interp)
        self._lines = make_lines(self._cone.pdescs(), emission)
        self._bands = make_bands(self._cone.pdescs(), emission)

    def check_lines(self, lines: tuple[str, ...] | None) -> None:
        if lines is not None and STARS in lines and self.kinematics() is None:
            raise ValueError(
                "a spectral cube sees the stars of a cone with kinematics "
                "only (its stars have no velocities to place in channels)")

    def kinematics(self) -> KinematicTraits | None:
        """Return its kinematic traits, if it moves."""
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

    def param_owners(self) -> dict[str, tuple[str | int, ...]]:
        owners = super().param_owners()
        owners.update({
            name: option
            for name, option in self._cone.trait_owners().items()
            if name in owners})
        return owners

    def constants(self) -> dict[str, Any]:
        return dict(
            rnodes=self._cone.rnodes(), subrnodes=self._cone.subrnodes())

    def plan(
            self,
            target: SceneTarget,
            emits: bool,
            opacity_ratios: np.ndarray | None,
            redshift: float
    ) -> ComponentPlan:
        driver, spectral, dtype = target.driver, target.spectral, target.dtype
        lines, band, data_unit = target.lines, target.band, target.data_unit
        emission = self.emission()
        return _build.cone_plan(
            self._cone, self._lines, self._bands,
            emission.unit() if emission is not None else None, driver,
            spectral, dtype, lines, band, emits, opacity_ratios, data_unit,
            redshift)
