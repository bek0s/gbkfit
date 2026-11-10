"""
Helpers of the cone component (Cone).
"""

from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import Driver
from gbkfit.grid import Grid
from gbkfit.utils import iterutils
from .. import traits
from .._detail import dump_name
from .._native import NativeComponentPlan, emission_setup
from .._nodal import (
    MAX_TRAITS, TraitTable, parse_component_rnode_args,
    with_default_height_traits)
from .. import _units
from .._units import DataUnit
from ..base import Component
from ..lines import Band, Bands, Lines
from ..nodes import Nodes
from ..parts import KinematicTraits
from ..roles import (
    DustOpacity, Emission, LineEmission, Opacity, emission_parser,
    opacity_parser)
from . import _cone


__all__ = [
    'CONE_TRAITS',
    'cone_plan',
    'dump_cone_component',
    'make_cone'
]


# The traits that the cones take: the polar traits of the radius and the
# azimuth alone (not the mixtures, maps and distortions of the plane of a
# disk), and every height trait and velocity trait but the rotation in the
# gravity of the galaxy (mass, of a disk) and the flows in wedges, along
# bars and of spiral waves of a disk, around the axis ('tan') or out of
# the apex ('out'); dispersion traits isotropic only (a cone has no
# radial, tangential and vertical directions of a disk)
CONE_TRAITS = TraitTable('cone', dict(
    rpt=(
        traits.RPTraitUniform, traits.RPTraitExponential,
        traits.RPTraitGauss, traits.RPTraitGGauss, traits.RPTraitLorentz,
        traits.RPTraitMoffat, traits.RPTraitSech2, traits.RPTraitNWUniform,
        traits.RPTraitNWHarmonic),
    rht=tuple(
        cls for cls in traits.rht_parser.registered_classes().values()
        if cls is not traits.RHTraitEquilibrium),
    vpt=tuple(
        cls for cls in traits.vpt_parser.registered_classes().values()
        if cls not in (
            traits.VPTraitMass, traits.VPTraitWedge, traits.VPTraitBar,
            traits.VPTraitNWBar, traits.VPTraitSpiral)),
    vht=tuple(
        cls for cls in traits.vht_parser.registered_classes().values()
        if cls is not traits.VHTraitMass),
    dpt=(
        traits.DPTraitUniform, traits.DPTraitExponential,
        traits.DPTraitGauss, traits.DPTraitGGauss, traits.DPTraitLorentz,
        traits.DPTraitMoffat, traits.DPTraitSech2, traits.DPTraitNWUniform,
        traits.DPTraitNWHarmonic),
    dht=tuple(traits.dht_parser.registered_classes().values()),
    spt=(
        traits.SPTraitRadialRange, traits.SPTraitAzimuthalRange,
        traits.SPTraitNWAzimuthalRange)),
    ('tan', 'out'), ('iso',))


def dump_cone_component(
        component: Component,
        cone: _cone.NodalCone,
        rnodes: Nodes,
        kinematics: KinematicTraits | None
) -> dict[str, Any]:
    """
    Return the configuration of a cone component made of the given cone,
    with its shells in the form they were given (rnodes).
    """
    info = dict(type=component.type()) | dump_name(component) | dict(
        sides=cone.sides(), rnodes=rnodes.dump(),
        rstep=cone.rstep(),
        interp=cone.interp().type(),
        rptraits=traits.rpt_parser.dump(list(cone.traits('rpt'))),
        rhtraits=traits.rht_parser.dump(
            list(cone.traits('rht')[:1 if cone.shared('rht') else None])))
    if kinematics is not None:
        info.update(kinematics.dump())
    if component.emission() is not None:
        info.update(emission=emission_parser.dump(component.emission()))
    if component.opacity() is not None:
        info.update(opacity=opacity_parser.dump(component.opacity()))
    if cone.traits('spt'):
        info.update(sptraits=traits.spt_parser.dump(list(cone.traits('spt'))))
    return info


def make_cone(
        cone_class: type[_cone.NodalCone],
        rptraits: tuple[traits.RPTrait, ...],
        rhtraits: tuple[traits.RHTrait, ...],
        kinematics: KinematicTraits | None,
        emission: Emission | None,
        opacity: Opacity | None,
        sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None,
        sides: str,
        rnodes: Nodes,
        rstep: float | None,
        interp: str
) -> _cone.NodalCone:
    """
    Return the cone of a cone component, of class cone_class, made of its
    traits (see Cone for the options). It takes the default height
    traits of its velocity and dispersion traits where it has none.

    Raises
    ------
    ValueError
        If the options are not valid, the cone does not take one of the
        traits (see CONE_TRAITS), or the parts and roles do not go
        together (e.g. lines without kinematics).
    """
    if isinstance(emission, LineEmission) and kinematics is None:
        raise ValueError(
            "a cone that emits lines needs kinematics: give it vptraits and "
            "dptraits")
    if opacity is not None and not isinstance(opacity, DustOpacity):
        raise ValueError("the opacity of a cone is dust (type: dust)")
    sptraits = iterutils.tuplify(sptraits)
    if len(sptraits) > MAX_TRAITS:
        raise ValueError(
            f"at most {MAX_TRAITS} sptraits are supported; {len(sptraits)} "
            f"were given")
    traits_ = dict(rpt=rptraits, rht=rhtraits, spt=sptraits)
    if kinematics is not None:
        traits_.update(
            vpt=kinematics.vptraits(), dpt=kinematics.dptraits(),
            vht=with_default_height_traits(
                kinematics.vhtraits(), traits.VHTraitOne),
            dht=with_default_height_traits(
                kinematics.dhtraits(), traits.DHTraitOne))
    CONE_TRAITS.check(traits_)
    node_args = parse_component_rnode_args(rnodes, rstep, interp)
    return cone_class(
        sides=sides, **node_args, traits_=traits_, prefixes={},
        density_units=dict(rpt=_units.density_unit(emission)))


def cone_plan(
        cone: _cone.NodalCone,
        lines: Lines | None,
        bands: Bands | None,
        unit: astropy.units.UnitBase | None,
        driver: Driver,
        spectral: Grid,
        dtype: np.dtype,
        names: tuple[str, ...] | None,
        band: Band | None,
        emits: bool,
        opacity_ratios: np.ndarray | None,
        data_unit: DataUnit | None,
        redshift: float
) -> NativeComponentPlan:
    """
    Return the plan of a cone component made of the given cone (see
    disks._detail.disk_plan for the arguments); a cone has no line
    opacity.
    """
    emission = emission_setup(
        lines, bands, unit, spectral, names, band, emits, data_unit,
        redshift, ())
    return NativeComponentPlan(
        driver, cone.plan(driver, emission.nlines, dtype), emission,
        opacity_ratios)
