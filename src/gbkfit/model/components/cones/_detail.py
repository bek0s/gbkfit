"""
Helpers of the cone components (SMCone).
"""

from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import Driver
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .. import traits
from .._detail import dump_name
from .._nodal import (
    MAX_TRAITS, NodalComponentPlan, TraitTable, emission_setup,
    parse_component_rnode_args, with_default_height_traits)
from .._units import DataUnit
from ..base import Component
from ..lines import Band, Bands, LineCatalogue, Lines
from ..parts import Density, Kinematics, density_parser, kinematics_parser
from ..roles import (
    DustOpacity, Emission, LineEmission, Opacity, emission_parser,
    opacity_parser)
from . import _cone


__all__ = [
    'CONE_TRAITS',
    'cone_plan',
    'dump_cone_component',
    'load_cone_component',
    'make_cone'
]


# The prefix of the parameters of the traits of each kind
TRAIT_PREFIXES = dict(
    rpt='pt', rht='ht', vpt='vpt', vht='vht', dpt='dpt', dht='dht',
    spt='spt')

# The traits that the cones take: the polar traits of the radius and the
# azimuth alone (not the mixtures, maps and distortions of the plane of a
# disk), and every height trait and velocity trait but the rotation in the
# gravity of the galaxy (mass, of a disk) and the flows in wedges, along
# bars and of spiral waves of a disk, around the axis ('tan') or out of
# the apex ('out')
CONE_TRAITS = TraitTable('cone', dict(
    rpt=(
        traits.RPTraitUniform, traits.RPTraitExponential,
        traits.RPTraitGauss, traits.RPTraitGGauss, traits.RPTraitLorentz,
        traits.RPTraitMoffat, traits.RPTraitSech2, traits.RPTraitNWUniform,
        traits.RPTraitNWHarmonic),
    rht=tuple(traits.rht_parser.registered_classes().values()),
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
    ('tan', 'out'))


def load_cone_component(
        cls: type[Component],
        info: dict[str, Any],
        catalogue: LineCatalogue | None
) -> dict[str, Any]:
    """
    Return the options of a cone component of class cls, with its parts,
    roles and traits loaded, and the lines its emission names from the
    catalogue (see Emission.load). The __init__ of cls declares the
    options.
    """
    for parser, key, required in (
            (density_parser, 'density', True),
            (kinematics_parser, 'kinematics', False),
            (opacity_parser, 'opacity', False),
            (traits.spt_parser, 'sptraits', False)):
        parseutils.load_option_and_update_info(
            parser, info, key, required=required)
    parseutils.load_option_and_update_info(
        emission_parser, info, 'emission', catalogue=catalogue)
    return parseutils.parse_options_for_callable(info, cls.__init__)


def dump_cone_component(
        component: Component,
        cone: _cone.Cone,
        density: Density,
        kinematics: Kinematics | None
) -> dict[str, Any]:
    """Return the configuration of a cone component made of the given cone."""
    info = dict(type=component.type()) | dump_name(component) | dict(
        sides=cone.sides(), rnodes=list(cone.rnodes()), rstep=cone.rstep(),
        interp=cone.interp().type(), density=density.dump())
    if kinematics is not None:
        info.update(kinematics=kinematics.dump())
    if component.emission() is not None:
        info.update(emission=emission_parser.dump(component.emission()))
    if component.opacity() is not None:
        info.update(opacity=opacity_parser.dump(component.opacity()))
    if cone.traits('spt'):
        info.update(sptraits=traits.spt_parser.dump(list(cone.traits('spt'))))
    return info


def make_cone(
        cone_class: type[_cone.Cone],
        density: Density,
        kinematics: Kinematics | None,
        emission: Emission | None,
        opacity: Opacity | None,
        sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None,
        sides: str,
        rnmin: float | None,
        rnmax: float | None,
        rnsep: float | None,
        rnlen: int | None,
        rnodes: Sequence[float] | None,
        rstep: float | None,
        interp: str
) -> _cone.Cone:
    """
    Return the cone of a cone component, of class cone_class, made of its
    parts (see SMCone for the options). It takes the default height
    traits of its velocity and dispersion traits where it has none.

    Raises
    ------
    ConfigError
        If the options are not valid, the cone does not take one of the
        traits (see CONE_TRAITS), or the parts and roles do not go
        together (e.g. lines without kinematics).
    """
    if sides not in _cone.SIDES:
        raise ConfigError(
            f"sides must be one of {list(_cone.SIDES)}; it is {sides!r}")
    if not density.thick():
        raise ConfigError(
            "a cone needs density height traits: its profile across its "
            "walls")
    if isinstance(emission, LineEmission) and kinematics is None:
        raise ConfigError("a cone that emits lines needs kinematics")
    if kinematics is not None and kinematics.equilibrium() is not None:
        raise ConfigError(
            "a cone has no vertical equilibrium: that is of disks")
    if opacity is not None and not isinstance(opacity, DustOpacity):
        raise ConfigError("the opacity of a cone is dust (type: dust)")
    sptraits = iterutils.tuplify(sptraits)
    if len(sptraits) > MAX_TRAITS:
        raise ConfigError(
            f"at most {MAX_TRAITS} sptraits are supported; {len(sptraits)} "
            f"were given")
    traits_ = dict(rpt=density.ptraits(), rht=density.htraits(), spt=sptraits)
    if kinematics is not None:
        traits_.update(
            vpt=kinematics.vptraits(), dpt=kinematics.dptraits(),
            vht=with_default_height_traits(
                kinematics.vhtraits(), len(kinematics.vptraits()),
                traits.VHTraitOne),
            dht=with_default_height_traits(
                kinematics.dhtraits(), len(kinematics.dptraits()),
                traits.DHTraitOne))
    CONE_TRAITS.check(traits_)
    node_args = parse_component_rnode_args(
        rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp)
    return cone_class(
        sides=sides, **node_args, traits_=traits_, prefixes=TRAIT_PREFIXES)


def cone_plan(
        cone: _cone.Cone,
        lines: Lines | None,
        bands: Bands | None,
        unit: astropy.units.UnitBase | None,
        driver: Driver,
        spectral: gridutils.Grid,
        dtype: np.dtype,
        names: tuple[str, ...] | None,
        band: Band | None,
        emits: bool,
        opacity_ratios: np.ndarray | None,
        data_unit: DataUnit | None,
        redshift: float
) -> NodalComponentPlan:
    """
    Return the plan of a cone component made of the given cone (see
    disks._detail.disk_plan for the arguments); a cone has no line
    opacity.
    """
    emission = emission_setup(
        lines, bands, unit, spectral, names, band, emits, data_unit,
        redshift, ())
    return NodalComponentPlan(
        driver, cone.plan(driver, emission.nlines, dtype), emission,
        opacity_ratios)
