"""
Helpers of the disk components (ThinDisk and ThickDisk).
"""

import logging
from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.config import warn
from gbkfit.driver import Driver
from gbkfit.grid import Grid
from gbkfit.params import ParamValues
from gbkfit.utils import iterutils
from .. import traits
from .._common import dump_name
from .._native import EmissionSetup, NativeComponentPlan, emission_setup
from .._nodal import (
    MAX_TRAITS, TraitTable, parse_component_rnode_args, trait_prefixes,
    with_default_height_traits)
from .. import _units
from .._units import DataUnit
from ..base import Component
from ..lines import Band, Bands, Lines
from ..nodes import Nodes
from ..roles import (
    Emission, LineEmission, Opacity, emission_parser, opacity_parser)
from . import _nodal
from .clouds import CloudDisk, Clouds
from .smooth import SmoothDisk
from .opacity import LineOpacity
from ..parts import KinematicTraits


__all__ = [
    'THICK_DISK_TRAITS',
    'THIN_DISK_TRAITS',
    'DiskComponentPlan',
    'disk_plan',
    'dump_disk_component',
    'make_disk'
]


_log = logging.getLogger(__name__)

# The share of the profile of a density height trait beyond its zmax
# above which a warning says that zmax shapes the density, and the
# azimuths where the displacements of vertical distortion traits are
# checked against their zmax (see DiskComponentPlan.warn_truncation)
_MAX_CUT = 0.01
_CUT_AZIMUTHS = 64


# The traits that the thick disks take: every type, along every
# direction, and the planar velocity traits, in their plane; dispersion
# traits isotropic or along each direction
THICK_DISK_TRAITS = TraitTable('thick disk', {
    kind: tuple(parser.registered_classes().values())
    for kind, parser in dict(
        rpt=traits.rpt_parser, rht=traits.rht_parser,
        vpt=traits.vpt_parser, vht=traits.vht_parser,
        dpt=traits.dpt_parser, dht=traits.dht_parser,
        zpt=traits.zpt_parser, spt=traits.spt_parser).items()},
    ('tan', 'rad', 'ver', 'away', 'out', 'los', 'plane'),
    tuple(traits.DPT_DIRECTIONS))

# The traits that the thin disks take: those of the thick ones, but the
# height traits, the vertical distortions and the dispersion of a vertical
# equilibrium (their kinds are there, empty), and velocity traits along
# every direction but away from the plane, whose faces only a thick disk
# has; of the density height traits, only the vertical equilibrium, which
# it does not draw (it gives its pressure support the density at its
# midplane; it is not among the traits that the disk evaluates, see
# NodalDisk.height_equilibrium)
_THIN_ONLY = dict(
    rht=(traits.RHTraitEquilibrium,), vht=(), dht=(), zpt=(),
    dpt=tuple(
        cls for cls in THICK_DISK_TRAITS.classes['dpt']
        if cls is not traits.DPTraitEquilibrium))
THIN_DISK_TRAITS = TraitTable('thin disk', {
    kind: _THIN_ONLY.get(kind, classes)
    for kind, classes in THICK_DISK_TRAITS.classes.items()},
    tuple(d for d in THICK_DISK_TRAITS.directions if d != 'away'),
    THICK_DISK_TRAITS.dispersion_directions)


def dump_disk_component(
        component: Component,
        disk: _nodal.NodalDisk,
        rnodes: Nodes,
        kinematics: KinematicTraits | None
) -> dict[str, Any]:
    """
    Return the configuration of a disk component made of the given disk,
    with its rings in the form they were given (rnodes).
    """
    info = dict(type=component.type()) | dump_name(component)
    info.update(
        loose=disk.loose(), tilted=disk.tilted(),
        rnodes=rnodes.dump(), rstep=disk.rstep(),
        interp=disk.interp().type(),
        rptraits=traits.rpt_parser.dump(list(disk.traits('rpt'))))
    if disk.traits('rht'):
        # (a shared height trait once)
        count = 1 if disk.shared('rht') else None
        info.update(rhtraits=traits.rht_parser.dump(
            list(disk.traits('rht')[:count])))
    elif (height := disk.height_equilibrium()) is not None:
        # (of a thin disk, not drawn)
        info.update(rhtraits=traits.rht_parser.dump(height))
    if kinematics is not None:
        info.update(kinematics.dump())
    if (support := disk.pressure_support()) is not None:
        info.update(
            pressure_support=traits.pressure_support_parser.dump(support))
    if fluid := disk.fluid():
        info.update(fluid=fluid[0] if len(fluid) == 1 else list(fluid))
    if component.emission() is not None:
        info.update(emission=emission_parser.dump(component.emission()))
    if component.opacity() is not None:
        info.update(opacity=opacity_parser.dump(component.opacity()))
    for key, kind, parser in (
            ('zptraits', 'zpt', traits.zpt_parser),
            ('sptraits', 'spt', traits.spt_parser)):
        if disk.traits(kind):
            info[key] = parser.dump(list(disk.traits(kind)))
    return info


def make_disk(
        rptraits: tuple[traits.RPTrait, ...],
        rhtraits: tuple[traits.RHTrait, ...],
        height: traits.RHTraitEquilibrium | None,
        kinematics: KinematicTraits | None,
        pressure_support: traits.PressureSupport | None,
        fluid: tuple[str, ...],
        emission: Emission | None,
        opacity: Opacity | None,
        zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None,
        sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None,
        thick: bool,
        loose: bool,
        tilted: bool,
        rnodes: Nodes,
        rstep: float | None,
        interp: str,
        clouds: Clouds | None
) -> _nodal.NodalDisk:
    """
    Return the disk of a disk component, made of its traits (see ThinDisk
    and ThickDisk for the options): smooth, or made of its clouds (a thick
    one). A thick disk takes the default height traits of its velocity
    and dispersion traits where it has none; a thin one has no height
    traits, opacity, vertical distortions or clouds.

    Raises
    ------
    ValueError
        If the options are not valid, or the parts and roles do not go
        together (e.g. lines without kinematics).
    """
    node_args = parse_component_rnode_args(
        rnodes=rnodes, rstep=rstep, interp=interp)
    traits_ = _disk_traits(
        rptraits, rhtraits, kinematics, emission, zptraits, sptraits, thick)
    (THICK_DISK_TRAITS if thick else THIN_DISK_TRAITS).check(traits_)
    _check_waves(traits_)
    # (not the coefficients of line opacity: a uniform one is common, e.g.
    # one spin temperature)
    check_traits_common(sum(traits_.values(), ()))
    opacity_traits, opacity_prefixes, opacity_lines = line_opacity_traits(
        opacity, emission)
    if opacity_traits:
        traits_.update(opt=opacity_traits)
    options = dict(
        loose=loose, tilted=tilted, **node_args,
        traits_=traits_, prefixes=dict(opt=opacity_prefixes),
        density_units=dict(
            rpt=_units.density_unit(emission),
            opt=_units.line_opacity_unit(emission)),
        opacity_lines=opacity_lines,
        equilibrium=kinematics.equilibrium() if kinematics is not None
        else None,
        height=height, pressure_support=pressure_support, fluid=fluid)
    if clouds is None:
        return SmoothDisk(**options)
    return CloudDisk(clouds, **options)


def _check_waves(traits_: dict[str, tuple[traits.Trait, ...]]) -> None:
    """
    Raise ValueError if a disk has a spiral density wave that takes its
    rotation (see traits.Spiral) but no tangential velocity
    traits.
    """
    waves = any(
        isinstance(trait, (traits.RPTraitSpiral, traits.VPTraitSpiral))
        and trait.takes_rotation()
        for kind in ('rpt', 'vpt') for trait in traits_.get(kind, ()))
    if waves and not any(
            trait.direction() == 'tan' for trait in traits_.get('vpt', ())):
        raise ValueError(
            "the traits 'spiral' are a density wave in the rotation of "
            "their disk: give the disk tangential velocity traits")


def line_opacity_traits(
        opacity: Opacity | None,
        emission: Emission | None
) -> tuple[tuple[traits.RPTrait, ...], tuple[str, ...], tuple[int, ...]]:
    """
    Return the coefficient traits of the line opacity of a disk component,
    if any (see LineOpacity), the prefix of the parameters of each (e.g.
    'op_' for one line without a name, 'op2_halpha_' for the second trait
    of the line halpha), and its line (its index among the lines of the
    emission).

    Raises
    ------
    ValueError
        If the component does not emit lines, or the coefficients do not
        go with its lines (the traits of one line without a name, or a
        mapping of each of its named lines to its traits), or there are
        more traits than the kernels take.
    """
    if not isinstance(opacity, LineOpacity):
        return (), (), ()
    if not isinstance(emission, LineEmission):
        raise ValueError(
            "line opacity is the opacity of the lines of a component: it "
            "needs emission lines (emission: {type: lines})")
    names = emission.lines().names()
    coefficients = opacity.coefficients()
    if not names:
        if isinstance(coefficients, dict):
            raise ValueError(
                "the emission has one line without a name: the coefficients "
                "of its line opacity are the traits of that line, not a "
                "mapping of lines")
        lines = [('', coefficients)]
    else:
        if not isinstance(coefficients, dict):
            raise ValueError(
                f"the emission has the lines {list(names)}: the coefficients "
                f"of its line opacity are a mapping of each line to its "
                f"traits")
        if missing := [name for name in names if name not in coefficients]:
            raise ValueError(
                f"the line opacity needs the coefficients of every line "
                f"(0 for a line that does not absorb); missing: {missing}")
        if unknown := [name for name in coefficients if name not in names]:
            raise ValueError(
                f"the line opacity has coefficients of lines that the "
                f"emission does not have: {unknown}; its lines: "
                f"{list(names)}")
        lines = [(f'{name}_', coefficients[name]) for name in names]
    result = ([], [], [])
    for index, (suffix, line_traits) in enumerate(lines):
        prefixes = trait_prefixes(
            [trait.name() for trait in line_traits], 'op')
        for trait, prefix in zip(line_traits, prefixes):
            if not isinstance(trait, traits.RPTrait):
                raise ValueError(
                    f"the coefficients of line opacity are density polar "
                    f"traits; {traits.trait_desc(trait.__class__)} is not")
            for values, value in zip(result, (trait, prefix + suffix, index)):
                values.append(value)
    if len(result[0]) > MAX_TRAITS:
        raise ValueError(
            f"line opacity takes at most {MAX_TRAITS} coefficient "
            f"traits; {len(result[0])} were given")
    return tuple(result[0]), tuple(result[1]), tuple(result[2])


def _disk_traits(
        rptraits: tuple[traits.RPTrait, ...],
        rhtraits: tuple[traits.RHTrait, ...],
        kinematics: KinematicTraits | None,
        emission: Emission | None,
        zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None,
        sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None,
        thick: bool
) -> dict[str, tuple[traits.Trait, ...]]:
    """
    Return the traits of a disk by kind (see _nodal.TRAIT_KINDS): those of
    its options, and of a thick one, the default height traits of its
    velocity and dispersion traits where it has none. Raise ValueError if
    the traits and roles do not go together.
    """
    if isinstance(emission, LineEmission) and kinematics is None:
        raise ValueError(
            "a disk that emits lines needs kinematics: give it vptraits and "
            "dptraits")
    zptraits = iterutils.tuplify(zptraits)
    sptraits = iterutils.tuplify(sptraits)
    for name, value in (('zptraits', zptraits), ('sptraits', sptraits)):
        if len(value) > MAX_TRAITS:
            raise ValueError(
                f"at most {MAX_TRAITS} {name} are supported; "
                f"{len(value)} were given")
    result = dict(
        rpt=rptraits, rht=rhtraits, zpt=zptraits, spt=sptraits)
    if kinematics is None:
        return result
    # (the kernels evaluate the fitted dispersion traits; its galaxy
    # derives that of the equilibrium)
    dptraits = kinematics.fitted_dptraits()
    result.update(vpt=kinematics.vptraits(), dpt=dptraits)
    if not thick:
        return result
    vhtraits, dhtraits = kinematics.vhtraits(), kinematics.dhtraits()
    # (above the density, pressure support divides by a density of 0:
    # the table of a mass trait above it was NaN, and its disk with it)
    zmax = max(trait.zmax() for trait in result['rht'])
    for trait in vhtraits or ():
        if isinstance(trait, traits.VHTraitMass) and trait.zmax() > zmax:
            raise ValueError(
                f"the zmax of a velocity height trait 'mass' is the top of "
                f"its table, which must not be above the zmax of the "
                f"density height traits, {zmax}; it is {trait.zmax()}")
    result.update(
        vht=with_default_height_traits(
            vhtraits, traits.VHTraitOne),
        dht=with_default_height_traits(
            dhtraits, traits.DHTraitOne) if dptraits else ())
    return result


def disk_plan(
        disk: _nodal.NodalDisk,
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
) -> 'DiskComponentPlan':
    """
    Return the plan of a disk component made of the given disk, with the
    given emission lines or bands (if any) and the unit of its light (if
    any), of which it evaluates, if it emits, the lines of the given names
    (all if None), at the redshift of its galaxy, or the given band (if
    known), dimmed by the opacity with the given ratios, in the unit of
    the data (see Component.plan and emission_setup).
    """
    emission = emission_setup(
        lines, bands, unit, spectral, names, band, emits, data_unit,
        redshift, disk.opacity_lines())
    return DiskComponentPlan(
        driver, disk,
        disk.plan(driver, emission.nlines, dtype, emission.opacity_rows),
        emission, opacity_ratios)


class DiskComponentPlan(NativeComponentPlan):
    """
    The evaluation of a component made of one disk (see
    NativeComponentPlan), whose galaxy can take the plan of its disk.
    """

    def __init__(
            self,
            driver: Driver,
            disk: _nodal.NodalDisk,
            disk_plan_: _nodal.DiskPlan,
            emission: EmissionSetup,
            opacity_ratios: np.ndarray | None = None
    ):
        super().__init__(driver, disk_plan_, emission, opacity_ratios)
        self._disk = disk
        self._disk_plan = disk_plan_

    def disk_plan(self) -> _nodal.DiskPlan:
        """Return the plan of its disk."""
        return self._disk_plan

    def warn_truncation(self, params: ParamValues, label: str) -> None:
        """
        Log a warning for each density height trait of its disk whose zmax
        cuts more than 1% of its profile at a ring, and for each vertical
        distortion trait whose zmax limits its displacement somewhere (at a
        ring, at azimuths around it): zmax then shapes the disk, rather
        than holding it (see RHTrait and ZPTrait).
        """
        disk = self._disk
        for index, cut, radius in _height_cuts(disk, params):
            if cut > _MAX_CUT:
                trait = disk.traits('rht')[index]
                _log.warning(
                    f"{label}: the zmax ({trait.zmax():g} arcsec) of its "
                    f"density height trait {index} ({trait.type()}) cuts "
                    f"{cut:.1%} of its profile at the radius {radius:g}: "
                    f"zmax shapes the density there; raise it")
        for index, radius in _limited_displacements(
                disk, self._disk_plan, params):
            trait = disk.traits('zpt')[index]
            _log.warning(
                f"{label}: the zmax ({trait.zmax():g} arcsec) of its "
                f"vertical distortion trait {index} ({trait.type()}) "
                f"limits its displacement at the radius {radius:g}: "
                f"zmax shapes the disk there; raise it")


def check_traits_common(traits_: Sequence[traits.Trait]) -> None:
    """Warn about the traits that are discouraged (see warn)."""
    for trait in traits_:
        trait_desc = traits.trait_desc(trait.__class__)
        if isinstance(trait, traits.RPTraitUniform):
            warn(
                f"the use of {trait_desc} is discouraged; "
                f"its main purpose is to facilitate software testing")
        if isinstance(trait, traits.RHTraitUniform):
            warn(
                f"the use of {trait_desc} is discouraged; "
                f"it may result in density overestimation due to aliasing")


def _height_cuts(
        disk: _nodal.NodalDisk, params: ParamValues
) -> list[tuple[int, float, float]]:
    """
    Return, for each density height trait of a disk (with its
    parameters), the largest share of its profile that its zmax cuts (see
    RHTrait), over its rings, and the radius where it does: (the index of
    the trait, the share, the radius).
    """
    radii = np.asarray(disk.rnodes())
    trait_params = disk.trait_params('rht')
    # (a shared height trait once)
    count = 1 if disk.shared('rht') else None
    cuts = []
    for index, (trait, names) in enumerate(zip(
            disk.traits('rht')[:count], trait_params.pnames)):
        values = [
            _at_radii(disk, params[names[name]], radii)
            for name in ('s', 'b') if name in names]
        zmax = trait.zmax()
        cut = 1 - (trait.cdf(zmax, *values) - trait.cdf(-zmax, *values))
        ring = int(np.argmax(cut))
        cuts.append((index, float(cut[ring]), float(radii[ring])))
    return cuts


def _limited_displacements(
        disk: _nodal.NodalDisk, plan: _nodal.DiskPlan, params: ParamValues
) -> list[tuple[int, float]]:
    """
    Return, for each vertical distortion trait of a disk (with its plan
    and parameters) whose zmax limits its displacement somewhere (see
    ZPTrait), over its rings and azimuths around them, the index of the
    trait and the first radius where it does.
    """
    if not disk.traits('zpt'):
        return []
    radii = np.asarray(disk.rnodes())
    azimuths = np.linspace(0, 360, _CUT_AZIMUTHS, endpoint=False)
    displacements = plan.evaluate_polar_traits(
        params, 'zpt', radii[:, None], azimuths)
    limited = []
    for index, (trait, values) in enumerate(zip(
            disk.traits('zpt'), displacements)):
        # (a limited displacement is zmax, in the precision of the kernels)
        zmax = values.dtype.type(trait.zmax())
        rings = np.flatnonzero((np.abs(values) >= zmax).any(axis=1))
        if rings.size:
            limited.append((index, float(radii[rings[0]])))
    return limited


def _at_radii(
        disk: _nodal.NodalDisk, values: float | np.ndarray, radii: np.ndarray
) -> np.ndarray:
    """
    Return the values of a parameter of a trait of a disk at the given
    radii: one value, or the node-wise values at the nodes or the
    subnodes of the disk, interpolated linearly.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(len(radii), float(values))
    nodes = disk.rnodes() if len(values) == len(disk.rnodes()) \
        else disk.subrnodes()
    return np.interp(radii, nodes, values)
