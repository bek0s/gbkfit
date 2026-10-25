"""
Helpers of the disk component (Disk).
"""

from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import Driver
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .. import traits
from .._detail import dump_geometry, dump_name, geometry_desc
from .._nodal import (
    MAX_TRAITS, EmissionSetup, NodalComponentPlan, TraitTable,
    emission_setup, parse_component_rnode_args, with_default_height_traits)
from .._units import DataUnit
from ..base import Component
from ..geometries import Geometry
from ..lines import Band, Bands, LineCatalogue, Lines
from ..roles import (
    Emission, LineEmission, Opacity, emission_parser, opacity_parser)
from . import _disk
from .clouds import CloudDisk, Clouds
from .smooth import SmoothDisk
from .opacity import LineOpacity
from ..parts import Density, Kinematics, density_parser, kinematics_parser


__all__ = [
    'DiskComponentPlan',
    'disk_plan',
    'dump_disk_component',
    'load_disk_component',
    'make_disk'
]


# The prefix of the parameters of the traits of each kind
TRAIT_PREFIXES = dict(
    rpt='pt', rht='ht', vpt='vpt', vht='vht', dpt='dpt', dht='dht',
    zpt='zpt', spt='spt')

# The traits that the disks take: every type, along every direction, and
# the planar velocity traits, in their plane; dispersion traits isotropic
# or along each direction
DISK_TRAITS = TraitTable('disk', {
    kind: tuple(parser.registered_classes().values())
    for kind, parser in dict(
        rpt=traits.rpt_parser, rht=traits.rht_parser,
        vpt=traits.vpt_parser, vht=traits.vht_parser,
        dpt=traits.dpt_parser, dht=traits.dht_parser,
        zpt=traits.zpt_parser, spt=traits.spt_parser).items()},
    ('tan', 'rad', 'ver', 'away', 'out', 'los', 'plane'),
    tuple(traits.DPT_DIRECTIONS))


def load_disk_component(
        cls: type[Component],
        info: dict[str, Any],
        catalogue: LineCatalogue | None
) -> dict[str, Any]:
    """
    Return the options of a disk component of class cls, with its parts,
    roles and traits loaded, and the lines its emission names from the
    catalogue (see Emission.load). The __init__ of cls declares the
    options.
    """
    for parser, key, required in (
            (density_parser, 'density', True),
            (kinematics_parser, 'kinematics', False),
            (opacity_parser, 'opacity', False),
            (traits.zpt_parser, 'zptraits', False),
            (traits.spt_parser, 'sptraits', False)):
        parseutils.load_option_and_update_info(
            parser, info, key, required=required)
    parseutils.load_option_and_update_info(
        emission_parser, info, 'emission', catalogue=catalogue)
    return parseutils.parse_options_for_callable(info, cls.__init__)


def dump_disk_component(
        component: Component,
        disk: _disk.NodalDisk,
        density: Density,
        kinematics: Kinematics | None
) -> dict[str, Any]:
    """
    Return the configuration of a disk component made of the given disk,
    without what its geometry (if any) gives.
    """
    geometry = component.geometry()
    if not isinstance(geometry, Geometry):
        geometry = None
    info = dict(type=component.type()) | dump_name(component) \
        | dump_geometry(component)
    if geometry is None:
        info.update(loose=disk.loose(), tilted=disk.tilted())
    if geometry is None or geometry.rnodes() is None:
        info.update(rnodes=list(disk.rnodes()))
    info.update(
        rstep=disk.rstep(), interp=disk.interp().type(),
        density=density.dump())
    if kinematics is not None:
        info.update(kinematics=kinematics.dump())
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
        density: Density,
        kinematics: Kinematics | None,
        emission: Emission | None,
        opacity: Opacity | None,
        zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None,
        sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None,
        geometry: Geometry | None,
        loose: bool | None,
        tilted: bool | None,
        rnmin: float | None,
        rnmax: float | None,
        rnsep: float | None,
        rnlen: int | None,
        rnodes: Sequence[float] | None,
        rstep: float | None,
        interp: str,
        clouds: Clouds | None
) -> _disk.NodalDisk:
    """
    Return the disk of a disk component, made of its parts (see Disk for
    the options): smooth, or made of its clouds. With a geometry, its
    warps and, if it is warped, its rings are those of the geometry, which
    the component must not give. A thick disk (with density height
    traits) takes the default height traits of its velocity and dispersion
    traits where it has none.

    Raises
    ------
    ConfigError
        If the options are not valid, conflict with the geometry, or the
        parts and roles do not go together (e.g. lines without
        kinematics, or what only a thick disk can have on a thin one).
    """
    rings = dict(rnmin=rnmin, rnmax=rnmax, rnsep=rnsep, rnlen=rnlen,
                 rnodes=rnodes)
    if geometry is not None:
        warps = dict(loose=geometry.loose(), tilted=geometry.tilted())
        if differ := [k for k, v in dict(loose=loose, tilted=tilted).items()
                      if v is not None and v != warps[k]]:
            raise ConfigError(
                f"the warps of a component are those of "
                f"{geometry_desc(geometry)}, which differ: {differ}; remove "
                f"them, or give the component geometry 'own'")
        loose, tilted = warps['loose'], warps['tilted']
        if geometry.rnodes() is not None:
            if given := [k for k, v in rings.items() if v is not None]:
                raise ConfigError(
                    f"the rings of a component are those of "
                    f"{geometry_desc(geometry)}, which is warped; remove "
                    f"{given}, or give the component geometry 'own'")
            rings = dict(rings, rnodes=geometry.rnodes())
    node_args = parse_component_rnode_args(
        **rings, rstep=rstep, interp=interp)
    traits_ = _disk_traits(
        density, kinematics, emission, opacity, zptraits, sptraits)
    DISK_TRAITS.check(traits_)
    _check_waves(traits_)
    # (not the coefficients of line opacity: a uniform one is common, e.g.
    # one spin temperature)
    check_traits_common(sum(traits_.values(), ()))
    opacity_traits, opacity_prefixes, opacity_lines = line_opacity_traits(
        opacity, emission)
    if opacity_traits:
        traits_.update(opt=opacity_traits)
    options = dict(
        loose=bool(loose), tilted=bool(tilted), **node_args,
        traits_=traits_, prefixes=TRAIT_PREFIXES | dict(opt=opacity_prefixes),
        opacity_lines=opacity_lines,
        equilibrium=None if kinematics is None else kinematics.equilibrium())
    if clouds is None:
        return SmoothDisk(**options)
    return CloudDisk(clouds, **options)


def _check_waves(traits_: dict[str, tuple[traits.Trait, ...]]) -> None:
    """
    Raise ConfigError if a disk has a spiral density wave that takes its
    rotation (see traits.Spiral) but no tangential velocity
    traits.
    """
    waves = any(
        isinstance(trait, (traits.RPTraitSpiral, traits.VPTraitSpiral))
        and trait.takes_rotation()
        for kind in ('rpt', 'vpt') for trait in traits_.get(kind, ()))
    if waves and not any(
            trait.direction() == 'tan' for trait in traits_.get('vpt', ())):
        raise ConfigError(
            "the traits 'spiral' are a density wave in the rotation of "
            "their disk: give the disk tangential velocity traits")


def line_opacity_traits(
        opacity: Opacity | None,
        emission: Emission | None
) -> tuple[tuple[traits.RPTrait, ...], tuple[str, ...], tuple[int, ...]]:
    """
    Return the coefficient traits of the line opacity of a disk component,
    if any (see LineOpacity), the prefix of the parameters of each (e.g.
    'op_' for one line without a name, 'op1_halpha_' for the second trait
    of the line halpha), and its line (its index among the lines of the
    emission).

    Raises
    ------
    ConfigError
        If the component does not emit lines, or the coefficients do not
        go with its lines (the traits of one line without a name, or a
        mapping of each of its named lines to its traits), or there are
        more traits than the kernels take.
    """
    if not isinstance(opacity, LineOpacity):
        return (), (), ()
    if not isinstance(emission, LineEmission):
        raise ConfigError(
            "line opacity is the opacity of the lines of a component: it "
            "needs emission lines (emission: {type: lines})")
    names = emission.lines().names()
    coefficients = opacity.coefficients()
    if not names:
        if isinstance(coefficients, dict):
            raise ConfigError(
                "the emission has one line without a name: the coefficients "
                "of its line opacity are the traits of that line, not a "
                "mapping of lines")
        lines = [('', coefficients)]
    else:
        if not isinstance(coefficients, dict):
            raise ConfigError(
                f"the emission has the lines {list(names)}: the coefficients "
                f"of its line opacity are a mapping of each line to its "
                f"traits")
        if missing := [name for name in names if name not in coefficients]:
            raise ConfigError(
                f"the line opacity needs the coefficients of every line "
                f"(0 for a line that does not absorb); missing: {missing}")
        if unknown := [name for name in coefficients if name not in names]:
            raise ConfigError(
                f"the line opacity has coefficients of lines that the "
                f"emission does not have: {unknown}; its lines: "
                f"{list(names)}")
        lines = [(f'{name}_', coefficients[name]) for name in names]
    result = ([], [], [])
    for index, (suffix, line_traits) in enumerate(lines):
        prefixes = parseutils.item_prefixes(
            [trait.name() for trait in line_traits], 'traits', 'op', True)
        for trait, prefix in zip(line_traits, prefixes):
            if not isinstance(trait, traits.RPTrait):
                raise ConfigError(
                    f"the coefficients of line opacity are density polar "
                    f"traits; {traits.trait_desc(trait.__class__)} is not")
            for values, value in zip(result, (trait, prefix + suffix, index)):
                values.append(value)
    if len(result[0]) > MAX_TRAITS:
        raise ConfigError(
            f"line opacity takes at most {MAX_TRAITS} coefficient "
            f"traits; {len(result[0])} were given")
    return tuple(result[0]), tuple(result[1]), tuple(result[2])


def _disk_traits(
        density: Density,
        kinematics: Kinematics | None,
        emission: Emission | None,
        opacity: Opacity | None,
        zptraits: traits.ZPTrait | Sequence[traits.ZPTrait] | None,
        sptraits: traits.SPTrait | Sequence[traits.SPTrait] | None
) -> dict[str, tuple[traits.Trait, ...]]:
    """
    Return the traits of a disk by kind (see _disk.TRAIT_KINDS). Raise
    ConfigError if the parts and roles do not go together.
    """
    thick = density.thick()
    if isinstance(emission, LineEmission) and kinematics is None:
        raise ConfigError("a disk that emits lines needs kinematics")
    if opacity is not None and not thick:
        raise ConfigError(
            "a disk with opacity needs density height traits (a thick "
            "disk)")
    zptraits = iterutils.tuplify(zptraits)
    sptraits = iterutils.tuplify(sptraits)
    for name, value in (('zptraits', zptraits), ('sptraits', sptraits)):
        if len(value) > MAX_TRAITS:
            raise ConfigError(
                f"at most {MAX_TRAITS} {name} are supported; "
                f"{len(value)} were given")
    if zptraits and not thick:
        raise ConfigError(
            "vertical distortion traits (zptraits) need density height "
            "traits (a thick disk)")
    result = dict(
        rpt=density.ptraits(), rht=density.htraits() or (),
        zpt=zptraits, spt=sptraits)
    if kinematics is not None:
        heights = dict(vht=kinematics.vhtraits(), dht=kinematics.dhtraits())
        if not thick and any(value is not None for value in heights.values()):
            raise ConfigError(
                "the height traits of the kinematics need density height "
                "traits (a thick disk)")
        if not thick and kinematics.equilibrium() is not None:
            raise ConfigError(
                "the vertical equilibrium of the kinematics needs density "
                "height traits (a thick disk): the height of the disk sets "
                "its dispersion")
        if not thick and any(
                trait.direction() == 'away'
                for trait in kinematics.vptraits()):
            raise ConfigError(
                "velocity traits away from the plane ('away') need density "
                "height traits (a thick disk, whose faces they leave)")
        result.update(vpt=kinematics.vptraits(), dpt=kinematics.dptraits())
        if thick:
            result.update(
                vht=with_default_height_traits(
                    heights['vht'], len(result['vpt']), traits.VHTraitOne),
                dht=with_default_height_traits(
                    heights['dht'], len(result['dpt']), traits.DHTraitOne))
    return result


def disk_plan(
        disk: _disk.NodalDisk,
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
        driver,
        disk.plan(driver, emission.nlines, dtype, emission.opacity_rows),
        emission, opacity_ratios)


class DiskComponentPlan(NodalComponentPlan):
    """
    The evaluation of a component made of one disk (see
    NodalComponentPlan), whose galaxy can take the plan of its disk.
    """

    def __init__(
            self,
            driver: Driver,
            disk_plan_: _disk.DiskPlan,
            emission: EmissionSetup,
            opacity_ratios: np.ndarray | None = None
    ):
        super().__init__(driver, disk_plan_, emission, opacity_ratios)
        self._disk_plan = disk_plan_

    def disk_plan(self) -> _disk.DiskPlan:
        return self._disk_plan


def check_traits_common(traits_: Sequence[traits.Trait]) -> None:
    """Warn about the traits that are discouraged (see parseutils.warn)."""
    for trait in traits_:
        trait_desc = traits.trait_desc(trait.__class__)
        if isinstance(trait, traits.RPTraitUniform):
            parseutils.warn(
                f"the use of {trait_desc} is discouraged; "
                f"its main purpose is to facilitate software testing")
        if isinstance(trait, traits.RHTraitUniform):
            parseutils.warn(
                f"the use of {trait_desc} is discouraged; "
                f"it may result in density overestimation due to aliasing")
