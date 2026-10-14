"""
Helpers shared by the disk components (SMDisk, MCDisk).
"""

from collections.abc import Sequence
from typing import Any

import astropy.units
import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.math import interpolation
from gbkfit.params import ParamValues
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .._detail import Fluxes, dump_geometry, dump_name, parse_nodes
from .._units import DataUnit, band_factor, line_factors
from ..base import Component, ComponentPlan, FieldsLayout, NativeGrid
from ..geometries import Geometry
from ..lines import Band, Bands, LineCatalogue, Lines
from ..roles import (
    BandEmission, Emission, LineEmission, Opacity, emission_parser,
    opacity_parser)
from . import _disk, traits
from .opacity import LineOpacity
from .parts import Density, Kinematics, density_parser, kinematics_parser


__all__ = [
    'DiskComponentPlan',
    'disk_plan',
    'dump_disk_component',
    'load_disk_component',
    'make_bands',
    'make_disk',
    'make_lines'
]


# The prefix of the parameters of the traits of each kind
TRAIT_PREFIXES = dict(
    rpt='pt', rht='ht', vpt='vpt', vht='vht', dpt='dpt', dht='dht',
    zpt='zpt', spt='spt')


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
        disk: _disk.Disk,
        density: Density,
        kinematics: Kinematics | None
) -> dict[str, Any]:
    """
    Return the configuration of a disk component made of the given disk,
    without what its geometry (if any) gives.
    """
    geometry = component.geometry()
    info = dict(type=component.type()) | dump_name(component) \
        | dump_geometry(component) | disk.options()
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
        disk_class: type[_disk.Disk],
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
        **disk_options: float
) -> _disk.Disk:
    """
    Return the disk of a disk component, of class disk_class, made of its
    parts (see SMDisk for the options). With a geometry, its warps and, if
    it is warped, its rings are those of the geometry, which the component
    must not give. A thick disk (with density height traits) takes the
    default height traits of its velocity and dispersion traits where it
    has none. disk_options are the options of the type of disk (e.g. cflux
    and seed of a CloudDisk).

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
        if given := [k for k, v in dict(
                loose=loose, tilted=tilted).items() if v is not None]:
            raise ConfigError(
                f"the warps of a component are those of its geometry "
                f"{geometry.name()!r}; remove {given}")
        loose, tilted = geometry.loose(), geometry.tilted()
        if geometry.rnodes() is not None:
            if given := [k for k, v in rings.items() if v is not None]:
                raise ConfigError(
                    f"the rings of a component are those of its warped "
                    f"geometry {geometry.name()!r}; remove {given}")
            rings = dict(rings, rnodes=geometry.rnodes())
    node_args = parse_component_rnode_args(**rings, rstep=rstep, interp=interp)
    traits_ = _disk_traits(
        density, kinematics, emission, opacity, zptraits, sptraits)
    # (not the coefficients of line opacity: a uniform one is common, e.g.
    # one spin temperature)
    check_traits_common(sum(traits_.values(), ()))
    opacity_traits, opacity_prefixes, opacity_lines = line_opacity_traits(
        opacity, emission)
    if opacity_traits:
        traits_.update(opt=opacity_traits)
    return disk_class(
        **disk_options,
        loose=bool(loose), tilted=bool(tilted), **node_args,
        traits_=traits_, prefixes=TRAIT_PREFIXES | dict(opt=opacity_prefixes),
        opacity_lines=opacity_lines)


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
            [None] * len(line_traits), 'traits', 'op', True)
        for trait, prefix in zip(line_traits, prefixes):
            if not isinstance(trait, traits.RPTrait):
                raise ConfigError(
                    f"the coefficients of line opacity are density polar "
                    f"traits; {traits.trait_desc(trait.__class__)} is not")
            for values, value in zip(result, (trait, prefix + suffix, index)):
                values.append(value)
    if len(result[0]) > _disk.MAX_TRAITS:
        raise ConfigError(
            f"line opacity takes at most {_disk.MAX_TRAITS} coefficient "
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
        if len(value) > _disk.MAX_TRAITS:
            raise ConfigError(
                f"at most {_disk.MAX_TRAITS} {name} are supported; "
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
        result.update(vpt=kinematics.vptraits(), dpt=kinematics.dptraits())
        if thick:
            result.update(
                vht=_with_defaults(
                    heights['vht'], len(result['vpt']), traits.VHTraitOne),
                dht=_with_defaults(
                    heights['dht'], len(result['dpt']), traits.DHTraitOne))
    return result


def _with_defaults(
        value: tuple[traits.HTrait | None, ...] | None,
        npolar: int,
        default: type[traits.HTrait]
) -> tuple[traits.HTrait, ...]:
    """
    Return height traits, one for each of npolar polar traits, with the
    missing ones (None, or all if value is None) of the default type.
    """
    if value is None:
        value = (None,) * npolar
    return tuple(default() if trait is None else trait for trait in value)


def make_lines(disk: _disk.Disk, emission: Emission | None) -> Lines | None:
    """
    Return the emission lines of a disk component (see Lines), if it emits
    lines, whose parameters must have other names than those of the disk.
    """
    if not isinstance(emission, LineEmission):
        return None
    lines = emission.lines()
    if repeated := sorted(set(lines.pdescs()) & set(disk.pdescs())):
        raise ConfigError(
            f"the parameters of the lines have the names of other "
            f"parameters: {repeated}; rename the lines")
    return lines


def make_bands(disk: _disk.Disk, emission: Emission | None) -> Bands | None:
    """
    Return the bands of a disk component (see Bands), if it emits in
    bands, whose parameters must have other names than those of the disk.
    """
    if not isinstance(emission, BandEmission):
        return None
    bands = emission.bands()
    if repeated := sorted(set(bands.pdescs()) & set(disk.pdescs())):
        raise ConfigError(
            f"the parameters of the bands have the names of other "
            f"parameters: {repeated}; rename the bands")
    return bands


def disk_plan(
        disk: _disk.Disk,
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
        data_unit: DataUnit | None
) -> 'DiskComponentPlan':
    """
    Return the plan of a disk component made of the given disk, with the
    given emission lines or bands (if any) and the unit of its light (if
    any), of which it evaluates, if it emits, the lines of the given names
    (all if None) or the given band (if known), dimmed by the opacity with
    the given ratios, in the unit of the data (see Component.plan).
    """
    if not emits:
        return DiskComponentPlan(driver, disk.plan(driver, 0, dtype))
    if lines is None:
        ratio = bands.ratio_name(band)
        return DiskComponentPlan(
            driver, disk.plan(driver, 0, dtype), opacity_ratios,
            Fluxes(1, [(0, ratio)] if ratio is not None else [],
                   scale=band_factor(unit, data_unit, band)))
    # (and the row of the line of each trait of line opacity among the
    # lines it evaluates)
    selected = lines.select(names)
    opacity_rows = [
        selected.index(line) if line in selected else -1
        for line in disk.opacity_lines()]
    rows = lines.values(spectral)[list(selected)]
    ratios = [
        (row, lines.ratio_name(index)) for row, index in enumerate(selected)
        if lines.ratio_name(index) is not None]
    return DiskComponentPlan(
        driver, disk.plan(driver, len(selected), dtype, opacity_rows),
        opacity_ratios,
        Fluxes(len(rows), ratios, line_factors(
            unit, data_unit, rows, spectral)),
        rows)


class DiskComponentPlan(ComponentPlan):
    """
    The evaluation of a component made of one disk: the plan of its disk,
    which adds its light to the outputs, dimmed by the opacity with the
    given ratios (see Component.plan), or its optical depth to a cube of
    opacity. Its light is that of the given lines (the offset and scale of
    each on the spectral axis, see Lines.values), or in an image, that of
    its band, of the given fluxes (see Fluxes).
    """

    def __init__(
            self,
            driver: Driver,
            disk_plan_: _disk.DiskPlan,
            opacity_ratios: np.ndarray | None = None,
            fluxes: Fluxes | None = None,
            lines: np.ndarray | None = None
    ):
        self._driver = driver
        self._disk_plan = disk_plan_
        self._fluxes = fluxes
        self._opacity_ratios = None
        if opacity_ratios is not None:
            self._opacity_ratios = driver.mem_copy_h2d(opacity_ratios)
        # The offset, scale and flux of each line (the fluxes are set at
        # each evaluation)
        self._lines = None if lines is None else np.array(lines, dtype=float)

    def disk_plan(self) -> _disk.DiskPlan:
        return self._disk_plan

    def line_values(self, params: ParamValues) -> np.ndarray:
        if self._lines is None:
            return np.array([[0.0, 0.0, self._image_scale(params)]])
        self._lines[:, 2] = self._fluxes(params)
        return self._lines

    def evaluate_emission(
            self,
            params: ParamValues,
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
    ) -> None:
        lines = self.line_values(params) if self._lines is not None \
            else None
        odata = outputs.get('odata')
        disk_outputs = dict(
            image=outputs.get('image'),
            scube=outputs.get('scube'),
            rdata=outputs.get('bdata'),
            opacity=odata,
            opacity_ratios=self._opacity_ratios if odata is not None
            else None,
            ordata=outputs.get('obdata'))
        self._disk_plan.evaluate(
            params, grid, disk_outputs, out_extra, 'bdata', lines,
            image_scale=self._image_scale(params))

    def fields_layout(self) -> FieldsLayout:
        return self._disk_plan.fields_layout()

    def evaluate_fields(
            self,
            params: ParamValues,
            grid: NativeGrid,
            fields: DeviceArray,
            out_extra: dict[str, Any] | None
    ) -> None:
        self._driver.mem_fill(fields, 0)
        self._disk_plan.evaluate(
            params, grid, dict(fields=fields), out_extra, 'bdata')

    def evaluate_opacity(
            self,
            params: ParamValues,
            grid: NativeGrid,
            odata: DeviceArray,
            scale: float,
            out_extra: dict[str, Any] | None
    ) -> None:
        # The density of the disk is its optical depth
        self._disk_plan.evaluate(
            params, grid, dict(rdata=odata), out_extra, 'odata',
            rdata_scale=scale)

    def _image_scale(self, params: ParamValues) -> float:
        """
        Return the flux of the light in the band of an image (see Fluxes);
        1 in a spectral cube, whose lines have theirs.
        """
        if self._lines is not None:
            return 1.0
        return float(self._fluxes(params)[0])


def parse_component_rnode_args(
        rnmin: float | None,
        rnmax: float | None,
        rnsep: float | None,
        rnlen: int | None,
        rnodes: Any,
        rstep: float | None,
        interp: str
) -> dict[str, Any]:
    """
    Return the rings of a disk: its radial nodes (see parse_nodes), the
    step of its subnodes (by default half the smallest separation of the
    nodes, at most 1), and the interpolation class of its node-wise
    parameters. Raise ConfigError for invalid options.
    """
    nodes = parse_nodes('r', rnmin, rnmax, rnsep, rnlen, rnodes)
    step = rstep
    if step is None:
        step = min(1, min(np.diff(nodes)) / 2)
    # (half the difference is allowed, also as it rounds)
    if step <= 0 or step > min(np.diff(nodes)) / 2 * (1 + 1e-9):
        raise ConfigError(
            "rstep must be greater than zero and less than half the "
            "smallest difference between two consecutive rnodes")
    interpolations = dict(
        linear=interpolation.InterpolatorLinear,
        akima=interpolation.InterpolatorAkima,
        pchip=interpolation.InterpolatorPCHIP)
    if interp not in interpolations:
        raise ConfigError(
            "interp must be one of the following: "
            f"{list(interpolations.keys())}")
    return dict(rnodes=nodes, rstep=step, interp=interpolations[interp])


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
