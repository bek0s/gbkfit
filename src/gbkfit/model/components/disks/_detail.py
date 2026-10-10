
import dataclasses
import inspect
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from gbkfit.driver import DeviceArray
from gbkfit.math import interpolation
from gbkfit.utils import gridutils, iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from .._detail import parse_nodes
from ..base import Component, ComponentPlan, NativeGrid
from ..geometries import Geometry
from ..lines import Line, Lines
from . import _disk, traits


__all__ = [
    'Slot',
    'DiskComponentPlan',
    'SpectralDiskComponentPlan',
    'BPT', 'BHT', 'OPT', 'OHT', 'VPT', 'VHT',
    'DPT', 'DHT', 'ZPT', 'SPT', 'WPT',
    'load_options',
    'make_disk',
    'make_lines',
    'dump_disk'
]



@dataclasses.dataclass(frozen=True)
class Slot:
    """
    The option of a component with its traits of one kind: its name, the
    kind of the traits in the disk (see _disk.TRAIT_KINDS), the prefix of
    their parameters, and their parser. A height trait slot must have as
    many traits as the polar trait slot it pairs with. If it has a default
    trait, it is used for each polar trait without a height trait.
    """
    key: str
    kind: str
    prefix: str
    parser: parseutils.TypedParser
    pairs_with: str | None = None
    default: type[traits.Trait] | None = None


# The trait slots. The density traits of a disk are surface brightness
# traits in brightness and spectral components, and opacity traits in
# opacity components.
BPT = Slot('bptraits', 'rpt', 'bpt', traits.bpt_parser)
BHT = Slot('bhtraits', 'rht', 'bht', traits.bht_parser, 'bptraits')
OPT = Slot('optraits', 'rpt', 'opt', traits.opt_parser)
OHT = Slot('ohtraits', 'rht', 'oht', traits.oht_parser, 'optraits')
VPT = Slot('vptraits', 'vpt', 'vpt', traits.vpt_parser)
VHT = Slot(
    'vhtraits', 'vht', 'vht', traits.vht_parser, 'vptraits',
    traits.VHTraitOne)
DPT = Slot('dptraits', 'dpt', 'dpt', traits.dpt_parser)
DHT = Slot(
    'dhtraits', 'dht', 'dht', traits.dht_parser, 'dptraits',
    traits.DHTraitOne)
ZPT = Slot('zptraits', 'zpt', 'zpt', traits.zpt_parser)
SPT = Slot('sptraits', 'spt', 'spt', traits.spt_parser)
WPT = Slot('wptraits', 'wpt', 'wpt', traits.wpt_parser)


class _DiskComponent(Protocol):
    """A component made of one disk."""

    def disk_outputs(
            self, outputs: dict[str, DeviceArray | None]
    ) -> dict[str, DeviceArray | None]:
        """Return the outputs of its disk, from those of the component."""
        ...


class DiskComponentPlan(ComponentPlan):
    """
    The evaluation of a component made of one disk: the plan of its disk,
    which adds to the outputs of the component that the component maps
    to those of the disk (Component.disk_outputs).
    """

    def __init__(
            self, component: _DiskComponent, disk_plan: _disk.DiskPlan):
        self._component = component
        self._disk_plan = disk_plan

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
    ) -> None:
        self._disk_plan.evaluate(
            params, grid, self._component.disk_outputs(outputs), out_extra)


class SpectralDiskComponentPlan(ComponentPlan):
    """
    The evaluation of a spectral component made of one disk: as
    DiskComponentPlan, and its disk adds the selected emission lines of
    the component (Lines; selected, their indices) to the spectral cube,
    at their places on the given spectral axis.
    """

    def __init__(
            self,
            component: _DiskComponent,
            disk_plan: _disk.DiskPlan,
            lines_: Lines,
            selected: Sequence[int],
            spectral: gridutils.Grid
    ):
        self._component = component
        self._disk_plan = disk_plan
        # The offset, scale and flux of each selected line; the fluxes of
        # the lines after the first are their ratios, which are parameters
        self._lines = lines_.values(spectral)[list(selected)]
        self._ratios = [
            (row, lines_.ratio_name(index))
            for row, index in enumerate(selected)
            if lines_.ratio_name(index) is not None]

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None
    ) -> None:
        for row, name in self._ratios:
            self._lines[row, 2] = params[name]
        self._disk_plan.evaluate(
            params, grid, self._component.disk_outputs(outputs), out_extra,
            self._lines)


def load_options(
        cls: type[Component], info: dict[str, Any], slots: Sequence['Slot']
) -> dict[str, Any]:
    """
    Return the options of a component of class cls that is made of one
    disk, with the traits of the given slots loaded. The __init__ of cls
    declares the options, and those without a default value are required.
    """
    for slot in slots:
        required = _is_required(cls, slot.key)
        parseutils.load_option_and_update_info(
            slot.parser, info, slot.key,
            required=required)
    return parseutils.parse_options_for_callable(info, cls.__init__)


def make_disk(
        cls: type[Component],
        disk_class: type[_disk.Disk],
        slots: Sequence['Slot'],
        rdata_key: str,
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
        traits_: dict[str, traits.Trait | Sequence[traits.Trait] | None],
        **disk_options: float
) -> _disk.Disk:
    """
    Return the disk of a component of class cls, of class disk_class, with
    the traits of the given slots, and its density map named rdata_key in
    the extra outputs (e.g. 'bdata'). With a geometry, its warps and, if
    it is warped, its rings are those of the geometry, which the component
    must not give. traits_ has the traits, keyed by option (e.g.
    'bptraits'). disk_options are the options of the type of disk (e.g.
    cflux and seed of MCDisk).
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
    traits_ = _parse_traits(cls, slots, traits_)
    check_traits_common(sum(traits_.values(), ()))
    return disk_class(
        **disk_options,
        loose=bool(loose), tilted=bool(tilted), **node_args,
        traits_={slot.kind: traits_[slot.key] for slot in slots},
        prefixes={slot.kind: slot.prefix for slot in slots},
        rdata_key=rdata_key)


def make_lines(disk: _disk.Disk, lines_: Sequence[Line] | None) -> Lines:
    """
    Return the emission lines (see Lines) of a spectral component made of
    the given disk, whose parameters must have other names.
    """
    result = Lines(lines_)
    if repeated := sorted(set(result.pdescs()) & set(disk.pdescs())):
        raise ConfigError(
            f"the parameters of the lines have the names of other "
            f"parameters: {repeated}; rename the lines")
    return result


def dump_disk(
        disk: _disk.Disk, slots: Sequence['Slot'], geometry: Geometry | None
) -> dict[str, Any]:
    """
    Return the options of a component made of the given disk, with the
    traits of the given slots, without those its geometry (if any) gives.
    """
    traits_ = {
        slot.key: slot.parser.dump(disk.traits(slot.kind))
        for slot in slots}
    info = dict(**disk.options())
    if geometry is None:
        info.update(loose=disk.loose(), tilted=disk.tilted())
    if geometry is None or geometry.rnodes() is None:
        info.update(rnodes=list(disk.rnodes()))
    return info | dict(
        rstep=disk.rstep(),
        interp=disk.interp().type(),
        **traits_)


def _is_required(cls: type[Component], option: str) -> bool:
    """Check whether an option of a component class has no default."""
    parameter = inspect.signature(cls.__init__).parameters[option]
    return parameter.default is inspect.Parameter.empty


def _parse_traits(
        cls: type[Component],
        slots: Sequence['Slot'],
        values: dict[str, traits.Trait | Sequence[traits.Trait] | None]
) -> dict[str, tuple[traits.Trait, ...]]:
    """
    Return the traits of each slot as a tuple, with the missing height
    traits replaced by the default ones. Raise ConfigError if there are
    none of a required slot, too many, or not one height trait for each
    polar trait.
    """
    result = {}
    for slot in slots:
        value = values[slot.key]
        if not value and _is_required(cls, slot.key):
            raise ConfigError(f"at least one {slot.key[:-1]} is required")
        value = iterutils.tuplify(value) if value else ()
        if len(value) > _disk.MAX_TRAITS:
            raise ConfigError(
                f"at most {_disk.MAX_TRAITS} {slot.key} are supported; "
                f"{len(value)} were given")
        if slot.pairs_with:
            npolar = len(result[slot.pairs_with])
            if slot.default:
                value = tuple(
                    slot.default() if trait is None else trait
                    for trait in value or (None,) * npolar)
            if len(value) != npolar:
                raise ConfigError(
                    f"the number of {slot.key} must be equal to "
                    f"the number of {slot.pairs_with} "
                    f"({len(value)} != {npolar})")
        result[slot.key] = value
    return result


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
    """
    Raise ConfigError for traits that are not implemented, and warn about
    those that are discouraged (see parseutils.warn).
    """
    unsupported_traits = (
        traits.WPTraitAxisRange)
    for trait in traits_:
        trait_desc = traits.trait_desc(trait.__class__)
        if isinstance(trait, unsupported_traits):
            raise ConfigError(f"{trait_desc} is not implemented yet")
        if isinstance(trait, traits.BPTraitUniform):
            parseutils.warn(
                f"the use of {trait_desc} is discouraged; "
                f"its main purpose is to facilitate software testing")
        if isinstance(trait, traits.BHTraitUniform):
            parseutils.warn(
                f"the use of {trait_desc} is discouraged; "
                f"it may result in density overestimation due to aliasing")
