
import dataclasses
import inspect

from gbkfit.utils import iterutils, parseutils
from . import _detail, _disk, common, lines, traits
from .base import ComponentPlan


__all__ = [
    'Slot',
    'DiskComponentPlan',
    'SpectralDiskComponentPlan',
    'BPT', 'BHT', 'OPT', 'OHT', 'VPT', 'VHT',
    'DPT', 'DHT', 'ZPT', 'SPT', 'WPT',
    'SPATIAL_NWMODES', 'SPECTRAL_NWMODES',
    'load_options',
    'make_disk',
    'make_lines',
    'dump_disk',
    'dump_lines',
    'dump_name'
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

# The geometric parameters that can have a node-wise mode: those of all
# components, and those of the spectral components, which also have a
# systemic velocity
SPATIAL_NWMODES = ('xpos', 'ypos', 'posa', 'incl')
SPECTRAL_NWMODES = ('vsys',) + SPATIAL_NWMODES


class DiskComponentPlan(ComponentPlan):
    """
    The evaluation of a component made of one disk: the plan of its disk,
    which adds to the outputs of the component that the component maps
    to those of the disk (Component.disk_outputs).
    """

    def __init__(self, component, disk_plan):
        self._component = component
        self._disk_plan = disk_plan

    def evaluate(self, params, grid, outputs, out_extra):
        self._disk_plan.evaluate(
            params, grid, self._component.disk_outputs(outputs), out_extra)


class SpectralDiskComponentPlan(ComponentPlan):
    """
    The evaluation of a spectral component made of one disk: as
    DiskComponentPlan, and its disk adds the selected emission lines of
    the component (Lines; selected, their indices) to the spectral cube,
    at their places on the given spectral axis.
    """

    def __init__(self, component, disk_plan, lines_, selected, spectral):
        self._component = component
        self._disk_plan = disk_plan
        # The offset, scale and flux of each selected line; the fluxes of
        # the lines after the first are their ratios, which are parameters
        self._lines = lines_.values(spectral)[list(selected)]
        self._ratios = [
            (row, lines_.ratio_name(index))
            for row, index in enumerate(selected)
            if lines_.ratio_name(index) is not None]

    def evaluate(self, params, grid, outputs, out_extra):
        for row, name in self._ratios:
            self._lines[row, 2] = params[name]
        self._disk_plan.evaluate(
            params, grid, self._component.disk_outputs(outputs), out_extra,
            self._lines)


def load_options(cls, info, slots, nwmodes):
    """
    The options of a component of class cls that is made of one disk,
    with its traits and node-wise modes loaded: those of the given trait
    slots and geometric parameters. The __init__ of cls declares the
    options, and those without a default value are required.
    """
    desc = parseutils.make_typed_desc(cls, 'gmodel component')
    for slot in slots:
        required = _is_required(cls, slot.key)
        parseutils.load_option_and_update_info(
            slot.parser, info, slot.key,
            required=required, allow_none=not required)
    for name in nwmodes:
        parseutils.load_option_and_update_info(
            common.nwmode_parser, info, f'{name}_nwmode')
    return parseutils.parse_options_for_callable(info, desc, cls.__init__)


def make_disk(
        cls, disk_class, slots, rdata_key,
        loose, tilted, rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp,
        nwmodes, traits_, **disk_options):
    """
    The disk of a component of class cls, of class disk_class, with the
    traits of the given slots, and its density map named rdata_key in the
    extra outputs (e.g. 'bdata'). nwmodes has the node-wise modes, keyed by
    geometric parameter (e.g. 'xpos'), and traits_ the traits, keyed by
    option (e.g. 'bptraits'). disk_options are the options of the type
    of disk (e.g. cflux and seed of MCDisk).
    """
    node_args = _detail.parse_component_rnode_args(
        rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp)
    nwmodes = _detail.validate_component_nwmodes(loose, tilted, nwmodes)
    traits_ = _parse_traits(cls, slots, traits_)
    _detail.check_traits_common(sum(traits_.values(), ()))
    return disk_class(
        **disk_options,
        loose=loose, tilted=tilted, **node_args,
        nwmodes=nwmodes,
        traits_={slot.kind: traits_[slot.key] for slot in slots},
        prefixes={slot.kind: slot.prefix for slot in slots},
        rdata_key=rdata_key)


def make_lines(disk, lines_):
    """
    The emission lines (see Lines) of a spectral component made of the
    given disk, whose parameters must have other names.
    """
    result = lines.Lines(lines_)
    if repeated := sorted(set(result.pdescs()) & set(disk.pdescs())):
        raise RuntimeError(
            f"the parameters of the lines have the names of other "
            f"parameters: {repeated}; rename the lines")
    return result


def dump_lines(lines_):
    """The lines option of a spectral component: none if not given."""
    if lines_.lines() is None:
        return {}
    return dict(lines=lines.line_parser.dump(list(lines_.lines())))


def dump_name(component):
    """The name option of a component: none if it has no name."""
    name = component.name()
    return dict(name=name) if name is not None else {}


def dump_disk(disk, slots, nwmodes):
    """
    The options of a component made of the given disk, with the traits
    of the given slots and the node-wise modes of the given geometric
    parameters.
    """
    nwmodes = {
        f'{name}_nwmode': common.nwmode_parser.dump(disk.nwmode(name))
        for name in nwmodes}
    traits_ = {
        slot.key: slot.parser.dump(disk.traits(slot.kind))
        for slot in slots}
    return dict(
        **disk.options(),
        loose=disk.loose(),
        tilted=disk.tilted(),
        rnodes=list(disk.rnodes()),
        rstep=disk.rstep(),
        interp=disk.interp().type(),
        **nwmodes,
        **traits_)


def _is_required(cls, option):
    parameter = inspect.signature(cls.__init__).parameters[option]
    return parameter.default is inspect.Parameter.empty


def _parse_traits(cls, slots, values):
    """
    Make a tuple with the traits of each slot, with the missing height
    traits replaced by the default ones, and check their number.
    """
    result = {}
    for slot in slots:
        value = values[slot.key]
        if not value and _is_required(cls, slot.key):
            raise RuntimeError(f"at least one {slot.key[:-1]} is required")
        value = iterutils.tuplify(value) if value else ()
        if len(value) > _disk.MAX_TRAITS:
            raise RuntimeError(
                f"at most {_disk.MAX_TRAITS} {slot.key} are supported; "
                f"{len(value)} were given")
        if slot.pairs_with:
            npolar = len(result[slot.pairs_with])
            if slot.default:
                value = tuple(
                    slot.default() if trait is None else trait
                    for trait in value or (None,) * npolar)
            if len(value) != npolar:
                raise RuntimeError(
                    f"the number of {slot.key} must be equal to "
                    f"the number of {slot.pairs_with} "
                    f"({len(value)} != {npolar})")
        result[slot.key] = value
    return result

