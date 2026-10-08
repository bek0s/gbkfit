
import dataclasses
import inspect

from gbkfit.utils import iterutils, parseutils
from . import _detail, common, traits


__all__ = [
    'Slot',
    'BPT', 'BHT', 'OPT', 'OHT', 'VPT', 'VHT',
    'DPT', 'DHT', 'ZPT', 'SPT', 'WPT',
    'SPATIAL_NWMODES', 'SPECTRAL_NWMODES',
    'load_options',
    'make_disk',
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

# The geometric parameters that can have a node-wise mode: those of all
# components, and those of the spectral components, which also have a
# systemic velocity
SPATIAL_NWMODES = ('xpos', 'ypos', 'posa', 'incl')
SPECTRAL_NWMODES = ('vsys',) + SPATIAL_NWMODES


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
    _check_traits(cls, disk_class, sum(traits_.values(), ()))
    return disk_class(
        **disk_options,
        loose=loose, tilted=tilted, **node_args,
        nwmodes=nwmodes,
        traits_={slot.kind: traits_[slot.key] for slot in slots},
        prefixes={slot.kind: slot.prefix for slot in slots},
        rdata_key=rdata_key)


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


def _check_traits(cls, disk_class, traits_):
    _detail.check_traits_common(traits_)
    for trait in traits_:
        if isinstance(trait, disk_class.unsupported_traits):
            cmp_desc = parseutils.make_typed_desc(cls, 'gmodel component')
            trait_desc = traits.trait_desc(trait.__class__)
            raise NotImplementedError(
                f"{cmp_desc} does not support {trait_desc} yet")
