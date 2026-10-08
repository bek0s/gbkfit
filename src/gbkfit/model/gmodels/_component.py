
import dataclasses
import inspect

from gbkfit.utils import iterutils, parseutils
from . import _detail, _disk, common, traits


__all__ = [
    'Slot',
    'BPT', 'BHT', 'OPT', 'OHT', 'VPT', 'VHT',
    'DPT', 'DHT', 'ZPT', 'SPT', 'WPT',
    'DiskComponent'
]


@dataclasses.dataclass(frozen=True)
class Slot:
    """
    The option of a component with its traits of one kind: its name, the
    kind of the traits in the disk (see _disk.TRAIT_KINDS), and their
    parser. A height trait slot must have as many traits as the polar
    trait slot it pairs with. If it has a default trait, it is used for
    each polar trait without a height trait.
    """
    key: str
    kind: str
    parser: parseutils.TypedParser
    pairs_with: str | None = None
    default: type[traits.Trait] | None = None


# The trait slots. The density traits of a disk are surface brightness
# traits in brightness and spectral components, and opacity traits in
# opacity components.
BPT = Slot('bptraits', 'rpt', traits.bpt_parser)
BHT = Slot('bhtraits', 'rht', traits.bht_parser, 'bptraits')
OPT = Slot('optraits', 'rpt', traits.opt_parser)
OHT = Slot('ohtraits', 'rht', traits.oht_parser, 'optraits')
VPT = Slot('vptraits', 'vpt', traits.vpt_parser)
VHT = Slot('vhtraits', 'vht', traits.vht_parser, 'vptraits', traits.VHTraitOne)
DPT = Slot('dptraits', 'dpt', traits.dpt_parser)
DHT = Slot('dhtraits', 'dht', traits.dht_parser, 'dptraits', traits.DHTraitOne)
ZPT = Slot('zptraits', 'zpt', traits.zpt_parser)
SPT = Slot('sptraits', 'spt', traits.spt_parser)
WPT = Slot('wptraits', 'wpt', traits.wpt_parser)


class DiskComponent:
    """
    The base of the gmodel components that are made of one disk.

    A subclass declares the class of its disk (_disk_class) and its trait
    slots (_slots, in the order of _disk.TRAIT_KINDS). Its __init__
    declares its options, which are also its configuration schema: the
    options without a default value are required. This class does the
    rest: load and dump, the validation of the options, and the disk.
    A component can override any of this, or not use this class at all.
    """

    _disk_class: type[_disk.Disk]
    _slots: tuple[Slot, ...]

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'gmodel component')
        for slot in cls._slots:
            required = cls._is_required(slot.key)
            parseutils.load_option_and_update_info(
                slot.parser, info, slot.key,
                required=required, allow_none=not required)
        for name in cls._nwmode_params():
            parseutils.load_option_and_update_info(
                common.nwmode_parser, info, f'{name}_nwmode')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        disk = self._disk
        nwmodes = {
            f'{name}_nwmode': common.nwmode_parser.dump(disk.nwmode(name))
            for name in self._nwmode_params()}
        traits_ = {
            slot.key: slot.parser.dump(disk.traits(slot.kind))
            for slot in self._slots}
        return dict(
            type=self.type(),
            **disk.options(),
            loose=disk.loose(),
            tilted=disk.tilted(),
            rnodes=disk.rnodes(),
            rstep=disk.rstep(),
            interp=disk.interp().type(),
            **nwmodes,
            **traits_)

    def __init__(
            self, loose, tilted,
            rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp,
            nwmodes, traits_, **disk_options):
        """
        nwmodes has the node-wise modes, keyed by geometric parameter
        (e.g. 'xpos'), and traits_ the traits, keyed by option (e.g.
        'bptraits'). disk_options are the options of the type of disk
        (e.g. cflux and seed of MCDisk).
        """
        node_args = _detail.parse_component_rnode_args(
            rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp)
        nwmodes = _detail.validate_component_nwmodes(loose, tilted, nwmodes)
        traits_ = self._parse_traits(traits_)
        self._check_traits(sum(traits_.values(), ()))
        self._disk = self._disk_class(
            **disk_options,
            loose=loose, tilted=tilted, **node_args,
            nwmodes=nwmodes,
            traits_={slot.kind: traits_[slot.key] for slot in self._slots})

    def pdescs(self):
        return self._disk.pdescs()

    def has_weights(self):
        return bool(self._disk.traits('wpt'))

    def constants(self):
        return dict(rnodes=self._disk.rnodes())

    def evaluate(self, driver, params, grid, outputs, dtype, out_extra):
        if OPT in self._slots:
            # The density of an opacity disk is the opacity
            disk_outputs = dict(rdata=outputs['odata'])
        else:
            # The density of the other disks is their brightness, which
            # the opacity absorbs
            disk_outputs = dict(
                opacity=outputs.get('odata'),
                image=outputs.get('image'),
                scube=outputs.get('scube'),
                wdata=outputs.get('wdata'),
                rdata=outputs.get('bdata'),
                ordata=outputs.get('obdata'))
        self._disk.evaluate(
            driver, params, grid, disk_outputs, dtype, out_extra)

    @classmethod
    def _is_required(cls, option):
        parameter = inspect.signature(cls.__init__).parameters[option]
        return parameter.default is inspect.Parameter.empty

    @classmethod
    def _nwmode_params(cls):
        # There is no systemic velocity without velocity traits
        has_velocity = any(slot.kind == 'vpt' for slot in cls._slots)
        return [
            name for name in _disk.GEOMETRY_PARAMS
            if name != 'vsys' or has_velocity]

    def _parse_traits(self, values):
        """
        Make a tuple with the traits of each slot, with the missing height
        traits replaced by the default ones, and check their number.
        """
        result = {}
        for slot in self._slots:
            value = values[slot.key]
            if not value and self._is_required(slot.key):
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

    def _check_traits(self, traits_):
        _detail.check_traits_common(traits_)
        for trait in traits_:
            if isinstance(trait, self._disk_class.unsupported_traits):
                cmp_desc = parseutils.make_typed_desc(
                    self.__class__, 'gmodel component')
                trait_desc = traits.trait_desc(trait.__class__)
                raise NotImplementedError(
                    f"{cmp_desc} does not support {trait_desc} yet")
