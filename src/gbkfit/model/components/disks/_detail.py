
import dataclasses
import inspect
import logging

import numpy as np

from gbkfit.math import interpolation
from gbkfit.utils import iterutils, numutils, parseutils
from ..base import ComponentPlan
from ..lines import Lines
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


_log = logging.getLogger(__name__)


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


def load_options(cls, info, slots):
    """
    The options of a component of class cls that is made of one disk,
    with the traits of the given slots loaded. The __init__ of cls
    declares the options, and those without a default value are required.
    """
    for slot in slots:
        required = _is_required(cls, slot.key)
        parseutils.load_option_and_update_info(
            slot.parser, info, slot.key,
            required=required)
    return parseutils.parse_options_for_callable(info, cls.__init__)


def make_disk(
        cls, disk_class, slots, rdata_key,
        loose, tilted, rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp,
        traits_, **disk_options):
    """
    The disk of a component of class cls, of class disk_class, with the
    traits of the given slots, and its density map named rdata_key in the
    extra outputs (e.g. 'bdata'). traits_ has the traits, keyed by option
    (e.g. 'bptraits'). disk_options are the options of the type of disk
    (e.g. cflux and seed of MCDisk).
    """
    node_args = parse_component_rnode_args(
        rnmin, rnmax, rnsep, rnlen, rnodes, rstep, interp)
    traits_ = _parse_traits(cls, slots, traits_)
    check_traits_common(sum(traits_.values(), ()))
    return disk_class(
        **disk_options,
        loose=loose, tilted=tilted, **node_args,
        traits_={slot.kind: traits_[slot.key] for slot in slots},
        prefixes={slot.kind: slot.prefix for slot in slots},
        rdata_key=rdata_key)


def make_lines(disk, lines_):
    """
    The emission lines (see Lines) of a spectral component made of the
    given disk, whose parameters must have other names.
    """
    result = Lines(lines_)
    if repeated := sorted(set(result.pdescs()) & set(disk.pdescs())):
        raise RuntimeError(
            f"the parameters of the lines have the names of other "
            f"parameters: {repeated}; rename the lines")
    return result


def dump_disk(disk, slots):
    """
    The options of a component made of the given disk, with the traits
    of the given slots.
    """
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


def _parse_component_node_args(
        prefix, nmin, nmax, nsep, nlen, nodes, step, interp):
    nodes_list = nodes is not None
    nodes_arange = [nmin, nmax, nsep].count(None) == 0
    nodes_linspace = [nmin, nmax, nlen].count(None) == 0
    if [nodes_list, nodes_arange, nodes_linspace].count(True) != 1:
        raise RuntimeError(
            f"only one of the following sets of options "
            f"must be defined: "
            f"(1) {prefix}nodes; "
            f"(2) {prefix}nmin, {prefix}nmax, {prefix}nsep; "
            f"(3) {prefix}nmin, {prefix}nmax, {prefix}nlen")
    if (nodes_arange or nodes_linspace) and not (0 <= nmin < nmax):
        raise RuntimeError(
            f"the following expression must be true: "
            f"0 <= {prefix}nmin < {prefix}nmax")
    if nodes_arange and not (0 < nsep <= nmax - nmin):
        raise RuntimeError(
            f"the following expression must be true: "
            f"0 < {prefix}nsep <= {prefix}nmax - {prefix}nmin")
    if nlen is not None and not 2 <= nlen:
        raise RuntimeError(
            f"the following expression must be true: "
            f"2 =< {prefix}nlen")
    if nodes_arange:
        # From nmin every nsep, to the first node at or beyond nmax; the
        # count tolerates the rounding of (nmax - nmin) / nsep, which made
        # np.arange to nmax + nsep add a node (e.g. 2.4 for 2.2 in 0.2s)
        count = int(np.ceil((nmax - nmin) / nsep - 1e-9)) + 1
        nodes = (nmin + nsep * np.arange(count)).tolist()
    elif nodes_linspace:
        nodes = np.linspace(nmin, nmax, nlen).tolist()
    nodes = tuple(nodes)
    if len(nodes) < 2:
        raise RuntimeError(f"at least two {prefix}nodes must be provided")
    if not iterutils.is_ascending(nodes):
        raise RuntimeError(f"{prefix}nodes must be ascending")
    if not numutils.all_positive(nodes, include_zero=True):
        raise RuntimeError(f"{prefix}nodes must not be negative")
    if not iterutils.all_unique(nodes):
        raise RuntimeError(f"{prefix}nodes must be unique")
    if step is None:
        step = min(1, min(np.diff(nodes)) / 2)
    # (half the difference is allowed, also as it rounds)
    if step <= 0 or step > min(np.diff(nodes)) / 2 * (1 + 1e-9):
        raise RuntimeError(
            f"{prefix}step must be greater than zero and less than half the "
            f"smallest difference between two consecutive {prefix}nodes")
    interpolations = dict(
        linear=interpolation.InterpolatorLinear,
        akima=interpolation.InterpolatorAkima,
        pchip=interpolation.InterpolatorPCHIP)
    if interp not in interpolations:
        raise RuntimeError(
            "interp must be one of the following: "
            f"{list(interpolations.keys())}")
    return {
        f'{prefix}nodes': nodes,
        f'{prefix}step': step,
        'interp': interpolations[interp]}


def parse_component_rnode_args(nmin, nmax, nsep, nlen, nodes, step, interp):
    return _parse_component_node_args(
        'r', nmin, nmax, nsep, nlen, nodes, step, interp)


def parse_component_hnode_args(nmin, nmax, nsep, nlen, nodes, step, interp):
    return _parse_component_node_args(
        'h', nmin, nmax, nsep, nlen, nodes, step, interp)


def check_traits_common(traits_):
    unsupported_traits = (
        traits.WPTraitAxisRange)
    for trait in traits_:
        trait_desc = traits.trait_desc(trait.__class__)
        if isinstance(trait, unsupported_traits):
            raise NotImplementedError(
                f"{trait_desc} is not implemented yet")
        if isinstance(trait, traits.BPTraitUniform):
            _log.warning(
                f"the use of {trait_desc} is discouraged; "
                f"its main purpose is to facilitate software testing")
        if isinstance(trait, traits.BHTraitUniform):
            _log.warning(
                f"the use of {trait_desc} is discouraged; "
                f"it may result in density overestimation due to aliasing")
