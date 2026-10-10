
import abc
import logging
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from gbkfit.driver import DeviceArray, Driver
from gbkfit.math.interpolation import Interpolator
from gbkfit.params.pdescs import ParamDesc, ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import iterutils, parseutils
from ..base import NativeGrid
from ..geometries import GEOMETRY_PARAMS
from .traits import Trait


_log = logging.getLogger(__name__)


# The kinds of traits of a disk: density (r), velocity (v) and dispersion
# (d) polar (p) and height (h) traits, and vertical distortion (z),
# selection (s) and weight (w) polar traits
TRAIT_KINDS = ('rpt', 'rht', 'vpt', 'vht', 'dpt', 'dht', 'zpt', 'spt', 'wpt')

# The most traits of one kind that the native kernels take (TRAIT_NUM_MAX
# in constants.hpp)
MAX_TRAITS = 4


def _make_param_descs(key: str, nnodes: int, nw: bool) -> dict[str, ParamDesc]:
    """Return the parameter of the given name, node-wise if nw."""
    return {key: ParamVectorDesc(key, nnodes) if nw else ParamScalarDesc(key)}


@dataclass
class TraitParams:
    """
    The parameters of the traits of one kind, with prefixed names, and for
    each one where its values are given if it is node-wise (its trait's
    sampling, see traits.SAMPLINGS) or None, and its name in its trait.
    The keys of all dicts are in the same order: the order of the traits,
    and for each trait, the smooth parameters before the node-wise ones.
    circular_velocity has the names of those whose values are the
    circular velocity of the mass model of the model (see
    traits.Trait.circular_velocity_params).
    """
    pdescs: dict[str, ParamDesc]
    sampling: dict[str, str | None]
    pnames: tuple[dict[str, str], ...]
    circular_velocity: tuple[str, ...]


def _trait_params(
        traits_: Sequence[Trait],
        prefix: str | None,
        nrnodes: int,
        nsubrnodes: int
) -> 'TraitParams':
    """
    Return the parameters of the traits of one kind. A node-wise parameter
    has a value for each node, or for each subnode if its trait samples it
    at the subnodes.
    """
    params_list = []
    for trait in traits_:
        sampling = trait.sampling()
        nnodes = nsubrnodes if sampling == 'subrings' else nrnodes
        params_sm = [(pdesc, None) for pdesc in trait.params_sm()]
        params_nw = [(pdesc, sampling) for pdesc in trait.params_rnw(nnodes)]
        params_list.append(
            {tuple_[0].name(): tuple_ for tuple_ in params_sm + params_nw})
    params, mappings = iterutils.merge_with_prefixes(
        params_list,
        parseutils.item_prefixes(
            [None] * len(params_list), 'traits', prefix, True))
    return TraitParams(
        pdescs={name: tuple_[0] for name, tuple_ in params.items()},
        sampling={name: tuple_[1] for name, tuple_ in params.items()},
        pnames=mappings,
        circular_velocity=tuple(
            mapping[name]
            for trait, mapping in zip(traits_, mappings)
            for name in trait.circular_velocity_params()))


def _trait_constants(
        traits_: Sequence[Trait], nnodes: int, nsubnodes: int
) -> tuple[list[int], list[float], list[int], list[int]]:
    """
    Return the uids, constant values and their counts, and the parameter
    value counts of a set of traits. Each node-wise parameter has one
    value for each subnode, because the node values are interpolated.
    """
    uids, cvalues, ccounts, pcounts = [], [], [], []
    for trait in traits_:
        consts = trait.consts()
        npvalues_sm = sum(p.size() for p in trait.params_sm())
        npvalues_nw = len(trait.params_rnw(nnodes)) * nsubnodes
        uids.append(trait.uid())
        cvalues += consts
        ccounts.append(len(consts))
        pcounts.append(npvalues_sm + npvalues_nw)
    return uids, cvalues, ccounts, pcounts


def _fill_param_values(
        values: np.ndarray,
        params: dict[str, float | np.ndarray],
        pdescs: dict[str, ParamDesc],
        sampling: dict[str, str | None],
        nodes: Sequence[float],
        subnodes: Sequence[float],
        interp: type[Interpolator]
) -> None:
    """
    Write the values of the given parameters one after the other into
    values. The values of the node-wise parameters given at the nodes are
    replaced in params with their values interpolated to the subnodes
    (sampling has where the values of each parameter are given, see
    TraitParams).
    """
    start = 0
    for name in pdescs:
        if sampling[name] == 'rnodes':
            params[name] = interp(nodes, params[name])(subnodes)
        stop = start + np.size(params[name])
        values[start:stop] = params[name]
        start = stop


def _weighted_mean(weighted_sum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """
    Return the mean of each voxel from its weighted sum; NaN without
    weight.
    """
    return np.divide(
        weighted_sum, weight, out=np.full_like(weighted_sum, np.nan),
        where=weight != 0)


class Disk(abc.ABC):

    def __init__(
            self,
            loose: bool,
            tilted: bool,
            rnodes: Sequence[float],
            rstep: float,
            interp: type[Interpolator],
            traits_: dict[str, Sequence[Trait]],
            prefixes: dict[str, str],
            rdata_key: str
    ):
        """
        traits_ has the traits of each kind, keyed as in TRAIT_KINDS, and
        can leave kinds out (no traits). prefixes has the prefix of the
        parameters of the traits of each kind in traits_ (e.g. 'opt' for
        the density traits of an opacity disk). rdata_key is the name of
        the density map in the extra outputs (e.g. 'odata').
        """

        nrnodes = len(rnodes)

        # The radial sub nodes: the first and the last node, and between
        # them the centres of rings of equal width, at most rstep, that
        # cover the radii between the two exactly
        nrings = max(1, int(np.ceil((rnodes[-1] - rnodes[0]) / rstep - 1e-9)))
        width = (rnodes[-1] - rnodes[0]) / nrings
        centres = rnodes[0] + (np.arange(nrings) + 0.5) * width
        subrnodes = np.concatenate(([rnodes[0]], centres, [rnodes[-1]]))
        subrnodes = tuple(typing.cast(list, subrnodes.tolist()))

        self._loose = loose
        self._tilted = tilted
        self._rnodes = rnodes
        self._nrnodes = nrnodes
        self._rstep = rstep
        self._subrnodes = subrnodes
        self._nsubrnodes = len(subrnodes)
        self._interp = interp
        self._traits = {kind: traits_.get(kind, ()) for kind in TRAIT_KINDS}
        self._rdata_key = rdata_key

        # Make descs for the geometric parameters. There is no systemic
        # velocity without velocity traits.
        switches = dict(loose=loose, tilted=tilted)
        self._geometry_isnw = {
            name: switches[switch] for name, switch in GEOMETRY_PARAMS.items()}
        self._geometry_pdescs = {
            name: _make_param_descs(name, nrnodes, self._geometry_isnw[name])
            for name in GEOMETRY_PARAMS
            if name != 'vsys' or self._traits['vpt']}

        # Make descs for the trait parameters (the kinds without traits
        # have no prefix, and no parameters)
        self._trait_params = {
            kind: _trait_params(
                traits_, prefixes.get(kind), nrnodes, self._nsubrnodes)
            for kind, traits_ in self._traits.items()}

        # The parameters whose values are the circular velocity of the
        # mass model of their model, and the radii of their values
        self._circular_velocity_params = {
            name: subrnodes if params.sampling[name] == 'subrings' else rnodes
            for params in self._trait_params.values()
            for name in params.circular_velocity}

        # Merge all parameter descs into the same dictionary, without those
        # that the model gives
        self._pdescs = {}
        for pdescs in self._geometry_pdescs.values():
            self._pdescs.update(pdescs)
        for params in self._trait_params.values():
            self._pdescs.update(params.pdescs)
        for name in self._circular_velocity_params:
            del self._pdescs[name]

    def loose(self) -> bool:
        return self._loose

    def tilted(self) -> bool:
        return self._tilted

    def rnodes(self) -> tuple[float, ...]:
        return self._rnodes

    def rstep(self) -> float:
        return self._rstep

    def subrnodes(self) -> tuple[float, ...]:
        return self._subrnodes

    def interp(self) -> type[Interpolator]:
        return self._interp

    def traits(self, kind: str) -> tuple[Trait, ...]:
        return self._traits[kind]

    def trait_params(self, kind: str) -> 'TraitParams':
        """Return the parameters of the traits of one kind (TraitParams)."""
        return self._trait_params[kind]

    def options(self) -> dict[str, Any]:
        """
        Return the options of this type of disk, besides those of all
        disks.
        """
        return {}

    def pdescs(self) -> dict[str, ParamDesc]:
        return self._pdescs

    def circular_velocity_params(self) -> dict[str, tuple[float, ...]]:
        """
        Return the parameters that are not in pdescs, but whose values are
        the circular velocity of the mass model of their model at the
        given radii (arcsec), by name. The disk plans need them with the
        others.
        """
        return self._circular_velocity_params

    @abc.abstractmethod
    def plan(self, driver: Driver, nlines: int, dtype: np.dtype) -> 'DiskPlan':
        """
        Plan the evaluation of the disk on the given driver and dtype (a
        DiskPlan), which owns the memory it needs. nlines is the number of
        emission lines it adds to spectral cubes (0 without them).
        """
        pass


class DiskPlan(abc.ABC):
    """
    The evaluation of a disk on a driver and dtype: the native description
    of the disk, and the host and device memory with its parameter values
    (geometry and traits, packed in one buffer, and the values of its
    emission lines), and the host view of each group of values in it.
    """

    def __init__(
            self, disk: Disk, driver: Driver, nlines: int, dtype: np.dtype):

        # The number of values of each group of parameters: one value for
        # each subnode for node-wise geometric parameters, and the values
        # of all the traits of each kind
        sizes = {
            name: disk._nsubrnodes if disk._geometry_isnw[name] else 1
            for name in disk._geometry_pdescs}
        constants = {}
        for kind, traits_ in disk._traits.items():
            constants[kind] = _trait_constants(
                traits_, disk._nrnodes, disk._nsubrnodes)
            sizes[kind] = sum(constants[kind][3])
        # The offset, scale and flux of each emission line
        sizes['lines'] = 3 * nlines

        # Pack all parameter values in one buffer, so that they can be
        # copied to the device with one copy per evaluation
        values_h, values_d = driver.mem_alloc_s(sum(sizes.values()), dtype)
        views_h, views_d = {}, {}
        start = 0
        for name, size in sizes.items():
            views_h[name] = values_h[start:start + size]
            views_d[name] = values_d[start:start + size]
            start += size

        def to_device(values, dtype_):
            return driver.mem_copy_h2d(np.array(values, dtype_))

        # Describe the disk to the native module: its (sub)nodes, the
        # device memory of its geometric parameters, and its traits
        trait_set_class = driver.native_class('TraitSet', dtype)
        trait_sets = {}
        for kind, (uids, cvalues, ccounts, pcounts) in constants.items():
            if disk._traits[kind]:
                trait_sets[kind] = trait_set_class(
                    uids=to_device(uids, np.int32),
                    cvalues=to_device(cvalues, dtype),
                    ccounts=to_device(ccounts, np.int32),
                    pvalues=views_d[kind],
                    pcounts=to_device(pcounts, np.int32))
        self._native_disk = driver.native_class('Disk', dtype)(
            loose=disk._loose,
            tilted=disk._tilted,
            rnodes=to_device(disk._subrnodes, dtype),
            **{name: views_d.get(name) for name in GEOMETRY_PARAMS},
            **trait_sets)

        self._disk = disk
        self._driver = driver
        self._dtype = dtype
        self._param_values = (values_h, values_d)
        self._param_views = views_h
        self._lines_h = views_h['lines'].reshape(nlines, 3)
        self._lines_d = views_d['lines'].reshape(nlines, 3) if nlines else None
        self._backend = driver.native_class('GModel', dtype)()

    def evaluate(
            self,
            params: dict[str, float | np.ndarray],
            grid: NativeGrid,
            outputs: dict[str, DeviceArray | None],
            out_extra: dict[str, Any] | None,
            lines: np.ndarray | None = None
    ) -> None:
        """
        Add the disk to the outputs. grid has the grid of the native
        evaluation functions (see ComponentPlan.evaluate), and outputs the
        arrays they add to (all optional): the opacity cube they read
        ('opacity'), the 'image' or 'scube', and the 3d spatial weights
        ('wdata'), density ('rdata') and density after the opacity
        ('ordata'). lines has the offset, scale and flux of each emission
        line (an array of shape (nlines, 3); see the native DiskArgs), if
        the plan has lines.
        """

        disk = self._disk
        driver = self._driver
        dtype = self._dtype

        # The node-wise parameter values are replaced below with their
        # values at the subnodes: work on a copy of the dict, and never
        # modify the caller's arrays in place
        params = dict(params)

        # Write the parameter values into the host memory, with the
        # nodewise parameters at the subnodes (interpolated if given at
        # the nodes), and copy them to the device
        def fill(name, pdescs, sampling):
            _fill_param_values(
                self._param_views[name], params, pdescs, sampling,
                disk._rnodes, disk._subrnodes, disk._interp)
        for name, pdescs in disk._geometry_pdescs.items():
            sampling = 'rnodes' if disk._geometry_isnw[name] else None
            fill(name, pdescs, dict.fromkeys(pdescs, sampling))
        for kind, trait_params in disk._trait_params.items():
            fill(kind, trait_params.pdescs, trait_params.sampling)
        if lines is not None:
            self._lines_h[:] = lines
        driver.mem_copy_h2d(*self._param_values)

        wdata_cmp = None
        rdata_cmp = None
        vdata_cmp = None
        ddata_cmp = None
        vdweight_cmp = None
        ordata_cmp = None

        odata = outputs.get('opacity')
        if out_extra is not None:
            shape = tuple(grid['spat_size'][::-1])
            if disk._traits['rpt']:
                rdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(rdata_cmp, 0)
            # The velocity and dispersion of each voxel are means weighted
            # by the absolute density: the kernels add to their weighted
            # sums and to the sum of the weights
            if disk._traits['vpt']:
                vdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(vdata_cmp, 0)
            if disk._traits['dpt']:
                ddata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(ddata_cmp, 0)
            if disk._traits['vpt'] or disk._traits['dpt']:
                vdweight_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(vdweight_cmp, 0)
            if disk._traits['wpt']:
                wdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(wdata_cmp, 1)
            if odata is not None:
                ordata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(ordata_cmp, 0)

        # The keyword arguments of the native evaluation functions
        grid_and_outputs = grid | outputs | dict(
            lines=self._lines_d,
            wdata_cmp=wdata_cmp, rdata_cmp=rdata_cmp, ordata_cmp=ordata_cmp,
            vdata_cmp=vdata_cmp, ddata_cmp=ddata_cmp,
            vdweight_cmp=vdweight_cmp)

        self._impl_evaluate(params, grid_and_outputs, out_extra)

        if out_extra is not None:

            rdata_key = disk._rdata_key

            if disk._traits['rpt']:
                out_extra[rdata_key] = driver.mem_copy_d2h(rdata_cmp)
            if vdweight_cmp is not None:
                weight = driver.mem_copy_d2h(vdweight_cmp)
            if disk._traits['vpt']:
                out_extra['vdata'] = _weighted_mean(
                    driver.mem_copy_d2h(vdata_cmp), weight)
            if disk._traits['dpt']:
                out_extra['ddata'] = _weighted_mean(
                    driver.mem_copy_d2h(ddata_cmp), weight)
            if disk._traits['wpt']:
                out_extra['wdata'] = driver.mem_copy_d2h(wdata_cmp)
            if odata is not None:
                out_extra['obdata'] = driver.mem_copy_d2h(ordata_cmp)
            if disk._traits['rpt']:
                sumabs = np.nansum(np.abs(out_extra[rdata_key]))
                _log.debug(f"sum(abs({rdata_key})): {sumabs}")
            if disk._traits['vpt']:
                sumabs = np.nansum(np.abs(out_extra['vdata']))
                _log.debug(f"sum(abs(vdata)): {sumabs}")
            if disk._traits['dpt']:
                sumabs = np.nansum(np.abs(out_extra['ddata']))
                _log.debug(f"sum(abs(ddata)): {sumabs}")

    @abc.abstractmethod
    def _impl_evaluate(
            self,
            params: dict[str, float | np.ndarray],
            grid_and_outputs: dict[str, Any],
            out_extra: dict[str, Any] | None
    ) -> None:
        """
        Evaluate the disk. params has the values of the node-wise
        parameters at the subnodes, and grid_and_outputs the keyword
        arguments of the grid and the outputs of the native evaluation
        functions.
        """
        pass
