
import abc
import logging
import typing
from dataclasses import dataclass

import numpy as np

from gbkfit.params.pdescs import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import miscutils, parseutils


_log = logging.getLogger(__name__)


# The kinds of traits of a disk: density (r), velocity (v) and dispersion
# (d) polar (p) and height (h) traits, and vertical distortion (z),
# selection (s) and weight (w) polar traits
TRAIT_KINDS = ('rpt', 'rht', 'vpt', 'vht', 'dpt', 'dht', 'zpt', 'spt', 'wpt')

# The most traits of one kind that the native kernels take (TRAIT_NUM_MAX
# in constants.hpp)
MAX_TRAITS = 4

# The geometric parameters of a disk, and the option that makes each of
# them node-wise: loose (vsys, xpos, ypos) or tilted (posa, incl)
NODEWISE_SWITCH = dict(
    vsys='loose', xpos='loose', ypos='loose', posa='tilted', incl='tilted')
GEOMETRY_PARAMS = tuple(NODEWISE_SWITCH)


def _make_param_descs(key, nnodes, nw):
    return {key: ParamVectorDesc(key, nnodes) if nw else ParamScalarDesc(key)}


@dataclass
class _TraitParams:
    """
    The parameters of the traits of one kind, with prefixed names, and for
    each one its node-wise mode (if any), whether it is node-wise, and its
    name in its trait. The keys of all dicts are in the same order: the
    order of the traits, and for each trait, the smooth parameters before
    the node-wise ones.
    """
    pdescs: dict
    nwmodes: dict
    isnw: dict
    pnames: list


def _trait_params(traits_, prefix, nrnodes):
    params_list = []
    for trait in traits_:
        params_sm = [(pdesc, None, False) for pdesc in trait.params_sm()]
        params_nw = [
            (pdesc, nwmode, True)
            for pdesc, nwmode in trait.params_rnw(nrnodes)]
        params_list.append(
            {tuple_[0].name(): tuple_ for tuple_ in params_sm + params_nw})
    params, mappings = miscutils.merge_with_prefixes(
        params_list,
        parseutils.item_prefixes([None] * len(params_list), prefix, True))
    return _TraitParams(
        pdescs={name: tuple_[0] for name, tuple_ in params.items()},
        nwmodes={name: tuple_[1] for name, tuple_ in params.items()},
        isnw={name: tuple_[2] for name, tuple_ in params.items()},
        pnames=mappings)


def _trait_constants(traits_, nnodes, nsubnodes):
    """
    The uids, constant values and their counts, and the parameter value
    counts of a set of traits. Each node-wise parameter has one value for
    each subnode, because the node values are interpolated.
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
        values, params, pdescs, isnw, nodes, subnodes, interp):
    """
    Write the values of the given parameters one after the other into
    values. The values of the node-wise parameters are replaced in params
    with their values interpolated to the subnodes.
    """
    start = 0
    for name in pdescs:
        if isnw[name]:
            params[name] = interp(nodes, params[name])(subnodes)
        stop = start + np.size(params[name])
        values[start:stop] = params[name]
        start = stop


def _weighted_mean(weighted_sum, weight):
    """The mean of each voxel from its weighted sum; NaN without weight."""
    return np.divide(
        weighted_sum, weight, out=np.full_like(weighted_sum, np.nan),
        where=weight != 0)


class Disk(abc.ABC):

    # The traits that this type of disk does not support (yet)
    unsupported_traits = ()

    def __init__(
            self, loose, tilted, rnodes, rstep, interp, nwmodes, traits_,
            prefixes, rdata_key):
        """
        nwmodes has the node-wise modes of the geometric parameters, and
        traits_ the traits of each kind. Both are keyed as in
        GEOMETRY_PARAMS and TRAIT_KINDS, and can leave keys out (no
        node-wise mode, no traits). prefixes has the prefix of the
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
        self._nwmodes = {name: nwmodes.get(name) for name in GEOMETRY_PARAMS}
        self._traits = {kind: traits_.get(kind, ()) for kind in TRAIT_KINDS}
        self._rdata_key = rdata_key

        # Make descs for the geometric parameters. There is no systemic
        # velocity without velocity traits.
        switches = dict(loose=loose, tilted=tilted)
        self._geometry_isnw = {
            name: switches[switch] for name, switch in NODEWISE_SWITCH.items()}
        self._geometry_pdescs = {
            name: _make_param_descs(name, nrnodes, self._geometry_isnw[name])
            for name in GEOMETRY_PARAMS
            if name != 'vsys' or self._traits['vpt']}

        # Make descs for the trait parameters (the kinds without traits
        # have no prefix, and no parameters)
        self._trait_params = {
            kind: _trait_params(traits_, prefixes.get(kind), nrnodes)
            for kind, traits_ in self._traits.items()}

        # Merge all parameter descs into the same dictionary
        self._pdescs = {}
        for pdescs in self._geometry_pdescs.values():
            self._pdescs.update(pdescs)
        for params in self._trait_params.values():
            self._pdescs.update(params.pdescs)

        # These are created by _prepare(): the native description of the
        # disk, the host and device memory with its parameter values
        # (geometry and traits, packed in one buffer), and the host view
        # of each group of parameter values in it
        self._native_disk = None
        self._param_values = None
        self._param_views = None

        self._dtype = None
        self._driver = None
        self._backend = None

    def loose(self):
        return self._loose

    def tilted(self):
        return self._tilted

    def rnodes(self):
        return self._rnodes

    def rstep(self):
        return self._rstep

    def interp(self):
        return self._interp

    def nwmode(self, name):
        return self._nwmodes[name]

    def traits(self, kind):
        return self._traits[kind]

    def options(self):
        """The options of this type of disk, besides those of all disks."""
        return {}

    def pdescs(self):
        return self._pdescs

    def _prepare(self, driver, dtype):

        # The number of values of each group of parameters: one value for
        # each subnode for node-wise geometric parameters, and the values
        # of all the traits of each kind
        sizes = {
            name: self._nsubrnodes if self._geometry_isnw[name] else 1
            for name in self._geometry_pdescs}
        constants = {}
        for kind, traits_ in self._traits.items():
            constants[kind] = _trait_constants(
                traits_, self._nrnodes, self._nsubrnodes)
            sizes[kind] = sum(constants[kind][3])

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
            if self._traits[kind]:
                trait_sets[kind] = trait_set_class(
                    uids=to_device(uids, np.int32),
                    cvalues=to_device(cvalues, dtype),
                    ccounts=to_device(ccounts, np.int32),
                    pvalues=views_d[kind],
                    pcounts=to_device(pcounts, np.int32))
        self._native_disk = driver.native_class('Disk', dtype)(
            loose=self._loose,
            tilted=self._tilted,
            rnodes=to_device(self._subrnodes, dtype),
            **{name: views_d.get(name) for name in GEOMETRY_PARAMS},
            **trait_sets)

        self._param_values = (values_h, values_d)
        self._param_views = views_h
        self._dtype = dtype
        self._driver = driver
        self._backend = driver.native_class('GModel', dtype)()

        # Perform preparation specific to the derived class
        self._impl_prepare(driver, dtype)

    def evaluate(self, driver, params, grid, outputs, dtype, out_extra):
        """
        Add the disk to the outputs. grid has the grid of the native
        evaluation functions (see Component.evaluate), and outputs the
        arrays they add to (all optional): the opacity cube they read
        ('opacity'), the 'image' or 'scube', and the 3d spatial weights
        ('wdata'), density ('rdata') and density after the opacity
        ('ordata').
        """

        if self._driver is not driver or self._dtype is not dtype:
            self._prepare(driver, dtype)

        # The parameter values are replaced below (nodewise mode
        # transforms, interpolation). Work on a copy of the dict, and
        # never modify the caller's arrays in place.
        params = dict(params)

        # Apply the nodewise mode transforms to the parameters
        for name, pdescs in self._geometry_pdescs.items():
            nwmode = self._nwmodes[name]
            if nwmode is not None:
                for pname in pdescs:
                    params[pname] = nwmode.transform(
                        params[pname], in_place=False)
        for trait_params in self._trait_params.values():
            for pname, nwmode in trait_params.nwmodes.items():
                if nwmode is not None:
                    params[pname] = nwmode.transform(
                        params[pname], in_place=False)

        # Write the parameter values into the host memory, with the
        # nodewise parameters interpolated to the subnodes, and copy
        # them to the device
        def fill(name, pdescs, isnw):
            _fill_param_values(
                self._param_views[name], params, pdescs, isnw,
                self._rnodes, self._subrnodes, self._interp)
        for name, pdescs in self._geometry_pdescs.items():
            isnw = self._geometry_isnw[name]
            fill(name, pdescs, dict.fromkeys(pdescs, isnw))
        for kind, trait_params in self._trait_params.items():
            fill(kind, trait_params.pdescs, trait_params.isnw)
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
            if self._traits['rpt']:
                rdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(rdata_cmp, 0)
            # The velocity and dispersion of each voxel are means weighted
            # by the absolute density: the kernels add to their weighted
            # sums and to the sum of the weights
            if self._traits['vpt']:
                vdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(vdata_cmp, 0)
            if self._traits['dpt']:
                ddata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(ddata_cmp, 0)
            if self._traits['vpt'] or self._traits['dpt']:
                vdweight_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(vdweight_cmp, 0)
            if self._traits['wpt']:
                wdata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(wdata_cmp, 1)
            if odata is not None:
                ordata_cmp = driver.mem_alloc_d(shape, dtype)
                driver.mem_fill(ordata_cmp, 0)

        # The keyword arguments of the native evaluation functions
        grid_and_outputs = grid | outputs | dict(
            wdata_cmp=wdata_cmp, rdata_cmp=rdata_cmp, ordata_cmp=ordata_cmp,
            vdata_cmp=vdata_cmp, ddata_cmp=ddata_cmp,
            vdweight_cmp=vdweight_cmp)

        self._impl_evaluate(driver, params, grid_and_outputs, out_extra)

        if out_extra is not None:

            rdata_key = self._rdata_key

            if self._traits['rpt']:
                out_extra[rdata_key] = driver.mem_copy_d2h(rdata_cmp)
            if vdweight_cmp is not None:
                weight = driver.mem_copy_d2h(vdweight_cmp)
            if self._traits['vpt']:
                out_extra['vdata'] = _weighted_mean(
                    driver.mem_copy_d2h(vdata_cmp), weight)
            if self._traits['dpt']:
                out_extra['ddata'] = _weighted_mean(
                    driver.mem_copy_d2h(ddata_cmp), weight)
            if self._traits['wpt']:
                out_extra['wdata'] = driver.mem_copy_d2h(wdata_cmp)
            if odata is not None:
                out_extra['obdata'] = driver.mem_copy_d2h(ordata_cmp)
            if self._traits['rpt']:
                sumabs = np.nansum(np.abs(out_extra[rdata_key]))
                _log.debug(f"sum(abs({rdata_key})): {sumabs}")
            if self._traits['vpt']:
                sumabs = np.nansum(np.abs(out_extra['vdata']))
                _log.debug(f"sum(abs(vdata)): {sumabs}")
            if self._traits['dpt']:
                sumabs = np.nansum(np.abs(out_extra['ddata']))
                _log.debug(f"sum(abs(ddata)): {sumabs}")

    @abc.abstractmethod
    def _impl_prepare(self, driver, dtype):
        pass

    @abc.abstractmethod
    def _impl_evaluate(self, driver, params, grid_and_outputs, out_extra):
        """
        Evaluate the disk. params has the values of the node-wise
        parameters interpolated to the subnodes, and grid_and_outputs
        the keyword arguments of the grid and the outputs of the native
        evaluation functions.
        """
        pass
