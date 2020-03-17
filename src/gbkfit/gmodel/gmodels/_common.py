
import abc

import numpy as np

from gbkfit.params import ParamScalarDesc, ParamVectorDesc
from gbkfit.utils import iterutils, parseutils
from . import traits


class DensityComponent2D(abc.ABC):

    @staticmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, backend, params, image, dtype,
            spat_size, spat_step, spat_zero,
            out_extra):
        pass


class DensityComponent3D(abc.ABC):

    @staticmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, backend, params, image, rcube, dtype,
            spat_size, spat_step, spat_zero,
            out_extra):
        pass


class SpectralComponent2D(abc.ABC):

    @staticmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, backend, params, scube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):
        pass


class SpectralComponent3D(abc.ABC):

    @staticmethod
    def type():
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, info):
        pass

    @abc.abstractmethod
    def dump(self):
        pass

    @abc.abstractmethod
    def params(self):
        pass

    @abc.abstractmethod
    def evaluate(
            self, backend, params, scube, rcube, dtype,
            spat_size, spat_step, spat_zero,
            spec_size, spec_step, spec_zero,
            out_extra):
        pass


density_component_2d_parser = parseutils.TypedParser(DensityComponent2D)
density_component_3d_parser = parseutils.TypedParser(DensityComponent3D)
spectral_component_2d_parser = parseutils.TypedParser(SpectralComponent2D)
spectral_component_3d_parser = parseutils.TypedParser(SpectralComponent3D)


def _make_component_params(components, prefix, force_prefix):
    params = {}
    mappings = []
    for i, cmp in enumerate(components):
        prefix = f'{prefix}{i}_' * (i > 0 or force_prefix)
        params.update({prefix + k: v for k, v in cmp.params().items()})
        mappings.append(dict(zip(cmp.params().keys(), params.keys())))
    return params, mappings


def make_model_2d_params(components):
    return _make_component_params(components, 'cmp', False)


def make_model_3d_params(components, tcomponents, tauto):
    params, mappings = _make_component_params(components, 'cmp', False)
    tparams, tmappings = _make_component_params(tcomponents, 'tcmp', True)
    return {**params, **tparams}, mappings, tmappings


def trait_param_info(traits, prefix, nrnodes):
    descs = {}
    mappings = []
    for i, trait in enumerate(traits):
        prefix += str(i + 1) * (i > 0)
        params = trait.params_sm() + trait.params_rnw(nrnodes)
        mapping = {}
        for param in params:
            old_pname = param.name()
            new_pname = f'{prefix}_{old_pname}'
            descs[new_pname] = param
            mapping[old_pname] = new_pname
        mappings.append(mapping)
    return descs, mappings






def prepare_trait_arrays(
        traits, nnodes, nsubnodes,
        ary_uids,
        ary_ccounts, ary_pcounts,
        ary_cvalues, ary_pvalues,
        dtype, driver):
    uids = []
    ccounts = []
    pcounts = []
    cvalues = []
    for trait in traits:
        consts = trait.consts()
        params_sm = trait.params_sm()
        params_nw = trait.params_rnw(nnodes)
        pcounts_sm = sum(p.size() for p in params_sm)
        pcounts_nw = len(params_nw) * nsubnodes
        uids.append(trait.uid())
        cvalues += consts
        ccounts += [len(consts)]
        pcounts += [pcounts_sm + pcounts_nw]
    ary_uids[:] = driver.mem_alloc_s(len(uids), np.int32)
    ary_ccounts[:] = driver.mem_alloc_s(len(ccounts), np.int32)
    ary_pcounts[:] = driver.mem_alloc_s(len(pcounts), np.int32)
    ary_cvalues[:] = driver.mem_alloc_s(len(cvalues), dtype)
    ary_pvalues[:] = driver.mem_alloc_s(sum(pcounts), dtype)
    ary_uids[0][:] = uids
    ary_ccounts[0][:] = ccounts
    ary_pcounts[0][:] = pcounts
    ary_cvalues[0][:] = cvalues
    driver.mem_copy_h2d(ary_uids[0], ary_uids[1])
    driver.mem_copy_h2d(ary_ccounts[0], ary_ccounts[1])
    driver.mem_copy_h2d(ary_pcounts[0], ary_pcounts[1])
    driver.mem_copy_h2d(ary_cvalues[0], ary_cvalues[1])



def make_param_descs(key, nnodes, pw):
    return {key: ParamVectorDesc(key, nnodes) if pw else ParamScalarDesc(key)}

# TODO all trait.params_nw occurrences

def prepare_traits_params_array(
        driver, params, ary, descs, nodes, subnodes, interp, mappings, traits):
    start = 0
    for trait, mapping in zip(traits, mappings):
        for pname in trait.params_sm():
            new_pname = mapping[pname.name()]
            stop = start + descs[new_pname].size()
            ary[0][start:stop] = params[new_pname]
            start = stop
        for pname in trait.params_rnw(len(nodes)):
            new_pname = mapping[pname.name()]
            stop = start + len(subnodes)
            params[new_pname] = interp(nodes, params[new_pname])(subnodes)
            ary[0][start:stop] = params[new_pname]
            start = stop
    driver.mem_copy_h2d(ary[0], ary[1])


def prepare_common_params_array(
        driver, params, ary, descs, nodes, subnodes, interp, nodewise):
    start = 0
    for pname, desc in descs.items():
        if nodewise:
            stop = start + len(subnodes)
            params[pname] = interp(nodes, params[pname])(subnodes)
            ary[0][start:stop] = params[pname]
        else:
            stop = start + desc.size()
            ary[0][start:stop] = params[pname]
        start = stop
    driver.mem_copy_h2d(ary[0], ary[1])


def prepare_rnode_array(backend, dtype, array_rnodes, rnodes):
    array_rnodes[:] = backend.mem_alloc_s(len(rnodes), dtype)
    array_rnodes[0][:] = rnodes
    backend.mem_copy_h2d(array_rnodes[0], array_rnodes[1])


def parse_node_args(prefix, nodes, nmin, nmax, nsep, nlen):
    nodes_list = nodes is not None
    nodes_arange = [nmin, nmax, nsep].count(None) == 0
    nodes_linspace = [nmin, nmax, nlen].count(None) == 0
    if [nodes_list, nodes_arange, nodes_linspace].count(True) != 1:
        raise RuntimeError(
            f'Only one of the following groups must be defined: '
            f'(1) {prefix}nodes; '
            f'(2) {prefix}nmin, {prefix}nmax, {prefix}nsep; '
            f'(3) {prefix}nmin, {prefix}nmax, {prefix}nlen')
    if nmin is not None and not (0 <= nmin < nmax):
        raise RuntimeError(
            f'The following expression must be true: '
            f'0 <= {prefix}nmin < {prefix}nmax')
    if nsep is not None and not (0 < nsep <= nmax - nmin):
        raise RuntimeError(
            f'The following expression must be true: '
            f'0 < {prefix}nsep <= {prefix}nmax - {prefix}nmin')
    if nlen is not None and not 2 <= nlen:
        raise RuntimeError(
            f'The following expression must be true: '
            f'2 =< {prefix}nlen')
    if nodes_arange:
        nodes = np.arange(nmin, nmax + nsep, nsep).tolist()
    elif nodes_linspace:
        nodes = np.linspace(nmin, nmax, nlen).tolist()
    return nodes


def parse_density_disk_2d_common_args(
        loose, tilted,
        rnmin, rnmax, rnsep, rnlen, rnodes,
        rptraits,
        sptraits):

    rnodes = parse_node_args('r', rnodes, rnmin, rnmax, rnsep, rnlen)

    if not rptraits:
        raise RuntimeError("at least one rptrait is required")

    rptraits = iterutils.tuplify(rptraits)
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()

    return dict(
        loose=loose, tilted=tilted, rnodes=rnodes,
        rptraits=rptraits,
        sptraits=sptraits)


def parse_density_disk_3d_common_args(
        loose, tilted,
        rnmin, rnmax, rnsep, rnlen, rnodes,
        rptraits, rhtraits,
        wptraits,
        sptraits):

    rnodes = parse_node_args('r', rnodes, rnmin, rnmax, rnsep, rnlen)

    if not rptraits:
        raise RuntimeError("at least one rptrait is required")
    if not rhtraits:
        raise RuntimeError("at least one rhtrait is required")

    rptraits = iterutils.tuplify(rptraits)
    rhtraits = iterutils.tuplify(rhtraits)
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()

    rptraits_len = len(rptraits)
    rhtraits_len = len(rhtraits)
    if rptraits_len != rhtraits_len:
        raise RuntimeError(
            f"the number of rhtraits must be equal to "
            f"the number of rptraits ({rhtraits_len} != {rptraits_len})")

    return dict(
        loose=loose, tilted=tilted, rnodes=rnodes,
        rptraits=rptraits, rhtraits=rhtraits,
        wptraits=wptraits,
        sptraits=sptraits)


def parse_spectral_disk_2d_common_args(
        loose, tilted,
        rnmin, rnmax, rnsep, rnlen, rnodes,
        rptraits,
        vptraits,
        dptraits,
        sptraits):

    rnodes = parse_node_args('r', rnodes, rnmin, rnmax, rnsep, rnlen)

    if not rptraits:
        raise RuntimeError("at least one rptrait is required")
    if not vptraits:
        raise RuntimeError("at least one vptrait is required")
    if not dptraits:
        raise RuntimeError("at least one dptrait is required")

    rptraits = iterutils.tuplify(rptraits)
    vptraits = iterutils.tuplify(vptraits)
    dptraits = iterutils.tuplify(dptraits)
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()

    return dict(
        loose=loose, tilted=tilted, rnodes=rnodes,
        rptraits=rptraits,
        vptraits=vptraits,
        dptraits=dptraits,
        sptraits=sptraits)


def parse_spectral_disk_3d_common_args(
        loose, tilted,
        rnmin, rnmax, rnsep, rnlen, rnodes,
        rptraits, rhtraits,
        vptraits, vhtraits,
        dptraits, dhtraits,
        wptraits,
        sptraits):

    rnodes = parse_node_args('r', rnodes, rnmin, rnmax, rnsep, rnlen)

    if not rptraits:
        raise RuntimeError("at least one rptrait is required")
    if not rhtraits:
        raise RuntimeError("at least one rhtrait is required")
    if not vptraits:
        raise RuntimeError("at least one vptrait is required")
    if not dptraits:
        raise RuntimeError("at least one dptrait is required")

    rptraits = iterutils.tuplify(rptraits)
    rhtraits = iterutils.tuplify(rhtraits)
    vptraits = iterutils.tuplify(vptraits)
    vhtraits = iterutils.tuplify(vhtraits) \
        if vhtraits else tuple([None] * len(vptraits))
    dptraits = iterutils.tuplify(dptraits)
    dhtraits = iterutils.tuplify(dhtraits) \
        if dhtraits else tuple([None] * len(dptraits))
    wptraits = iterutils.tuplify(wptraits) if wptraits else tuple()
    sptraits = iterutils.tuplify(sptraits) if sptraits else tuple()

    if None in vhtraits:
        vhtraits = iterutils.replace_items_and_copy(
            vhtraits, None, traits.VHTraitOne())
    if None in dhtraits:
        dhtraits = iterutils.replace_items_and_copy(
            dhtraits, None, traits.DHTraitOne())

    rptraits_len = len(rptraits)
    rhtraits_len = len(rhtraits)
    vptraits_len = len(vptraits)
    vhtraits_len = len(vhtraits)
    dptraits_len = len(dptraits)
    dhtraits_len = len(dhtraits)
    if rptraits_len != rhtraits_len:
        raise RuntimeError(
            f"the number of rhtraits must be equal to "
            f"the number of rptraits ({rhtraits_len} != {rptraits_len})")
    if vptraits_len != vhtraits_len:
        raise RuntimeError(
            f"the number of vhtraits must be equal to "
            f"the number of vptraits ({vhtraits_len} != {vptraits_len})")
    if len(dptraits) != len(dhtraits):
        raise RuntimeError(
            f"the number of dhtraits must be equal to "
            f"the number of dptraits ({dhtraits_len} != {dptraits_len})")

    return dict(
        loose=loose, tilted=tilted, rnodes=rnodes,
        rptraits=rptraits, rhtraits=rhtraits,
        vptraits=vptraits, vhtraits=vhtraits,
        dptraits=dptraits, dhtraits=dhtraits,
        wptraits=wptraits,
        sptraits=sptraits)
