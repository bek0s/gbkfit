
import numpy as np

from gbkfit.math import interpolation
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import iterutils
from . import traits


def _parse_component_node_args(
        prefix, nmin, nmax, nsep, nlen, nodes, nstep, interp):
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
        nodes = np.arange(nmin, nmax + nsep, nsep).tolist()
    elif nodes_linspace:
        nodes = np.linspace(nmin, nmax, nlen).tolist()
    if len(nodes) < 2:
        raise RuntimeError(f"at least two {prefix}nodes must be provided")
    if not iterutils.is_ascending(nodes):
        raise RuntimeError(f"{prefix}nodes must be ascending")
    if not iterutils.all_positive(nodes):
        raise RuntimeError(f"{prefix}nodes must be positive")
    if not iterutils.all_unique(nodes):
        raise RuntimeError(f"{prefix}nodes must be unique")
    if nstep is None:
        nstep = min(np.diff(nodes)) / 2
    if nstep <= 0 or nstep > min(np.diff(nodes)) / 2:
        raise RuntimeError(
            f"{prefix}nstep must be greater than zero and less than half the "
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
        f'{prefix}nstep': nstep,
        'interp': interpolations[interp]}


def parse_component_rnode_args(nmin, nmax, nsep, nlen, nodes, nstep, interp):
    return _parse_component_node_args(
        'r', nmin, nmax, nsep, nlen, nodes, nstep, interp)


def parse_component_hnode_args(nmin, nmax, nsep, nlen, nodes, nstep, interp):
    return _parse_component_node_args(
        'h', nmin, nmax, nsep, nlen, nodes, nstep, interp)


def parse_component_d2d_trait_args(
        rptraits,
        sptraits,
        wptraits):
    if not rptraits:
        raise RuntimeError("at least one rptrait is required")
    rptraits = iterutils.tuplify(rptraits)
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    return dict(
        rptraits=rptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def parse_component_s2d_trait_args(
        rptraits,
        vptraits,
        dptraits,
        sptraits,
        wptraits):
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
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    return dict(
        rptraits=rptraits,
        vptraits=vptraits,
        dptraits=dptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def parse_component_d3d_trait_args(
        rptraits, rhtraits,
        zptraits,
        sptraits,
        wptraits):
    if not rptraits:
        raise RuntimeError("at least one rptrait is required")
    if not rhtraits:
        raise RuntimeError("at least one rhtrait is required")
    rptraits = iterutils.tuplify(rptraits)
    rhtraits = iterutils.tuplify(rhtraits)
    zptraits = iterutils.tuplify(zptraits) if zptraits is not None else ()
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    rptraits_len = len(rptraits)
    rhtraits_len = len(rhtraits)
    if rptraits_len != rhtraits_len:
        raise RuntimeError(
            f"the number of rhtraits must be equal to "
            f"the number of rptraits ({rhtraits_len} != {rptraits_len})")
    return dict(
        rptraits=rptraits, rhtraits=rhtraits,
        zptraits=zptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def parse_component_s3d_trait_args(
        rptraits, rhtraits,
        vptraits, vhtraits,
        dptraits, dhtraits,
        zptraits,
        sptraits,
        wptraits):
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
    zptraits = iterutils.tuplify(zptraits) if zptraits else tuple()
    sptraits = iterutils.tuplify(sptraits) if sptraits else tuple()
    wptraits = iterutils.tuplify(wptraits) if wptraits else tuple()
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
        rptraits=rptraits, rhtraits=rhtraits,
        vptraits=vptraits, vhtraits=vhtraits,
        dptraits=dptraits, dhtraits=dhtraits,
        zptraits=zptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def _make_gmodel_params_cmp(components, prefix, force_prefix):
    import gbkfit.utils.miscutils
    return gbkfit.utils.miscutils.merge_dicts_and_make_mappings(
        [cmp.params() for cmp in components], prefix, force_prefix)


def make_gmodel_2d_params(components):
    return _make_gmodel_params_cmp(components, 'cmp', False)


def make_gmodel_3d_params(components, tcomponents, tauto):
    params, mappings = _make_gmodel_params_cmp(components, 'cmp', False)
    tparams, tmappings = _make_gmodel_params_cmp(tcomponents, 'tcmp', True)
    params.update(tparams)
    params.update(dict(tauto=ParamScalarDesc('tauto')) if tauto else {})
    return params, mappings, tmappings
