
import copy
import logging

import numpy as np

from gbkfit.math import interpolation
from gbkfit.params.pdescs import ParamScalarDesc
from gbkfit.utils import iterutils, parseutils
from . import traits


_log = logging.getLogger(__name__)


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
        nodes = np.arange(nmin, nmax + nsep, nsep).tolist()
    elif nodes_linspace:
        nodes = np.linspace(nmin, nmax, nlen).tolist()
    nodes = tuple(nodes)
    if len(nodes) < 2:
        raise RuntimeError(f"at least two {prefix}nodes must be provided")
    if not iterutils.is_ascending(nodes):
        raise RuntimeError(f"{prefix}nodes must be ascending")
    if not iterutils.all_positive(nodes):
        raise RuntimeError(f"{prefix}nodes must be positive")
    if not iterutils.all_unique(nodes):
        raise RuntimeError(f"{prefix}nodes must be unique")
    if step is None:
        step = min(1, min(np.diff(nodes)) / 2)
    if step <= 0 or step > min(np.diff(nodes)) / 2:
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


def parse_nwmode(info, info_key):
    error_header = \
        "could not parse options for node-wise parameter mode (aka nwmode)"
    relative_types = ['relative1', 'relative2']
    allowed_types = ['absolute'] + relative_types
    result = dict()
    # Make sure the nwmode value is a mapping or None
    if not (info is None or iterutils.is_mapping(info)):
        raise RuntimeError(
            f"{error_header}; "
            f"{info_key} (value={info}) must be a mapping or None/null")
    # Use 'absolute' mode by default
    info = dict(type='absolute') if info is None else copy.deepcopy(info)
    # Ensure type is provided
    if 'type' not in info:
        raise RuntimeError(
            f"{error_header}; "
            f"key 'type' is not provided; "
            f"allowed values: {allowed_types}")
    # Create a shortcut of the type, for convenience
    type_ = info.get('type')
    # Ensure a valid type is provided
    if type_ not in allowed_types:
        raise RuntimeError(
            f"{error_header}; "
            f"key 'type' set to an invalid value; "
            f"allowed values: {allowed_types}; "
            f"provided value: '{type_}'")
    # type is now known and final - update result
    result.update(type=type_)
    # If type is relative, validate and extract origin
    if type_ in relative_types:
        if 'origin' not in info:
            raise RuntimeError(
                f"{error_header}; "
                f"key 'origin' is not provided; "
                f"its value must be an integer referencing a radial node; "
                f"negative indexing is supported")
        result.update(origin=info['origin'])
    return result


def _parse_component_nwmode(enabled, enabled_name, nwmode, nwmode_name):
    nwmode = parse_nwmode(nwmode, nwmode_name)
    if not enabled and nwmode['type'] in ['relative1', 'relative2']:
        raise RuntimeError(
            f"when {enabled_name} is False {nwmode_name} must be either "
            f"not set or set to 'absolute'")
    return nwmode


def parse_component_nwmodes_for_geometry(
        loose, tilted, xpos_nwmode, ypos_nwmode, posa_nwmode, incl_nwmode):
    xpos_nwmode = _parse_component_nwmode(
        loose, 'loose', xpos_nwmode, 'xpos_nwmode')
    ypos_nwmode = _parse_component_nwmode(
        loose, 'loose', ypos_nwmode, 'ypos_nwmode')
    posa_nwmode = _parse_component_nwmode(
        tilted, 'tilted', posa_nwmode, 'posa_nwmode')
    incl_nwmode = _parse_component_nwmode(
        tilted, 'tilted', incl_nwmode, 'incl_nwmode')
    return dict(
        xpos_nwmode=xpos_nwmode,
        ypos_nwmode=ypos_nwmode,
        posa_nwmode=posa_nwmode,
        incl_nwmode=incl_nwmode)


def parse_component_nwmodes_for_velocity(
        loose, vsys_nwmode):
    vsys_nwmode = _parse_component_nwmode(
        loose, 'loose', vsys_nwmode, 'vsys_nwmode')
    return dict(
        vsys_nwmode=vsys_nwmode)


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


def check_traits_common(traits_):
    for trait in traits_:
        trait_desc = traits.trait_desc(trait.__class__)
        if isinstance(trait, traits.RPTraitUniform):
            _log.warning(
                f"the use of {trait_desc} is discouraged; "
                f"its main purpose is to facilitate software testing")
        if isinstance(trait, traits.RHTraitUniform):
            _log.warning(
                f"the use of {trait_desc} is discouraged; "
                f"it may result in density overestimation due to aliasing")


def check_traits_mcdisk(component, traits_):
    # mcdisk components do not support mixture traits yet
    unsupported_traits = (
        traits.RPTraitMixtureExponential,
        traits.RPTraitMixtureGauss,
        traits.RPTraitMixtureGGauss,
        traits.RPTraitMixtureMoffat,
        traits.DPTraitMixtureExponential,
        traits.DPTraitMixtureGauss,
        traits.DPTraitMixtureGGauss,
        traits.DPTraitMixtureMoffat)
    for trait in traits_:
        if isinstance(trait, unsupported_traits):
            cmp_class = component.__class__
            cmp_label = 'gmodel component'
            cmp_desc = parseutils.make_typed_desc(cmp_class, cmp_label)
            trait_desc = traits.trait_desc(trait.__class__)
            raise RuntimeError(f"{cmp_desc} does not support {trait_desc} yet")


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
