
import logging

import numpy as np

from gbkfit.math import interpolation
from gbkfit.utils import iterutils, parseutils
from gbkfit.utils.parseutils import ConfigError
from . import _disk, traits


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


def _validate_component_nwmode(enabled, enabled_name, nwmode, nwmode_name):
    if nwmode is not None and not enabled:
        _log.warning(
            f"{nwmode_name} is set to '{nwmode.type()}', "
            f"but it will be ignored because {enabled_name} is not set to True")
        # ignore this nwmode
        nwmode = None
    return nwmode


def validate_component_nwmodes(loose, tilted, nwmodes):
    """
    The node-wise modes of the geometric parameters of a component (e.g.
    'xpos'), without those of the parameters that are not node-wise:
    vsys, xpos and ypos if the component is not loose, posa and incl if
    it is not tilted.
    """
    switches = dict(loose=loose, tilted=tilted)
    result = {}
    for name, nwmode in nwmodes.items():
        switch = _disk.NODEWISE_SWITCH[name]
        result[name] = _validate_component_nwmode(
            switches[switch], switch, nwmode, f'{name}_nwmode')
    return result


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


def component_prefixes(components, label, prefix, prefix_first):
    """
    The prefix of the parameters, constants and extra outputs of each
    component of a list: its name, or its position if the components have
    no names (e.g. 'cmp1_'; see parseutils.item_prefixes).
    """
    return parseutils.item_prefixes(
        [cmp.name() for cmp in components], label, prefix, prefix_first)


def select_components(components, names):
    """
    The indices of the components of the given names, in their order (all
    of them if names is None). Raise ConfigError for unknown names.
    """
    if names is None:
        return tuple(range(len(components)))
    known = [component.name() for component in components]
    if not names:
        raise ConfigError("at least one component must be selected")
    if len(set(names)) != len(names):
        raise ConfigError(f"the selected components repeat names: {names}")
    if unknown := [name for name in names if name not in known]:
        raise ConfigError(
            f"there are no components named {unknown}; the named components "
            f"are {[name for name in known if name is not None]}")
    return tuple(sorted(known.index(name) for name in names))


def evaluate_components(
        components, plans, mappings, params, grid, outputs, out_extra,
        out_extra_label, extra):
    """
    Evaluate the components of a gmodel through their plans, each with its
    parameters. Their extra outputs are named after the given label and
    their name, or their index if they have none (e.g. 'opacity_dust_odata'
    or 'opacity_component0_odata'), and made from their data by extra
    (e.g. with the coordinates of the grid).
    """
    for i, (component, plan, mapping) in enumerate(
            zip(components, plans, mappings)):
        name = component.name()
        label = name if name is not None else f'component{i}'
        prefix = f'{out_extra_label}{label}_'
        component_params = {p: params[mapping[p]] for p in component.pdescs()}
        component_out_extra = {} if out_extra is not None else None
        plan.evaluate(component_params, grid, outputs, component_out_extra)
        if component_out_extra is not None:
            for k, v in component_out_extra.items():
                out_extra[f'{prefix}{k}'] = extra(v)
