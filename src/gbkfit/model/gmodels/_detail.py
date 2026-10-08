
import logging

import numpy as np

from gbkfit.math import interpolation
from gbkfit.utils import iterutils, miscutils
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
    result = {}
    for name, nwmode in nwmodes.items():
        enabled_name = 'tilted' if name in ('posa', 'incl') else 'loose'
        enabled = tilted if enabled_name == 'tilted' else loose
        result[name] = _validate_component_nwmode(
            enabled, enabled_name, nwmode, f'{name}_nwmode')
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


def _make_gmodel_params_cmp(components, prefix, force_prefix):
    return miscutils.merge_dicts_and_make_mappings(
        [cmp.pdescs() for cmp in components], prefix, force_prefix)


def make_gmodel_2d_params(components):
    return _make_gmodel_params_cmp(components, 'cmp', False)


def make_gmodel_3d_params(components, ocomponents):
    params, mappings = _make_gmodel_params_cmp(components, 'cmp', False)
    oparams, omappings = _make_gmodel_params_cmp(ocomponents, 'ocmp', True)
    return params | oparams, mappings, omappings


def is_gmodel_weighted(components):
    return any([bool(cmp.has_weights()) for cmp in components])


def evaluate_components_b2d(
        components, driver, params, mappings,
        image, wdata, bdata,
        spat_size, spat_step, spat_zero, spat_rota,
        dtype, out_extra, out_extra_label):
    for i, (component, mapping) in enumerate(zip(components, mappings)):
        component_params = {p: params[mapping[p]] for p in component.pdescs()}
        component_out_extra = {} if out_extra is not None else None
        component.evaluate(
            driver, component_params,
            image, wdata, bdata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, component_out_extra)
        if component_out_extra is not None:
            for k, v in component_out_extra.items():
                out_extra[f'{out_extra_label}component{i}_{k}'] = v


def evaluate_components_b3d(
        components, driver, params, mappings, odata,
        image, wdata, bdata, obdata,
        spat_size, spat_step, spat_zero, spat_rota,
        dtype, out_extra, out_extra_label):
    for i, (component, mapping) in enumerate(zip(components, mappings)):
        component_params = {p: params[mapping[p]] for p in component.pdescs()}
        component_out_extra = {} if out_extra is not None else None
        component.evaluate(
            driver, component_params, odata,
            image, wdata, bdata, obdata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, component_out_extra)
        if component_out_extra is not None:
            for k, v in component_out_extra.items():
                out_extra[f'{out_extra_label}component{i}_{k}'] = v


def evaluate_components_s2d(
        components, driver, params, mappings,
        scube, wdata, bdata,
        spat_size, spat_step, spat_zero, spat_rota,
        spec_size, spec_step, spec_zero,
        dtype, out_extra, out_extra_label):
    for i, (component, mapping) in enumerate(zip(components, mappings)):
        component_params = {p: params[mapping[p]] for p in component.pdescs()}
        component_out_extra = {} if out_extra is not None else None
        component.evaluate(
            driver, component_params,
            scube, wdata, bdata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, component_out_extra)
        if component_out_extra is not None:
            for k, v in component_out_extra.items():
                out_extra[f'{out_extra_label}component{i}_{k}'] = v


def evaluate_components_s3d(
        components, driver, params, mappings, odata,
        scube, wdata, bdata, obdata,
        spat_size, spat_step, spat_zero, spat_rota,
        spec_size, spec_step, spec_zero,
        dtype, out_extra, out_extra_label):
    for i, (component, mapping) in enumerate(zip(components, mappings)):
        component_params = {p: params[mapping[p]] for p in component.pdescs()}
        component_out_extra = {} if out_extra is not None else None
        component.evaluate(
            driver, component_params, odata,
            scube, wdata, bdata, obdata,
            spat_size, spat_step, spat_zero, spat_rota,
            spec_size, spec_step, spec_zero,
            dtype, component_out_extra)
        if component_out_extra is not None:
            for k, v in component_out_extra.items():
                out_extra[f'{out_extra_label}component{i}_{k}'] = v


def evaluate_components_o3d(
        components, driver, params, mappings, odata,
        spat_size, spat_step, spat_zero, spat_rota,
        dtype, out_extra, out_extra_label):
    for i, (component, mapping) in enumerate(zip(components, mappings)):
        component_params = {p: params[mapping[p]] for p in component.pdescs()}
        component_out_extra = {} if out_extra is not None else None
        component.evaluate(
            driver, component_params, odata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, component_out_extra)
        if component_out_extra is not None:
            for k, v in component_out_extra.items():
                out_extra[f'{out_extra_label}component{i}_{k}'] = v
