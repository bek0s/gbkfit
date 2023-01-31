
import logging

import numpy as np

from gbkfit.math import interpolation
from gbkfit.utils import iterutils, miscutils, parseutils
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


def validate_component_nwmodes_for_geometry(
        loose, tilted, xpos_nwmode, ypos_nwmode, posa_nwmode, incl_nwmode):
    xpos_nwmode = _validate_component_nwmode(  # noqa
        loose, 'loose', xpos_nwmode, 'xpos_nwmode')
    ypos_nwmode = _validate_component_nwmode(  # noqa
        loose, 'loose', ypos_nwmode, 'ypos_nwmode')
    posa_nwmode = _validate_component_nwmode(  # noqa
        tilted, 'tilted', posa_nwmode, 'posa_nwmode')
    incl_nwmode = _validate_component_nwmode(  # noqa
        tilted, 'tilted', incl_nwmode, 'incl_nwmode')
    return dict(
        xpos_nwmode=xpos_nwmode,
        ypos_nwmode=ypos_nwmode,
        posa_nwmode=posa_nwmode,
        incl_nwmode=incl_nwmode)


def validate_component_nwmodes_for_velocity(
        loose, vsys_nwmode):
    vsys_nwmode = _validate_component_nwmode(  # noqa
        loose, 'loose', vsys_nwmode, 'vsys_nwmode')
    return dict(
        vsys_nwmode=vsys_nwmode)


def parse_component_b2d_traits(
        bptraits,
        sptraits,
        wptraits):
    if not bptraits:
        raise RuntimeError("at least one bptrait is required")
    bptraits = iterutils.tuplify(bptraits)
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    return dict(
        bptraits=bptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def parse_component_s2d_traits(
        bptraits,
        vptraits,
        dptraits,
        sptraits,
        wptraits):
    if not bptraits:
        raise RuntimeError("at least one bptrait is required")
    if not vptraits:
        raise RuntimeError("at least one vptrait is required")
    if not dptraits:
        raise RuntimeError("at least one dptrait is required")
    bptraits = iterutils.tuplify(bptraits)
    vptraits = iterutils.tuplify(vptraits)
    dptraits = iterutils.tuplify(dptraits)
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    return dict(
        bptraits=bptraits,
        vptraits=vptraits,
        dptraits=dptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def parse_component_b3d_traits(
        bptraits, bhtraits,
        zptraits,
        sptraits,
        wptraits):
    if not bptraits:
        raise RuntimeError("at least one bptrait is required")
    if not bhtraits:
        raise RuntimeError("at least one bhtrait is required")
    bptraits = iterutils.tuplify(bptraits)
    bhtraits = iterutils.tuplify(bhtraits)
    zptraits = iterutils.tuplify(zptraits) if zptraits is not None else ()
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    bptraits_len = len(bptraits)
    bhtraits_len = len(bhtraits)
    if bptraits_len != bhtraits_len:
        raise RuntimeError(
            f"the number of bhtraits must be equal to "
            f"the number of bptraits ({bhtraits_len} != {bptraits_len})")
    return dict(
        bptraits=bptraits, bhtraits=bhtraits,
        zptraits=zptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def parse_component_s3d_traits(
        bptraits, bhtraits,
        vptraits, vhtraits,
        dptraits, dhtraits,
        zptraits,
        sptraits,
        wptraits):
    if not bptraits:
        raise RuntimeError("at least one bptrait is required")
    if not bhtraits:
        raise RuntimeError("at least one bhtrait is required")
    if not vptraits:
        raise RuntimeError("at least one vptrait is required")
    if not dptraits:
        raise RuntimeError("at least one dptrait is required")
    bptraits = iterutils.tuplify(bptraits)
    bhtraits = iterutils.tuplify(bhtraits)
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
        vhtraits = iterutils.tuplify(iterutils.replace_items_and_copy(
            vhtraits, None, traits.VHTraitOne()))
    if None in dhtraits:
        dhtraits = iterutils.tuplify(iterutils.replace_items_and_copy(
            dhtraits, None, traits.DHTraitOne()))
    bptraits_len = len(bptraits)
    bhtraits_len = len(bhtraits)
    vptraits_len = len(vptraits)
    vhtraits_len = len(vhtraits)
    dptraits_len = len(dptraits)
    dhtraits_len = len(dhtraits)
    if bptraits_len != bhtraits_len:
        raise RuntimeError(
            f"the number of bhtraits must be equal to "
            f"the number of bptraits ({bhtraits_len} != {bptraits_len})")
    if vptraits_len != vhtraits_len:
        raise RuntimeError(
            f"the number of vhtraits must be equal to "
            f"the number of vptraits ({vhtraits_len} != {vptraits_len})")
    if len(dptraits) != len(dhtraits):
        raise RuntimeError(
            f"the number of dhtraits must be equal to "
            f"the number of dptraits ({dhtraits_len} != {dptraits_len})")
    return dict(
        bptraits=bptraits, bhtraits=bhtraits,
        vptraits=vptraits, vhtraits=vhtraits,
        dptraits=dptraits, dhtraits=dhtraits,
        zptraits=zptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def parse_component_o3d_traits(
        optraits, ohtraits,
        zptraits,
        sptraits,
        wptraits):
    if not optraits:
        raise RuntimeError("at least one optrait is required")
    if not ohtraits:
        raise RuntimeError("at least one ohtrait is required")
    optraits = iterutils.tuplify(optraits)
    ohtraits = iterutils.tuplify(ohtraits)
    zptraits = iterutils.tuplify(zptraits) if zptraits is not None else ()
    sptraits = iterutils.tuplify(sptraits) if sptraits is not None else ()
    wptraits = iterutils.tuplify(wptraits) if wptraits is not None else ()
    optraits_len = len(optraits)
    ohtraits_len = len(ohtraits)
    if optraits_len != ohtraits_len:
        raise RuntimeError(
            f"the number of ohtraits must be equal to "
            f"the number of optraits ({ohtraits_len} != {optraits_len})")
    return dict(
        optraits=optraits, ohtraits=ohtraits,
        zptraits=zptraits,
        sptraits=sptraits,
        wptraits=wptraits)


def rename_bx_to_rx_traits(traits_):
    if 'bptraits' in traits_:
        traits_.update(rptraits=traits_.pop('bptraits'))
    if 'bhtraits' in traits_:
        traits_.update(rhtraits=traits_.pop('bhtraits'))
    return traits_


def rename_ox_to_rx_traits(traits_):
    if 'optraits' in traits_:
        traits_.update(rptraits=traits_.pop('optraits'))
    if 'ohtraits' in traits_:
        traits_.update(rhtraits=traits_.pop('ohtraits'))
    return traits_


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


def check_traits_mcdisk(component, traits_):
    unsupported_traits = (
        traits.BPTraitMixtureExponential,
        traits.BPTraitMixtureGauss,
        traits.BPTraitMixtureGGauss,
        traits.BPTraitMixtureMoffat,
        traits.BPTraitNWDistortion,
        traits.DPTraitMixtureExponential,
        traits.DPTraitMixtureGauss,
        traits.DPTraitMixtureGGauss,
        traits.DPTraitMixtureMoffat,
        traits.DPTraitNWDistortion,
        traits.OPTraitMixtureExponential,
        traits.OPTraitMixtureGauss,
        traits.OPTraitMixtureGGauss,
        traits.OPTraitMixtureMoffat,
        traits.OPTraitNWDistortion)
    for trait in traits_:
        if isinstance(trait, unsupported_traits):
            cmp_class = component.__class__
            cmp_label = 'gmodel component'
            cmp_desc = parseutils.make_typed_desc(cmp_class, cmp_label)
            trait_desc = traits.trait_desc(trait.__class__)
            raise NotImplementedError(
                f"{cmp_desc} does not support {trait_desc} yet")


def _make_gmodel_params_cmp(components, prefix, force_prefix):
    return miscutils.merge_dicts_and_make_mappings(
        [cmp.params() for cmp in components], prefix, force_prefix)


def make_gmodel_2d_params(components):
    return _make_gmodel_params_cmp(components, 'cmp', False)


def make_gmodel_3d_params(components, ocomponents):
    params, mappings = _make_gmodel_params_cmp(components, 'cmp', False)
    oparams, omappings = _make_gmodel_params_cmp(ocomponents, 'ocmp', True)
    return params | oparams, mappings, omappings


def is_gmodel_weighted(components):
    return any([bool(cmp.is_weighted()) for cmp in components])


def evaluate_components_b2d(
        components, driver, params, mappings,
        image, wdata, bdata,
        spat_size, spat_step, spat_zero, spat_rota,
        dtype, out_extra, out_extra_label):
    for i, (component, mapping) in enumerate(zip(components, mappings)):
        component_params = {p: params[mapping[p]] for p in component.params()}
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
        component_params = {p: params[mapping[p]] for p in component.params()}
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
        component_params = {p: params[mapping[p]] for p in component.params()}
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
        component_params = {p: params[mapping[p]] for p in component.params()}
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
        component_params = {p: params[mapping[p]] for p in component.params()}
        component_out_extra = {} if out_extra is not None else None
        component.evaluate(
            driver, component_params, odata,
            spat_size, spat_step, spat_zero, spat_rota,
            dtype, component_out_extra)
        if component_out_extra is not None:
            for k, v in component_out_extra.items():
                out_extra[f'{out_extra_label}component{i}_{k}'] = v
