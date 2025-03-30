import copy
import json
import logging
import os
from typing import Any, Literal

import astropy.io.fits as fits
import numpy as np
import pandas as pd
import ruamel.yaml

import gbkfit.dataset
import gbkfit.driver
import gbkfit.model
import gbkfit.objective
import gbkfit.params
from gbkfit.params import ParamDesc
from gbkfit.utils import iterutils, timeutils
from . import _detail


_log = logging.getLogger(__name__)


# Use this object to load and dump yaml
yaml = ruamel.yaml.YAML()

# This is needed for dumping dicts with correct order
ruamel.yaml.add_representer(dict, lambda self, data: self.represent_mapping(
    'tag:yaml.org,2002:map', data.items()))


def _prepare_params(
        info: dict[str, Any],
        pdescs: dict[str, ParamDesc]
) -> dict[str, Any]:
    """
    Prepares parameter information for model evaluation.

    This function ensures that parameter keys are valid, expands any
    parameters that require explosion, and converts parameter
    configurations originally intended for model fitting into a format
    suitable for model evaluation. This allows the same JSON/YAML file
    to be used in both modes without requiring manual modifications.

    In model fitting mode, parameters may include metadata such as
    "min", "max", and "value" (where "value" represents the initial
    fitting value). In model evaluation mode, only a numeric value or
    a string reference to another variable is needed. This function
    extracts and converts the necessary values accordingly.

    Parameters
    ----------
    info : dict[str, Any]
        The parameter configuration dictionary.
    pdescs : dict[str, ParamDesc]
        Parameter descriptors used for validation and expansion.

    Returns
    -------
    dict[str, Any]
        A dictionary with processed parameters, ready for model evaluation.
    """

    # Copy info for safety
    info = copy.deepcopy(info)

    # Prepare the supplied parameter properties. This will:
    # - Ensure the parameter keys are valid
    # - Explode all parameter names that are can be exploded
    parameters = gbkfit.params.parse_param_info(
        info.get('properties'), pdescs).info

    recovery_failed = []
    recovery_succeed = {}

    def recover_value(
            value_dict: dict[str, Any], key_: str, index_: int | str | None
    ) -> int | float | str | None:
        value = value_dict.get('value')
        # Create a value id to facilitate logging
        value_id = key if index_ is None else (key_, index_)
        # We use the value of key 'value' as the parameter value
        if 'value' in value_dict:
            recovery_succeed[value_id] = value
        else:
            recovery_failed.append(value_id)
        return value

    # Try to recover the parameter values from all dicts.
    # We are only interested in dicts and lists of dicts.
    # Everything else can't possibly be recovered or need recovering.
    for key, val in parameters.items():
        if iterutils.is_mapping(val):
            parameters[key] = recover_value(val, key, None)
        elif iterutils.is_sequence(val):
            parameters[key] = [recover_value(ival, key, i)
                               if iterutils.is_mapping(ival) else ival
                               for i, ival in enumerate(val)]

    # Report successful recoveries
    if recovery_succeed:
        _log.info(
            f"successfully recovered values "
            f"for {len(recovery_succeed)} parameter keys: {recovery_succeed}")

    # Check for errors (failed recoveries)
    if recovery_failed:
        raise RuntimeError(
            f"failed to recover values "
            f"for {len(recovery_failed)} parameter keys: {recovery_failed}")

    # Update parameter info and return it
    return info | dict(parameters=parameters)


def eval_(
        mode: Literal['model', 'objective'],
        config: str,
        profile_iters: int,
        output_dir: str,
        output_dir_mode: Literal['terminate', 'overwrite', 'unique']):

    #
    # Read configuration file and
    # perform all necessary validation/preparation
    #

    config = os.path.abspath(config)
    _log.info(f"reading configuration from file: {config}...")

    try:
        cfg = yaml.load(open(config))
    except Exception as e:
        raise RuntimeError(
            f"error while reading configuration file {config}; "
            f"see reason below:\n{e}") from e

    _log.info("preparing configuration...")
    # This is not a full-fledged validation. It just tries to catch
    # and inform the user about the really obvious mistakes.
    required_sections = ('models', 'params')
    optional_sections = ('pdescs',)
    if mode == 'model':
        optional_sections += ('datasets',)
    elif mode == 'objective':
        required_sections += ('datasets',)
        optional_sections += ('objective',)
    else:
        raise RuntimeError("impossible")
    cfg = _detail.prepare_config(cfg, required_sections, optional_sections)

    #
    # Ensure an output directory is available for the outputs
    #

    _log.info("preparing output directory...")
    output_dir = _detail.make_output_dir(output_dir, output_dir_mode)
    _log.info(f"output will be stored under directory: {output_dir}")

    #
    # Setup all the components described in the configuration.
    # After running the configuration through _detail.prepare_config():
    # - datasets and models configurations are lists
    # - objective, pdescs, and params configurations are dicts
    #

    datasets = None
    if 'datasets' in cfg:
        _log.info("setting up datasets...")
        datasets = gbkfit.dataset.dataset_parser.load(cfg['datasets'])

    _log.info("setting up models...")
    models = gbkfit.model.model_parser.load(cfg['models'], dataset=datasets)
    model_group = gbkfit.model.ModelGroup(models)

    objective = None
    if mode == 'objective':
        _log.info("setting up objective...")
        objective = gbkfit.objective.objective_parser.load(
            cfg.get('objective', {}), datasets=datasets, models=model_group)

    _log.info("setting up pdescs...")
    pdescs = objective.pdescs() \
        if objective is not None else model_group.pdescs()
    if 'pdescs' in cfg:
        user_pdescs = gbkfit.params.load_pdescs_dict(cfg['pdescs'])
        pdescs = _detail.merge_pdescs(pdescs, user_pdescs)

    _log.info("setting up params...")
    cfg['params'] = _prepare_params(cfg['params'], pdescs)
    params = gbkfit.params.evaluation_params_parser.load(
        cfg['params'], pdescs=pdescs)

    #
    # Calculate model parameters
    #

    _log.info("calculating model parameters...")

    exploded_param_values = {}
    param_values = params.evaluate(out_exploded_params=exploded_param_values)
    params_info = iterutils.nativify(dict(
        params=param_values,
        eparams=exploded_param_values))
    filename = os.path.join(output_dir, 'gbkfit_eval_params')
    _detail.dump_dict(json, yaml, params_info, filename)

    #
    # Evaluate objective
    #

    # Always evaluate model
    model_extra = {}
    model_data = []
    if mode == 'model':
        print(params)
        model_data = model_group.model_h(param_values, model_extra)

    resid_u_extra = {}
    resid_u_data = []
    resid_w_extra = {}
    resid_w_data = []
    if mode == 'objective':
        resid_u_data = objective.residual_nddata_h(param_values, resid_u_extra)
        resid_w_data = []  # objective.residual_nddata_h(params, True, resid_w_extra)
        foo = objective.residual_scalar(param_values, True)
        print(params)
        print("residual:", foo)

    #
    # Gather objective outputs
    #

    _log.info("gathering outputs...")

    outputs = {}
    model_prefix = 'model'
    resid_u_prefix = 'residual'
    resid_w_prefix = 'wresidual'

    # Store model
    for i, data_i in enumerate(model_data):
        # prefix_i = model_prefix + f'_{i}' * bool(model.nitems() > 0)
        prefix_i = model_prefix + f'_{i}'
        for key, value in data_i.items():
            outputs |= {
                f'{prefix_i}_{key}_d.fits': value.get('d'),
                f'{prefix_i}_{key}_m.fits': value.get('m'),
                f'{prefix_i}_{key}_w.fits': value.get('w')}
    # Store residual (if available)
    for i, data_i in enumerate(resid_u_data):
        prefix_i = resid_u_prefix + f'_{i}' * bool(objective.nitems() > 1)
        for key, value in data_i.items():
            outputs |= {f'{prefix_i}_{key}_d.fits': value}
    for i, data_i in enumerate(resid_w_data):
        prefix_i = resid_w_prefix + f'_{i}' * bool(objective.nitems() > 1)
        for key, value in data_i.items():
            outputs |= {f'{prefix_i}_{key}_d.fits': value}
    # Store model extra
    for key, value in model_extra.items():
        outputs |= {f'{model_prefix}_extra_{key}.fits': value}
    # Store residual extra (if available)
    for key, value in resid_u_extra.items():
        outputs |= {f'{resid_u_prefix}_extra_{key}.fits': value}
    for key, value in resid_w_extra.items():
        outputs |= {f'{resid_u_prefix}_extra_{key}.fits': value}

    # #
    # # Calculate outputs statistics
    # #
    #
    # _log.info("calculating statistics for outputs...")
    #
    # outputs_stats = {}
    # for filename, data in outputs.items():
    #     if data is not None:
    #         sum_ = np.nansum(data)
    #         min_ = np.nanmin(data)
    #         max_ = np.nanmax(data)
    #         mean = np.nanmean(data)
    #         stddev = np.nanstd(data)
    #         median = np.nanmedian(data)
    #         outputs_stats.update({filename: dict(
    #             sum=sum_, min=min_, max=max_, mean=mean, stddev=stddev,
    #             median=median)})
    #
    # filename = os.path.join(output_dir, 'gbkfit_eval_outputs')
    # outputs_stats = iterutils.nativify(outputs_stats)
    # _detail.dump_dict(json, yaml, outputs_stats, filename)

    #
    # Store outputs
    #

    _log.info("storing outputs to the filesystem...")

    for filename, data in outputs.items():
        if isinstance(data, np.ndarray):
            hdu = fits.PrimaryHDU(data)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(filename, overwrite=True)

    #
    # Run performance tests
    #

    if profile_iters > 0:
        _log.info("running performance test...")
        for i in range(profile_iters):
            if mode == 'model':
                model.model_d(params)
            if mode == 'objective':
                objective.log_likelihood(params)
                objective.residual_scalar(params, squared=True)
        _log.info("calculating timing statistics...")
        time_stats = iterutils.nativify(timeutils.get_time_stats())
        _log.info(pd.DataFrame.from_dict(time_stats, orient='index'))
        filename = os.path.join(output_dir, 'gbkfit_eval_timings')
        _detail.dump_dict(json, yaml, time_stats, filename)
